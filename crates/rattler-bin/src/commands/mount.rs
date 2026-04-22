//! `rattler mount` — mount a lockfile as a virtual conda environment.
//!
//! Wraps the [`rattler_fs`] library API. `rattler_fs` itself is now a thin
//! VFS primitive with no knowledge of lock files or package caches — the
//! orchestration (parse lockfile, fetch packages, derive python info, build
//! `PackageSource`s) lives here in the CLI so the library can stay focused.

use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use miette::{IntoDiagnostic, Result};
use rattler::install::PythonInfo;
use rattler_cache::{default_cache_dir, package_cache::PackageCache};
use rattler_conda_types::Platform;
use rattler_fs::{
    build_and_mount,
    package_source::{CondaPackage, PackageSource},
    Layout, MountConfig, Transport, VirtualFile,
};
use rattler_lock::{LockFile, DEFAULT_ENVIRONMENT_NAME};

/// Mount a lockfile as a virtual conda environment.
#[derive(Debug, Parser)]
pub struct Opt {
    /// Lock file to mount (e.g. `pixi.lock`).
    pub lock_file: PathBuf,

    /// Where the virtual environment should appear.
    pub mount_point: PathBuf,

    /// Environment name in the lock file.
    #[clap(long, default_value = DEFAULT_ENVIRONMENT_NAME)]
    pub environment: String,

    /// Persistent writable overlay directory. If omitted, the mount is read-only.
    /// Required for ProjFS (which is always writable).
    #[clap(long)]
    pub overlay: Option<PathBuf>,

    /// Transport backend. Defaults to the platform's preferred transport.
    #[clap(long, value_enum, default_value_t = TransportArg::Auto)]
    pub transport: TransportArg,

    /// Allow other users to access the mount (FUSE only; requires
    /// `user_allow_other` in `/etc/fuse.conf`).
    #[clap(long)]
    pub allow_other: bool,
}

/// Transport selector exposed on the CLI. Mirrors [`rattler_fs::Transport`]
/// but with snake-case clap-friendly variants.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TransportArg {
    /// Platform default: FUSE on Linux, NFS on macOS, ProjFS on Windows.
    Auto,
    /// FUSE via libfuse3 (Linux) or macFUSE (macOS).
    Fuse,
    /// NFSv3 userspace server on localhost.
    Nfs,
    /// Windows Projected File System.
    Projfs,
}

impl From<TransportArg> for Transport {
    fn from(t: TransportArg) -> Self {
        match t {
            TransportArg::Auto => Transport::Auto,
            TransportArg::Fuse => Transport::Fuse,
            TransportArg::Nfs => Transport::Nfs,
            TransportArg::Projfs => Transport::ProjFs,
        }
    }
}

/// Convert any `anyhow::Error` (used by rattler_fs) into a miette diagnostic.
fn anyhow_to_miette(e: anyhow::Error) -> miette::Report {
    miette::miette!("{e}")
}

pub async fn mount(opt: Opt) -> Result<()> {
    let lockfile = LockFile::from_path(&opt.lock_file).into_diagnostic()?;
    let platform = Platform::current();
    let environment = lockfile
        .environment(&opt.environment)
        .ok_or_else(|| miette::miette!("environment '{}' not found in lock file", opt.environment))?;
    let env_hash = environment
        .content_hash(platform)
        .ok_or_else(|| miette::miette!("platform '{platform}' not in environment '{}'", opt.environment))?;

    let cache_dir = default_cache_dir().map_err(anyhow_to_miette)?.join("pkgs");
    let package_cache = PackageCache::new(cache_dir);

    let mount_point = opt.mount_point.canonicalize().into_diagnostic()?;
    let transport: Transport = opt.transport.into();

    // Fetch every package in parallel, then build CondaPackage sources from
    // the extracted cache dirs. The work used to live inside rattler_fs; it
    // belongs up here now that the library takes a list of `PackageSource`s.
    let packages = fetch_conda_packages(&lockfile, &opt.environment, platform, &package_cache)
        .await
        .map_err(anyhow_to_miette)?;

    let layout = Layout::new()
        .with_packages(packages)
        .with_virtual_files(vec![VirtualFile::new(
            "conda-meta/rattler-fs_env",
            env_hash.clone().into_bytes(),
        )]);

    let config = match opt.overlay {
        Some(overlay) => {
            MountConfig::new_writable(mount_point.clone(), Some(overlay), transport, env_hash)
        }
        None => MountConfig::new_read_only(mount_point.clone(), transport, env_hash),
    };
    let config = config.with_allow_other(opt.allow_other);

    let handle = build_and_mount(&layout, &config)
        .await
        .map_err(anyhow_to_miette)?;

    eprintln!(
        "mounted {} at {}",
        opt.lock_file.display(),
        mount_point.display()
    );

    // Wait for ctrl-c (or sigterm on Unix) then explicitly unmount so any
    // unmount failures surface as exit-1 instead of being swallowed by Drop.
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .into_diagnostic()?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = sigterm.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.into_diagnostic()?;

    handle.unmount().await.map_err(anyhow_to_miette)?;
    Ok(())
}

/// Resolve every conda package in `lockfile[environment][platform]`, fetch it
/// into the package cache (parallel, semaphore-bounded), then construct a
/// `CondaPackage` source for each. Moved out of `rattler_fs` so the library
/// no longer depends on `rattler_lock`/`rattler_cache`.
async fn fetch_conda_packages(
    lockfile: &LockFile,
    environment_name: &str,
    platform: Platform,
    package_cache: &PackageCache,
) -> anyhow::Result<Vec<Box<dyn PackageSource>>> {
    let environment = lockfile
        .environment(environment_name)
        .ok_or_else(|| anyhow::anyhow!("environment '{environment_name}' not found"))?;
    let lock_platform = lockfile
        .platform(&platform.to_string())
        .ok_or_else(|| anyhow::anyhow!("platform '{platform}' not found"))?;
    let package_refs: Vec<_> = environment
        .packages(lock_platform)
        .ok_or_else(|| {
            anyhow::anyhow!("no packages for platform {platform} in environment '{environment_name}'")
        })?
        .collect();

    let python_info = package_refs
        .iter()
        .filter_map(|p| p.as_binary_conda())
        .find(|p| p.package_record.name.as_normalized() == "python")
        .map(|p| PythonInfo::from_python_record(&p.package_record, platform))
        .transpose()
        .map_err(|e| anyhow::anyhow!("failed to get python info: {e}"))?;

    let mut conda_packages: Vec<_> = package_refs
        .iter()
        .filter_map(|p| p.as_binary_conda())
        .collect();

    // Largest first so long downloads start early (mirrors rattler's installer
    // pattern at installer/mod.rs:600).
    conda_packages.sort_by(|a, b| {
        b.package_record
            .size
            .unwrap_or(0)
            .cmp(&a.package_record.size.unwrap_or(0))
    });

    // Lazy HTTP client: on macOS `ClientWithMiddleware::default()` walks the
    // keychain, which takes seconds. Deferring it means warm-cache mounts
    // skip that work entirely.
    let client = rattler_networking::LazyClient::new(
        reqwest_middleware::ClientWithMiddleware::default,
    );
    let concurrency = Arc::new(tokio::sync::Semaphore::new(16));
    let mut join_set = tokio::task::JoinSet::new();

    for package_data in &conda_packages {
        let cache = package_cache.clone();
        let client = client.clone();
        let record = package_data.package_record.clone();
        let location = package_data.location.clone();
        let is_noarch_python = package_data.package_record.noarch.is_python();
        let name = record.name.as_normalized().to_string();
        let py = python_info.clone();
        let sem = concurrency.clone();

        join_set.spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("concurrency semaphore closed: {e}"))?;
            let url = location
                .as_url()
                .ok_or_else(|| anyhow::anyhow!("package has no URL"))?
                .clone();
            let cache_metadata = cache
                .get_or_fetch_from_url_with_retry(
                    &record,
                    url,
                    client,
                    rattler_networking::retry_policies::default_retry_policy(),
                    None,
                )
                .await?;

            // Parse paths.json / link.json off the main runtime thread.
            let extracted_path = cache_metadata.path().to_path_buf();
            let python_for_pkg = if is_noarch_python { py } else { None };
            let pkg: Box<dyn PackageSource> = Box::new(CondaPackage::from_extracted(
                name,
                &extracted_path,
                python_for_pkg,
            )?);
            Ok::<_, anyhow::Error>(pkg)
        });
    }

    let mut out: Vec<Box<dyn PackageSource>> = Vec::with_capacity(conda_packages.len());
    while let Some(result) = join_set.join_next().await {
        out.push(result.map_err(|e| anyhow::anyhow!("fetch task failed: {e}"))??);
    }
    Ok(out)
}
