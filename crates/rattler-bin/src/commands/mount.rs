//! `rattler mount` — mount a lockfile as a virtual conda environment.
//!
//! Wraps the [`rattler_fs`] library API. The library is the supported
//! integration point for downstream consumers (pixi); this subcommand exists
//! as a developer convenience for mounting a lockfile from the shell and as
//! the canonical CLI surface for the rattler binary.

use std::path::PathBuf;

use clap::Parser;
use miette::{IntoDiagnostic, Result};
use rattler_cache::{default_cache_dir, package_cache::PackageCache};
use rattler_conda_types::Platform;
use rattler_fs::{build_and_mount, MountConfig, Transport};
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

    let config = match opt.overlay {
        Some(overlay) => {
            MountConfig::new_writable(mount_point.clone(), Some(overlay), transport, env_hash)
        }
        None => MountConfig::new_read_only(mount_point.clone(), transport, env_hash),
    };
    let config = config.with_allow_other(opt.allow_other);

    let handle = build_and_mount(
        &lockfile,
        &opt.environment,
        platform,
        &package_cache,
        &config,
    )
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
