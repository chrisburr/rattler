#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
pub mod package_cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod run_exports_cache;

#[cfg(not(target_arch = "wasm32"))]
pub mod validation;

mod consts;
pub use consts::{PACKAGE_CACHE_DIR, REPODATA_CACHE_DIR, RUN_EXPORTS_CACHE_DIR};

/// Determines the default cache directories for rattler.
///
/// The environment variable `RATTLER_CACHE_DIR` is consulted first. Its value
/// is split using the platform's `PATH` separator (`:` on Unix, `;` on
/// Windows), allowing multiple cache roots to be configured. The first entry
/// is the primary (writable) cache; subsequent entries act as additional
/// lookup layers and are only used as writable fallbacks if earlier layers
/// are read-only.
///
/// If `RATTLER_CACHE_DIR` is unset, falls back to a single default at
/// `dirs::cache_dir()/rattler/cache`.
///
/// Only the package cache is layered. Other caches (repodata, run-exports,
/// etc.) use only the first entry.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_cache_dirs() -> anyhow::Result<Vec<PathBuf>> {
    if let Ok(value) = std::env::var("RATTLER_CACHE_DIR") {
        let dirs: Vec<PathBuf> = std::env::split_paths(&value)
            .filter(|p| !p.as_os_str().is_empty())
            .collect();
        if dirs.is_empty() {
            anyhow::bail!("RATTLER_CACHE_DIR is set but contains no valid paths");
        }
        return Ok(dirs);
    }

    let dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine cache directory for current platform"))
        .map(|mut p| {
            p.push("rattler");
            p.push("cache");
            p
        })?;
    Ok(vec![dir])
}

/// Determines the default cache directory for rattler.
///
/// Returns the primary (first) entry from [`default_cache_dirs`]. Use
/// [`default_cache_dirs`] when constructing a layered package cache.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_cache_dir() -> anyhow::Result<PathBuf> {
    Ok(default_cache_dirs()?
        .into_iter()
        .next()
        .expect("default_cache_dirs returns a non-empty Vec"))
}

/// Creates the cache directory if it doesn't exist and excludes it from backups.
///
/// This function:
/// 1. Creates the directory and all parent directories if they don't exist
/// 2. Creates a `CACHEDIR.TAG` file to exclude from backup tools (borg, restic, etc.)
/// 3. On macOS, marks the directory as excluded from Time Machine
///
/// This is idempotent - calling it multiple times on the same directory is safe.
#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_cache_dir(path: &Path) -> std::io::Result<()> {
    fs_err::create_dir_all(path)?;
    rattler_conda_types::backup::exclude_from_backups(path)?;
    Ok(())
}
