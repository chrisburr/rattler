//! Abstraction over anything that contributes files to a mounted environment.
//!
//! Each implementor of [`PackageSource`] emits a collection of [`PackageFile`]s
//! describing what it wants mounted, without knowing anything about the other
//! packages in the environment or how the VFS itself works. The tree-builder
//! in [`crate::build_metadata_tree`] iterates sources, collects files, and
//! wires them into a single [`crate::MetadataTree`].
//!
//! Today the one built-in implementor is [`CondaPackage`]. PyPI wheel support
//! will add more impls without changes to the core tree-builder.
//!
//! # Design: why owned values?
//!
//! [`PackageSource::files`] returns `Vec<PackageFile>` rather than an iterator
//! to keep the API simple: tree-building is a one-shot mount-time operation,
//! allocations here are swamped by package-cache I/O and won't appear on the
//! read hot path. If profiling later shows this matters, an iterator variant
//! can be added non-breakingly.
//!
//! [`PackageFile`] owns its `PathBuf` for the same reason.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use rattler::install::{python_entry_point_template, PythonInfo};
use rattler_conda_types::package::{
    EntryPoint, LinkJson, NoArchLinks, PackageFile as _, PathType, PathsJson,
};

use crate::metadata_tree::{ContentTransform, DEFAULT_FILE_MODE, EXECUTABLE_FILE_MODE};

/// A single file that a [`PackageSource`] wants mounted.
#[derive(Debug)]
pub struct PackageFile {
    /// Path relative to the mount root (e.g.
    /// `lib/python3.11/site-packages/foo/__init__.py`). May include leading
    /// path components that don't yet exist in the tree — the tree-builder
    /// creates intermediate directories on demand.
    pub relative_path: PathBuf,
    pub content: FileContent,
    /// POSIX-style mode bits for the file.
    pub mode: u32,
    pub path_type: PathType,
}

/// How the bytes of a [`PackageFile`] are produced.
#[derive(Debug)]
pub enum FileContent {
    /// Bytes served lazily from an on-disk file, optionally transformed on read.
    CachedBytes {
        /// Root of the extracted package cache directory.
        cache_path: Arc<Path>,
        /// Override for the cache-side path prefix when the virtual path differs
        /// from the on-disk layout. Used for noarch python packages where e.g.
        /// `python-scripts/foo` on disk is exposed as `bin/foo` in the mount.
        cache_prefix: Option<PathBuf>,
        /// Optional byte-level transform (e.g. prefix replacement).
        transform: Option<ContentTransform>,
    },
    /// Bytes materialized at tree-build time.
    Inline(Vec<u8>),
}

/// Anything that contributes files to a mounted environment.
pub trait PackageSource: Send + Sync {
    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str;

    /// All files this package contributes to the environment.
    ///
    /// `mount_point` is the absolute path the environment will be mounted at,
    /// needed for things like entry-point shebangs that bake in the python
    /// interpreter's final location.
    fn files(&self, mount_point: &Path) -> anyhow::Result<Vec<PackageFile>>;
}

/// A conda package already extracted into the package cache.
///
/// Reads `paths.json` (and `link.json` for noarch-python packages with entry
/// points) at construction; `files()` just walks the pre-parsed metadata.
pub struct CondaPackage {
    name: String,
    extracted_path: Arc<Path>,
    paths_json: PathsJson,
    entry_points: Vec<EntryPoint>,
    /// `Some` iff this is a noarch-python package that needs `site-packages/`
    /// and `python-scripts/` path rewriting to match the target Python layout.
    python_info: Option<PythonInfo>,
}

impl CondaPackage {
    /// Build a `CondaPackage` from an extracted conda package directory.
    ///
    /// `python_info` should be `Some` iff this package is noarch-python AND the
    /// target environment has a Python interpreter — in that case paths are
    /// rewritten (`site-packages/` → `lib/pythonX.Y/site-packages/`,
    /// `python-scripts/` → `bin/`) and entry points are read from `link.json`.
    pub fn from_extracted(
        name: impl Into<String>,
        extracted_path: &Path,
        python_info: Option<PythonInfo>,
    ) -> anyhow::Result<Self> {
        let paths_json =
            PathsJson::from_package_directory_with_deprecated_fallback(extracted_path)?;
        let entry_points = if python_info.is_some() {
            LinkJson::from_package_directory(extracted_path)
                .ok()
                .and_then(|lj| match lj.noarch {
                    NoArchLinks::Python(ep) => Some(ep.entry_points),
                    NoArchLinks::Generic => None,
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(Self::from_parts(
            name,
            extracted_path,
            paths_json,
            entry_points,
            python_info,
        ))
    }

    /// Construct a `CondaPackage` from already-parsed parts without touching
    /// the filesystem. Intended for tests and for callers that have cached
    /// `paths.json` / `link.json` elsewhere.
    pub fn from_parts(
        name: impl Into<String>,
        extracted_path: &Path,
        paths_json: PathsJson,
        entry_points: Vec<EntryPoint>,
        python_info: Option<PythonInfo>,
    ) -> Self {
        Self {
            name: name.into(),
            extracted_path: Arc::from(extracted_path),
            paths_json,
            entry_points,
            python_info,
        }
    }
}

impl PackageSource for CondaPackage {
    fn name(&self) -> &str {
        &self.name
    }

    fn files(&self, mount_point: &Path) -> anyhow::Result<Vec<PackageFile>> {
        let mut out =
            Vec::with_capacity(self.paths_json.paths.len() + self.entry_points.len());

        for path in &self.paths_json.paths {
            // Noarch-python path rewriting: `site-packages/*` →
            // `lib/pythonX.Y/site-packages/*`, `python-scripts/*` → `bin/*`.
            // When the path is rewritten, `cache_prefix` remembers the original
            // on-disk parent so `_getpath` can still find the source bytes.
            let (virtual_path, cache_prefix) = match &self.python_info {
                Some(info) => {
                    let rewritten = info.get_python_noarch_target_path(&path.relative_path);
                    if rewritten.as_ref() == path.relative_path {
                        (path.relative_path.clone(), None)
                    } else {
                        let original_parent = path
                            .relative_path
                            .parent()
                            .map_or_else(|| PathBuf::from("."), |p| PathBuf::from(".").join(p));
                        (rewritten.into_owned(), Some(original_parent))
                    }
                }
                None => (path.relative_path.clone(), None),
            };

            out.push(PackageFile {
                relative_path: virtual_path,
                content: FileContent::CachedBytes {
                    cache_path: self.extracted_path.clone(),
                    cache_prefix,
                    transform: path
                        .prefix_placeholder
                        .clone()
                        .map(ContentTransform::PrefixReplace),
                },
                mode: DEFAULT_FILE_MODE,
                path_type: path.path_type,
            });
        }

        // Entry-point scripts (only exist for noarch-python packages with link.json).
        if let Some(python_info) = &self.python_info {
            let target_prefix = mount_point.to_string_lossy();
            for ep in &self.entry_points {
                let content =
                    python_entry_point_template(&target_prefix, false, ep, python_info);
                out.push(PackageFile {
                    relative_path: PathBuf::from("bin").join(ep.command.as_str()),
                    content: FileContent::Inline(content.into_bytes()),
                    mode: EXECUTABLE_FILE_MODE,
                    path_type: PathType::HardLink,
                });
            }
        }

        Ok(out)
    }
}
