//! Virtual conda environment mounts.
//!
//! `rattler_fs` presents a conda environment as a virtual filesystem, serving
//! files directly from the package cache with on-the-fly prefix replacement.
//! No files are copied to disk for read-only use; a persistent copy-on-write
//! overlay enables writes (e.g. `pip install`) without modifying the cache.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use rattler_fs::{Layout, MountConfig, Transport, VirtualFile, build_and_mount};
//! use rattler_fs::package_source::{CondaPackage, PackageSource};
//! # async fn example() -> anyhow::Result<()> {
//! // Caller is responsible for fetching packages and providing extracted paths.
//! let layout = Layout::new().with_packages(vec![
//!     Box::new(CondaPackage::from_extracted("foo", "/cache/foo-1.0".as_ref(), None)?)
//!         as Box<dyn PackageSource>,
//! ]);
//!
//! let config = MountConfig::new_read_only(
//!     PathBuf::from("/path/to/env"),
//!     Transport::Auto,
//!     "env-hash".to_string(),
//! );
//! let handle = build_and_mount(&layout, &config).await?;
//! // Environment is live at /path/to/env.
//! // Dropping `handle` unmounts; call `handle.unmount().await` for explicit error handling.
//! # Ok(())
//! # }
//! ```
//!
//! # Platform support
//!
//! | Platform | Default backend | Available |
//! |----------|-----------------|-----------|
//! | Linux | FUSE | FUSE, NFS |
//! | macOS | NFS | NFS, FUSE (requires [macFUSE]) |
//! | Windows | [ProjFS] | `ProjFS` |
//!
//! [`Transport::Auto`] selects the default for the current platform.
//!
//! **Why NFS on macOS?** FUSE on macOS requires [macFUSE], a third-party
//! kernel extension that needs System Integrity Protection (SIP) to be
//! reduced on Apple Silicon. Additionally, FUSE mounts lose all kernel vnode
//! code-signature caches on unmount, causing a significant Gatekeeper
//! re-verification penalty on every remount. The NFS transport uses macOS's
//! built-in NFS client, avoiding both issues.
//!
//! **Why FUSE on Linux?** FUSE has lower overhead than NFS on Linux (no TCP
//! stack, no marshalling) and supports kernel-level page caching and passthrough
//! I/O. The NFS backend is available as a fallback.
//!
//! **Why `ProjFS` on Windows?** [ProjFS] is built into Windows 10+ and mounts to
//! any directory without elevation. Its demand-driven callback model
//! ("materialize files when accessed") maps naturally onto virtual environments.
//! NFS is not supported as a transport on Windows due to client limitations
//! (portmapper requirements, drive-letter-only mounts, `NFSv2` fallback).
//!
//! **macOS alternatives under investigation:** [FSKit] is Apple's modern
//! successor to kernel extensions for filesystems, but current known
//! implementations target block-style storage rather than projected/virtual
//! filesystems.
//!
//! [macFUSE]: https://osxfuse.github.io/
//! [ProjFS]: https://learn.microsoft.com/en-us/windows/win32/projfs/projected-file-system
//! [FSKit]: https://developer.apple.com/documentation/fskit

#[cfg(target_os = "macos")]
pub mod codesign;
#[cfg(any(target_os = "linux", feature = "fuse"))]
pub mod fuse_adapter;
pub(crate) mod metadata_tree;
#[cfg(feature = "nfs")]
pub mod nfs_adapter;
pub mod overlay;
pub mod overlay_fs;
pub mod package_source;
pub mod prefix_replacement;
#[cfg(target_os = "windows")]
pub mod projfs_adapter;
pub mod vfs_ops;
pub mod virtual_fs;

use std::{
    collections::HashMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

use metadata_tree::{MetadataNode, DEFAULT_FILE_MODE};
use package_source::{FileContent, PackageFile, PackageSource};
use rattler_conda_types::package::PathType;
use virtual_fs::VirtualFS;

pub use package_source::CondaPackage;

// ---------------------------------------------------------------------------
// Structured errors for downstream consumers (pixi)
// ---------------------------------------------------------------------------

/// Errors from `rattler_fs` that downstream consumers can match on.
///
/// Most `rattler_fs` functions return `anyhow::Result` for convenience, with
/// these variants as the underlying cause when a structured match is needed.
/// Use `anyhow::Error::downcast_ref::<MountError>()` to extract them.
#[derive(Debug, thiserror::Error)]
pub enum MountError {
    /// The `ProjFS` optional Windows feature is not enabled.
    #[error(
        "Windows Projected File System (ProjFS) is not available.\n\
         Enable it with (requires Administrator):\n\n  \
         Enable-WindowsOptionalFeature -Online -FeatureName Client-ProjFS -NoRestart\n"
    )]
    ProjFsDllMissing,

    /// `ProjFS` does not support read-only mode.
    #[error(
        "ProjFS does not support read-only mode: it lacks a pre-creation \
         notification, so new files can always be created. Use Mode::Writable \
         instead (overlay_dir is ignored on ProjFS)."
    )]
    ProjFsReadOnlyUnsupported,

    /// Linux NFS mount requires passwordless sudo.
    #[error(
        "Linux NFS mount requires passwordless sudo (the NFS client needs \
         CAP_SYS_ADMIN). Configure passwordless sudo for `mount -t nfs` or \
         use Transport::Fuse instead."
    )]
    SudoRequired,

    /// The overlay's environment hash doesn't match the current lock file.
    #[error(
        "the overlay was created for a different environment (expected hash \
         '{expected}', found '{found}'). The overlay may contain files you \
         want to keep.\n\
         To reset the overlay for the new environment, remove it and remount:\n  \
         rm -rf <overlay_dir>"
    )]
    OverlayEnvHashMismatch { expected: String, found: String },

    /// The overlay directory was created by a different transport.
    #[error(
        "overlay was created with transport '{found}' but the current mount \
         requested '{expected}'. Remove the overlay manually or switch back \
         to the original transport."
    )]
    OverlayTransportMismatch { expected: String, found: String },

    /// The requested transport is not available on this platform.
    #[error("transport {transport:?} not available (missing feature or unsupported platform)")]
    TransportNotAvailable { transport: Transport },
}

/// A synthetic file materialized at tree-build time and mounted alongside the
/// package contents. Used for caller-supplied markers such as the env-hash
/// file that downstream tools (e.g. pixi) read to detect stale caches.
#[derive(Debug, Clone)]
pub struct VirtualFile {
    /// Path relative to the mount root (e.g. `conda-meta/rattler-fs_env`).
    pub relative_path: PathBuf,
    pub content: Vec<u8>,
    /// POSIX-style mode bits (default `0o644` if the caller doesn't care).
    pub mode: u32,
}

impl VirtualFile {
    pub fn new(relative_path: impl Into<PathBuf>, content: impl Into<Vec<u8>>) -> Self {
        Self {
            relative_path: relative_path.into(),
            content: content.into(),
            mode: DEFAULT_FILE_MODE,
        }
    }
}

/// How the tree-builder should resolve two sources contributing the same
/// `(parent_directory, file_name)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionPolicy {
    /// The first source to insert a given path wins; subsequent inserts are
    /// silently dropped (logged at `debug`). Matches rattler-fs's pre-refactor
    /// implicit behavior and is the right default for conda-only mounts.
    FirstWins,
    /// The last source to insert a given path wins; earlier nodes are
    /// replaced and a `warn` is logged. Intended for wheel-over-conda
    /// shadowing once pypi support lands.
    LastWins,
    /// Any collision aborts the build with a structured error. Useful for
    /// tests and for callers who want to surface overlapping package
    /// contents explicitly.
    Error,
}

impl Default for CollisionPolicy {
    fn default() -> Self {
        Self::FirstWins
    }
}

/// The "what to serve" half of a mount, bundled so it can be built and
/// validated independently of the mount transport and overlay options in
/// [`MountConfig`].
pub struct Layout {
    /// Package sources to mount. Each source's files are inserted in iteration
    /// order; collisions within or across sources follow `collision_policy`.
    pub packages: Vec<Box<dyn PackageSource>>,
    /// Caller-injected synthetic files overlaid on top of the package
    /// contributions (inserted last, so `LastWins` callers can use them to
    /// shadow package files).
    pub virtual_files: Vec<VirtualFile>,
    /// Collision handling for overlapping `(parent, name)` inserts.
    pub collision_policy: CollisionPolicy,
}

impl Layout {
    /// Empty layout with the default [`CollisionPolicy::FirstWins`].
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
            virtual_files: Vec::new(),
            collision_policy: CollisionPolicy::default(),
        }
    }

    /// Builder-style package list setter.
    pub fn with_packages(mut self, packages: Vec<Box<dyn PackageSource>>) -> Self {
        self.packages = packages;
        self
    }

    /// Builder-style virtual-files list setter.
    pub fn with_virtual_files(mut self, virtual_files: Vec<VirtualFile>) -> Self {
        self.virtual_files = virtual_files;
        self
    }

    /// Builder-style collision-policy setter.
    pub fn with_collision_policy(mut self, policy: CollisionPolicy) -> Self {
        self.collision_policy = policy;
        self
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self::new()
    }
}

/// Ensure a directory exists in the metadata tree, creating it if necessary.
/// Returns the index of the directory.
fn ensure_directory(
    dir_path: PathBuf,
    parent_index: usize,
    env_paths: &mut Vec<MetadataNode>,
    directory_indices: &mut HashMap<PathBuf, usize>,
) -> usize {
    if let Some(&index) = directory_indices.get(&dir_path) {
        return index;
    }
    let new_dir = MetadataNode::new_directory(dir_path.clone(), parent_index);
    let child_index = env_paths.len();
    env_paths.push(new_dir);
    env_paths[parent_index]
        .as_directory_mut()
        .expect("parent is a directory")
        .children
        .push(child_index);
    directory_indices.insert(dir_path, child_index);
    child_index
}

/// Walk `file.relative_path`, creating any missing ancestor directories, then
/// insert the file as a leaf under its parent. `file_indices` memoises the
/// `(parent_index, file_name)` pairs that already have a node so collision
/// checks stay O(1) no matter how many files share a directory — critical for
/// directories like site-packages that routinely hold thousands of entries.
fn upsert_file(
    env_paths: &mut Vec<MetadataNode>,
    directory_indices: &mut HashMap<PathBuf, usize>,
    file_indices: &mut HashMap<(usize, OsString), usize>,
    policy: CollisionPolicy,
    file: PackageFile,
) -> anyhow::Result<()> {
    // Resolve (or create) every ancestor directory so we end up with a valid
    // parent index. Paths in `PackageFile::relative_path` are treated as
    // relative to the mount root (represented here by the "." directory at
    // index 0).
    let parent_directory = file.relative_path.parent().unwrap_or(Path::new("."));
    let mut parent_index = 0;
    for component in parent_directory.components() {
        let current_path = env_paths[parent_index]
            .as_directory()
            .expect("parent is always a directory")
            .prefix_path
            .join(component);
        parent_index =
            ensure_directory(current_path, parent_index, env_paths, directory_indices);
    }

    let file_name: OsString = file
        .relative_path
        .file_name()
        .expect("files always have names")
        .to_os_string();

    // Collision check against an already-inserted file at the same
    // (parent, name). Directories are tracked separately via
    // `directory_indices` and are deduped by `ensure_directory`, so we only
    // worry about file-vs-file overlap here.
    let key = (parent_index, file_name.clone());
    if let Some(&existing_index) = file_indices.get(&key) {
        match policy {
            CollisionPolicy::FirstWins => {
                tracing::debug!(
                    "collision at {:?}: keeping first-inserted node (policy=FirstWins)",
                    file.relative_path
                );
                return Ok(());
            }
            CollisionPolicy::LastWins => {
                tracing::warn!(
                    "collision at {:?}: replacing earlier node (policy=LastWins)",
                    file.relative_path
                );
                env_paths[existing_index] = build_file_node(file_name, parent_index, file);
                return Ok(());
            }
            CollisionPolicy::Error => {
                anyhow::bail!(
                    "file collision at {:?} (policy=Error)",
                    file.relative_path
                );
            }
        }
    }

    let file_index = env_paths.len();
    env_paths.push(build_file_node(file_name.clone(), parent_index, file));
    env_paths[parent_index]
        .as_directory_mut()
        .expect("parent is a directory")
        .children
        .push(file_index);
    file_indices.insert((parent_index, file_name), file_index);
    Ok(())
}

/// Construct the concrete `MetadataNode::File` for a [`PackageFile`]. Split
/// out of [`upsert_file`] so both the insert path and the replace path
/// (`LastWins` collisions) share the construction logic.
fn build_file_node(
    file_name: OsString,
    parent_index: usize,
    file: PackageFile,
) -> MetadataNode {
    match file.content {
        FileContent::CachedBytes {
            cache_path,
            cache_prefix,
            transform,
        } => {
            let mut node = MetadataNode::new_file(
                file_name,
                parent_index,
                cache_path,
                file.path_type,
                // new_file currently takes Option<PrefixPlaceholder>; extract
                // the inner placeholder from the content-transform. Once new
                // transform variants exist this will need to fan out.
                transform.and_then(|t| match t {
                    metadata_tree::ContentTransform::PrefixReplace(p) => Some(p),
                }),
            );
            if let Some(override_path) = cache_prefix {
                let f = node.as_file_mut().expect("just built a file node");
                f.cache_prefix_path = Some(override_path);
            }
            let f = node.as_file_mut().expect("just built a file node");
            f.mode = file.mode;
            node
        }
        FileContent::Inline(content) => {
            let file_mode = file.mode;
            // Pick the virtual-file constructor whose default mode matches the
            // caller's request. Either way `f.mode` is stamped afterwards to
            // preserve any mode bits the constructor doesn't cover.
            let mut node = if file_mode == metadata_tree::EXECUTABLE_FILE_MODE {
                MetadataNode::new_virtual_executable(file_name, parent_index, content)
            } else {
                MetadataNode::new_virtual_file(file_name, parent_index, content)
            };
            let f = node.as_file_mut().expect("just built a file node");
            f.mode = file_mode;
            node
        }
    }
}

/// Initialise the root directory and index for a new virtual filesystem tree.
pub(crate) fn new_empty_tree() -> (Vec<MetadataNode>, HashMap<PathBuf, usize>) {
    let env_paths = vec![MetadataNode::new_directory(PathBuf::from("."), 0)];
    let mut directory_indices = HashMap::new();
    directory_indices.insert(PathBuf::from("."), 0);
    (env_paths, directory_indices)
}

// ---------------------------------------------------------------------------
// Library API: mount orchestration
// ---------------------------------------------------------------------------

/// Opaque metadata tree produced by [`build_metadata_tree`].
///
/// Pass to [`mount`] or [`build_and_mount`]; the internal representation is
/// not stable and is intentionally not exposed. The newtype wrapper means
/// downstream consumers cannot construct one directly — guaranteeing every
/// mount went through `build_metadata_tree`'s validation.
#[derive(Debug)]
pub struct MetadataTree(pub(crate) Vec<MetadataNode>);

/// Transport backend for the virtual filesystem.
///
/// See the [crate-level docs](crate#platform-support) for why each platform
/// has a different default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// Auto-detect the best backend for the current platform: FUSE on Linux,
    /// NFS on macOS, `ProjFS` on Windows.
    Auto,
    /// `NFSv3` userspace server on localhost. Works on all platforms without
    /// kernel extensions. On Windows, constrained to port 2049 and drive letters.
    ///
    /// **Linux note:** `mount -t nfs` requires `CAP_SYS_ADMIN`, which is not
    /// granted in unprivileged user namespaces. The adapter probes for
    /// passwordless `sudo` before attempting the mount and fails fast if it's
    /// not available. On Linux, prefer [`Transport::Fuse`] unless you
    /// specifically need NFS parity with macOS — [`Transport::Auto`] already
    /// picks FUSE.
    Nfs,
    /// FUSE via libfuse3 (Linux) or macFUSE (macOS, requires `fuse` feature).
    /// Not available on Windows.
    Fuse,
    /// Windows Projected File System. Demand-driven: files are materialized on
    /// first access. Only available on Windows 10 version 1809+.
    ProjFs,
}

impl Transport {
    /// Short name for state file tracking.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Nfs => "nfs",
            Self::Fuse => "fuse",
            Self::ProjFs => "projfs",
        }
    }

    /// Resolve `Auto` to the platform-appropriate transport.
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                if cfg!(target_os = "windows") {
                    Self::ProjFs
                } else if cfg!(target_os = "macos") {
                    Self::Nfs
                } else {
                    Self::Fuse
                }
            }
            other => other,
        }
    }

    /// Whether this transport is available on the current platform and build.
    ///
    /// Use this at config-parse time to reject invalid combinations early
    /// instead of waiting for [`mount`] to fail at runtime.
    pub fn is_available(self) -> bool {
        match self.resolve() {
            Self::Auto => unreachable!("resolve() never returns Auto"),
            Self::Fuse => cfg!(any(target_os = "linux", feature = "fuse")),
            Self::Nfs => cfg!(feature = "nfs"),
            Self::ProjFs => cfg!(target_os = "windows"),
        }
    }
}

/// Whether the mount is read-only or writable, and where the writable
/// overlay lives.
///
/// Replaces the old `overlay_dir: Option<PathBuf>` convention with an explicit
/// distinction between read-only and writable mounts. `ProjFS` does not support
/// read-only mode (it lacks a pre-creation notification, so new files can
/// always be created via the virtualization root) — passing [`Mode::ReadOnly`]
/// to a `ProjFS` mount returns a clear error.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Mode {
    /// Read-only mount. Writes return `EROFS`. Not supported on `ProjFS` —
    /// use [`Mode::ReadOnlyIfSupported`] for cross-platform configs.
    ReadOnly,

    /// Read-only if the transport supports it, otherwise writable.
    ///
    /// On FUSE/NFS this behaves identically to [`Mode::ReadOnly`].
    /// On `ProjFS` (which cannot enforce read-only) this silently falls
    /// through to writable mode and logs a warning. Use this in pixi
    /// configs where `mount-read-only = true` should work cross-platform.
    ReadOnlyIfSupported,

    /// Writable mount. Writes go to a persistent copy-on-write overlay.
    ///
    /// For FUSE/NFS, `overlay_dir` is a separate persistent directory pinned
    /// to a specific environment via [`MountConfig::env_hash`].
    ///
    /// For `ProjFS`, `overlay_dir` is ignored — `ProjFS` writes hydrated
    /// content directly to the virtualization root (the mount point) and
    /// tracks deletions via tombstones. Pass `overlay_dir: None` for `ProjFS`.
    Writable {
        /// Persistent overlay directory for FUSE/NFS. `None` is valid only
        /// for `ProjFS`, which uses the mount point itself.
        overlay_dir: Option<PathBuf>,
    },
}

/// Configuration for mounting a virtual environment.
///
/// Marked `#[non_exhaustive]` so new fields can be added without a `SemVer`
/// break. Construct via [`MountConfig::new_read_only`] or
/// [`MountConfig::new_writable`], optionally chaining
/// [`with_allow_other`](MountConfig::with_allow_other).
#[non_exhaustive]
pub struct MountConfig {
    /// Directory where the virtual environment will appear.
    pub mount_point: PathBuf,

    /// Read-only or writable, and where the overlay lives.
    pub mode: Mode,

    /// Transport backend. Use [`Transport::Auto`] to let the platform decide.
    pub transport: Transport,

    /// Identity hash of the resolved environment, used to detect when the
    /// environment has changed and the overlay needs to be reset. Callers
    /// typically derive this from their lock-file representation (e.g.
    /// `rattler_lock::Environment::content_hash` when using rattler_lock).
    pub env_hash: String,

    /// Allow other users to access the mount. Only applies to FUSE; requires
    /// `user_allow_other` in `/etc/fuse.conf`. Most use cases don't need this.
    pub allow_other: bool,
}

impl MountConfig {
    /// Read-only mount. Writes return `EROFS`.
    pub fn new_read_only(mount_point: PathBuf, transport: Transport, env_hash: String) -> Self {
        Self {
            mount_point,
            mode: Mode::ReadOnly,
            transport,
            env_hash,
            allow_other: false,
        }
    }

    /// Writable mount with a persistent COW overlay.
    ///
    /// `overlay_dir` is required for FUSE/NFS and must be a separate directory
    /// from `mount_point`. Pass `None` for `ProjFS` — `ProjFS` uses the mount
    /// point itself as the virtualization root.
    pub fn new_writable(
        mount_point: PathBuf,
        overlay_dir: Option<PathBuf>,
        transport: Transport,
        env_hash: String,
    ) -> Self {
        Self {
            mount_point,
            mode: Mode::Writable { overlay_dir },
            transport,
            env_hash,
            allow_other: false,
        }
    }

    /// Allow other users to access the mount (FUSE only).
    pub fn with_allow_other(mut self, allow_other: bool) -> Self {
        self.allow_other = allow_other;
        self
    }
}

/// Handle to a running mount.
///
/// The mount stays live for as long as this handle exists. Dropping it
/// triggers a best-effort unmount and stops the background server (NFS) or
/// session (FUSE). Use [`MountHandle::unmount`] for explicit, error-returning
/// unmount; prefer it over relying on Drop when error handling matters.
///
/// Marked `#[non_exhaustive]` so new transport variants can be added without
/// a `SemVer` break. Downstream `match` arms must include `_ =>` to be exhaustive.
#[non_exhaustive]
pub enum MountHandle {
    #[cfg(feature = "nfs")]
    Nfs(nfs_adapter::NfsMountHandle),
    #[cfg(any(target_os = "linux", feature = "fuse"))]
    Fuse(fuser::BackgroundSession),
    #[cfg(target_os = "windows")]
    ProjFs(projfs_adapter::ProjFsHandle),
}

impl MountHandle {
    /// Whether the mount's backing server is still running.
    ///
    /// Currently only meaningful for the NFS transport, where the userspace
    /// server task can exit unexpectedly (panic, I/O error, or unexpected
    /// clean return) leaving the kernel mount stale. FUSE and `ProjFS` mounts
    /// are managed by the kernel and always report healthy from userspace.
    ///
    /// Pixi's `MountGuard` can poll this to detect a dead server before
    /// handing out a reference to the environment.
    #[allow(unreachable_patterns)]
    pub fn is_healthy(&self) -> bool {
        match self {
            #[cfg(feature = "nfs")]
            Self::Nfs(h) => h.is_healthy(),
            _ => true,
        }
    }

    /// Explicitly unmount the filesystem and shut down the backing server.
    ///
    /// This is async so the NFS unmount path can use `tokio::process::Command`
    /// instead of blocking the runtime. FUSE and `ProjFS` unmount synchronously
    /// (kernel-managed, no subprocess) so the async boundary is free for them.
    ///
    /// Prefer this over relying on Drop when error handling matters (e.g.
    /// sidecar shutdown, CI cleanup, signal-handling paths). Drop stays as a
    /// best-effort fallback that logs failures but cannot return them.
    #[allow(unreachable_patterns)]
    pub async fn unmount(self) -> anyhow::Result<()> {
        match self {
            #[cfg(feature = "nfs")]
            Self::Nfs(h) => h.unmount().await,
            #[cfg(any(target_os = "linux", feature = "fuse"))]
            Self::Fuse(session) => {
                // fuser does not expose a Result-returning unmount; dropping
                // the BackgroundSession is the documented shutdown path.
                drop(session);
                Ok(())
            }
            #[cfg(target_os = "windows")]
            Self::ProjFs(h) => h.unmount(),
            _ => Ok(()),
        }
    }
}

/// Build the in-memory metadata tree from a [`Layout`].
///
/// `rattler_fs` has no knowledge of lock files, package caches, or remote
/// fetches — those concerns belong to the caller (see `rattler-bin`'s `mount`
/// subcommand for a working example). Each [`PackageSource`] in the layout is
/// queried for the files it wants to expose; [`Layout::virtual_files`] are
/// layered on top for synthetic markers (e.g. a content-hash file for
/// stale-cache detection). Overlapping paths are resolved according to
/// [`Layout::collision_policy`].
///
/// `mount_point` is passed through to each source so generated content that
/// bakes in absolute paths (entry-point shebangs, etc.) can reference the
/// final mount location.
pub fn build_metadata_tree(
    layout: &Layout,
    mount_point: &Path,
) -> anyhow::Result<MetadataTree> {
    let (mut env_paths, mut directory_indices) = new_empty_tree();
    let mut file_indices: HashMap<(usize, OsString), usize> = HashMap::new();

    for pkg in &layout.packages {
        let files = pkg
            .files(mount_point)
            .map_err(|e| anyhow::anyhow!("package '{}': {e}", pkg.name()))?;
        for file in files {
            upsert_file(
                &mut env_paths,
                &mut directory_indices,
                &mut file_indices,
                layout.collision_policy,
                file,
            )?;
        }
        tracing::debug!(
            "parsed package '{}': {} metadata entries so far",
            pkg.name(),
            env_paths.len()
        );
    }

    for vf in &layout.virtual_files {
        let package_file = PackageFile {
            relative_path: vf.relative_path.clone(),
            content: FileContent::Inline(vf.content.clone()),
            mode: vf.mode,
            path_type: PathType::HardLink,
        };
        upsert_file(
            &mut env_paths,
            &mut directory_indices,
            &mut file_indices,
            layout.collision_policy,
            package_file,
        )?;
    }

    Ok(MetadataTree(env_paths))
}

/// Mount a pre-built metadata tree. Returns a handle that unmounts on drop.
pub async fn mount(metadata: MetadataTree, config: &MountConfig) -> anyhow::Result<MountHandle> {
    let transport = config.transport.resolve();

    // ProjFS-specific pre-flight checks: DLL availability, mode validity,
    // overlay state. Done before VFS construction so we don't waste offset
    // computation if ProjFS isn't installed or the user passed Mode::ReadOnly.
    #[cfg(target_os = "windows")]
    if matches!(transport, Transport::ProjFs) {
        // Verify that the ProjFS optional feature is enabled before calling
        // any ProjFS API.  The `windows` crate delay-loads the DLL, so a
        // missing feature won't crash the process, but the first API call
        // would return a confusing "not found" HRESULT.  Give users a clear
        // message instead.
        {
            use std::os::windows::ffi::OsStrExt;
            let dll: Vec<u16> = std::ffi::OsStr::new("projectedfslib.dll")
                .encode_wide()
                .chain(Some(0))
                .collect();
            let handle = unsafe {
                windows::Win32::System::LibraryLoader::LoadLibraryW(windows::core::PCWSTR(
                    dll.as_ptr(),
                ))
            };
            if handle.is_err() {
                return Err(MountError::ProjFsDllMissing.into());
            }
        }

        // ProjFS is always writable — it writes hydrated content directly
        // to the virtualization root and tracks deletions via tombstones.
        // There is no read-only mode: ProjFS lacks a pre-creation
        // notification, so new files can always be created.
        if matches!(config.mode, Mode::ReadOnly) {
            return Err(MountError::ProjFsReadOnlyUnsupported.into());
        }
        if matches!(config.mode, Mode::ReadOnlyIfSupported) {
            tracing::warn!("ProjFS does not support read-only mode; falling through to writable");
        }

        // Validate overlay state (env hash) to reject stale mounts.
        {
            use crate::overlay::{OverlayError, OverlayState};
            match OverlayState::load(
                config.mount_point.clone(),
                config.env_hash.clone(),
                "projfs".to_string(),
            ) {
                Ok(_) => {} // hash matches or fresh overlay
                Err(OverlayError::EnvHashMismatch {
                    expected, found, ..
                }) => {
                    return Err(MountError::OverlayEnvHashMismatch { expected, found }.into());
                }
                Err(OverlayError::TransportMismatch {
                    expected, found, ..
                }) => {
                    return Err(MountError::OverlayTransportMismatch { expected, found }.into());
                }
                Err(e) => anyhow::bail!("overlay state check failed: {e}"),
            }
        }
    }

    // Construct the VirtualFS once. Each transport branch consumes it.
    // VFS construction does eager prefix-offset computation, so we want
    // exactly one call per mount.
    let vfs = VirtualFS::new(metadata.0, &config.mount_point);

    match transport {
        #[cfg(feature = "nfs")]
        Transport::Nfs => Ok(MountHandle::Nfs(mount_nfs(vfs, config).await?)),
        #[cfg(any(target_os = "linux", feature = "fuse"))]
        Transport::Fuse => Ok(MountHandle::Fuse(mount_fuse(vfs, config)?)),
        #[cfg(target_os = "windows")]
        Transport::ProjFs => {
            let adapter = projfs_adapter::ProjFsAdapter::new(vfs);
            let handle = adapter.start(&config.mount_point)?;
            Ok(MountHandle::ProjFs(handle))
        }
        #[allow(unreachable_patterns)]
        _ => Err(MountError::TransportNotAvailable { transport })?,
    }
}

/// Convenience wrapper that builds the metadata tree and then mounts it.
///
/// See [`build_metadata_tree`] for the division of responsibilities between
/// the layout (what to serve) and the mount config (how to serve it).
pub async fn build_and_mount(
    layout: &Layout,
    config: &MountConfig,
) -> anyhow::Result<MountHandle> {
    let metadata = build_metadata_tree(layout, &config.mount_point)?;
    mount(metadata, config).await
}


// ---------------------------------------------------------------------------
// Internal: transport-specific mount helpers
// ---------------------------------------------------------------------------

/// Mount via FUSE, with optional writable overlay.
///
/// The overlay is retried once if the env hash mismatches (environment updated).
#[cfg(any(target_os = "linux", feature = "fuse"))]
fn mount_fuse(vfs: VirtualFS, config: &MountConfig) -> anyhow::Result<fuser::BackgroundSession> {
    use fuse_adapter::FuseAdapter;
    use fuser::{Config as FuserConfig, MountOption, SessionACL};

    let mut fuser_config = FuserConfig::default();
    fuser_config.mount_options = vec![
        MountOption::FSName("conda-packages".to_string()),
        MountOption::CUSTOM("noatime".to_string()),
    ];
    if matches!(config.mode, Mode::ReadOnly | Mode::ReadOnlyIfSupported) {
        fuser_config.mount_options.push(MountOption::RO);
    }
    if config.allow_other {
        fuser_config.acl = SessionACL::All;
    }

    match &config.mode {
        Mode::Writable {
            overlay_dir: Some(overlay_dir),
        } => {
            let overlay = create_overlay(vfs, overlay_dir, &config.env_hash, "fuse")?;
            let adapter = FuseAdapter::new(overlay);
            Ok(fuser::spawn_mount2(
                adapter,
                &config.mount_point,
                &fuser_config,
            )?)
        }
        Mode::Writable { overlay_dir: None } => {
            anyhow::bail!(
                "FUSE writable mode requires an overlay directory. Use \
                 MountConfig::new_writable(.., Some(overlay_dir), ..) or \
                 MountConfig::new_read_only(..) for a read-only mount."
            );
        }
        Mode::ReadOnly | Mode::ReadOnlyIfSupported => {
            let adapter = FuseAdapter::new(vfs);
            Ok(fuser::spawn_mount2(
                adapter,
                &config.mount_point,
                &fuser_config,
            )?)
        }
    }
}

/// Mount via NFS, with optional writable overlay.
///
/// The overlay is retried once if the env hash mismatches (environment updated).
#[cfg(feature = "nfs")]
async fn mount_nfs(
    vfs: VirtualFS,
    config: &MountConfig,
) -> anyhow::Result<nfs_adapter::NfsMountHandle> {
    use nfs_adapter::NfsAdapter;

    let read_only = matches!(config.mode, Mode::ReadOnly | Mode::ReadOnlyIfSupported);

    let bind_port = 0u16;

    let server_handle = match &config.mode {
        Mode::Writable {
            overlay_dir: Some(overlay_dir),
        } => {
            let overlay = create_overlay(vfs, overlay_dir, &config.env_hash, "nfs")?;
            NfsAdapter::new(overlay).serve(bind_port).await?
        }
        Mode::Writable { overlay_dir: None } => {
            anyhow::bail!(
                "NFS writable mode requires an overlay directory. Use \
                 MountConfig::new_writable(.., Some(overlay_dir), ..) or \
                 MountConfig::new_read_only(..) for a read-only mount."
            );
        }
        Mode::ReadOnly | Mode::ReadOnlyIfSupported => NfsAdapter::new(vfs).serve(bind_port).await?,
    };

    let port = server_handle.port();

    let mut opts = format!("noacl,nolock,vers=3,tcp,port={port},mountport={port},rsize=1048576");
    if read_only {
        opts.push_str(",ro");
    } else {
        opts.push_str(",wsize=1048576");
    }

    #[cfg(target_os = "macos")]
    {
        let mnt = config.mount_point.display().to_string();
        let status = tokio::process::Command::new("mount_nfs")
            .args(["-o", &opts, "localhost:/", &mnt])
            .status()
            .await?;
        if !status.success() {
            server_handle.abort();
            anyhow::bail!("NFS mount failed with exit status {status}");
        }
    }

    #[cfg(target_os = "linux")]
    {
        let mnt = config.mount_point.display().to_string();
        // Probe passwordless sudo first — `mount -t nfs` needs CAP_SYS_ADMIN
        // which isn't available in unprivileged user namespaces, so there's no
        // userspace fallback we can reach for. Fail loudly instead of letting
        // sudo prompt interactively (terrible UX in `pixi run`).
        //
        // TODO(bind-mount): investigate `unshare -Urm` + `mount --bind` as a
        // rootless alternative transport on Linux. That would work in rootless
        // containers where neither FUSE nor sudo is available.
        let probe = tokio::process::Command::new("sudo")
            .args(["-n", "true"])
            .status()
            .await;
        match probe {
            Ok(s) if s.success() => {}
            _ => {
                server_handle.abort();
                return Err(MountError::SudoRequired.into());
            }
        }

        let status = tokio::process::Command::new("sudo")
            .args(["mount", "-t", "nfs", "-o", &opts, "localhost:/", &mnt])
            .status()
            .await?;
        if !status.success() {
            server_handle.abort();
            anyhow::bail!("NFS mount failed with exit status {status}");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        server_handle.abort();
        anyhow::bail!("NFS mount is not supported on this platform. Use ProjFS on Windows.");
    }

    tracing::info!("mounted via NFS on {}", config.mount_point.display());

    Ok(nfs_adapter::NfsMountHandle {
        mount_point: config.mount_point.clone(),
        server_handle,
        unmounted: false,
    })
}

/// Create an overlay, wiping and retrying transparently on state-version
/// mismatch (internal schema change). Returns a structured error on env-hash
/// mismatch so the caller can decide whether to wipe — the overlay may contain
/// user work. Refuses (does not wipe) on transport mismatch.
///
/// Acquires the directory lock once and carries it through the wipe-and-retry
/// path so no other process can sneak in between the wipe and the reload.
#[cfg(any(feature = "nfs", target_os = "linux", feature = "fuse"))]
fn create_overlay(
    vfs: VirtualFS,
    overlay_dir: &Path,
    env_hash: &str,
    transport: &str,
) -> anyhow::Result<overlay_fs::OverlayFS<VirtualFS>> {
    use crate::overlay::{OverlayError, OverlayState};

    // Acquire the lock once, before the first load attempt. The lock handle
    // is passed through both the initial load and the retry so the wipe
    // step is protected.
    let lock = OverlayState::acquire_lock(overlay_dir)
        .map_err(|e| anyhow::anyhow!("failed to acquire overlay lock: {e}"))?;

    let state = match OverlayState::load_with_lock(
        overlay_dir.to_path_buf(),
        env_hash.to_string(),
        transport.to_string(),
        lock,
    ) {
        Ok(state) => state,
        Err(OverlayError::EnvHashMismatch {
            expected, found, ..
        }) => {
            return Err(MountError::OverlayEnvHashMismatch { expected, found }.into());
        }
        Err(OverlayError::VersionMismatch { lock, .. }) => {
            tracing::info!("overlay state version changed; wiping and recreating");
            // Lock is still held — safe to wipe without a race.
            if overlay_dir.exists() {
                std::fs::remove_dir_all(overlay_dir)?;
            }
            OverlayState::load_with_lock(
                overlay_dir.to_path_buf(),
                env_hash.to_string(),
                transport.to_string(),
                lock,
            )
            .map_err(|e| anyhow::anyhow!("failed to recreate overlay state: {e}"))?
        }
        Err(OverlayError::TransportMismatch {
            expected, found, ..
        }) => {
            return Err(MountError::OverlayTransportMismatch { expected, found }.into());
        }
        Err(e) => anyhow::bail!("failed to load overlay state: {e}"),
    };

    overlay_fs::OverlayFS::wrap(vfs, state)
        .map_err(|e| anyhow::anyhow!("failed to wrap VFS with overlay: {e}"))
}

/// Force unmount a mount point.
///
/// Best-effort cleanup for stale mounts (e.g. after a crash).  The
/// `transport` hint selects the right teardown method:
///
/// | Platform | Transport | Method |
/// |----------|-----------|--------|
/// | Linux | FUSE | `fusermount3 -uz` |
/// | Linux | NFS | `sudo umount -f` (requires passwordless sudo) |
/// | macOS | FUSE / NFS | `umount -f` |
/// | Windows | `ProjFS` | Not yet supported — returns an error |
///
/// Pass [`Transport::Auto`] to use the platform default.
///
/// **NFS on Linux note:** `umount -f` requires `CAP_SYS_ADMIN`.  If
/// passwordless sudo is not available, this will fail.  Consider switching
/// to [`Transport::Fuse`] where possible.
///
/// **`ProjFS` note:** stale `ProjFS` mounts are structurally different — the
/// virtualization context died with the owning process, but hydrated files
/// remain.  Recovery currently requires wiping the directory and remounting.
/// A future version may support re-attaching to an existing virtualization
/// root.
pub fn force_unmount(mount_point: &Path, transport: Transport) -> anyhow::Result<()> {
    let transport = transport.resolve();
    let mnt = mount_point.display().to_string();

    match transport {
        #[cfg(any(target_os = "linux", feature = "fuse"))]
        Transport::Fuse => {
            #[cfg(target_os = "linux")]
            {
                let status = std::process::Command::new("fusermount3")
                    .args(["-uz", &mnt])
                    .status()?;
                if !status.success() {
                    anyhow::bail!("fusermount3 -uz {mnt} failed (exit {status})");
                }
            }
            #[cfg(target_os = "macos")]
            {
                let status = std::process::Command::new("umount")
                    .args(["-f", &mnt])
                    .status()?;
                if !status.success() {
                    anyhow::bail!("umount -f {mnt} failed (exit {status})");
                }
            }
        }
        #[cfg(feature = "nfs")]
        Transport::Nfs => {
            #[cfg(target_os = "macos")]
            {
                let status = std::process::Command::new("umount")
                    .args(["-f", &mnt])
                    .status()?;
                if !status.success() {
                    anyhow::bail!("umount -f {mnt} failed (exit {status})");
                }
            }
            #[cfg(target_os = "linux")]
            {
                let status = std::process::Command::new("sudo")
                    .args(["umount", "-f", &mnt])
                    .status()?;
                if !status.success() {
                    anyhow::bail!(
                        "sudo umount -f {mnt} failed (exit {status}). \
                         NFS force-unmount on Linux requires passwordless sudo."
                    );
                }
            }
        }
        #[cfg(target_os = "windows")]
        Transport::ProjFs => {
            anyhow::bail!(
                "ProjFS stale-mount recovery is not yet supported. \
                 The virtualization context died with the owning process; \
                 hydrated files remain at {mnt}. Remove the directory \
                 manually and remount, or wait for re-attach support."
            );
        }
        _ => {
            anyhow::bail!(
                "force_unmount: transport {transport:?} is not available on this platform"
            );
        }
    }

    #[allow(unreachable_code)]
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rattler::install::PythonInfo;
    use rattler_conda_types::package::{EntryPoint, PathType, PathsEntry, PathsJson};
    use std::path::PathBuf;

    fn make_paths_json(paths: Vec<&str>) -> PathsJson {
        PathsJson {
            paths: paths
                .into_iter()
                .map(|p| PathsEntry {
                    relative_path: PathBuf::from(p),
                    path_type: PathType::HardLink,
                    prefix_placeholder: None,
                    no_link: false,
                    sha256: None,
                    size_in_bytes: None,
                })
                .collect(),
            paths_version: 1,
        }
    }

    fn make_python_info() -> PythonInfo {
        use rattler_conda_types::Version;
        use std::str::FromStr;
        PythonInfo::from_version(
            &Version::from_str("3.11.0").unwrap(),
            None,
            rattler_conda_types::Platform::Linux64,
        )
        .unwrap()
    }

    fn make_entry_points() -> Vec<EntryPoint> {
        use std::str::FromStr;
        vec![
            EntryPoint::from_str("ipython = IPython:start_ipython").unwrap(),
            EntryPoint::from_str("ipython3 = IPython:start_ipython").unwrap(),
        ]
    }

    /// Wrap a single `CondaPackage` into a [`Layout`], drive it through
    /// `build_metadata_tree`, and return the flat metadata vec.
    fn build_from_single_package(pkg: CondaPackage, mount_point: &Path) -> Vec<MetadataNode> {
        let layout = Layout::new().with_packages(vec![Box::new(pkg)]);
        let tree = build_metadata_tree(&layout, mount_point).expect("tree build ok");
        tree.0
    }

    #[test]
    fn test_single_file_at_root() {
        let paths_json = make_paths_json(vec!["foo.txt"]);
        let pkg = CondaPackage::from_parts(
            "pkg",
            Path::new("/cache/pkg"),
            paths_json,
            vec![],
            None,
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));

        assert_eq!(env_paths.len(), 2); // root + foo.txt
        let root = env_paths[0].as_directory().unwrap();
        assert_eq!(root.children.len(), 1);
        let file = env_paths[root.children[0]].as_file().unwrap();
        assert_eq!(file.file_name, "foo.txt");
        assert_eq!(&*file.cache_base_path, Path::new("/cache/pkg"));
    }

    #[test]
    fn test_nested_directories() {
        let paths_json = make_paths_json(vec!["a/b/c.txt"]);
        let pkg = CondaPackage::from_parts(
            "pkg",
            Path::new("/cache/pkg"),
            paths_json,
            vec![],
            None,
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));

        // root, dir "a", dir "a/b", file "c.txt"
        assert_eq!(env_paths.len(), 4);

        let root = env_paths[0].as_directory().unwrap();
        assert_eq!(root.children.len(), 1);

        let dir_a = env_paths[root.children[0]].as_directory().unwrap();
        assert_eq!(dir_a.prefix_path, PathBuf::from("./a"));
        assert_eq!(dir_a.children.len(), 1);

        let dir_b = env_paths[dir_a.children[0]].as_directory().unwrap();
        assert_eq!(dir_b.prefix_path, PathBuf::from("./a/b"));
        assert_eq!(dir_b.children.len(), 1);

        let file = env_paths[dir_b.children[0]].as_file().unwrap();
        assert_eq!(file.file_name, "c.txt");
    }

    #[test]
    fn test_directory_dedup() {
        let paths_json = make_paths_json(vec!["lib/foo", "lib/bar"]);
        let pkg = CondaPackage::from_parts(
            "pkg",
            Path::new("/cache/pkg"),
            paths_json,
            vec![],
            None,
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));

        // root, dir "lib", file "foo", file "bar"
        assert_eq!(env_paths.len(), 4);

        let root = env_paths[0].as_directory().unwrap();
        assert_eq!(root.children.len(), 1); // single lib dir

        let lib_dir = env_paths[root.children[0]].as_directory().unwrap();
        assert_eq!(lib_dir.children.len(), 2); // foo and bar
    }

    #[test]
    fn test_multiple_packages() {
        let pkg1 = CondaPackage::from_parts(
            "pkg1",
            Path::new("/cache/pkg1"),
            make_paths_json(vec!["lib/foo.so"]),
            vec![],
            None,
        );
        let pkg2 = CondaPackage::from_parts(
            "pkg2",
            Path::new("/cache/pkg2"),
            make_paths_json(vec!["lib/bar.so"]),
            vec![],
            None,
        );

        let layout = Layout::new().with_packages(vec![Box::new(pkg1), Box::new(pkg2)]);
        let tree =
            build_metadata_tree(&layout, Path::new("/prefix")).expect("tree build ok");
        let env_paths = tree.0;

        // root, dir "lib", file "foo.so", file "bar.so"
        assert_eq!(env_paths.len(), 4);

        let lib_dir = env_paths[1].as_directory().unwrap();
        assert_eq!(lib_dir.children.len(), 2);

        let foo = env_paths[lib_dir.children[0]].as_file().unwrap();
        assert_eq!(&*foo.cache_base_path, Path::new("/cache/pkg1"));

        let bar = env_paths[lib_dir.children[1]].as_file().unwrap();
        assert_eq!(&*bar.cache_base_path, Path::new("/cache/pkg2"));
    }

    #[test]
    fn test_empty_paths_json() {
        let pkg = CondaPackage::from_parts(
            "pkg",
            Path::new("/cache/pkg"),
            make_paths_json(vec![]),
            vec![],
            None,
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));

        assert_eq!(env_paths.len(), 1); // root only
        let root = env_paths[0].as_directory().unwrap();
        assert_eq!(root.children.len(), 0);
    }

    /// Index every directory in a built tree by its virtual path. Mirrors the
    /// old `directory_indices` map that path-parse tests used to inspect.
    fn collect_directory_indices(env_paths: &[MetadataNode]) -> HashMap<PathBuf, usize> {
        let mut out = HashMap::new();
        for (i, n) in env_paths.iter().enumerate() {
            if let Some(d) = n.as_directory() {
                out.insert(d.prefix_path.clone(), i);
            }
        }
        out
    }

    #[test]
    fn test_entry_points_creates_bin_dir() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec![]),
            make_entry_points(),
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        assert!(dir_indices.contains_key(&PathBuf::from("./bin")));
        let root = env_paths[0].as_directory().unwrap();
        assert_eq!(root.children.len(), 1); // bin dir
    }

    #[test]
    fn test_entry_points_adds_files() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec![]),
            make_entry_points(),
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        let bin_idx = dir_indices[&PathBuf::from("./bin")];
        let bin_dir = env_paths[bin_idx].as_directory().unwrap();
        assert_eq!(bin_dir.children.len(), 2);

        let names: Vec<_> = bin_dir
            .children
            .iter()
            .map(|&i| env_paths[i].file_name().to_str().unwrap().to_string())
            .collect();
        assert!(names.contains(&"ipython".to_string()));
        assert!(names.contains(&"ipython3".to_string()));
    }

    #[test]
    fn test_entry_points_virtual_content() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec![]),
            make_entry_points(),
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        let bin_idx = dir_indices[&PathBuf::from("./bin")];
        let bin_dir = env_paths[bin_idx].as_directory().unwrap();
        let file = env_paths[bin_dir.children[0]].as_file().unwrap();

        let content = file
            .virtual_content
            .as_ref()
            .expect("should have virtual content");
        let text = std::str::from_utf8(content).unwrap();
        assert!(
            text.contains("#!/prefix/bin/python3.11"),
            "shebang missing: {text}"
        );
        assert!(
            text.contains("from IPython import"),
            "import missing: {text}"
        );
        assert!(
            text.contains("start_ipython()"),
            "function call missing: {text}"
        );
    }

    #[test]
    fn test_entry_points_dedup_bin_dir() {
        // Package 1 already has a bin/ directory with one file in it; package
        // 2 (the noarch-python one) should merge its entry-point scripts into
        // the same bin/ rather than creating a duplicate node.
        let pkg1 = CondaPackage::from_parts(
            "other",
            Path::new("/cache/pkg"),
            make_paths_json(vec!["bin/existing"]),
            vec![],
            None,
        );
        let pkg2 = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec![]),
            make_entry_points(),
            Some(make_python_info()),
        );
        let layout = Layout::new().with_packages(vec![Box::new(pkg1), Box::new(pkg2)]);
        let tree =
            build_metadata_tree(&layout, Path::new("/prefix")).expect("tree build ok");
        let env_paths = tree.0;
        let dir_indices = collect_directory_indices(&env_paths);

        let bin_idx = dir_indices[&PathBuf::from("./bin")];
        let bin_dir = env_paths[bin_idx].as_directory().unwrap();
        assert_eq!(bin_dir.children.len(), 3); // existing + ipython + ipython3

        let root = env_paths[0].as_directory().unwrap();
        assert_eq!(root.children.len(), 1);
    }

    // --- noarch Python path rewriting tests ---

    #[test]
    fn test_noarch_python_rewrites_site_packages() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec![
                "site-packages/foo/__init__.py",
                "site-packages/foo/bar.py",
            ]),
            vec![],
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        assert!(dir_indices.contains_key(&PathBuf::from("./lib")));
        assert!(dir_indices.contains_key(&PathBuf::from("./lib/python3.11")));
        assert!(dir_indices.contains_key(&PathBuf::from("./lib/python3.11/site-packages")));
        assert!(dir_indices.contains_key(&PathBuf::from("./lib/python3.11/site-packages/foo")));
        assert!(!dir_indices.contains_key(&PathBuf::from("./site-packages")));
    }

    #[test]
    fn test_noarch_python_rewrites_python_scripts() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec!["python-scripts/mycmd"]),
            vec![],
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        assert!(dir_indices.contains_key(&PathBuf::from("./bin")));
        let bin_idx = dir_indices[&PathBuf::from("./bin")];
        let bin_dir = env_paths[bin_idx].as_directory().unwrap();
        assert_eq!(bin_dir.children.len(), 1);
        let file = env_paths[bin_dir.children[0]].as_file().unwrap();
        assert_eq!(file.file_name, "mycmd");
    }

    #[test]
    fn test_noarch_python_preserves_cache_path() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/noarch-pkg"),
            make_paths_json(vec!["site-packages/foo/bar.py"]),
            vec![],
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        let foo_idx = dir_indices[&PathBuf::from("./lib/python3.11/site-packages/foo")];
        let foo_dir = env_paths[foo_idx].as_directory().unwrap();
        let file = env_paths[foo_dir.children[0]].as_file().unwrap();

        assert_eq!(&*file.cache_base_path, Path::new("/cache/noarch-pkg"));
        assert_eq!(
            file.cache_prefix_path.as_deref(),
            Some(Path::new("./site-packages/foo"))
        );
    }

    #[test]
    fn test_noarch_non_rewritten_paths_unchanged() {
        let pkg = CondaPackage::from_parts(
            "noarch-py",
            Path::new("/cache/pkg"),
            make_paths_json(vec!["share/data/file.txt"]),
            vec![],
            Some(make_python_info()),
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        assert!(dir_indices.contains_key(&PathBuf::from("./share")));
        assert!(dir_indices.contains_key(&PathBuf::from("./share/data")));
        let data_idx = dir_indices[&PathBuf::from("./share/data")];
        let file = env_paths[env_paths[data_idx].as_directory().unwrap().children[0]]
            .as_file()
            .unwrap();
        assert_eq!(file.file_name, "file.txt");
        assert!(file.cache_prefix_path.is_none());
    }

    #[test]
    fn test_non_noarch_no_rewrite() {
        // Without python_info, site-packages/ stays as-is.
        let pkg = CondaPackage::from_parts(
            "pkg",
            Path::new("/cache/pkg"),
            make_paths_json(vec!["site-packages/foo/bar.py"]),
            vec![],
            None,
        );
        let env_paths = build_from_single_package(pkg, Path::new("/prefix"));
        let dir_indices = collect_directory_indices(&env_paths);

        assert!(dir_indices.contains_key(&PathBuf::from("./site-packages")));
        assert!(!dir_indices.contains_key(&PathBuf::from("./lib")));
    }

    #[test]
    fn test_virtual_file_injection() {
        let pkg = CondaPackage::from_parts(
            "pkg",
            Path::new("/cache/pkg"),
            make_paths_json(vec![]),
            vec![],
            None,
        );
        let layout = Layout::new()
            .with_packages(vec![Box::new(pkg)])
            .with_virtual_files(vec![VirtualFile::new(
                "conda-meta/rattler-fs_env",
                b"abc123".to_vec(),
            )]);
        let tree =
            build_metadata_tree(&layout, Path::new("/prefix")).expect("tree build ok");
        let env_paths = tree.0;
        let dir_indices = collect_directory_indices(&env_paths);

        let conda_meta_idx = dir_indices[&PathBuf::from("./conda-meta")];
        let conda_meta = env_paths[conda_meta_idx].as_directory().unwrap();
        let file = env_paths[conda_meta.children[0]].as_file().unwrap();
        assert_eq!(file.file_name, "rattler-fs_env");
        assert_eq!(file.virtual_content.as_deref(), Some(b"abc123".as_slice()));
    }

    // --- CollisionPolicy tests ---

    fn make_colliding_packages() -> (Box<dyn PackageSource>, Box<dyn PackageSource>) {
        let pkg1 = CondaPackage::from_parts(
            "pkg1",
            Path::new("/cache/pkg1"),
            make_paths_json(vec!["lib/shared.so"]),
            vec![],
            None,
        );
        let pkg2 = CondaPackage::from_parts(
            "pkg2",
            Path::new("/cache/pkg2"),
            make_paths_json(vec!["lib/shared.so"]),
            vec![],
            None,
        );
        (Box::new(pkg1), Box::new(pkg2))
    }

    #[test]
    fn test_collision_first_wins_keeps_pkg1() {
        let (pkg1, pkg2) = make_colliding_packages();
        let layout = Layout::new()
            .with_packages(vec![pkg1, pkg2])
            .with_collision_policy(CollisionPolicy::FirstWins);
        let tree =
            build_metadata_tree(&layout, Path::new("/prefix")).expect("tree build ok");
        let env_paths = tree.0;

        // There should be exactly one file under lib/ and it should point at
        // pkg1's cache path, not pkg2's.
        let lib_dir = env_paths
            .iter()
            .find_map(|n| {
                n.as_directory()
                    .filter(|d| d.prefix_path == PathBuf::from("./lib"))
            })
            .expect("lib dir present");
        assert_eq!(lib_dir.children.len(), 1);
        let file = env_paths[lib_dir.children[0]].as_file().unwrap();
        assert_eq!(&*file.cache_base_path, Path::new("/cache/pkg1"));
    }

    #[test]
    fn test_collision_last_wins_keeps_pkg2() {
        let (pkg1, pkg2) = make_colliding_packages();
        let layout = Layout::new()
            .with_packages(vec![pkg1, pkg2])
            .with_collision_policy(CollisionPolicy::LastWins);
        let tree =
            build_metadata_tree(&layout, Path::new("/prefix")).expect("tree build ok");
        let env_paths = tree.0;

        let lib_dir = env_paths
            .iter()
            .find_map(|n| {
                n.as_directory()
                    .filter(|d| d.prefix_path == PathBuf::from("./lib"))
            })
            .expect("lib dir present");
        assert_eq!(lib_dir.children.len(), 1);
        let file = env_paths[lib_dir.children[0]].as_file().unwrap();
        assert_eq!(&*file.cache_base_path, Path::new("/cache/pkg2"));
    }

    #[test]
    fn test_collision_error_aborts() {
        let (pkg1, pkg2) = make_colliding_packages();
        let layout = Layout::new()
            .with_packages(vec![pkg1, pkg2])
            .with_collision_policy(CollisionPolicy::Error);
        let err =
            build_metadata_tree(&layout, Path::new("/prefix")).expect_err("should collide");
        assert!(
            err.to_string().contains("collision"),
            "unexpected error: {err}"
        );
    }
}
