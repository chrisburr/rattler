//! Persistent overlay state management.
//!
//! Tracks environment identity and transport in a small JSON state file,
//! and tracks whiteouts (deleted files) and opaque-dir markers as
//! `.wh.{name}` and `.wh..wh..opq` files inside the upper directory —
//! the same on-disk convention used by the Linux kernel's overlayfs and
//! by many userspace overlays (podman/buildah, fuse-overlayfs).
//!
//! Storing whiteouts on disk rather than in JSON has two benefits:
//!
//! 1. Out-of-band tools (`tar`, `rsync`, manual file inspection)
//!    preserve deletion state when copying the upper directory.
//! 2. Each whiteout/opaque add is a single atomic file create, instead
//!    of a full state-file rewrite + fsync — `rm -rf` of N files is now
//!    O(N) writes instead of O(N²).
//!
//! The state file is still written atomically (write-tmp → fsync →
//! rename), but only on mount/unmount and overlay-version migration —
//! not on every whiteout.

use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    ffi::{OsStr, OsString},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub(crate) const STATE_FILENAME: &str = ".rattler_fs_state.json";
pub(crate) const STATE_TMP_FILENAME: &str = ".rattler_fs_state.tmp";
pub(crate) const STATE_LOCK_FILENAME: &str = ".rattler_fs_state.lock";

/// Filename prefix used for whiteout markers (`.wh.{name}` deletes
/// `{name}` in the parent dir).
pub(crate) const WHITEOUT_PREFIX: &str = ".wh.";

/// Marker filename for an opaque directory (placed *inside* the dir;
/// hides every entry in the lower-layer counterpart).
pub(crate) const OPAQUE_MARKER: &str = ".wh..wh..opq";

#[derive(Debug)]
pub enum OverlayError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// Env hash mismatch.  The `lock` field carries the directory lock.
    /// `create_overlay` returns this as a structured `MountError` so the
    /// caller can decide whether to wipe — the overlay may contain user work.
    EnvHashMismatch {
        expected: String,
        found: String,
        lock: fs::File,
    },
    /// Transport mismatch.  The `lock` field carries the directory lock.
    TransportMismatch {
        expected: String,
        found: String,
        lock: fs::File,
    },
    /// State file version mismatch.  The `lock` field carries the directory lock.
    VersionMismatch {
        expected: u32,
        found: u32,
        lock: fs::File,
    },
}

impl std::fmt::Display for OverlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "overlay I/O error: {e}"),
            Self::Json(e) => write!(f, "overlay state file error: {e}"),
            Self::EnvHashMismatch {
                expected, found, ..
            } => {
                write!(
                    f,
                    "overlay environment mismatch: expected {expected}, found {found}"
                )
            }
            Self::TransportMismatch {
                expected, found, ..
            } => {
                write!(
                    f,
                    "overlay transport mismatch: expected {expected}, found {found}"
                )
            }
            Self::VersionMismatch {
                expected, found, ..
            } => {
                write!(
                    f,
                    "overlay state version mismatch: expected {expected}, found {found}"
                )
            }
        }
    }
}

impl From<std::io::Error> for OverlayError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for OverlayError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Current state file format version. Bump when the format changes
/// incompatibly. Older versions for which we know how to migrate
/// in-place are handled in [`migrate_v1_to_v2`].
const STATE_VERSION: u32 = 2;

/// State file schema (v2 onwards): metadata only. Whiteouts and opaque
/// markers live on disk as `.wh.*` files in the upper directory.
#[derive(Serialize, Deserialize)]
struct StateFile {
    #[serde(default)]
    version: u32,
    env_hash: String,
    /// Transport that created this overlay (e.g. "fuse", "nfs").
    /// Used to detect incompatible overlay/transport combinations.
    #[serde(default)]
    transport: String,
}

/// V1 state file schema. Read once during migration to materialise
/// in-tree `.wh.*` markers, then discarded. Only the fields we still
/// care about are listed — `env_hash` and `transport` were already
/// validated by parsing the file as the current [`StateFile`] header.
#[derive(Deserialize)]
struct StateFileV1 {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    whiteouts: Vec<PathBuf>,
    #[serde(default)]
    opaque_dirs: Vec<PathBuf>,
}

/// Persistent overlay state: in-memory cache of whiteouts/opaque
/// directories backed by `.wh.*` markers on disk, plus environment
/// identity and transport.
///
/// Holds an exclusive file lock on `.rattler_fs_state.lock` for its entire
/// lifetime, preventing concurrent access to the same overlay directory.
/// The lock is released automatically on drop (or on process crash, since
/// advisory file locks are released by the OS).
pub struct OverlayState {
    dir: PathBuf,
    pub(crate) whiteouts: HashSet<PathBuf>,
    pub(crate) opaque_dirs: HashSet<PathBuf>,
    state_path: PathBuf,
    env_hash: String,
    transport: String,
    /// Exclusive file lock held for the lifetime of this state.  Dropping
    /// the `File` releases the lock.
    _lock: fs::File,
}

impl std::fmt::Debug for OverlayState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayState")
            .field("dir", &self.dir)
            .field("env_hash", &self.env_hash)
            .field("transport", &self.transport)
            .field("whiteouts", &self.whiteouts.len())
            .field("opaque_dirs", &self.opaque_dirs.len())
            .finish()
    }
}

impl OverlayState {
    /// Acquire the overlay directory lock without loading state.
    ///
    /// Returns the lock file handle. Use with [`Self::load_with_lock`] when the
    /// caller needs to hold the lock across a wipe-and-retry cycle.
    pub fn acquire_lock(dir: &Path) -> Result<fs::File, OverlayError> {
        fs::create_dir_all(dir)?;
        let lock_path = dir.join(STATE_LOCK_FILENAME);
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        if lock.try_lock_exclusive().is_err() {
            tracing::warn!(
                "waiting for overlay state lock at {} — another process holds the lock",
                lock_path.display(),
            );
            lock.lock_exclusive()?;
        }
        Ok(lock)
    }

    /// Load an existing overlay or create a new one.
    ///
    /// Acquires the directory lock internally.  If you need to hold the lock
    /// across a wipe-and-retry cycle, use [`Self::acquire_lock`] +
    /// [`Self::load_with_lock`] instead.
    pub fn load(dir: PathBuf, env_hash: String, transport: String) -> Result<Self, OverlayError> {
        let lock = Self::acquire_lock(&dir)?;
        Self::load_with_lock(dir, env_hash, transport, lock)
    }

    /// Load an existing overlay using a pre-acquired lock.
    ///
    /// The `lock` must have been obtained via [`Self::acquire_lock`] on the same
    /// directory.  This variant exists so `create_overlay` can hold the lock
    /// across the wipe-and-retry path without a race window.
    ///
    /// Performs in-place migration of v1 state files (which carried
    /// whiteouts inline as JSON) to v2 (whiteouts as on-disk `.wh.*`
    /// markers).
    pub fn load_with_lock(
        dir: PathBuf,
        env_hash: String,
        transport: String,
        lock: fs::File,
    ) -> Result<Self, OverlayError> {
        fs::create_dir_all(&dir)?;
        let state_path = dir.join(STATE_FILENAME);

        if state_path.exists() {
            let content = fs::read_to_string(&state_path)?;
            let header: StateFile = serde_json::from_str(&content)?;
            if header.env_hash != env_hash {
                return Err(OverlayError::EnvHashMismatch {
                    expected: env_hash,
                    found: header.env_hash,
                    lock,
                });
            }
            if !header.transport.is_empty() && header.transport != transport {
                return Err(OverlayError::TransportMismatch {
                    expected: transport,
                    found: header.transport,
                    lock,
                });
            }
            match header.version {
                STATE_VERSION => {}
                1 => migrate_v1_to_v2(&dir, &content)?,
                v => {
                    return Err(OverlayError::VersionMismatch {
                        expected: STATE_VERSION,
                        found: v,
                        lock,
                    });
                }
            }
        }

        let (whiteouts, opaque_dirs) = scan_markers(&dir)?;

        let overlay = OverlayState {
            dir,
            whiteouts,
            opaque_dirs,
            state_path,
            env_hash,
            transport,
            _lock: lock,
        };
        // Write fresh state header (v2 metadata only) — covers both the
        // brand-new case and the post-migration case.
        overlay.flush()?;
        Ok(overlay)
    }

    /// Check if a virtual path has been whiteout'd (deleted).
    pub fn is_whiteout(&self, path: &Path) -> bool {
        self.whiteouts.contains(path)
    }

    /// Mark a virtual path as deleted.
    ///
    /// Creates the on-disk `.wh.{name}` marker file in the upper-layer
    /// counterpart of `path`'s parent directory, then updates the
    /// in-memory cache. The marker is an empty regular file — the kernel
    /// overlayfs convention is a 0:0 char device, but those require root
    /// privileges; userspace overlays (podman, fuse-overlayfs) use plain
    /// files instead.
    pub fn add_whiteout(&mut self, path: PathBuf) -> Result<(), OverlayError> {
        let marker = whiteout_marker_path(&self.dir, &path)?;
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent)?;
        }
        // O_CREAT | O_TRUNC — atomic create, idempotent if the marker
        // already exists.
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&marker)?;
        self.whiteouts.insert(path);
        Ok(())
    }

    /// Remove a whiteout (e.g. when recreating a deleted file).
    ///
    /// Deletes the on-disk marker (no-op if missing) and updates the
    /// in-memory cache.
    pub fn remove_whiteout(&mut self, path: &Path) -> Result<(), OverlayError> {
        let was_present = self.whiteouts.remove(path);
        let marker = whiteout_marker_path(&self.dir, path)?;
        match fs::remove_file(&marker) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Marker missing — fine if the in-memory entry was also
                // missing (idempotent remove).
                if was_present {
                    tracing::debug!(
                        "remove_whiteout: in-memory entry present but marker file missing at {}",
                        marker.display()
                    );
                }
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Get the path in the upper layer for a given virtual path.
    pub fn upper_path(&self, virtual_path: &Path) -> PathBuf {
        self.dir.join(virtual_path)
    }

    /// Check if a file exists in the upper layer.
    pub fn has_upper(&self, virtual_path: &Path) -> bool {
        self.upper_path(virtual_path).exists()
    }

    /// Check if a directory is opaque (lower layer hidden entirely).
    pub fn is_opaque(&self, path: &Path) -> bool {
        self.opaque_dirs.contains(path)
    }

    /// Mark a directory as opaque.
    ///
    /// Creates the `.wh..wh..opq` marker file *inside* the upper-layer
    /// counterpart of `path`, then updates the in-memory cache.
    pub fn add_opaque_dir(&mut self, path: PathBuf) -> Result<(), OverlayError> {
        let marker = opaque_marker_path(&self.dir, &path);
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&marker)?;
        self.opaque_dirs.insert(path);
        Ok(())
    }

    /// Remove opaque marker from a directory.
    pub fn remove_opaque_dir(&mut self, path: &Path) -> Result<(), OverlayError> {
        let was_present = self.opaque_dirs.remove(path);
        let marker = opaque_marker_path(&self.dir, path);
        match fs::remove_file(&marker) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if was_present {
                    tracing::debug!(
                        "remove_opaque_dir: in-memory entry present but marker missing at {}",
                        marker.display()
                    );
                }
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The overlay directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Clean up every whiteout/opaque marker that lives directly under
    /// `virtual_dir` in the upper layer, and the corresponding in-memory
    /// entries.
    ///
    /// Used by `rmdir` to drain the marker files inside a directory
    /// that's about to be removed (otherwise `fs::remove_dir` would fail
    /// because the directory still contains those markers). The
    /// to-be-removed dir's own whiteout (recorded on its *parent*) takes
    /// over the hiding-from-the-lower-layer responsibility, so the
    /// per-child markers become redundant.
    pub fn clear_dir_markers(&mut self, virtual_dir: &Path) -> Result<(), OverlayError> {
        let upper = self.upper_path(virtual_dir);
        match fs::read_dir(&upper) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    if is_marker_name(&name) {
                        // Best-effort: ignore NotFound (race with concurrent remove).
                        match fs::remove_file(entry.path()) {
                            Ok(()) => {}
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                            Err(e) => return Err(e.into()),
                        }
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }

        // Drop the in-memory entries that lived under this dir.
        let dir_owned = virtual_dir.to_path_buf();
        self.whiteouts
            .retain(|p| p.parent().map(Path::to_path_buf) != Some(dir_owned.clone()));
        self.opaque_dirs.retain(|p| p != &dir_owned);
        Ok(())
    }

    /// Atomically write the v2 state header (`env_hash` + `transport` +
    /// `version`) to disk. Called on mount and on overlay-version
    /// migration. Whiteouts are NOT written here — they live as on-disk
    /// markers.
    pub(crate) fn flush(&self) -> Result<(), OverlayError> {
        let state = StateFile {
            version: STATE_VERSION,
            env_hash: self.env_hash.clone(),
            transport: self.transport.clone(),
        };
        let json = serde_json::to_string_pretty(&state)?;

        let tmp_path = self.state_path.with_extension("tmp");
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp_path, &self.state_path)?;
        Ok(())
    }
}

/// Translate `virtual_path` ("lib/foo.py") to the on-disk marker file
/// path ("{upper}/lib/.wh.foo.py").
///
/// Returns `Err` if `virtual_path` has no file name (e.g. is empty or
/// ends in `..`) — those paths cannot meaningfully be deleted.
fn whiteout_marker_path(upper_dir: &Path, virtual_path: &Path) -> Result<PathBuf, OverlayError> {
    let name = virtual_path.file_name().ok_or_else(|| {
        OverlayError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "cannot create whiteout for path with no file name: {}",
                virtual_path.display()
            ),
        ))
    })?;
    let parent = virtual_path.parent().unwrap_or_else(|| Path::new(""));
    let mut marker_name = OsString::from(WHITEOUT_PREFIX);
    marker_name.push(name);
    Ok(upper_dir.join(parent).join(marker_name))
}

/// Translate `virtual_dir` ("lib/python") to the opaque marker path
/// ("{upper}/lib/python/.wh..wh..opq").
fn opaque_marker_path(upper_dir: &Path, virtual_dir: &Path) -> PathBuf {
    upper_dir.join(virtual_dir).join(OPAQUE_MARKER)
}

/// Walk `upper_dir` recursively and collect any `.wh.*` markers into
/// (`whiteouts`, `opaque_dirs`) sets.
///
/// `upper_dir` not existing yet is fine — returns empty sets.
fn scan_markers(upper_dir: &Path) -> Result<(HashSet<PathBuf>, HashSet<PathBuf>), OverlayError> {
    let mut whiteouts = HashSet::new();
    let mut opaques = HashSet::new();
    if upper_dir.exists() {
        scan_markers_recursive(upper_dir, upper_dir, &mut whiteouts, &mut opaques)?;
    }
    Ok((whiteouts, opaques))
}

fn scan_markers_recursive(
    base: &Path,
    cur: &Path,
    whiteouts: &mut HashSet<PathBuf>,
    opaques: &mut HashSet<PathBuf>,
) -> Result<(), OverlayError> {
    let entries = match fs::read_dir(cur) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_owned();
        // Skip our own state files
        if name == STATE_FILENAME || name == STATE_TMP_FILENAME || name == STATE_LOCK_FILENAME {
            continue;
        }

        let relative = cur.strip_prefix(base).unwrap_or_else(|_| Path::new(""));
        if let Some(orig) = decode_marker(&name) {
            match orig {
                MarkerKind::Whiteout(target) => {
                    whiteouts.insert(relative.join(target));
                }
                MarkerKind::Opaque => {
                    opaques.insert(relative.to_path_buf());
                }
            }
            // Don't recurse into a marker (it's a regular file).
            continue;
        }

        if path.is_dir() {
            scan_markers_recursive(base, &path, whiteouts, opaques)?;
        }
    }
    Ok(())
}

enum MarkerKind {
    Whiteout(OsString),
    Opaque,
}

/// If `name` is a whiteout marker, return the original (un-whited-out)
/// filename; if it is an opaque marker, return [`MarkerKind::Opaque`];
/// otherwise return `None`.
fn decode_marker(name: &OsStr) -> Option<MarkerKind> {
    let s = name.to_str()?;
    if s == OPAQUE_MARKER {
        return Some(MarkerKind::Opaque);
    }
    if let Some(rest) = s.strip_prefix(WHITEOUT_PREFIX) {
        if rest.is_empty() {
            // ".wh." with no suffix — malformed, ignore.
            return None;
        }
        return Some(MarkerKind::Whiteout(OsString::from(rest)));
    }
    None
}

/// True if `name` is one of our internal markers (whiteout or opaque).
/// Callers serving readdir output to clients use this to filter the
/// upper directory.
pub(crate) fn is_marker_name(name: &OsStr) -> bool {
    decode_marker(name).is_some()
}

/// One-shot v1 → v2 migration.
///
/// Reads the v1 state file (which embedded the whiteout/opaque lists in
/// JSON) and materialises each entry as a `.wh.*` marker in the upper
/// directory. The state file itself is rewritten as v2 by the caller's
/// subsequent `flush()`.
fn migrate_v1_to_v2(upper_dir: &Path, v1_content: &str) -> Result<(), OverlayError> {
    let v1: StateFileV1 = serde_json::from_str(v1_content)?;
    debug_assert_eq!(v1.version, 1);
    tracing::info!(
        "migrating overlay state from v1 → v2 in {}: {} whiteouts, {} opaque dirs",
        upper_dir.display(),
        v1.whiteouts.len(),
        v1.opaque_dirs.len(),
    );
    for path in v1.whiteouts {
        let marker = whiteout_marker_path(upper_dir, &path)?;
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent)?;
        }
        // create_new is intentional: if a file with the same name already
        // exists in upper (the user did create-then-delete), the marker
        // collision is a real conflict and the migration should error.
        match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&marker)
        {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // Idempotent if migration was interrupted previously.
                tracing::debug!("migration: marker already exists at {}", marker.display());
            }
            Err(e) => return Err(e.into()),
        }
    }
    for path in v1.opaque_dirs {
        let marker = opaque_marker_path(upper_dir, &path);
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&marker)
        {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_creates_new_state() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        let state = OverlayState::load(dir.clone(), "hash123".into(), "test".into()).unwrap();

        assert!(state.whiteouts.is_empty());
        assert!(dir.join(STATE_FILENAME).exists());

        let content = fs::read_to_string(dir.join(STATE_FILENAME)).unwrap();
        let parsed: StateFile = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.env_hash, "hash123");
        assert_eq!(parsed.version, STATE_VERSION);
    }

    #[test]
    fn test_load_verifies_hash() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");

        OverlayState::load(dir.clone(), "hash_a".into(), "test".into()).unwrap();
        let err = OverlayState::load(dir, "hash_b".into(), "test".into()).unwrap_err();
        assert!(matches!(err, OverlayError::EnvHashMismatch { .. }));
    }

    #[test]
    fn test_load_accepts_matching_hash() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");

        OverlayState::load(dir.clone(), "hash_a".into(), "test".into()).unwrap();
        let state = OverlayState::load(dir, "hash_a".into(), "test".into()).unwrap();
        assert!(state.whiteouts.is_empty());
    }

    #[test]
    fn test_whiteout_add_remove() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        let mut state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();

        let path = PathBuf::from("lib/foo.py");
        assert!(!state.is_whiteout(&path));

        state.add_whiteout(path.clone()).unwrap();
        assert!(state.is_whiteout(&path));
        // Marker file present on disk
        assert!(dir.join("lib/.wh.foo.py").exists());

        state.remove_whiteout(&path).unwrap();
        assert!(!state.is_whiteout(&path));
        assert!(!dir.join("lib/.wh.foo.py").exists());
    }

    #[test]
    fn test_whiteout_persists_across_reload() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");

        {
            let mut state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();
            state.add_whiteout(PathBuf::from("lib/deleted.py")).unwrap();
        }

        let state = OverlayState::load(dir, "hash".into(), "test".into()).unwrap();
        assert!(state.is_whiteout(Path::new("lib/deleted.py")));
    }

    #[test]
    fn test_opaque_dir_add_remove() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        let mut state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();

        let path = PathBuf::from("lib/python");
        assert!(!state.is_opaque(&path));

        state.add_opaque_dir(path.clone()).unwrap();
        assert!(state.is_opaque(&path));
        assert!(dir.join("lib/python/.wh..wh..opq").exists());

        state.remove_opaque_dir(&path).unwrap();
        assert!(!state.is_opaque(&path));
        assert!(!dir.join("lib/python/.wh..wh..opq").exists());
    }

    #[test]
    fn test_opaque_persists_across_reload() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");

        {
            let mut state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();
            state.add_opaque_dir(PathBuf::from("lib/python")).unwrap();
        }

        let state = OverlayState::load(dir, "hash".into(), "test".into()).unwrap();
        assert!(state.is_opaque(Path::new("lib/python")));
    }

    #[test]
    fn test_upper_path() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        let state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();

        assert_eq!(
            state.upper_path(Path::new("lib/foo.py")),
            dir.join("lib/foo.py")
        );
    }

    #[test]
    fn test_has_upper() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        let state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();

        assert!(!state.has_upper(Path::new("lib/foo.py")));

        fs::create_dir_all(dir.join("lib")).unwrap();
        fs::write(dir.join("lib/foo.py"), b"content").unwrap();

        assert!(state.has_upper(Path::new("lib/foo.py")));
    }

    #[test]
    fn test_flush_produces_metadata_only_json() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        let mut state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();

        state.add_whiteout(PathBuf::from("a/b.py")).unwrap();
        state.add_whiteout(PathBuf::from("c/d.py")).unwrap();

        // The state file is metadata only; whiteouts are out-of-band markers.
        let content = fs::read_to_string(dir.join(STATE_FILENAME)).unwrap();
        assert!(!content.contains("whiteouts"));
        assert!(!content.contains("opaque_dirs"));
        let parsed: StateFile = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.env_hash, "hash");
        assert_eq!(parsed.version, STATE_VERSION);

        // Markers are on disk.
        assert!(dir.join("a/.wh.b.py").exists());
        assert!(dir.join("c/.wh.d.py").exists());
    }

    #[test]
    fn test_v1_state_file_migrates() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        fs::create_dir_all(&dir).unwrap();

        // Write a v1 state file by hand.
        let v1 = serde_json::json!({
            "version": 1,
            "env_hash": "hash",
            "transport": "test",
            "whiteouts": ["lib/python/foo.py", "bin/oldscript"],
            "opaque_dirs": ["lib/etc"],
        });
        fs::write(
            dir.join(STATE_FILENAME),
            serde_json::to_string_pretty(&v1).unwrap(),
        )
        .unwrap();

        // Loading should migrate.
        let state = OverlayState::load(dir.clone(), "hash".into(), "test".into()).unwrap();
        assert!(state.is_whiteout(Path::new("lib/python/foo.py")));
        assert!(state.is_whiteout(Path::new("bin/oldscript")));
        assert!(state.is_opaque(Path::new("lib/etc")));

        // The state file is now v2 and contains no inline whiteouts.
        let content = fs::read_to_string(dir.join(STATE_FILENAME)).unwrap();
        assert!(!content.contains("whiteouts"));
        assert!(!content.contains("opaque_dirs"));
        let parsed: StateFile = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.version, STATE_VERSION);

        // Markers are on disk.
        assert!(dir.join("lib/python/.wh.foo.py").exists());
        assert!(dir.join("bin/.wh.oldscript").exists());
        assert!(dir.join("lib/etc/.wh..wh..opq").exists());
    }

    #[test]
    fn test_unknown_future_version_rejected() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        fs::create_dir_all(&dir).unwrap();
        let v999 = serde_json::json!({
            "version": 999,
            "env_hash": "hash",
            "transport": "test",
        });
        fs::write(
            dir.join(STATE_FILENAME),
            serde_json::to_string_pretty(&v999).unwrap(),
        )
        .unwrap();
        let err = OverlayState::load(dir, "hash".into(), "test".into()).unwrap_err();
        assert!(matches!(
            err,
            OverlayError::VersionMismatch { found: 999, .. }
        ));
    }

    #[test]
    fn test_whiteout_marker_path_components() {
        let dir = PathBuf::from("/upper");
        assert_eq!(
            whiteout_marker_path(&dir, Path::new("lib/foo.py")).unwrap(),
            PathBuf::from("/upper/lib/.wh.foo.py")
        );
        assert_eq!(
            whiteout_marker_path(&dir, Path::new("foo")).unwrap(),
            PathBuf::from("/upper/.wh.foo")
        );
    }

    #[test]
    fn test_decode_marker() {
        assert!(matches!(
            decode_marker(OsStr::new(".wh.foo")),
            Some(MarkerKind::Whiteout(_))
        ));
        assert!(matches!(
            decode_marker(OsStr::new(".wh..wh..opq")),
            Some(MarkerKind::Opaque)
        ));
        assert!(decode_marker(OsStr::new("foo")).is_none());
        assert!(decode_marker(OsStr::new(".wh.")).is_none());
    }

    #[test]
    fn test_is_marker_name() {
        assert!(is_marker_name(OsStr::new(".wh.foo")));
        assert!(is_marker_name(OsStr::new(".wh..wh..opq")));
        assert!(!is_marker_name(OsStr::new("regular_file")));
    }

    #[test]
    fn test_scan_markers_finds_existing() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("overlay");
        fs::create_dir_all(dir.join("lib/python")).unwrap();
        fs::create_dir_all(dir.join("etc")).unwrap();
        fs::write(dir.join("lib/python/.wh.foo.py"), b"").unwrap();
        fs::write(dir.join(".wh.toplevel"), b"").unwrap();
        fs::write(dir.join("etc/.wh..wh..opq"), b"").unwrap();
        fs::write(dir.join("regular.txt"), b"hi").unwrap();

        let (wos, opqs) = scan_markers(&dir).unwrap();
        assert!(wos.contains(Path::new("lib/python/foo.py")));
        assert!(wos.contains(Path::new("toplevel")));
        assert!(opqs.contains(Path::new("etc")));
        assert_eq!(wos.len(), 2);
        assert_eq!(opqs.len(), 1);
    }
}
