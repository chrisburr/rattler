use std::{
    fmt::{Debug, Formatter},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use digest::generic_array::GenericArray;
use fs4::fs_std::FileExt;
use rattler_conda_types::package::{IndexJson, PathsJson};
use rattler_digest::Sha256Hash;

use crate::package_cache::PackageCacheLayerError;

/// A validated cache entry with its associated metadata.
///
/// This struct represents a cache entry that has been validated and is ready for use.
/// It holds the cache entry's path, revision number, and optional SHA256 hash.
///
/// Note: Concurrent access is coordinated via the global cache lock mechanism
/// (see [`CacheGlobalLock`]). Individual cache entries do not hold locks.
pub struct CacheMetadata {
    pub(super) revision: u64,
    pub(super) sha256: Option<Sha256Hash>,
    pub(super) path: PathBuf,
    pub(super) index_json: Option<IndexJson>,
    pub(super) paths_json: Option<PathsJson>,
}

impl Debug for CacheMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheMetadata")
            .field("path", &self.path)
            .field("revision", &self.revision)
            .field("sha256", &self.sha256)
            .finish()
    }
}

impl CacheMetadata {
    /// Returns the path to the cache entry on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the revision of the cache entry. This revision indicates the
    /// number of times the cache entry has been updated.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the cached `index.json` data if it was read during validation.
    pub fn index_json(&self) -> Option<&IndexJson> {
        self.index_json.as_ref()
    }

    /// Returns the cached `paths.json` data if it was read during validation.
    pub fn paths_json(&self) -> Option<&PathsJson> {
        self.paths_json.as_ref()
    }
}

/// A global lock for the entire package cache.
///
/// This can be used to reduce lock overhead when performing many package
/// operations by acquiring a single global lock instead of individual per-package locks.
pub struct CacheGlobalLock {
    file: std::fs::File,
}

impl Debug for CacheGlobalLock {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheGlobalLock").finish()
    }
}

impl Drop for CacheGlobalLock {
    fn drop(&mut self) {
        // Ensure that the lock is released when the lock is dropped.
        let _ = fs4::fs_std::FileExt::unlock(&self.file);
    }
}

impl CacheGlobalLock {
    /// Acquires a global write lock on the package cache.
    ///
    /// This lock should be used to coordinate access across multiple package
    /// operations to reduce the overhead of acquiring individual locks.
    pub async fn acquire(path: &Path) -> Result<Self, PackageCacheLayerError> {
        let lock_file_path = path.to_path_buf();
        let acquire_lock_fut = simple_spawn_blocking::tokio::run_blocking_task(move || {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .read(true)
                .open(&lock_file_path)
                .map_err(|e| {
                    PackageCacheLayerError::LockError(
                        format!(
                            "failed to open global cache lock for writing: '{}'",
                            lock_file_path.display()
                        ),
                        e,
                    )
                })?;

            file.lock_exclusive().map_err(move |e| {
                PackageCacheLayerError::LockError(
                    format!(
                        "failed to acquire write lock on global cache lock file: '{}'",
                        lock_file_path.display()
                    ),
                    e,
                )
            })?;

            Ok(CacheGlobalLock { file })
        });

        tokio::select!(
            lock = acquire_lock_fut => lock,
            _ = warn_timeout_future(
                "Blocking waiting for global file lock on package cache".to_string()
            ) => unreachable!("warn_timeout_future should never finish")
        )
    }
}

/// A handle to a cache metadata file.
///
/// This struct manages access to a `.lock` file that stores metadata about a cache entry,
/// including its revision number and optional SHA256 hash. It does not provide filesystem
/// locking - concurrent access should be coordinated via [`CacheGlobalLock`].
///
/// The handle may be backed by no file at all when the layer is read-only and the
/// metadata file is absent or cannot be opened — in that case reads return defaults
/// and writes are rejected.
pub struct CacheMetadataFile {
    file: Option<Arc<std::fs::File>>,
}

impl CacheMetadataFile {
    /// Acquires a handle to the cache metadata file.
    ///
    /// Opens the file with both read and write permissions. Since concurrent access
    /// is coordinated via [`CacheGlobalLock`], this single method is sufficient for
    /// all metadata operations.
    pub async fn acquire(path: &Path) -> Result<Self, PackageCacheLayerError> {
        let lock_file_path = path.to_path_buf();

        simple_spawn_blocking::tokio::run_blocking_task(move || {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&lock_file_path)
                .map_err(|e| {
                    PackageCacheLayerError::LockError(
                        format!(
                            "failed to open cache metadata file: '{}'",
                            lock_file_path.display()
                        ),
                        e,
                    )
                })?;

            Ok(CacheMetadataFile {
                file: Some(Arc::new(file)),
            })
        })
        .await
    }

    /// Acquires a read-only handle to the cache metadata file.
    ///
    /// Opens the file without creating it and without requesting write access, so it
    /// works for layers stored on read-only filesystems (e.g. CVMFS, NFS-RO,
    /// squashfs). When the file cannot be opened — because it does not exist or the
    /// open is otherwise rejected — a handle with no underlying file is returned;
    /// subsequent reads then yield defaults (revision `0`, no sha256). Writes are
    /// rejected for read-only handles.
    pub async fn acquire_readonly(path: &Path) -> Self {
        let lock_file_path = path.to_path_buf();

        let file: Result<Option<std::fs::File>, PackageCacheLayerError> =
            simple_spawn_blocking::tokio::run_blocking_task(move || {
                Ok(std::fs::OpenOptions::new()
                    .read(true)
                    .open(&lock_file_path)
                    .ok())
            })
            .await;

        CacheMetadataFile {
            file: file.ok().flatten().map(Arc::new),
        }
    }
}

impl CacheMetadataFile {
    pub async fn write_revision_and_sha(
        &mut self,
        revision: u64,
        sha256: Option<&Sha256Hash>,
    ) -> Result<(), PackageCacheLayerError> {
        let Some(file) = self.file.clone() else {
            return Err(PackageCacheLayerError::LockError(
                "cannot write to a read-only cache metadata handle".to_string(),
                std::io::Error::from(std::io::ErrorKind::ReadOnlyFilesystem),
            ));
        };

        let sha256 = sha256.cloned();
        simple_spawn_blocking::tokio::run_blocking_task(move || {
            // Ensure we write from the start of the file
            (&*file).rewind().map_err(|e| {
                PackageCacheLayerError::LockError(
                    "failed to rewind cache lock for reading revision".to_string(),
                    e,
                )
            })?;

            // Write the bytes of the revision
            let revision_bytes = revision.to_be_bytes();
            (&*file).write_all(&revision_bytes).map_err(|e| {
                PackageCacheLayerError::LockError(
                    "failed to write revision from cache lock".to_string(),
                    e,
                )
            })?;

            // Write the bytes of the sha256 hash
            let sha_bytes = if let Some(sha) = sha256 {
                let len = sha.len();
                let sha = &sha[..];
                (&*file).write_all(sha).map_err(|e| {
                    PackageCacheLayerError::LockError(
                        "failed to write sha256 from cache lock".to_string(),
                        e,
                    )
                })?;
                len
            } else {
                0
            };

            // Ensure all bytes are written to disk
            (&*file).flush().map_err(|e| {
                PackageCacheLayerError::LockError(
                    "failed to flush cache lock after writing revision".to_string(),
                    e,
                )
            })?;

            // Update the length of the file
            let file_length = revision_bytes.len() + sha_bytes;
            file.set_len(file_length as u64).map_err(|e| {
                PackageCacheLayerError::LockError(
                    "failed to truncate cache lock after writing revision".to_string(),
                    e,
                )
            })?;

            Ok(())
        })
        .await
    }

    /// Reads the revision from the cache metadata file.
    pub fn read_revision(&mut self) -> Result<u64, PackageCacheLayerError> {
        let Some(file) = self.file.as_ref() else {
            return Ok(0);
        };
        (&**file).rewind().map_err(|e| {
            PackageCacheLayerError::LockError(
                "failed to rewind cache lock for reading revision".to_string(),
                e,
            )
        })?;
        let mut buf = [0; 8];
        match (&**file).read_exact(&mut buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(0);
            }
            Err(e) => {
                return Err(PackageCacheLayerError::LockError(
                    "failed to read revision from cache lock".to_string(),
                    e,
                ));
            }
        }
        Ok(u64::from_be_bytes(buf))
    }

    /// Reads the sha256 hash from the cache metadata file.
    pub fn read_sha256(&mut self) -> Result<Option<Sha256Hash>, PackageCacheLayerError> {
        const SHA256_LEN: usize = 32;
        const REVISION_LEN: u64 = 8;
        let Some(file) = self.file.as_ref() else {
            return Ok(None);
        };
        (&**file).rewind().map_err(|e| {
            PackageCacheLayerError::LockError(
                "failed to rewind cache lock for reading sha256".to_string(),
                e,
            )
        })?;
        let mut buf = [0; SHA256_LEN];
        let _ = (&**file)
            .seek(SeekFrom::Start(REVISION_LEN))
            .map_err(|e| {
                PackageCacheLayerError::LockError(
                    "failed to seek to sha256 in cache lock".to_string(),
                    e,
                )
            })?;
        match (&**file).read_exact(&mut buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(e) => {
                return Err(PackageCacheLayerError::LockError(
                    "failed to read sha256 from cache lock".to_string(),
                    e,
                ));
            }
        }
        Ok(Some(GenericArray::clone_from_slice(&buf)))
    }
}

async fn warn_timeout_future(message: String) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        tracing::warn!("{}", &message);
    }
}

#[cfg(test)]
mod tests {
    use rattler_digest::{parse_digest_from_hex, Sha256};

    use super::CacheMetadataFile;

    #[tokio::test]
    async fn cache_metadata_serialize_deserialize() {
        // Temporarily create a metadata file and write a revision and sha to it
        let temp_dir = tempfile::tempdir().unwrap();
        let metadata_file = temp_dir.path().join("foo.lock");
        // Acquire a handle on the file
        let mut metadata = CacheMetadataFile::acquire(&metadata_file).await.unwrap();
        // Write a revision and sha to the lock file
        let sha = parse_digest_from_hex::<Sha256>(
            "4dd9893f1eee45e1579d1a4f5533ef67a84b5e4b7515de7ed0db1dd47adc6bc8",
        );
        metadata
            .write_revision_and_sha(1, sha.as_ref())
            .await
            .unwrap();
        // Read back the revision and sha from the metadata file
        let revision = metadata.read_revision().unwrap();
        assert_eq!(revision, 1);
        let read_sha = metadata.read_sha256().unwrap();
        assert_eq!(sha, read_sha);
    }
}
