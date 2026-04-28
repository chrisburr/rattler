//! FUSE transport adapter.
//!
//! Thin layer that maps `fuser::Filesystem` callbacks to `VfsOps` trait methods.
//! Handles FUSE-specific concerns: capability negotiation, passthrough, reply types.

use fuser::{
    Errno, FileHandle, Filesystem, FopenFlags, INodeNo, InitFlags, KernelConfig, LockOwner,
    OpenFlags, ReplyAttr, ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyXattr,
    Request,
};
use libc::EIO;
use memmap2::Mmap;
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::File,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::Duration,
};

use crate::vfs_ops::{ContentSource, FileAttr, FileKind, VfsOps};

/// Convert mutex `PoisonError` into an errno on the FUSE reply.
///
/// Mirrors the helper in [`crate::overlay_fs`]: poisoning one lock should
/// fail the current syscall, not propagate a panic across the FUSE FFI
/// boundary and kill the session. Locally-scoped here so the FUSE adapter
/// stays self-contained.
trait MutexExt<T> {
    fn lock_or_eio(&self) -> Result<MutexGuard<'_, T>, i32>;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_or_eio(&self) -> Result<MutexGuard<'_, T>, i32> {
        self.lock().map_err(|e| {
            tracing::error!("FUSE adapter mutex poisoned: {e}");
            EIO
        })
    }
}

// ---------------------------------------------------------------------------
// Conversions between crate-local types and fuser types
// ---------------------------------------------------------------------------

impl From<FileKind> for fuser::FileType {
    fn from(kind: FileKind) -> fuser::FileType {
        match kind {
            FileKind::RegularFile => fuser::FileType::RegularFile,
            FileKind::Directory => fuser::FileType::Directory,
            FileKind::Symlink => fuser::FileType::Symlink,
        }
    }
}

impl From<FileAttr> for fuser::FileAttr {
    fn from(attr: FileAttr) -> fuser::FileAttr {
        fuser::FileAttr {
            ino: INodeNo(attr.ino),
            size: attr.size,
            blocks: attr.blocks,
            atime: attr.atime,
            mtime: attr.mtime,
            ctime: attr.ctime,
            crtime: attr.atime,
            kind: attr.kind.into(),
            perm: attr.perm,
            nlink: attr.nlink,
            uid: attr.uid,
            gid: attr.gid,
            rdev: 0,
            flags: 0,
            blksize: 512,
        }
    }
}

/// Content of an open file managed by the adapter.
enum OpenFileContent {
    /// Materialized content from the VFS (e.g. after prefix replacement).
    Materialized(Vec<u8>),
    /// Raw mmap of the cache file (no prefix replacement needed).
    Mapped(Mmap),
    /// Kernel handles reads directly via the backing fd (Linux 6.9+).
    /// The `BackingId` must stay alive until `release()`.
    Passthrough { _backing_id: Arc<fuser::BackingId> },
    /// VFS-managed handle (overlay writable files). Reads/writes/release
    /// must be delegated back to the VFS using this key.
    VfsManaged(u64),
    /// Inode-based reads for transformed/virtual content. Each `read()`
    /// call delegates to `VfsOps::read(ino, offset, size)`.
    InodeRead(u64),
}

impl OpenFileContent {
    fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Materialized(buf) => Some(buf),
            Self::Mapped(mmap) => Some(mmap.as_ref()),
            Self::Passthrough { .. } | Self::VfsManaged(_) | Self::InodeRead(_) => None,
        }
    }
}

/// An open file as the adapter sees it. `bound_ino` is the inode the fd was
/// opened on — it's preserved across overlay COW so `fstat()` on this fh
/// continues to return the original inode's attrs even after a sibling fd
/// promotes the path to upper.
struct OpenFile {
    bound_ino: u64,
    content: OpenFileContent,
}

/// FUSE transport adapter, generic over any `VfsOps` implementation.
pub struct FuseAdapter<T: VfsOps> {
    vfs: T,
    open_files: Mutex<HashMap<u64, OpenFile>>,
    next_fh: AtomicU64,
    passthrough_enabled: AtomicBool,
}

impl<T: VfsOps> FuseAdapter<T> {
    pub fn new(vfs: T) -> Self {
        FuseAdapter {
            vfs,
            open_files: Mutex::new(HashMap::new()),
            next_fh: AtomicU64::new(1),
            passthrough_enabled: AtomicBool::new(false),
        }
    }

    /// Allocate a new fh and stash the open file. Returns `EIO` if the
    /// `open_files` mutex is poisoned.
    fn store_open(&self, bound_ino: u64, content: OpenFileContent) -> Result<u64, i32> {
        let fh = self.next_fh.fetch_add(1, Ordering::Relaxed);
        self.open_files
            .lock_or_eio()?
            .insert(fh, OpenFile { bound_ino, content });
        Ok(fh)
    }

    /// Read the bound inode out of the `open_files` map. Used by metadata
    /// callbacks that receive `fh` so they can route through `getattr_strict`.
    fn bound_ino_for(&self, fh: FileHandle) -> Option<u64> {
        self.open_files
            .lock_or_eio()
            .ok()
            .and_then(|m| m.get(&fh.0).map(|f| f.bound_ino))
    }

    /// Pick the VFS method to use for a `getattr` request.
    ///
    /// When `fh` is present, the fd is bound to a specific inode that must
    /// not be redirected through the overlay's `promoted` map (otherwise
    /// `fstat(fd)` would shift to the upper inode after a sibling fd
    /// triggers COW and disagree with `read(fd)`). Without an `fh` it's a
    /// path-side stat and the redirect is the right behavior.
    fn resolve_attr_for_fh(&self, ino: u64, fh: Option<FileHandle>) -> Result<FileAttr, i32> {
        match fh.and_then(|fh| self.bound_ino_for(fh)) {
            Some(bound_ino) => self.vfs.getattr_strict(bound_ino),
            None => self.vfs.getattr(ino),
        }
    }
}

impl<T: VfsOps> Filesystem for FuseAdapter<T> {
    fn init(&mut self, _req: &Request, config: &mut KernelConfig) -> std::io::Result<()> {
        if config.add_capabilities(InitFlags::FUSE_PASSTHROUGH).is_ok() {
            self.passthrough_enabled.store(true, Ordering::Relaxed);
            tracing::info!("FUSE passthrough enabled");
        }
        Ok(())
    }

    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        match self.vfs.lookup(parent.0, name) {
            Ok(attr) => {
                let fattr: fuser::FileAttr = attr.into();
                reply.entry(&Duration::MAX, &fattr, fuser::Generation(0));
            }
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, fh: Option<FileHandle>, reply: ReplyAttr) {
        match self.resolve_attr_for_fh(ino.0, fh) {
            Ok(attr) => {
                let fattr: fuser::FileAttr = attr.into();
                reply.attr(&Duration::MAX, &fattr);
            }
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn readlink(&self, _req: &Request, ino: INodeNo, reply: ReplyData) {
        match self.vfs.readlink(ino.0) {
            Ok(target) => reply.data(target.as_os_str().as_encoded_bytes()),
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn open(&self, _req: &Request, ino: INodeNo, flags: OpenFlags, reply: ReplyOpen) {
        let write = matches!(
            flags.acc_mode(),
            fuser::OpenAccMode::O_WRONLY | fuser::OpenAccMode::O_RDWR
        );

        if write {
            // Write path: delegate to VFS open_write
            match self.vfs.open_write(ino.0) {
                Ok(vfs_fh) => match self.store_open(ino.0, OpenFileContent::VfsManaged(vfs_fh)) {
                    Ok(fh) => reply.opened(FileHandle(fh), FopenFlags::empty()),
                    Err(e) => {
                        self.vfs.release_write(vfs_fh);
                        reply.error(Errno::from_i32(e));
                    }
                },
                Err(e) => reply.error(Errno::from_i32(e)),
            }
            return;
        }

        // Read path: use content_source to decide strategy
        match self.vfs.content_source(ino.0) {
            Ok(ContentSource::Direct(path)) => {
                if self.passthrough_enabled.load(Ordering::Relaxed) {
                    let file = match File::open(&path) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::warn!("passthrough open failed for {}: {}", path.display(), e);
                            reply.error(Errno::from_i32(EIO));
                            return;
                        }
                    };
                    match reply.open_backing(&file) {
                        Ok(id) => {
                            let backing_id = Arc::new(id);
                            let fh = self.next_fh.fetch_add(1, Ordering::Relaxed);
                            reply.opened_passthrough(
                                FileHandle(fh),
                                FopenFlags::FOPEN_KEEP_CACHE,
                                &backing_id,
                            );
                            let entry = OpenFile {
                                bound_ino: ino.0,
                                content: OpenFileContent::Passthrough {
                                    _backing_id: backing_id,
                                },
                            };
                            match self.open_files.lock_or_eio() {
                                Ok(mut m) => {
                                    m.insert(fh, entry);
                                }
                                Err(_) => {
                                    tracing::error!(
                                        "FUSE passthrough open: open_files poisoned, leaking fh={fh}"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            // Passthrough negotiated but not usable at runtime;
                            // disable and fall back to mmap for this and future opens.
                            tracing::warn!(
                                "FUSE passthrough unavailable at runtime ({}), falling back to mmap",
                                e
                            );
                            self.passthrough_enabled.store(false, Ordering::Relaxed);
                            let mmap = match unsafe { Mmap::map(&file) } {
                                Ok(m) => m,
                                Err(e) => {
                                    tracing::warn!("failed to mmap {}: {}", path.display(), e);
                                    reply.error(Errno::from_i32(EIO));
                                    return;
                                }
                            };
                            match self.store_open(ino.0, OpenFileContent::Mapped(mmap)) {
                                Ok(fh) => {
                                    reply.opened(FileHandle(fh), FopenFlags::FOPEN_KEEP_CACHE)
                                }
                                Err(e) => reply.error(Errno::from_i32(e)),
                            }
                        }
                    };
                } else {
                    // Passthrough not available — mmap and serve normally
                    let file = match File::open(&path) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::warn!("failed to open {}: {}", path.display(), e);
                            reply.error(Errno::from_i32(EIO));
                            return;
                        }
                    };
                    let mmap = match unsafe { Mmap::map(&file) } {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::warn!("failed to mmap {}: {}", path.display(), e);
                            reply.error(Errno::from_i32(EIO));
                            return;
                        }
                    };
                    match self.store_open(ino.0, OpenFileContent::Mapped(mmap)) {
                        Ok(fh) => reply.opened(FileHandle(fh), FopenFlags::FOPEN_KEEP_CACHE),
                        Err(e) => reply.error(Errno::from_i32(e)),
                    }
                }
            }
            Ok(ContentSource::Transformed | ContentSource::Virtual) => {
                // Inode-based reads — store the inode so read() can call VFS
                match self.store_open(ino.0, OpenFileContent::InodeRead(ino.0)) {
                    Ok(fh) => reply.opened(FileHandle(fh), FopenFlags::FOPEN_KEEP_CACHE),
                    Err(e) => reply.error(Errno::from_i32(e)),
                }
            }
            Err(e) => {
                // Directories/symlinks — open with empty content
                let _ = e;
                match self.store_open(ino.0, OpenFileContent::Materialized(vec![])) {
                    Ok(fh) => reply.opened(FileHandle(fh), FopenFlags::FOPEN_KEEP_CACHE),
                    Err(e) => reply.error(Errno::from_i32(e)),
                }
            }
        }
    }

    fn release(
        &self,
        _req: &Request,
        _ino: INodeNo,
        fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        if fh != FileHandle(0) {
            // A poisoned mutex here would leak the entry but the kernel
            // already considers the fh closed; nothing useful to surface.
            let removed = match self.open_files.lock_or_eio() {
                Ok(mut m) => m.remove(&fh.0),
                Err(_) => None,
            };
            // Propagate release to VFS for overlay-managed handles
            if let Some(OpenFile {
                content: OpenFileContent::VfsManaged(vfs_fh),
                ..
            }) = removed
            {
                self.vfs.release_write(vfs_fh);
            }
        }
        reply.ok();
    }

    fn read(
        &self,
        _req: &Request,
        _ino: INodeNo,
        fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let lock = match self.open_files.lock_or_eio() {
            Ok(l) => l,
            Err(e) => {
                reply.error(Errno::from_i32(e));
                return;
            }
        };
        let Some(open_file) = lock.get(&fh.0) else {
            reply.error(Errno::from_i32(EIO));
            return;
        };

        match &open_file.content {
            OpenFileContent::VfsManaged(vfs_fh) => {
                let vfs_fh = *vfs_fh;
                drop(lock); // release adapter lock before calling into VFS
                match self.vfs.read_handle(vfs_fh, offset, size) {
                    Ok(data) => reply.data(&data),
                    Err(e) => reply.error(Errno::from_i32(e)),
                }
            }
            OpenFileContent::InodeRead(ino) => {
                let ino = *ino;
                drop(lock); // release adapter lock before calling into VFS
                match self.vfs.read(ino, offset, size) {
                    Ok(data) => reply.data(&data),
                    Err(e) => reply.error(Errno::from_i32(e)),
                }
            }
            content => match content.as_bytes() {
                Some(data) => {
                    let start = (offset as usize).min(data.len());
                    let end = (start + size as usize).min(data.len());
                    reply.data(&data[start..end]);
                }
                None => {
                    // Passthrough — kernel handles reads, shouldn't reach here
                    reply.error(Errno::from_i32(EIO));
                }
            },
        }
    }

    fn getxattr(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _name: &OsStr,
        _size: u32,
        reply: ReplyXattr,
    ) {
        reply.error(Errno::NO_XATTR);
    }

    fn listxattr(&self, _req: &Request, _ino: INodeNo, size: u32, reply: ReplyXattr) {
        if size == 0 {
            reply.size(0);
        } else {
            reply.data(&[]);
        }
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        match self.vfs.readdir(ino.0, offset) {
            Ok(entries) => {
                for (i, entry) in entries.into_iter().enumerate() {
                    let ftype: fuser::FileType = entry.kind.into();
                    if reply.add(
                        INodeNo(entry.ino),
                        offset + i as u64 + 1,
                        ftype,
                        &entry.name,
                    ) {
                        break;
                    }
                }
                reply.ok();
            }
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    // Write operations — delegate to VfsOps (EROFS for read-only, overlay handles them)

    fn create(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: fuser::ReplyCreate,
    ) {
        match self.vfs.create(parent.0, name, mode) {
            Ok((attr, vfs_fh)) => {
                let fattr: fuser::FileAttr = attr.into();
                match self.store_open(attr.ino, OpenFileContent::VfsManaged(vfs_fh)) {
                    Ok(fh) => reply.created(
                        &Duration::MAX,
                        &fattr,
                        fuser::Generation(0),
                        FileHandle(fh),
                        FopenFlags::empty(),
                    ),
                    Err(e) => {
                        self.vfs.release_write(vfs_fh);
                        reply.error(Errno::from_i32(e));
                    }
                }
            }
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn write(
        &self,
        _req: &Request,
        _ino: INodeNo,
        fh: FileHandle,
        offset: u64,
        data: &[u8],
        _write_flags: fuser::WriteFlags,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: fuser::ReplyWrite,
    ) {
        let vfs_fh = match self.open_files.lock_or_eio() {
            Ok(map) => match map.get(&fh.0) {
                Some(OpenFile {
                    content: OpenFileContent::VfsManaged(vfs_fh),
                    ..
                }) => *vfs_fh,
                _ => {
                    reply.error(Errno::from_i32(EIO));
                    return;
                }
            },
            Err(e) => {
                reply.error(Errno::from_i32(e));
                return;
            }
        };
        match self.vfs.write(vfs_fh, offset, data) {
            Ok(written) => reply.written(written),
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn unlink(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        match self.vfs.unlink(parent.0, name) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn mkdir(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        match self.vfs.mkdir(parent.0, name, mode) {
            Ok(attr) => {
                let fattr: fuser::FileAttr = attr.into();
                reply.entry(&Duration::MAX, &fattr, fuser::Generation(0));
            }
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn rmdir(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        match self.vfs.rmdir(parent.0, name) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn rename(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        newparent: INodeNo,
        newname: &OsStr,
        flags: fuser::RenameFlags,
        reply: ReplyEmpty,
    ) {
        match self
            .vfs
            .rename(parent.0, name, newparent.0, newname, flags.bits())
        {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }

    fn setattr(
        &self,
        _req: &Request,
        ino: INodeNo,
        mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<fuser::TimeOrNow>,
        _mtime: Option<fuser::TimeOrNow>,
        _ctime: Option<std::time::SystemTime>,
        _fh: Option<FileHandle>,
        _crtime: Option<std::time::SystemTime>,
        _chgtime: Option<std::time::SystemTime>,
        _bkuptime: Option<std::time::SystemTime>,
        _flags: Option<fuser::BsdFileFlags>,
        reply: ReplyAttr,
    ) {
        match self.vfs.setattr(ino.0, size, mode) {
            Ok(attr) => {
                let fattr: fuser::FileAttr = attr.into();
                reply.attr(&Duration::MAX, &fattr);
            }
            Err(e) => reply.error(Errno::from_i32(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs_ops::{DirEntry, FileAttr, FileKind};
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    /// Records which `getattr` variant was called and with which inode.
    /// Used to verify the FUSE adapter dispatches through the right
    /// `VfsOps` method based on whether an `fh` was provided.
    struct DispatchSpyVfs {
        getattr_calls: Mutex<Vec<u64>>,
        getattr_strict_calls: Mutex<Vec<u64>>,
        // Distinct sizes so the caller can tell which method served the
        // reply without inspecting the call log.
        getattr_size: u64,
        getattr_strict_size: u64,
    }

    impl DispatchSpyVfs {
        fn new() -> Self {
            Self {
                getattr_calls: Mutex::new(Vec::new()),
                getattr_strict_calls: Mutex::new(Vec::new()),
                getattr_size: 100,
                getattr_strict_size: 42,
            }
        }

        fn make_attr(&self, ino: u64, size: u64) -> FileAttr {
            FileAttr {
                ino,
                size,
                blocks: 0,
                atime: UNIX_EPOCH,
                mtime: UNIX_EPOCH,
                ctime: UNIX_EPOCH,
                kind: FileKind::RegularFile,
                perm: 0o644,
                nlink: 1,
                uid: 0,
                gid: 0,
            }
        }
    }

    impl VfsOps for DispatchSpyVfs {
        fn lookup(&self, _parent: u64, _name: &OsStr) -> Result<FileAttr, i32> {
            Err(libc::ENOENT)
        }
        fn getattr(&self, ino: u64) -> Result<FileAttr, i32> {
            self.getattr_calls.lock().unwrap().push(ino);
            Ok(self.make_attr(ino, self.getattr_size))
        }
        fn getattr_strict(&self, ino: u64) -> Result<FileAttr, i32> {
            self.getattr_strict_calls.lock().unwrap().push(ino);
            Ok(self.make_attr(ino, self.getattr_strict_size))
        }
        fn readlink(&self, _ino: u64) -> Result<PathBuf, i32> {
            Err(libc::ENOENT)
        }
        fn read(&self, _ino: u64, _offset: u64, _size: u32) -> Result<Vec<u8>, i32> {
            Ok(vec![])
        }
        fn content_source(&self, _ino: u64) -> Result<ContentSource, i32> {
            Err(libc::ENOENT)
        }
        fn readdir(&self, _ino: u64, _offset: u64) -> Result<Vec<DirEntry>, i32> {
            Ok(vec![])
        }
    }

    /// `resolve_attr_for_fh(ino, None)` must call `getattr` (path-side stat).
    #[test]
    fn dispatch_no_fh_calls_getattr() {
        let adapter = FuseAdapter::new(DispatchSpyVfs::new());

        let attr = adapter.resolve_attr_for_fh(7, None).unwrap();
        assert_eq!(attr.size, adapter.vfs.getattr_size);
        assert_eq!(adapter.vfs.getattr_calls.lock().unwrap().as_slice(), &[7]);
        assert!(adapter.vfs.getattr_strict_calls.lock().unwrap().is_empty());
    }

    /// `resolve_attr_for_fh(ino, Some(fh))` must dispatch through the inode
    /// the fh was bound to, via `getattr_strict`. This is the regression for
    /// the fstat-after-COW inconsistency: a fd opened on lower inode L and
    /// later promoted to upper U must keep seeing L's attrs.
    #[test]
    fn dispatch_with_fh_uses_bound_ino_and_strict() {
        let adapter = FuseAdapter::new(DispatchSpyVfs::new());

        // Bind a fh to inode 7 and call resolve_attr with a *different*
        // path-side ino (mimicking the kernel passing a redirected ino).
        let fh = adapter
            .store_open(7, OpenFileContent::InodeRead(7))
            .unwrap();
        let attr = adapter
            .resolve_attr_for_fh(99, Some(FileHandle(fh)))
            .unwrap();

        // Strict size proves we went via getattr_strict, not getattr.
        assert_eq!(attr.size, adapter.vfs.getattr_strict_size);
        // And we used the bound ino (7), not the path-side ino (99).
        assert_eq!(
            adapter.vfs.getattr_strict_calls.lock().unwrap().as_slice(),
            &[7]
        );
        assert!(adapter.vfs.getattr_calls.lock().unwrap().is_empty());
    }

    /// If the fh isn't in the open_files map (e.g. stale/invalid), fall back
    /// to path-side `getattr` so we still respond cleanly. This matches the
    /// pre-fix behavior for unknown fhs and avoids surfacing EIO for what
    /// the kernel may legitimately ask via a stale handle.
    #[test]
    fn dispatch_with_unknown_fh_falls_back_to_getattr() {
        let adapter = FuseAdapter::new(DispatchSpyVfs::new());

        let attr = adapter
            .resolve_attr_for_fh(11, Some(FileHandle(0xdead_beef)))
            .unwrap();
        assert_eq!(attr.size, adapter.vfs.getattr_size);
        assert_eq!(adapter.vfs.getattr_calls.lock().unwrap().as_slice(), &[11]);
    }

    /// `release` removes the open_file entry. After release, `resolve_attr_for_fh`
    /// with the now-stale fh falls back to path-side `getattr`.
    #[test]
    fn release_removes_bound_ino() {
        let adapter = FuseAdapter::new(DispatchSpyVfs::new());
        let fh = adapter
            .store_open(7, OpenFileContent::InodeRead(7))
            .unwrap();

        // Pre-release: fh resolves to bound ino 7.
        assert_eq!(adapter.bound_ino_for(FileHandle(fh)), Some(7));

        // Drop the entry directly (mirrors what `release` does).
        adapter.open_files.lock_or_eio().unwrap().remove(&fh);

        // Post-release: bound_ino_for returns None, dispatch falls back.
        assert_eq!(adapter.bound_ino_for(FileHandle(fh)), None);
        let attr = adapter
            .resolve_attr_for_fh(11, Some(FileHandle(fh)))
            .unwrap();
        assert_eq!(attr.size, adapter.vfs.getattr_size);
    }
}
