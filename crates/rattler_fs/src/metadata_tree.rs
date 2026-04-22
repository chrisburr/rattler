//! In-memory metadata tree shared by all transports.
//!
//! Each [`MetadataNode`] is either a [`DirectoryNode`] (children indices into
//! the same `Vec<MetadataNode>`) or a [`FileNode`] (a leaf with cache path,
//! prefix-replacement metadata, and optional materialized content). The tree
//! is built once by `build_metadata_tree` and consumed by `VirtualFS::new`,
//! after which it backs FUSE, NFS, and `ProjFS` reads alike.
//!
//! Despite the historical `fuse_directory.rs` filename, this is **not**
//! FUSE-specific.

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    sync::Arc,
};

use rattler_conda_types::package::{PathType, PrefixPlaceholder};

/// A byte-level transformation applied to a file's content on read.
///
/// Today the only variant is prefix replacement (used by both conda packages
/// and pypi wheel `.data/scripts/` shebang rewriting); additional variants can
/// be added without touching the read path — each variant plugs into
/// `VirtualFS`'s offset-caching logic the same way.
#[derive(Debug)]
pub enum ContentTransform {
    /// Replace a placeholder prefix string with the mount point on read.
    PrefixReplace(PrefixPlaceholder),
}

impl ContentTransform {
    /// Borrow the inner [`PrefixPlaceholder`] if this is a `PrefixReplace` transform.
    pub fn as_prefix_replace(&self) -> Option<&PrefixPlaceholder> {
        match self {
            ContentTransform::PrefixReplace(p) => Some(p),
        }
    }
}

#[derive(Debug)]
pub struct DirectoryNode {
    pub prefix_path: PathBuf,
    pub parent: usize,
    pub children: Vec<usize>,
}

impl DirectoryNode {
    fn new(prefix_path: PathBuf, parent: usize) -> Self {
        DirectoryNode {
            prefix_path,
            parent,
            children: vec![],
        }
    }
}

/// Default POSIX mode for regular files (`rw-r--r--`).
pub const DEFAULT_FILE_MODE: u32 = 0o644;
/// POSIX mode for executable files (`rwxr-xr-x`).
pub const EXECUTABLE_FILE_MODE: u32 = 0o755;

#[derive(Debug)]
pub struct FileNode {
    pub file_name: OsString,
    pub parent: usize,
    pub cache_base_path: Arc<Path>,
    pub path_type: PathType,
    /// Optional byte-level transform applied when serving this file's content.
    /// Kept as `Option<_>` (not `Vec<_>`) because in practice no file ever needs
    /// more than one transform — avoiding the per-node heap allocation a `Vec`
    /// would impose on every file in the tree.
    pub transform: Option<ContentTransform>,
    /// POSIX-style mode bits for this file. Virtual files and the stat-failure
    /// fallback use this value; cache-backed files in the happy path read mode
    /// from the underlying extracted file on disk (via `FileAttr::from_metadata`).
    pub mode: u32,
    /// Pre-materialized content for virtual files (e.g. generated entry point scripts).
    /// When set, the FUSE layer serves this content directly instead of reading from disk.
    pub virtual_content: Option<Vec<u8>>,
    /// Pre-computed file size after prefix replacement (for text-mode files).
    /// When set, `_getattr()` uses this instead of the on-disk file size.
    pub computed_size: Option<u64>,
    /// Override for the cache directory path used by `_getpath()`.
    /// When set (e.g. for noarch Python files where the virtual path differs from
    /// the on-disk cache path), `_getpath()` uses this instead of `parent.prefix_path`.
    pub cache_prefix_path: Option<PathBuf>,
}

impl FileNode {
    fn new(
        file_name: OsString,
        parent: usize,
        cache_base_path: Arc<Path>,
        path_type: PathType,
        prefix_placeholder: Option<PrefixPlaceholder>,
        mode: u32,
    ) -> Self {
        FileNode {
            file_name,
            parent,
            cache_base_path,
            path_type,
            transform: prefix_placeholder.map(ContentTransform::PrefixReplace),
            mode,
            virtual_content: None,
            computed_size: None,
            cache_prefix_path: None,
        }
    }

    fn new_virtual(file_name: OsString, parent: usize, content: Vec<u8>, mode: u32) -> Self {
        FileNode {
            file_name,
            parent,
            cache_base_path: Arc::from(Path::new("")),
            path_type: PathType::HardLink,
            transform: None,
            mode,
            virtual_content: Some(content),
            computed_size: None,
            cache_prefix_path: None,
        }
    }

    /// Borrow the prefix-replacement placeholder for this file, if any.
    pub fn prefix_placeholder(&self) -> Option<&PrefixPlaceholder> {
        self.transform
            .as_ref()
            .and_then(ContentTransform::as_prefix_replace)
    }
}

#[derive(Debug)]
pub enum MetadataNode {
    Directory(DirectoryNode),
    File(FileNode),
}

impl MetadataNode {
    pub fn file_name(&self) -> &OsStr {
        match self {
            Self::Directory(directory) => directory
                .prefix_path
                .file_name()
                .unwrap_or(std::ffi::OsStr::new(".")),
            Self::File(file) => &file.file_name,
        }
    }

    pub fn new_directory(prefix_path: PathBuf, parent: usize) -> Self {
        MetadataNode::Directory(DirectoryNode::new(prefix_path, parent))
    }

    pub fn new_file(
        file_name: OsString,
        parent: usize,
        cache_base_path: Arc<Path>,
        path_type: PathType,
        prefix_placeholder: Option<PrefixPlaceholder>,
    ) -> Self {
        MetadataNode::File(FileNode::new(
            file_name,
            parent,
            cache_base_path,
            path_type,
            prefix_placeholder,
            DEFAULT_FILE_MODE,
        ))
    }

    pub fn new_virtual_file(file_name: OsString, parent: usize, content: Vec<u8>) -> Self {
        MetadataNode::File(FileNode::new_virtual(
            file_name,
            parent,
            content,
            DEFAULT_FILE_MODE,
        ))
    }

    /// Virtual file with the executable bit set (mode `0o755`). Use for generated
    /// entry-point scripts and any other synthetic files that must be executable.
    pub fn new_virtual_executable(file_name: OsString, parent: usize, content: Vec<u8>) -> Self {
        MetadataNode::File(FileNode::new_virtual(
            file_name,
            parent,
            content,
            EXECUTABLE_FILE_MODE,
        ))
    }

    pub fn as_directory(&self) -> Option<&DirectoryNode> {
        if let Self::Directory(directory) = self {
            Some(directory)
        } else {
            None
        }
    }

    pub fn as_directory_mut(&mut self) -> Option<&mut DirectoryNode> {
        if let Self::Directory(directory) = self {
            Some(directory)
        } else {
            None
        }
    }

    pub fn as_file(&self) -> Option<&FileNode> {
        if let Self::File(file) = self {
            Some(file)
        } else {
            None
        }
    }

    pub fn as_file_mut(&mut self) -> Option<&mut FileNode> {
        if let Self::File(file) = self {
            Some(file)
        } else {
            None
        }
    }
}
