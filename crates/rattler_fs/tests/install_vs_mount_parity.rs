//! Install-vs-mount byte parity tests.
//!
//! These tests verify that `rattler::install::link`'s prefix-replacement
//! routines (which write transformed bytes to a destination Writer) and
//! `rattler_fs::prefix_replacement`'s ranged-read routines (which return
//! transformed bytes from a source slice) produce **byte-identical** output
//! for the same input.
//!
//! This catches drift between the install-time and mount-time prefix
//! replacement code paths — the same package on disk vs. mounted should be
//! indistinguishable.

use std::io::Cursor;

use rattler_conda_types::{package::FileMode, Platform};
use rattler_fs::prefix_replacement::{binary_ranged_read, text_ranged_read};

/// Compute placeholder offsets in `source` by scanning for `placeholder` bytes.
fn placeholder_offsets(source: &[u8], placeholder: &str) -> Vec<usize> {
    use memchr::memmem;
    memmem::find_iter(source, placeholder.as_bytes()).collect()
}

/// Run install-time prefix replacement and return the resulting bytes.
fn install_replace(
    source: &[u8],
    placeholder: &str,
    target_prefix: &str,
    file_mode: FileMode,
    platform: Platform,
) -> Vec<u8> {
    let mut output = Cursor::new(Vec::<u8>::new());
    rattler::install::link::copy_and_replace_placeholders(
        source,
        &mut output,
        placeholder,
        target_prefix,
        &platform,
        file_mode,
    )
    .expect("install-time replacement should succeed");
    output.into_inner()
}

/// Run mount-time ranged-read replacement over the full output range.
fn mount_replace_full(
    source: &[u8],
    placeholder: &str,
    target_prefix: &str,
    file_mode: FileMode,
) -> Vec<u8> {
    let offsets = placeholder_offsets(source, placeholder);
    // Pick an end position large enough to cover any possible transformed
    // length (text mode can grow if target_prefix is longer than placeholder).
    let huge = source.len() + target_prefix.len() * (offsets.len() + 1) + 1024;
    match file_mode {
        FileMode::Text => text_ranged_read(
            source,
            placeholder.as_bytes(),
            target_prefix.as_bytes(),
            &offsets,
            0,
            huge,
        ),
        FileMode::Binary => binary_ranged_read(
            source,
            placeholder.as_bytes(),
            target_prefix.as_bytes(),
            &offsets,
            0,
            huge,
        ),
    }
}

// ---------------------------------------------------------------------------
// Text mode parity
// ---------------------------------------------------------------------------

#[test]
fn text_mode_simple_replacement_matches_install() {
    let placeholder = "/old/conda/prefix";
    let target = "/new/longer/conda/prefix";
    let source = format!("hello {placeholder} world\n");

    let install_bytes = install_replace(
        source.as_bytes(),
        placeholder,
        target,
        FileMode::Text,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(source.as_bytes(), placeholder, target, FileMode::Text);

    assert_eq!(
        install_bytes, mount_bytes,
        "install-time and mount-time text replacement diverged"
    );
}

#[test]
fn text_mode_multiple_replacements_match_install() {
    let placeholder = "/p";
    let target = "/QQQQ";
    // Three placeholders separated by literal text.
    let source = b"a/p b/p c/p d";

    let install_bytes = install_replace(
        source,
        placeholder,
        target,
        FileMode::Text,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(source, placeholder, target, FileMode::Text);

    assert_eq!(install_bytes, mount_bytes);
}

#[test]
fn text_mode_no_replacement_match_install() {
    let placeholder = "/old/conda/prefix";
    let target = "/new/conda/prefix";
    let source = b"completely unrelated content with no placeholder\n";

    let install_bytes = install_replace(
        source,
        placeholder,
        target,
        FileMode::Text,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(source, placeholder, target, FileMode::Text);

    assert_eq!(install_bytes, mount_bytes);
}

#[test]
fn text_mode_shorter_target_matches_install() {
    let placeholder = "/long/old/prefix/path";
    let target = "/short";
    let source = format!("{placeholder}/bin/python\n");

    let install_bytes = install_replace(
        source.as_bytes(),
        placeholder,
        target,
        FileMode::Text,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(source.as_bytes(), placeholder, target, FileMode::Text);

    assert_eq!(install_bytes, mount_bytes);
}

// ---------------------------------------------------------------------------
// Binary mode parity (c-string with null terminator and padding)
// ---------------------------------------------------------------------------

/// Build a binary blob containing a c-string with a placeholder, terminated
/// by a null byte. Returns the bytes — caller passes them to install vs mount
/// replacement.
fn build_cstring(placeholder: &str, suffix: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(placeholder.as_bytes());
    buf.extend_from_slice(suffix.as_bytes());
    buf.push(0u8);
    // Some trailing context so the test catches replacement past the null.
    buf.extend_from_slice(b"\x01\x02\x03tail\x00");
    buf
}

#[test]
fn binary_mode_cstring_with_padding_matches_install() {
    let placeholder = "/long/old/prefix";
    let target = "/short";
    let source = build_cstring(placeholder, "/lib/foo.so");

    let install_bytes = install_replace(
        &source,
        placeholder,
        target,
        FileMode::Binary,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(&source, placeholder, target, FileMode::Binary);

    assert_eq!(
        install_bytes, mount_bytes,
        "install-time and mount-time binary replacement diverged for c-string"
    );
}

#[test]
fn binary_mode_no_replacement_matches_install() {
    let placeholder = "/long/old/prefix";
    let target = "/short";
    let source: &[u8] = b"\x7fELF unrelated binary contents\x00\x01\x02\x00";

    let install_bytes = install_replace(
        source,
        placeholder,
        target,
        FileMode::Binary,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(source, placeholder, target, FileMode::Binary);

    assert_eq!(install_bytes, mount_bytes);
}

#[test]
fn binary_mode_multiple_cstrings_match_install() {
    let placeholder = "/long/old/prefix";
    let target = "/p";
    let mut source = Vec::new();
    source.extend_from_slice(placeholder.as_bytes());
    source.extend_from_slice(b"/a\x00");
    source.extend_from_slice(placeholder.as_bytes());
    source.extend_from_slice(b"/b\x00");
    source.extend_from_slice(b"unrelated\x00");

    let install_bytes = install_replace(
        &source,
        placeholder,
        target,
        FileMode::Binary,
        Platform::Linux64,
    );
    let mount_bytes = mount_replace_full(&source, placeholder, target, FileMode::Binary);

    assert_eq!(install_bytes, mount_bytes);
}

// ---------------------------------------------------------------------------
// Ranged-read parity: mount-time read of an arbitrary slice should equal the
// corresponding slice of the install-time output.
// ---------------------------------------------------------------------------

#[test]
fn ranged_read_text_matches_install_slice() {
    let placeholder = "/old/conda/prefix";
    let target = "/new/longer/conda/prefix";
    let source = format!("hello {placeholder} middle {placeholder} tail\n");

    let install_bytes = install_replace(
        source.as_bytes(),
        placeholder,
        target,
        FileMode::Text,
        Platform::Linux64,
    );
    let offsets = placeholder_offsets(source.as_bytes(), placeholder);

    // Sample several arbitrary byte ranges and confirm they match the
    // corresponding window of the full install output.
    let cases = [(0usize, 5), (3, 20), (7, install_bytes.len()), (0, 1)];
    for (start, end) in cases {
        let mount_slice = text_ranged_read(
            source.as_bytes(),
            placeholder.as_bytes(),
            target.as_bytes(),
            &offsets,
            start,
            end,
        );
        let expected = &install_bytes[start.min(install_bytes.len())..end.min(install_bytes.len())];
        assert_eq!(
            mount_slice, expected,
            "ranged read [{start}, {end}) diverged from install"
        );
    }
}

#[test]
fn ranged_read_binary_matches_install_slice() {
    let placeholder = "/long/old/prefix";
    let target = "/p";
    let source = build_cstring(placeholder, "/bin/foo");

    let install_bytes = install_replace(
        &source,
        placeholder,
        target,
        FileMode::Binary,
        Platform::Linux64,
    );
    let offsets = placeholder_offsets(&source, placeholder);

    let cases = [(0usize, 4), (2, 16), (0, install_bytes.len()), (10, 20)];
    for (start, end) in cases {
        let mount_slice = binary_ranged_read(
            &source,
            placeholder.as_bytes(),
            target.as_bytes(),
            &offsets,
            start,
            end,
        );
        let expected = &install_bytes[start.min(install_bytes.len())..end.min(install_bytes.len())];
        assert_eq!(
            mount_slice, expected,
            "binary ranged read [{start}, {end}) diverged from install"
        );
    }
}
