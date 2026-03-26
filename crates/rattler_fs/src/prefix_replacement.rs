//! Ranged prefix replacement for FUSE/NFS reads.
//!
//! Serves a byte range `[start, end)` from a source file with prefix
//! placeholder replacements applied on the fly, without materializing the
//! entire transformed file in memory.

use memchr::memmem;

/// Read a range from a text file with prefix replacements applied.
///
/// Text-mode replacement changes `old_prefix` → `new_prefix` at each offset.
/// Because the replacement can change length, output byte positions shift
/// relative to source positions. The `offsets` array gives the source-file
/// positions of each placeholder occurrence.
///
/// Returns the bytes in the output range `[start, end)`.
pub fn text_ranged_read(
    source: &[u8],
    old_prefix: &[u8],
    new_prefix: &[u8],
    offsets: &[usize],
    start: usize,
    end: usize,
) -> Vec<u8> {
    let delta = new_prefix.len() as isize - old_prefix.len() as isize;
    let transformed_len = (source.len() as isize + delta * offsets.len() as isize).max(0) as usize;

    let actual_end = end.min(transformed_len);
    let actual_start = start.min(transformed_len);
    if actual_start >= actual_end {
        return vec![];
    }

    let capacity = actual_end - actual_start;
    let mut buffer = Vec::with_capacity(capacity);

    // Walk through the source, tracking the current position in both
    // source space and output (transformed) space.
    let mut src_pos = 0usize;
    let mut out_pos = 0usize;
    let mut offset_idx = 0usize;

    while src_pos < source.len() && buffer.len() < capacity {
        if offset_idx < offsets.len() && src_pos == offsets[offset_idx] {
            // At a replacement site: emit new_prefix bytes
            for &b in new_prefix {
                if out_pos >= actual_start && out_pos < actual_end {
                    buffer.push(b);
                }
                out_pos += 1;
                if buffer.len() >= capacity {
                    return buffer;
                }
            }
            // Skip the old prefix in the source
            src_pos += old_prefix.len();
            offset_idx += 1;
        } else {
            // Regular byte: copy through
            if out_pos >= actual_start && out_pos < actual_end {
                buffer.push(source[src_pos]);
            }
            out_pos += 1;
            src_pos += 1;
        }
    }

    buffer
}

/// Read a range from a binary file with prefix replacements applied.
///
/// Binary-mode replacement swaps `old_prefix` → `new_prefix` and pads with
/// null bytes to maintain the same total length. Multiple consecutive
/// replacements that share a c-string (no null terminator between them)
/// accumulate padding which is emitted at the next null byte.
///
/// Output length always equals source length.
///
/// Returns the bytes in the output range `[start, end)`.
pub fn binary_ranged_read(
    source: &[u8],
    old_prefix: &[u8],
    new_prefix: &[u8],
    offsets: &[usize],
    start: usize,
    end: usize,
) -> Vec<u8> {
    assert!(
        new_prefix.len() <= old_prefix.len(),
        "new prefix cannot be longer than old prefix in binary mode"
    );

    let actual_end = end.min(source.len());
    let actual_start = start.min(source.len());
    if actual_start >= actual_end {
        return vec![];
    }

    let length_change = old_prefix.len() - new_prefix.len();
    let capacity = actual_end - actual_start;
    let mut buffer = Vec::with_capacity(capacity);

    // Find the first offset at or after `start` via binary search.
    let first_offset_idx = offsets.partition_point(|&o| o < actual_start);

    // Count "unfinished replacements" — replacements before `start` whose
    // null-padding hasn't been emitted yet (no null byte between the
    // replacement and `start`).
    let unfinished = count_unfinished_before(source, offsets, first_offset_idx, actual_start);

    // Compute where in the source to start reading, accounting for
    // accumulated padding from unfinished replacements that shifts our
    // position forward.
    let src_start = actual_start + unfinished * length_change;

    let mut src_pos = src_start;
    let mut offset_idx = first_offset_idx;
    let mut pending_padding = unfinished * length_change;

    while src_pos < source.len() && buffer.len() < capacity {
        if offset_idx < offsets.len() && src_pos == offsets[offset_idx] {
            // At a replacement site
            offset_idx += 1;

            // Emit the new prefix
            for &b in new_prefix {
                if buffer.len() >= capacity {
                    return buffer;
                }
                buffer.push(b);
            }

            // Skip the old prefix in source
            src_pos += old_prefix.len();
            pending_padding += length_change;

            // Determine the boundary for this c-string: next offset or end
            let boundary = offsets.get(offset_idx).copied().unwrap_or(source.len());

            // Copy bytes after the old prefix until null byte or next offset
            while src_pos < source.len()
                && src_pos < boundary
                && source[src_pos] != 0
                && buffer.len() < capacity
            {
                buffer.push(source[src_pos]);
                src_pos += 1;
            }

            // If we hit a null byte, emit the accumulated padding
            if src_pos < source.len() && source[src_pos] == 0 {
                // Emit the original null byte position's worth of padding
                for _ in 0..pending_padding {
                    if buffer.len() >= capacity {
                        return buffer;
                    }
                    buffer.push(0);
                }
                pending_padding = 0;
            }
        } else if source[src_pos] == 0 && pending_padding > 0 {
            // Null byte with accumulated padding from previous replacements
            buffer.push(0);
            src_pos += 1;
            for _ in 0..pending_padding {
                if buffer.len() >= capacity {
                    return buffer;
                }
                buffer.push(0);
            }
            pending_padding = 0;
        } else {
            // Regular byte
            buffer.push(source[src_pos]);
            src_pos += 1;
        }
    }

    // Emit any remaining padding at EOF (e.g. file ends without a null byte
    // after replacements)
    while pending_padding > 0 && buffer.len() < capacity {
        buffer.push(0);
        pending_padding -= 1;
    }

    buffer
}

/// Count replacements before `first_offset_idx` that haven't had their
/// null-padding emitted yet (i.e. no null byte between the last replacement
/// before `start` and `start` itself).
fn count_unfinished_before(
    source: &[u8],
    offsets: &[usize],
    first_offset_idx: usize,
    start: usize,
) -> usize {
    if first_offset_idx == 0 {
        return 0;
    }

    // Find the last null byte before `start` in the source
    let region = &source[..start.min(source.len())];
    let last_null = memchr::memrchr(0, region);

    // Count offsets that are after the last null byte (or from the start if no null)
    let threshold = last_null.map_or(0, |pos| pos + 1);
    offsets[..first_offset_idx]
        .iter()
        .rev()
        .take_while(|&&o| o >= threshold)
        .count()
}

/// Compute replacement offsets by scanning the source for the placeholder.
/// Used when paths.json doesn't provide offsets (legacy v1 format).
pub fn collect_offsets(source: &[u8], placeholder: &[u8]) -> Vec<usize> {
    memmem::find_iter(source, placeholder).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Text mode tests ──────────────────────────────────────────────

    fn text_test(
        placeholder: &[u8],
        prefix: &[u8],
        source: &[u8],
        expected: &[u8],
        start: usize,
        end: usize,
    ) {
        let offsets = collect_offsets(source, placeholder);
        let result = text_ranged_read(source, placeholder, prefix, &offsets, start, end);
        assert_eq!(
            result, expected,
            "text replacement [{start}..{end}] of {source:?}: expected {expected:?}, got {result:?}"
        );
    }

    fn binary_test(
        placeholder: &[u8],
        prefix: &[u8],
        source: &[u8],
        expected: &[u8],
        start: usize,
        end: usize,
    ) {
        let offsets = collect_offsets(source, placeholder);
        let result = binary_ranged_read(source, placeholder, prefix, &offsets, start, end);
        assert_eq!(
            result, expected,
            "binary replacement [{start}..{end}] of {source:?}: expected {expected:?}, got {result:?}"
        );
    }

    // Full-file text replacements

    #[test]
    fn text_full_file() {
        text_test(
            b"ABCD",
            b"XY",
            b"01ABCD23456ABCD7890",
            b"01XY23456XY7890",
            0,
            19,
        );
    }

    #[test]
    fn text_only_placeholder() {
        text_test(b"ABCD", b"XY", b"ABCD", b"XY", 0, 4);
    }

    #[test]
    fn text_consecutive_placeholders() {
        text_test(b"ABCD", b"XY", b"ABCDABCD", b"XYXY", 0, 8);
    }

    #[test]
    fn text_no_placeholders() {
        text_test(b"ABCD", b"XY", b"0123456789", b"0123456789", 0, 10);
    }

    #[test]
    fn text_same_length() {
        text_test(
            b"ABCD",
            b"WXYZ",
            b"01ABCD6789012ABCD7890",
            b"01WXYZ6789012WXYZ7890",
            0,
            21,
        );
    }

    #[test]
    fn text_empty_file() {
        text_test(b"ABCD", b"XY", b"", b"", 0, 0);
    }

    #[test]
    fn text_many_placeholders() {
        let mut source = Vec::new();
        let mut expected = Vec::new();
        for i in 0..10u8 {
            source.extend_from_slice(&[i + b'0', i + b'0']);
            source.extend_from_slice(b"ABCD");
            expected.extend_from_slice(&[i + b'0', i + b'0']);
            expected.extend_from_slice(b"XY");
        }
        text_test(b"ABCD", b"XY", &source, &expected, 0, source.len());
    }

    // Partial-range text replacements

    #[test]
    fn text_partial_range() {
        text_test(
            b"ABCD",
            b"XY",
            b"ABCD0ABCD5ABCD0ABCD5ABCD",
            b"0XY5XY0",
            2,
            9,
        );
    }

    #[test]
    fn text_start_after_prefix() {
        // Output: "XY01234XY56789" (14 chars) — start at index 5 = "34XY56789"
        text_test(b"ABCD", b"XY", b"ABCD01234ABCD56789", b"34XY56789", 5, 18);
    }

    #[test]
    fn text_start_between_placeholders() {
        // Output: "XY0123XY5678XY" — start at index 5 = "3XY5678XY"
        text_test(b"ABCD", b"XY", b"ABCD0123ABCD5678ABCD", b"3XY5678XY", 5, 20);
    }

    #[test]
    fn text_start_at_placeholder() {
        // Output: "01234XY6789XY" — start at index 5 = "XY6789XY"
        text_test(b"ABCD", b"XY", b"01234ABCD6789ABCD", b"XY6789XY", 5, 17);
    }

    #[test]
    fn text_longer_placeholder() {
        text_test(
            b"ABCDEFGH",
            b"XYZ",
            b"01ABCDEFGH234ABCDEFGH567",
            b"01XYZ234XYZ567",
            0,
            24,
        );
    }

    // ── Binary mode tests ────────────────────────────────────────────

    #[test]
    fn binary_full_file() {
        binary_test(
            b"ABCD",
            b"XY",
            b"01ABCD23\x00456ABCD78\x0090",
            b"01XY23\x00\x00\x00456XY78\x00\x00\x0090",
            0,
            21,
        );
    }

    #[test]
    fn binary_only_placeholder() {
        binary_test(b"ABCD", b"XY", b"ABCD", b"XY\x00\x00", 0, 4);
    }

    #[test]
    fn binary_no_placeholders() {
        binary_test(b"ABCD", b"XY", b"0123456789", b"0123456789", 0, 10);
    }

    #[test]
    fn binary_same_length() {
        binary_test(
            b"ABCD",
            b"WXYZ",
            b"01ABCD6789012ABCD7890",
            b"01WXYZ6789012WXYZ7890",
            0,
            21,
        );
    }

    #[test]
    fn binary_consecutive_placeholders() {
        binary_test(b"ABCD", b"XY", b"ABCDABCD", b"XYXY\x00\x00\x00\x00", 0, 8);
    }

    #[test]
    fn binary_multiple_placeholders() {
        let source = b"\x00\x00ABCDZ\x00\x00\x00ABCDEFABCDEF\x00\x00\x00ABCDMNOPQRSABCDMNOPQRSABCDMNOPQRS\x00\x00";
        let expected = b"\x00\x00XYZ\x00\x00\x00\x00\x00XYEFXYEF\x00\x00\x00\x00\x00\x00\x00XYMNOPQRSXYMNOPQRSXYMNOPQRS\x00\x00\x00\x00\x00\x00\x00\x00";
        binary_test(b"ABCD", b"XY", source, expected, 0, source.len());
    }

    #[test]
    fn binary_empty_file() {
        binary_test(b"ABCD", b"XY", b"", b"", 0, 0);
    }

    #[test]
    fn binary_many_placeholders() {
        let mut source = Vec::new();
        let mut expected = Vec::new();
        for i in 0..10u8 {
            source.extend_from_slice(&[i + b'0', i + b'0']);
            source.extend_from_slice(b"ABCD\x00");
            expected.extend_from_slice(&[i + b'0', i + b'0']);
            expected.extend_from_slice(b"XY\x00\x00\x00");
        }
        binary_test(b"ABCD", b"XY", &source, &expected, 0, source.len());
    }

    // Partial-range binary replacements

    #[test]
    fn binary_partial_range() {
        binary_test(
            b"ABCD",
            b"XY",
            b"ABCD\x000ABCD\x005ABCD\x000ABCD\x005ABCD\x00",
            b"0XY\x00\x00",
            5,
            10,
        );
    }

    #[test]
    fn binary_start_after_prefix() {
        binary_test(
            b"ABCD",
            b"XY",
            b"ABCD01234ABCD\x0056789",
            b"34XY\x00\x00\x00\x00\x0056789",
            5,
            19,
        );
    }

    #[test]
    fn binary_start_between_placeholders() {
        binary_test(
            b"ABCD",
            b"XY",
            b"ABCD012\x003ABCD5678ABCD",
            b"012\x00\x00\x003XY5678XY\x00\x00\x00\x00",
            2,
            21,
        );
    }

    #[test]
    fn binary_start_at_placeholder() {
        binary_test(
            b"ABCD",
            b"XY",
            b"01234ABCD\x006789ABCD\x00",
            b"XY\x00\x00\x006789XY\x00\x00\x00",
            5,
            19,
        );
    }

    #[test]
    fn binary_longer_placeholder() {
        binary_test(
            b"ABCDEFGH",
            b"XYZ",
            b"01ABCDEFGH234ABCDEFGH567",
            b"01XYZ234XYZ567\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
            0,
            24,
        );
    }

    // ── Offset collection ────────────────────────────────────────────

    #[test]
    fn collect_offsets_basic() {
        assert_eq!(collect_offsets(b"01ABCD56ABCD", b"ABCD"), vec![2, 8]);
    }

    #[test]
    fn collect_offsets_none() {
        assert_eq!(collect_offsets(b"0123456789", b"ABCD"), Vec::<usize>::new());
    }

    #[test]
    fn collect_offsets_consecutive() {
        assert_eq!(collect_offsets(b"ABCDABCD", b"ABCD"), vec![0, 4]);
    }
}
