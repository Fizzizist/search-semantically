pub fn truncate_with_ellipsis(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    if max_bytes <= 3 {
        let mut i = max_bytes;
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        return s[..i].to_string();
    }
    let target = max_bytes - 3;
    let mut i = target;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    format!("{}...", &s[..i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_string_unchanged() {
        assert_eq!(truncate_with_ellipsis("hello", 10), "hello");
    }

    #[test]
    fn exact_length_unchanged() {
        assert_eq!(truncate_with_ellipsis("hello", 5), "hello");
    }

    #[test]
    fn long_ascii_truncated() {
        let s = "a".repeat(200);
        let result = truncate_with_ellipsis(&s, 120);
        assert_eq!(result.len(), 120);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn multibyte_at_cut_is_valid_utf8() {
        // U+2019 RIGHT SINGLE QUOTATION MARK is 3 bytes: 0xE2 0x80 0x99
        let prefix = "a".repeat(115);
        let s = format!("{prefix}\u{2019}trailing text to make it long enough");
        let result = truncate_with_ellipsis(&s, 120);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
        assert!(result.len() <= 120);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn emoji_at_cut_is_valid_utf8() {
        // Emoji are 4 bytes
        let prefix = "b".repeat(118);
        let s = format!("{prefix}\u{1F600}more text here");
        let result = truncate_with_ellipsis(&s, 120);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
        assert!(result.len() <= 120);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn max_bytes_less_than_or_equal_3_no_ellipsis() {
        let s = "hello world";
        let result = truncate_with_ellipsis(s, 2);
        assert!(result.len() <= 2);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    #[test]
    fn max_bytes_zero() {
        let result = truncate_with_ellipsis("hello", 0);
        assert_eq!(result, "");
    }
}
