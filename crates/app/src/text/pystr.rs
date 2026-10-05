//! Python `str` predicates the ported modules rely on (`str.isupper`, `str.islower`).

/// `str.isupper()`: at least one cased character and no lowercase ones.
pub fn is_upper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() {
            return false;
        }
        cased |= c.is_uppercase();
    }
    cased
}

/// `str.islower()`: at least one cased character and no uppercase ones.
pub fn is_lower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            return false;
        }
        cased |= c.is_lowercase();
    }
    cased
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cased() {
        assert!(is_upper("HELLO, WORLD 2"));
        assert!(!is_upper("Hello"));
        assert!(!is_upper("123"));
        assert!(is_lower("vizcom's"));
        assert!(!is_lower("Vizcom"));
        assert!(!is_lower("--"));
    }
}
