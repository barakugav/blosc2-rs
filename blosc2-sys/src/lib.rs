//! Unsafe Rust bindings for blosc2 - a fast, compressed, persistent binary data store library.
//!
//! ## Features
//! Cargo features enable or disable support for various compression codecs such as `zstd` and
//! `zlib`.
//!
//! ## Error Handling
//! Errors are represented by int codes. In addition, if the environment variable
//! `BLOSC_TRACE` is set, it will print detailed trace during failures which is useful for
//! debugging.

mod c_bridge {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(rustdoc::invalid_html_tags)]
    #![allow(rustdoc::broken_intra_doc_links)]
    #![allow(rustdoc::bare_urls)]
    #![allow(missing_docs)]
    #![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
    #![allow(clippy::suspicious_doc_comments)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub use c_bridge::*;

#[cfg(test)]
mod tests {
    #[test]
    fn check_linking() {
        let data: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dest = [0u8; 100];
        unsafe {
            crate::blosc2_compress(
                6,
                crate::BLOSC_SHUFFLE as _,
                1,
                data.as_ptr() as *const core::ffi::c_void,
                data.len() as i32,
                dest.as_mut_ptr() as *mut core::ffi::c_void,
                dest.len() as i32,
            );
        }
    }
}
