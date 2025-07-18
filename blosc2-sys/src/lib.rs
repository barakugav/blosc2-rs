// Set the BLOSC_TRACE environment variable
//  * for getting more info on what is happening. If the error is not related with
//  * wrong params, please report it back together with the buffer data causing this,
//  * as well as the compression params used.

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
