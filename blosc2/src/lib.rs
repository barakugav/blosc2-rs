#![cfg_attr(deny_warnings, deny(warnings))]
// #![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

// Set the BLOSC_TRACE environment variable
//  * for getting more info on what is happening. If the error is not related with
//  * wrong params, please report it back together with the buffer data causing this,
//  * as well as the compression params used.

mod error;
pub use error::Error;

mod encode;
pub use encode::*;

mod misc;
pub use misc::*;

mod chunk;
pub use chunk::*;

mod schunk;
pub use schunk::*;

mod global;
pub mod util;

mod tracing;
pub(crate) use tracing::trace;

/// The version of the crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The version of the underlying C-blosc2 library used by this crate.
pub const BLOSC2_C_VERSION: &str = {
    match blosc2_sys::BLOSC2_VERSION_STRING.to_str() {
        Ok(s) => s,
        Err(_) => unreachable!(),
    }
};
