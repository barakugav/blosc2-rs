use std::cell::Cell;

struct Option(Cell<OptionInner>);
#[derive(Clone, Copy)]
enum OptionInner {
    Uninit,
    Enabled,
    Disabled,
}
impl Option {
    const fn new() -> Self {
        Self(Cell::new(OptionInner::Uninit))
    }

    fn get(&self, init: impl FnOnce() -> bool) -> bool {
        match self.0.get() {
            OptionInner::Enabled => true,
            OptionInner::Disabled => false,
            OptionInner::Uninit => {
                let enabled = init();
                self.0.set(if enabled {
                    OptionInner::Enabled
                } else {
                    OptionInner::Disabled
                });
                enabled
            }
        }
    }
}

thread_local! {
    static TRACE_ENABLED: Option = Option::new();
}

#[inline(never)]
#[cold]
pub(crate) fn is_trace_enabled() -> bool {
    TRACE_ENABLED.with(|opt| opt.get(|| std::env::var_os("BLOSC_TRACE").is_some()))
}

macro_rules! trace {
    ($($arg:tt)*) => {{
        if $crate::tracing::is_trace_enabled() {
            eprintln!($($arg)*)
        }
    }};
}
pub(crate) use trace;
