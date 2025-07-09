use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Once;

struct InitGuard;
impl InitGuard {
    fn new() -> Self {
        unsafe { blosc2_sys::blosc2_init() };
        Self
    }
}
impl Drop for InitGuard {
    fn drop(&mut self) {
        unsafe { blosc2_sys::blosc2_destroy() };
    }
}

// TODO: use LazyOnce instead of this custom struct when we bump MSRV to 1.80
struct InitGuardSync {
    lock: Once,
    inner: UnsafeCell<MaybeUninit<InitGuard>>,
}
impl InitGuardSync {
    const fn new() -> Self {
        InitGuardSync {
            lock: Once::new(),
            inner: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}
impl Drop for InitGuardSync {
    fn drop(&mut self) {
        if self.lock.is_completed() {
            let inner = self.inner.get_mut();
            unsafe { MaybeUninit::assume_init_drop(inner) };
        }
    }
}
unsafe impl Sync for InitGuardSync {}

static GLOBAL_GUARD: InitGuardSync = InitGuardSync::new();

pub(crate) fn global_init() {
    GLOBAL_GUARD.lock.call_once(|| {
        let guard = unsafe { &mut *GLOBAL_GUARD.inner.get() };
        guard.write(InitGuard::new());
    });
}
