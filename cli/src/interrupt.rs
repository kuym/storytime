//! Process-wide Ctrl-C / SIGTERM handling for storytime's long-running work
//! (one-shot synthesis and `clone` training).
//!
//! We do **not** trust the default SIGINT disposition, because in practice a
//! long run can be launched with Ctrl-C already defeated in several ways, all
//! of which look identical to "Ctrl-C does nothing":
//!
//!   - **Inherited `SIG_IGN`.** A background shell job sets SIGINT to ignored
//!     and children inherit it across `exec`; some launchers/IDicators do too.
//!   - **A blocked signal mask.** A backend library can block SIGINT
//!     process-wide; a pending-but-blocked signal is never delivered.
//!   - **A library's own handler.** A GPU/inference backend may install a
//!     SIGINT handler of its own during initialization.
//!
//! `install_*` defeats all three: re-`signal()`-ing reclaims the signal from an
//! inherited `SIG_IGN` or another handler, and `unblock` clears it from the
//! process signal mask. Install **after** backend initialization so a library
//! installed during init can't clobber us. Everything the handler does is
//! async-signal-safe (an atomic store, or `_exit`).

use std::sync::atomic::{AtomicBool, Ordering};

static REQUESTED: AtomicBool = AtomicBool::new(false);

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
    fn _exit(code: i32) -> !;
}

/// Graceful handler: the first interrupt flips a flag the work loop polls (so
/// it can stop cleanly — e.g. write a clone checkpoint); a second interrupt,
/// for an impatient user, hard-exits immediately.
extern "C" fn on_signal_graceful(_sig: i32) {
    if REQUESTED.swap(true, Ordering::SeqCst) {
        unsafe { _exit(130) };
    }
}

/// Abort handler: any interrupt exits immediately. For one-shot synthesis,
/// where a half-rendered WAV isn't worth keeping.
extern "C" fn on_signal_abort(_sig: i32) {
    unsafe { _exit(130) };
}

/// Clear SIGINT/SIGTERM from the process signal mask (in case a library blocked
/// them) so the handler can actually be delivered.
#[cfg(target_os = "macos")]
fn unblock() {
    // macOS `sigset_t` is a `u32` bitmask with bit `(signo - 1)` set per signal.
    extern "C" {
        fn sigprocmask(how: i32, set: *const u32, oldset: *mut u32) -> i32;
    }
    const SIG_UNBLOCK: i32 = 2;
    let mask: u32 = (1 << (SIGINT - 1)) | (1 << (SIGTERM - 1));
    unsafe {
        sigprocmask(SIG_UNBLOCK, &mask, std::ptr::null_mut());
    }
}

/// Linux `sigset_t` is a larger, layout-specific struct; build it via the libc
/// helpers so the size is correct, then unblock SIGINT/SIGTERM.
#[cfg(target_os = "linux")]
fn unblock() {
    // `sigset_t` is up to 128 bytes on Linux; over-allocate and let the kernel
    // helpers populate it rather than guessing the layout.
    #[repr(C)]
    struct SigSet([u64; 16]);
    extern "C" {
        fn sigemptyset(set: *mut SigSet) -> i32;
        fn sigaddset(set: *mut SigSet, signum: i32) -> i32;
        fn sigprocmask(how: i32, set: *const SigSet, oldset: *mut SigSet) -> i32;
    }
    const SIG_UNBLOCK: i32 = 1; // Linux value
    let mut set = SigSet([0; 16]);
    unsafe {
        sigemptyset(&mut set);
        sigaddset(&mut set, SIGINT);
        sigaddset(&mut set, SIGTERM);
        sigprocmask(SIG_UNBLOCK, &set, std::ptr::null_mut());
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn unblock() {}

fn install(handler: extern "C" fn(i32)) {
    let h = handler as *const () as usize;
    unsafe {
        signal(SIGINT, h);
        signal(SIGTERM, h);
    }
    unblock();
}

/// Install the graceful handler (first Ctrl-C → stop after polling, second →
/// hard exit). Pair with [`requested`]. Used by `clone`.
pub fn install_graceful() {
    install(on_signal_graceful);
}

/// Install the abort handler (any Ctrl-C → immediate exit). Used by one-shot
/// synthesis.
pub fn install_abort() {
    install(on_signal_abort);
}

/// True once the user has asked to stop (first Ctrl-C / SIGTERM under the
/// graceful handler).
pub fn requested() -> bool {
    REQUESTED.load(Ordering::Relaxed)
}
