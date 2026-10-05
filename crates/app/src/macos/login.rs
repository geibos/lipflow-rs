//! Open at login (macOS 13+ `SMAppService.mainAppService`): the hotkey only works while Lipflow
//! runs, so it starts with your session unless you turn that off in Settings.

use objc2_service_management::{SMAppService, SMAppServiceStatus};

/// Lipflow is started at login.
pub fn enabled() -> bool {
    // SAFETY: class method and status getter of ServiceManagement, no arguments.
    unsafe { SMAppService::mainAppService().status() == SMAppServiceStatus::Enabled }
}

/// Start (or stop starting) Lipflow at login. Only meaningful for the installed app bundle.
pub fn set(on: bool) -> Result<(), String> {
    // SAFETY: as above; register/unregister report failures through NSError.
    unsafe {
        let s = SMAppService::mainAppService();
        let r = if on { s.registerAndReturnError() } else { s.unregisterAndReturnError() };
        r.map_err(|e| e.localizedDescription().to_string())
    }
}

/// Running from the app bundle (not `cargo run`).
pub fn in_bundle() -> bool {
    std::env::current_exe().is_ok_and(|p| p.to_string_lossy().contains(".app/Contents/MacOS/"))
}
