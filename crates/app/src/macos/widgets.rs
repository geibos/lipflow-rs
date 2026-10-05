//! Shared window pieces for Settings and setup: a borderless glass window that can become key,
//! wrapping labels and hand-drawn capsule buttons (system buttons don't render on glass).

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Sel};
use objc2::{MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSBackingStoreType, NSButton, NSColor, NSFont, NSFontAttributeName, NSFontWeightRegular, NSForegroundColorAttributeName,
    NSMutableParagraphStyle, NSParagraphStyleAttributeName, NSTextAlignment, NSTextField, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_core_foundation::CGFloat;
use objc2_foundation::{MainThreadMarker, NSAttributedString, NSDictionary, NSRect, NSString};

use super::hud::{ACCENT, cg, dark, glass, rect, rgb, weight_semibold};

define_class!(
    // SAFETY: NSWindow subclass on the main thread; borderless windows must opt in to key status.
    #[unsafe(super = NSWindow)]
    #[thread_kind = MainThreadOnly]
    pub struct GlassWindow;

    impl GlassWindow {
        #[unsafe(method(canBecomeKeyWindow))]
        fn can_become_key(&self) -> bool {
            true
        }

        #[unsafe(method(canBecomeMainWindow))]
        fn can_become_main(&self) -> bool {
            true
        }
    }
);

/// (window, root content view) — a rounded glass window, movable by its background.
pub fn glass_window(w: f64, h: f64, title: &str, mtm: MainThreadMarker) -> (Retained<GlassWindow>, Retained<NSView>) {
    let this = GlassWindow::alloc(mtm).set_ivars(());
    // SAFETY: NSWindow's designated initializer.
    let win: Retained<GlassWindow> = unsafe {
        msg_send![super(this), initWithContentRect: rect(0.0, 0.0, w, h), styleMask: NSWindowStyleMask::Borderless, backing: NSBackingStoreType::Buffered, defer: false]
    };
    // SAFETY: we own the window for the app's lifetime.
    unsafe { win.setReleasedWhenClosed(false) };
    win.setTitle(&NSString::from_str(title));
    win.setMovableByWindowBackground(true);
    win.setOpaque(false);
    win.setBackgroundColor(Some(&NSColor::clearColor()));
    win.setHasShadow(true);
    dark(&win);
    let (outer, root) = glass(rect(0.0, 0.0, w, h), 26.0, 0.72, mtm);
    win.setContentView(Some(&outer));
    win.center();
    (win, root)
}

/// Float the window above others and bring the (menu-bar-only) app forward.
pub fn present(win: &NSWindow, mtm: MainThreadMarker) {
    const NS_FLOATING_WINDOW_LEVEL: isize = 3;
    win.setLevel(NS_FLOATING_WINDOW_LEVEL);
    win.center();
    win.orderFrontRegardless();
    win.makeKeyWindow();
    #[allow(deprecated)]
    NSApplication::sharedApplication(mtm).activateIgnoringOtherApps(true);
}

pub struct TextStyle {
    pub size: f64,
    pub weight: CGFloat,
    pub alpha: f64,
    pub center: bool,
    pub color: Option<Retained<NSColor>>,
}

impl TextStyle {
    pub fn new(size: f64) -> Self {
        // SAFETY: reading an immutable AppKit constant.
        Self { size, weight: unsafe { NSFontWeightRegular }, alpha: 0.95, center: false, color: None }
    }
    pub fn weight(mut self, w: CGFloat) -> Self {
        self.weight = w;
        self
    }
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }
    pub fn center(mut self) -> Self {
        self.center = true;
        self
    }
    pub fn color(mut self, c: Retained<NSColor>) -> Self {
        self.color = Some(c);
        self
    }
}

/// A wrapping, non-selectable label.
pub fn text(parent: &NSView, frame: NSRect, s: &str, style: TextStyle, mtm: MainThreadMarker) -> Retained<NSTextField> {
    let t = NSTextField::wrappingLabelWithString(&NSString::from_str(s), mtm);
    t.setFrame(frame);
    t.setFont(Some(&NSFont::systemFontOfSize_weight(style.size, style.weight)));
    t.setTextColor(Some(&style.color.unwrap_or_else(|| rgb((1.0, 1.0, 1.0), style.alpha))));
    t.setSelectable(false);
    if style.center {
        t.setAlignment(NSTextAlignment::Center);
    }
    parent.addSubview(&t);
    t
}

/// A capsule button drawn by hand; `primary` is filled with the accent colour.
pub fn capsule(parent: &NSView, title: &str, target: &AnyObject, action: Sel, frame: NSRect, primary: bool, mtm: MainThreadMarker) -> Retained<NSButton> {
    // SAFETY: target/action pair; the target outlives the button (both owned by the window).
    let b = unsafe { NSButton::buttonWithTitle_target_action(&NSString::from_str(title), Some(target), Some(action), mtm) };
    b.setBordered(false);
    b.setFrame(frame);
    b.setWantsLayer(true);
    if let Some(l) = b.layer() {
        l.setCornerRadius(frame.size.height / 2.0);
        let bg = if primary { cg(ACCENT, 1.0) } else { cg((1.0, 1.0, 1.0), 0.12) };
        l.setBackgroundColor(Some(&bg));
        l.setBorderWidth(if primary { 0.0 } else { 0.5 });
        l.setBorderColor(Some(&cg((1.0, 1.0, 1.0), 0.18)));
    }
    let para = NSMutableParagraphStyle::new();
    para.setAlignment(NSTextAlignment::Center);
    let font = NSFont::systemFontOfSize_weight(if frame.size.height >= 40.0 { 14.0 } else { 13.0 }, weight_semibold());
    let color = rgb((1.0, 1.0, 1.0), if primary { 1.0 } else { 0.9 });
    // SAFETY: reading immutable AppKit attribute-name constants.
    let (kf, kc, kp) = unsafe { (NSFontAttributeName, NSForegroundColorAttributeName, NSParagraphStyleAttributeName) };
    let keys: [&NSString; 3] = [kf, kc, kp];
    let vals: [&AnyObject; 3] = [&font, &color, &para];
    let attrs = NSDictionary::from_slices(&keys, &vals);
    // SAFETY: attributes of the documented value types.
    let s = unsafe { NSAttributedString::new_with_attributes(&NSString::from_str(title), &attrs) };
    b.setAttributedTitle(&s);
    if primary {
        b.setKeyEquivalent(&NSString::from_str("\r"));
    }
    parent.addSubview(&b);
    b
}

/// The round × in the top-right corner.
pub fn close_button(parent: &NSView, w: f64, h: f64, target: &AnyObject, action: Sel, mtm: MainThreadMarker) -> Retained<NSButton> {
    // SAFETY: as in `capsule`.
    let b = unsafe { NSButton::buttonWithTitle_target_action(&NSString::from_str(""), Some(target), Some(action), mtm) };
    b.setBordered(false);
    b.setImage(super::hud::symbol("xmark", 11.0, weight_semibold()).as_deref());
    b.setContentTintColor(Some(&rgb((1.0, 1.0, 1.0), 0.7)));
    b.setFrame(rect(w - 44.0, h - 44.0, 26.0, 26.0));
    b.setWantsLayer(true);
    if let Some(l) = b.layer() {
        l.setCornerRadius(13.0);
        l.setBackgroundColor(Some(&cg((1.0, 1.0, 1.0), 0.1)));
    }
    parent.addSubview(&b);
    b
}
