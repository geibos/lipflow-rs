//! The floating pill at the bottom of the screen plus the live mouth video above it
//! (`lipflow/hud.py`). States: listening (mouth video + lip-motion meter + live words), reading
//! (travelling wave), done (the pasted text, fades out), error (fades out). Main thread only.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSAnimatablePropertyContainer, NSAnimationContext, NSAppearance, NSAppearanceCustomization, NSAppearanceNameDarkAqua, NSBackingStoreType, NSBezierPath, NSColor, NSFont, NSFontWeightMedium,
    NSFontWeightSemibold, NSGraphicsContext, NSImage, NSImageScaling, NSImageSymbolConfiguration, NSImageView, NSLineBreakMode, NSPanel,
    NSScreen, NSTextField, NSView, NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState, NSVisualEffectView,
    NSWindowCollectionBehavior, NSWindowStyleMask,
};
use objc2_core_foundation::{CFData, CGFloat};
use objc2_core_graphics::{
    CGBitmapInfo, CGColor, CGColorRenderingIntent, CGColorSpace, CGContext, CGDataProvider, CGImage, CGImageAlphaInfo,
};
use objc2_foundation::{MainThreadMarker, NSPoint, NSRect, NSSize, NSString, NSTimer};

use crate::mouth_view::MouthView;

const W: f64 = 420.0;
const H: f64 = 60.0;
const VW: f64 = 232.0;
const VH: f64 = 144.0;
const BARS: usize = 16;
pub const ACCENT: (f64, f64, f64) = (1.0, 0.33, 0.45); // lip pink
pub const GREEN: (f64, f64, f64) = (0.30, 0.85, 0.55);
pub const AMBER: (f64, f64, f64) = (1.0, 0.72, 0.25);
const NS_STATUS_WINDOW_LEVEL: isize = 25;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Idle,
    Listening,
    Reading,
    Done,
    Error,
}

impl Mode {
    fn symbol(self) -> (&'static str, (f64, f64, f64)) {
        match self {
            Mode::Listening => ("mouth.fill", ACCENT),
            Mode::Done => ("checkmark", GREEN),
            Mode::Error => ("exclamationmark.triangle.fill", AMBER),
            Mode::Reading | Mode::Idle => ("waveform", (1.0, 1.0, 1.0)),
        }
    }

    fn live(self) -> bool {
        matches!(self, Mode::Listening | Mode::Reading)
    }
}

pub fn rgb(c: (f64, f64, f64), a: f64) -> Retained<NSColor> {
    NSColor::colorWithSRGBRed_green_blue_alpha(c.0, c.1, c.2, a)
}

pub fn cg(c: (f64, f64, f64), a: f64) -> objc2_core_foundation::CFRetained<CGColor> {
    CGColor::new_srgb(c.0, c.1, c.2, a)
}

pub fn rect(x: f64, y: f64, w: f64, h: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(w, h))
}

/// SF Symbol at a point size and weight.
pub fn symbol(name: &str, size: f64, weight: CGFloat) -> Option<Retained<NSImage>> {
    let img = NSImage::imageWithSystemSymbolName_accessibilityDescription(&NSString::from_str(name), None)?;
    let cfg = NSImageSymbolConfiguration::configurationWithPointSize_weight(size, weight);
    img.imageWithSymbolConfiguration(&cfg)
}

pub fn weight_semibold() -> CGFloat {
    // SAFETY: reading an immutable AppKit constant.
    unsafe { NSFontWeightSemibold }
}

fn weight_medium() -> CGFloat {
    // SAFETY: reading an immutable AppKit constant.
    unsafe { NSFontWeightMedium }
}

pub fn label(parent: &NSView, frame: NSRect, size: f64, weight: CGFloat, color: &NSColor, mtm: MainThreadMarker) -> Retained<NSTextField> {
    let t = NSTextField::initWithFrame(NSTextField::alloc(mtm), frame);
    t.setBezeled(false);
    t.setDrawsBackground(false);
    t.setEditable(false);
    t.setSelectable(false);
    t.setTextColor(Some(color));
    t.setFont(Some(&NSFont::systemFontOfSize_weight(size, weight)));
    parent.addSubview(&t);
    t
}

/// A borderless, non-activating, click-through panel above everything, on every Space.
fn panel(frame: NSRect, mtm: MainThreadMarker) -> Retained<NSPanel> {
    let p = NSPanel::initWithContentRect_styleMask_backing_defer(
        NSPanel::alloc(mtm),
        frame,
        NSWindowStyleMask::Borderless | NSWindowStyleMask::NonactivatingPanel,
        NSBackingStoreType::Buffered,
        false,
    );
    // SAFETY: disable auto-release on close, required for windows we own.
    unsafe { p.setReleasedWhenClosed(false) };
    p.setLevel(NS_STATUS_WINDOW_LEVEL);
    p.setOpaque(false);
    p.setBackgroundColor(Some(&NSColor::clearColor()));
    p.setHasShadow(true);
    p.setIgnoresMouseEvents(true);
    p.setHidesOnDeactivate(false);
    dark(&p);
    p.setCollectionBehavior(NSWindowCollectionBehavior::CanJoinAllSpaces | NSWindowCollectionBehavior::FullScreenAuxiliary);
    p
}

pub fn dark(w: &objc2_app_kit::NSWindow) {
    // SAFETY: reading an immutable AppKit constant.
    let name = unsafe { NSAppearanceNameDarkAqua };
    w.setAppearance(NSAppearance::appearanceNamed(name).as_deref());
}

/// (outer view, content view): Liquid Glass (NSGlassEffectView, macOS 26+) when available,
/// otherwise the frosted HUD material. The content carries a dark scrim so white text stays
/// readable on any background.
pub fn glass(frame: NSRect, radius: f64, scrim: f64, mtm: MainThreadMarker) -> (Retained<NSView>, Retained<NSView>) {
    let content = NSView::initWithFrame(NSView::alloc(mtm), rect(0.0, 0.0, frame.size.width, frame.size.height));
    content.setWantsLayer(true);
    if let Some(layer) = content.layer() {
        layer.setCornerRadius(radius);
        layer.setMasksToBounds(true);
        layer.setBackgroundColor(Some(&cg((0.06, 0.06, 0.09), scrim)));
    }
    if let Some(cls) = AnyClass::get(c"NSGlassEffectView") {
        // SAFETY: NSGlassEffectView (macOS 26) is an NSView subclass with these methods:
        // initWithFrame:, setStyle: (NSInteger), setCornerRadius: (CGFloat), setTintColor:,
        // setContentView:. The returned object is retained by `Retained`.
        unsafe {
            let g: Option<Retained<NSView>> = msg_send![msg_send![cls, alloc], initWithFrame: frame];
            if let Some(g) = g {
                let _: () = msg_send![&*g, setStyle: 0isize];
                let _: () = msg_send![&*g, setCornerRadius: radius];
                let tint = rgb((0.05, 0.05, 0.08), 0.5);
                let _: () = msg_send![&*g, setTintColor: &*tint];
                let _: () = msg_send![&*g, setContentView: &*content];
                return (g, content);
            }
        }
    }
    let v = NSVisualEffectView::initWithFrame(NSVisualEffectView::alloc(mtm), frame);
    v.setMaterial(NSVisualEffectMaterial::HUDWindow);
    v.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
    v.setState(NSVisualEffectState::Active);
    v.setWantsLayer(true);
    if let Some(layer) = v.layer() {
        layer.setCornerRadius(radius);
        layer.setMasksToBounds(true);
    }
    v.addSubview(&content);
    (Retained::into_super(v), content)
}

// -- meter ---------------------------------------------------------------------------

pub struct MeterIvars {
    levels: RefCell<[f64; BARS]>,
    mode: Cell<Mode>,
    phase: Cell<f64>,
}

define_class!(
    // SAFETY: NSView has no subclassing requirements beyond being used on the main thread.
    #[unsafe(super = NSView)]
    #[thread_kind = MainThreadOnly]
    #[ivars = MeterIvars]
    struct MeterView;

    impl MeterView {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty: NSRect) {
            let iv = self.ivars();
            let mode = iv.mode.get();
            if !mode.live() {
                return;
            }
            let b = self.bounds();
            let step = b.size.width / BARS as f64;
            let mid = b.size.height / 2.0;
            let phase = iv.phase.get();
            for (i, &lv0) in iv.levels.borrow().iter().enumerate() {
                let (lv, color) = if mode == Mode::Reading {
                    let s = (phase * 2.0 - i as f64 * 0.5).sin();
                    (0.22 + 0.22 * s, rgb((1.0, 1.0, 1.0), 0.55 + 0.35 * s.max(0.0)))
                } else {
                    (lv0, rgb(ACCENT, 0.45 + 0.55 * lv0))
                };
                let h = 3.0 + lv * (b.size.height - 6.0);
                color.setFill();
                NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(rect(i as f64 * step + (step - 2.5) / 2.0, mid - h / 2.0, 2.5, h), 1.25, 1.25).fill();
            }
        }
    }
);

impl MeterView {
    fn new(frame: NSRect, mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(MeterIvars { levels: RefCell::new([0.0; BARS]), mode: Cell::new(Mode::Idle), phase: Cell::new(0.0) });
        // SAFETY: NSView's designated initializer.
        unsafe { msg_send![super(this), initWithFrame: frame] }
    }
}

// -- video ---------------------------------------------------------------------------

pub struct VideoIvars {
    frame: RefCell<Option<MouthView>>,
    hint: RefCell<Option<Retained<NSTextField>>>,
}

define_class!(
    // SAFETY: NSView subclass used on the main thread only.
    #[unsafe(super = NSView)]
    #[thread_kind = MainThreadOnly]
    #[ivars = VideoIvars]
    pub struct VideoView;

    impl VideoView {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty: NSRect) {
            let b = self.bounds();
            let frame = self.ivars().frame.borrow();
            let Some(v) = frame.as_ref() else { return };
            let Some(ctx) = NSGraphicsContext::currentContext() else { return };
            let cgctx: Retained<CGContext> = ctx.CGContext();
            let Some(img) = cg_image(v) else { return };
            // aspect fit
            let s = (b.size.width / v.width as f64).min(b.size.height / v.height as f64);
            let (dw, dh) = (v.width as f64 * s, v.height as f64 * s);
            let (ox, oy) = ((b.size.width - dw) / 2.0, (b.size.height - dh) / 2.0);
            CGContext::draw_image(Some(&cgctx), objc2_core_foundation::CGRect::new(objc2_core_foundation::CGPoint::new(ox, oy), objc2_core_foundation::CGSize::new(dw, dh)), Some(&img));
            let to_view = |p: [f32; 2]| NSPoint::new(ox + f64::from(p[0]) * s, oy + dh - f64::from(p[1]) * s);
            if v.face {
                for contour in [&v.outer, &v.inner] {
                    let path = NSBezierPath::bezierPath();
                    for (i, &p) in contour.iter().enumerate() {
                        if i == 0 { path.moveToPoint(to_view(p)) } else { path.lineToPoint(to_view(p)) }
                    }
                    path.closePath();
                    path.setLineWidth(1.0);
                    rgb((1.0, 150.0 / 255.0, 170.0 / 255.0), 0.7).setStroke();
                    path.stroke();
                }
                rgb((250.0 / 255.0, 92.0 / 255.0, 115.0 / 255.0), 1.0).setFill();
                for &p in &v.points {
                    let c = to_view(p);
                    NSBezierPath::bezierPathWithOvalInRect(rect(c.x - 2.0, c.y - 2.0, 4.0, 4.0)).fill();
                }
            }
        }
    }
);

fn cg_image(v: &MouthView) -> Option<objc2_core_foundation::CFRetained<CGImage>> {
    let data = CFData::from_bytes(&v.rgba);
    let provider = CGDataProvider::with_cf_data(Some(&data))?;
    let space = CGColorSpace::new_device_rgb()?;
    // SAFETY: the provider holds w*h*4 bytes of RGBX rows, matching the arguments.
    unsafe {
        CGImage::new(
            v.width,
            v.height,
            8,
            32,
            v.width * 4,
            Some(&space),
            CGBitmapInfo(CGImageAlphaInfo::NoneSkipLast.0),
            Some(&provider),
            std::ptr::null(),
            true,
            CGColorRenderingIntent::RenderingIntentDefault,
        )
    }
}

impl VideoView {
    fn new(frame: NSRect, mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(VideoIvars { frame: RefCell::new(None), hint: RefCell::new(None) });
        // SAFETY: NSView's designated initializer.
        let this: Retained<Self> = unsafe { msg_send![super(this), initWithFrame: frame] };
        let hint = label(&this, rect(14.0, 8.0, frame.size.width - 20.0, 16.0), 10.5, weight_medium(), &rgb((0.92, 0.92, 0.92), 1.0), mtm);
        hint.setStringValue(&NSString::from_str("looking for your face..."));
        hint.setHidden(true);
        *this.ivars().hint.borrow_mut() = Some(hint);
        this
    }

    pub fn set(&self, v: Option<MouthView>) {
        let looking = v.as_ref().is_some_and(|v| !v.face);
        if let Some(h) = self.ivars().hint.borrow().as_ref() {
            h.setHidden(!looking);
        }
        *self.ivars().frame.borrow_mut() = v;
        self.setNeedsDisplay(true);
    }
}

pub fn make_video_view(frame: NSRect, mtm: MainThreadMarker) -> Retained<VideoView> {
    VideoView::new(frame, mtm)
}

// -- the HUD -------------------------------------------------------------------------

pub struct Hud {
    panel: Retained<NSPanel>,
    badge: Retained<NSView>,
    icon: Retained<NSImageView>,
    title: Retained<NSTextField>,
    pub body: Retained<NSTextField>,
    meter: Retained<MeterView>,
    video_panel: Retained<NSPanel>,
    video: Retained<VideoView>,
    rec_dot: Retained<NSView>,
    hide_timer: RefCell<Option<Retained<NSTimer>>>,
    anim_timer: RefCell<Option<Retained<NSTimer>>>,
    mode: Cell<Mode>,
}

impl Hud {
    pub fn new(mtm: MainThreadMarker) -> Self {
        let screen = NSScreen::mainScreen(mtm).map_or_else(|| rect(0.0, 0.0, 1440.0, 900.0), |s| s.visibleFrame());
        let x = screen.origin.x + (screen.size.width - W) / 2.0;
        let y = screen.origin.y + 28.0;

        let pill = panel(rect(x, y, W, H), mtm);
        let (outer, content) = glass(rect(0.0, 0.0, W, H), H / 2.0, 0.8, mtm);
        pill.setContentView(Some(&outer));
        let badge_size = 36.0;
        let badge = NSView::initWithFrame(NSView::alloc(mtm), rect(12.0, (H - badge_size) / 2.0, badge_size, badge_size));
        badge.setWantsLayer(true);
        if let Some(l) = badge.layer() {
            l.setCornerRadius(badge_size / 2.0);
        }
        content.addSubview(&badge);
        let icon = NSImageView::initWithFrame(NSImageView::alloc(mtm), rect(0.0, 0.0, badge_size, badge_size));
        icon.setImageScaling(NSImageScaling::ScaleNone);
        badge.addSubview(&icon);
        let tx = 12.0 + badge_size + 12.0;
        let meter = MeterView::new(rect(W - 18.0 - BARS as f64 * 5.0, (H - 28.0) / 2.0, BARS as f64 * 5.0, 28.0), mtm);
        content.addSubview(&meter);
        let title = label(&content, rect(tx, H / 2.0 + 2.0, W - tx - 20.0, 15.0), 10.5, weight_semibold(), &rgb((1.0, 1.0, 1.0), 0.68), mtm);
        let body = label(&content, rect(tx, H / 2.0 - 19.0, W - tx - 20.0, 19.0), 14.0, weight_medium(), &rgb((1.0, 1.0, 1.0), 0.95), mtm);

        let video_panel = panel(rect(x + (W - VW) / 2.0, y + H + 10.0, VW, VH), mtm);
        let (vouter, vcontent) = glass(rect(0.0, 0.0, VW, VH), 22.0, 0.7, mtm);
        video_panel.setContentView(Some(&vouter));
        let video = VideoView::new(rect(6.0, 6.0, VW - 12.0, VH - 12.0), mtm);
        video.setWantsLayer(true);
        if let Some(l) = video.layer() {
            l.setCornerRadius(16.0);
            l.setMasksToBounds(true);
            l.setBorderWidth(1.5);
            l.setBorderColor(Some(&cg(ACCENT, 0.75)));
        }
        vcontent.addSubview(&video);
        let tag = NSView::initWithFrame(NSView::alloc(mtm), rect(14.0, VH - 32.0, 44.0, 18.0));
        tag.setWantsLayer(true);
        if let Some(l) = tag.layer() {
            l.setCornerRadius(9.0);
            l.setBackgroundColor(Some(&cg((0.0, 0.0, 0.0), 0.55)));
        }
        vcontent.addSubview(&tag);
        let dot = NSView::initWithFrame(NSView::alloc(mtm), rect(7.0, 6.0, 6.0, 6.0));
        dot.setWantsLayer(true);
        if let Some(l) = dot.layer() {
            l.setCornerRadius(3.0);
            l.setBackgroundColor(Some(&cg(ACCENT, 1.0)));
        }
        tag.addSubview(&dot);
        label(&tag, rect(15.0, 1.0, 28.0, 14.0), 9.5, weight_semibold(), &rgb((1.0, 1.0, 1.0), 0.9), mtm).setStringValue(&NSString::from_str("REC"));

        Self {
            panel: pill,
            badge,
            icon,
            title,
            body,
            meter,
            video_panel,
            video,
            rec_dot: dot,
            hide_timer: RefCell::new(None),
            anim_timer: RefCell::new(None),
            mode: Cell::new(Mode::Idle),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode.get()
    }

    pub fn body_text(&self) -> String {
        self.body.stringValue().to_string()
    }

    pub fn show(&'static self, mode: Mode, title: &str, body: &str, hide_after: Option<f64>) {
        self.cancel_hide();
        self.mode.set(mode);
        let (sym, tint) = mode.symbol();
        self.icon.setImage(symbol(sym, 15.0, weight_semibold()).as_deref());
        self.icon.setContentTintColor(Some(&rgb(tint, 1.0)));
        if let Some(l) = self.badge.layer() {
            l.setBackgroundColor(Some(&cg(tint, 0.16)));
        }
        self.title.setStringValue(&NSString::from_str(&title.to_uppercase()));
        self.body.setStringValue(&NSString::from_str(body));
        let live = mode.live();
        if let Some(cell) = self.body.cell() {
            cell.setLineBreakMode(if live { NSLineBreakMode::ByTruncatingHead } else { NSLineBreakMode::ByTruncatingTail });
        }
        let tx = self.body.frame().origin.x;
        let width = W - tx - 20.0 - if live { BARS as f64 * 5.0 + 14.0 } else { 0.0 };
        self.body.setFrameSize(NSSize::new(width, self.body.frame().size.height));
        let iv = self.meter.ivars();
        iv.mode.set(mode);
        if mode != Mode::Listening {
            *iv.levels.borrow_mut() = [0.0; BARS];
        }
        self.meter.setNeedsDisplay(true);
        if live {
            self.start_anim();
        } else {
            self.stop_anim();
        }
        self.panel.setAlphaValue(1.0);
        self.panel.orderFrontRegardless();
        if mode == Mode::Listening {
            self.video_panel.setAlphaValue(1.0);
            self.video_panel.orderFrontRegardless();
        } else {
            self.video_panel.orderOut(None);
            self.video.set(None);
        }
        if let Some(secs) = hide_after {
            let block = RcBlock::new(move |_t: NonNull<NSTimer>| self.fade_out());
            // SAFETY: the block only touches `self`, which is 'static and main-thread.
            let t = unsafe { NSTimer::scheduledTimerWithTimeInterval_repeats_block(secs, false, &block) };
            *self.hide_timer.borrow_mut() = Some(t);
        }
    }

    pub fn set_text(&self, body: &str) {
        self.body.setStringValue(&NSString::from_str(body));
    }

    /// A new video frame and lip-motion level while listening.
    pub fn set_frame(&self, view: Option<MouthView>, level: f64) {
        if self.mode.get() == Mode::Listening && view.is_some() {
            self.video.set(view);
        }
        let mut lv = self.meter.ivars().levels.borrow_mut();
        lv.rotate_left(1);
        lv[BARS - 1] = level.clamp(0.0, 1.0);
    }

    pub fn hide(&self) {
        self.cancel_hide();
        self.stop_anim();
        self.mode.set(Mode::Idle);
        self.panel.orderOut(None);
        self.video_panel.orderOut(None);
        self.video.set(None);
    }

    fn fade_out(&'static self) {
        *self.hide_timer.borrow_mut() = None;
        let block = RcBlock::new(move |ctx: NonNull<NSAnimationContext>| {
            // SAFETY: AppKit passes a valid animation context for the duration of the block.
            unsafe { ctx.as_ref() }.setDuration(0.3);
            self.panel.animator().setAlphaValue(0.0);
            self.video_panel.animator().setAlphaValue(0.0);
        });
        let done = RcBlock::new(move || {
            if self.panel.alphaValue() < 0.05 {
                self.hide();
            }
        });
        NSAnimationContext::runAnimationGroup_completionHandler(&block, Some(&done));
    }

    fn start_anim(&'static self) {
        if self.anim_timer.borrow().is_some() {
            return;
        }
        let block = RcBlock::new(move |_t: NonNull<NSTimer>| self.tick());
        // SAFETY: as in `show`.
        let t = unsafe { NSTimer::scheduledTimerWithTimeInterval_repeats_block(1.0 / 30.0, true, &block) };
        *self.anim_timer.borrow_mut() = Some(t);
    }

    fn stop_anim(&self) {
        if let Some(t) = self.anim_timer.borrow_mut().take() {
            t.invalidate();
        }
    }

    fn cancel_hide(&self) {
        if let Some(t) = self.hide_timer.borrow_mut().take() {
            t.invalidate();
        }
    }

    fn tick(&self) {
        let iv = self.meter.ivars();
        iv.phase.set(iv.phase.get() + 0.18);
        self.meter.setNeedsDisplay(true);
        if let Some(l) = self.rec_dot.layer() {
            l.setOpacity((0.55 + 0.45 * (0.5 + 0.5 * (iv.phase.get() * 1.4).sin())) as f32);
        }
    }
}

