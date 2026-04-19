use nih_plug::prelude::ParamSetter;
use nih_plug_egui::egui::{
    self, Align2, Color32, Context, FontId, Frame, Id, Pos2, Rect, RichText,
    CornerRadius, Sense, Shape, Stroke, Vec2,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::TcnLa2aParams;

// ─── Meter state shared between audio thread and GUI editor ──────────────────

pub struct MeterState {
    pub input_peak:    AtomicU32, // f32 bits, linear amplitude
    pub output_peak:   AtomicU32, // f32 bits, linear amplitude
    pub gain_reduction: AtomicU32, // f32 bits, dB (≤ 0)
}

impl Default for MeterState {
    fn default() -> Self {
        Self {
            input_peak:    AtomicU32::new(0.0f32.to_bits()),
            output_peak:   AtomicU32::new(0.0f32.to_bits()),
            gain_reduction: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

// ─── Meter ballistics ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct MeterBallistics {
    input_db:  f32,
    output_db: f32,
    gr_db:     f32,
}

impl Default for MeterBallistics {
    fn default() -> Self {
        Self { input_db: -120.0, output_db: -120.0, gr_db: 0.0 }
    }
}

fn smooth(current: f32, target: f32, attack: f32, decay: f32) -> f32 {
    let coeff = if target > current { attack } else { decay };
    current + (target - current) * coeff
}

// ─── Palette ──────────────────────────────────────────────────────────────────

const BG:         Color32 = Color32::from_rgb(24, 24, 28);
const METER_BG:   Color32 = Color32::from_rgb(13, 13, 16);
const GREEN:      Color32 = Color32::from_rgb(60, 195, 80);
const AMBER:      Color32 = Color32::from_rgb(255, 168, 48);
const RED:        Color32 = Color32::from_rgb(228, 48, 48);
const ACCENT:     Color32 = Color32::from_rgb(80, 140, 220);
const TEXT:       Color32 = Color32::from_rgb(215, 215, 225);
const TEXT_DIM:   Color32 = Color32::from_rgb(110, 110, 125);
const KNOB_BODY:  Color32 = Color32::from_rgb(50, 50, 58);
const KNOB_RIM:   Color32 = Color32::from_rgb(78, 78, 90);
const DIVIDER:    Color32 = Color32::from_rgb(45, 45, 52);

pub const GUI_WIDTH:  u32 = 480;
pub const GUI_HEIGHT: u32 = 340;

// ─── Entry point ──────────────────────────────────────────────────────────────

pub fn draw_ui(
    ctx: &Context,
    setter: &ParamSetter,
    params: &TcnLa2aParams,
    meters: &Arc<MeterState>,
) {
    ctx.request_repaint(); // meters need continuous refresh

    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG;
    visuals.window_fill = BG;
    visuals.override_text_color = Some(TEXT);
    ctx.set_visuals(visuals);

    egui::CentralPanel::default()
        .frame(Frame::new().fill(BG).inner_margin(egui::Margin::same(10)))
        .show(ctx, |ui| {
            ui.style_mut().spacing.item_spacing = Vec2::new(10.0, 2.0);

            // ── Header ──────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label(RichText::new("micro-TCN").color(ACCENT).size(13.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new("Neural Compressor  ·  LA-2A")
                            .color(TEXT_DIM)
                            .size(10.0),
                    );
                });
            });

            ui.add_space(4.0);
            ui.painter().hline(
                ui.available_rect_before_wrap().x_range(),
                ui.cursor().top(),
                Stroke::new(1.0, DIVIDER),
            );
            ui.add_space(6.0);

            // ── Meters ──────────────────────────────────────────────────
            let target_in_db  = lin_to_db(f32::from_bits(meters.input_peak.load(Ordering::Relaxed)));
            let target_out_db = lin_to_db(f32::from_bits(meters.output_peak.load(Ordering::Relaxed)));
            let target_gr_db  = f32::from_bits(meters.gain_reduction.load(Ordering::Relaxed));

            let meters_id = Id::new("meter_ballistics");
            let mut b = ctx.data(|d| d.get_temp::<MeterBallistics>(meters_id).unwrap_or_default());
            b.input_db  = smooth(b.input_db,  target_in_db,  0.5, 0.07);
            b.output_db = smooth(b.output_db, target_out_db, 0.5, 0.07);
            b.gr_db = smooth(b.gr_db, target_gr_db, 0.5, 0.07);
            ctx.data_mut(|d| d.insert_temp(meters_id, b.clone()));

            let (input_db, output_db, gr_db) = (b.input_db, b.output_db, b.gr_db);

            ui.horizontal(|ui| {
                ui.add_space(4.0);
                level_meter(ui, "IN",  input_db,  48.0, 140.0, GREEN);
                ui.add_space(8.0);
                gr_meter(ui, "GR",   gr_db,   72.0, 140.0);
                ui.add_space(8.0);
                level_meter(ui, "OUT", output_db, 48.0, 140.0, GREEN);
            });

            ui.add_space(6.0);
            ui.painter().hline(
                ui.available_rect_before_wrap().x_range(),
                ui.cursor().top(),
                Stroke::new(1.0, DIVIDER),
            );
            ui.add_space(6.0);

            // ── Controls ─────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.add_space(16.0);

                let mut pr = params.peak_reduction.value();
                if knob(ui, "PEAK RED", &mut pr, 0.0..=1.0, |v| {
                    format!("{:.0}%", v * 100.0)
                })
                .changed()
                {
                    setter.begin_set_parameter(&params.peak_reduction);
                    setter.set_parameter(&params.peak_reduction, pr);
                    setter.end_set_parameter(&params.peak_reduction);
                }

                ui.add_space(24.0);

                let is_limit = params.limit.value();
                if let Some(new_limit) = mode_button(ui, is_limit) {
                    setter.begin_set_parameter(&params.limit);
                    setter.set_parameter(&params.limit, new_limit);
                    setter.end_set_parameter(&params.limit);
                }
            });
        });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn lin_to_db(lin: f32) -> f32 {
    if lin < 1e-6 { -120.0 } else { 20.0 * lin.log10() }
}

// ─── Widgets ──────────────────────────────────────────────────────────────────

fn level_meter(ui: &mut egui::Ui, label: &str, db: f32, w: f32, h: f32, color: Color32) {
    let total = Vec2::new(w + 20.0, h + 32.0);
    let (rect, _) = ui.allocate_exact_size(total, Sense::hover());
    if !ui.is_rect_visible(rect) { return; }
    let p = ui.painter();
    let bar = Rect::from_min_size(rect.min, Vec2::new(w, h));

    p.rect_filled(bar, CornerRadius::same(2), METER_BG);
    p.rect_stroke(bar, CornerRadius::same(2), Stroke::new(0.5, DIVIDER), egui::StrokeKind::Outside);

    let db_range = 60.0f32;
    let norm = ((db + db_range) / db_range).clamp(0.0, 1.0);
    let fill_h = norm * bar.height();
    if fill_h > 1.0 {
        let fill = Rect::from_min_max(
            Pos2::new(bar.min.x + 1.0, bar.max.y - fill_h),
            Pos2::new(bar.max.x - 1.0, bar.max.y - 1.0),
        );
        let c = if db > -3.0 { RED } else if db > -12.0 { AMBER } else { color };
        p.rect_filled(fill, CornerRadius::same(1), c);
    }

    for tick_db in [-6.0f32, -18.0, -36.0] {
        let n = ((tick_db + db_range) / db_range).clamp(0.0, 1.0);
        let y = bar.max.y - n * bar.height();
        p.line_segment([Pos2::new(bar.min.x, y), Pos2::new(bar.max.x, y)], Stroke::new(0.5, DIVIDER));
        p.text(Pos2::new(bar.max.x + 3.0, y), Align2::LEFT_CENTER, format!("{}", tick_db as i32), FontId::monospace(7.5), TEXT_DIM);
    }

    let val = if db < -99.0 { "-∞".into() } else { format!("{:.1}", db) };
    p.text(Pos2::new(bar.center().x, bar.max.y + 5.0), Align2::CENTER_TOP, label, FontId::proportional(9.0), TEXT_DIM);
    p.text(Pos2::new(bar.center().x, bar.max.y + 17.0), Align2::CENTER_TOP, val, FontId::monospace(9.0), TEXT);
}

fn gr_meter(ui: &mut egui::Ui, label: &str, gr_db: f32, w: f32, h: f32) {
    let total = Vec2::new(w + 20.0, h + 32.0);
    let (rect, _) = ui.allocate_exact_size(total, Sense::hover());
    if !ui.is_rect_visible(rect) { return; }
    let p = ui.painter();
    let bar = Rect::from_min_size(rect.min, Vec2::new(w, h));

    p.rect_filled(bar, CornerRadius::same(2), METER_BG);
    p.rect_stroke(bar, CornerRadius::same(2), Stroke::new(0.5, DIVIDER), egui::StrokeKind::Outside);

    let gr_range = 30.0f32;
    let norm = (gr_db.abs() / gr_range).clamp(0.0, 1.0);
    let fill_h = norm * bar.height();
    if fill_h > 1.0 {
        let fill = Rect::from_min_max(
            Pos2::new(bar.min.x + 1.0, bar.min.y + 1.0),
            Pos2::new(bar.max.x - 1.0, bar.min.y + fill_h),
        );
        p.rect_filled(fill, CornerRadius::same(1), AMBER);
    }

    p.line_segment([Pos2::new(bar.min.x, bar.min.y + 1.0), Pos2::new(bar.max.x, bar.min.y + 1.0)], Stroke::new(0.5, Color32::from_rgb(70, 70, 80)));
    p.text(Pos2::new(bar.max.x + 3.0, bar.min.y + 1.0), Align2::LEFT_CENTER, "0", FontId::monospace(7.5), TEXT_DIM);

    for tick in [-6.0f32, -12.0, -20.0] {
        let n = (tick.abs() / gr_range).clamp(0.0, 1.0);
        let y = bar.min.y + n * bar.height();
        p.line_segment([Pos2::new(bar.min.x, y), Pos2::new(bar.max.x, y)], Stroke::new(0.5, DIVIDER));
        p.text(Pos2::new(bar.max.x + 3.0, y), Align2::LEFT_CENTER, format!("{}", tick as i32), FontId::monospace(7.5), TEXT_DIM);
    }

    let val = if gr_db.abs() < 0.05 { "0.0".into() } else { format!("{:.1}", gr_db) };
    p.text(Pos2::new(bar.center().x, bar.max.y + 5.0), Align2::CENTER_TOP, label, FontId::proportional(9.0), TEXT_DIM);
    p.text(Pos2::new(bar.center().x, bar.max.y + 17.0), Align2::CENTER_TOP, val, FontId::monospace(9.0), TEXT);
}

fn knob(
    ui: &mut egui::Ui,
    name: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    fmt: impl Fn(f32) -> String,
) -> egui::Response {
    let size = Vec2::new(68.0, 76.0);
    let (rect, mut resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let center = Pos2::new(rect.center().x, rect.min.y + 28.0);
    let radius = 22.0f32;

    if resp.dragged() {
        let range_span = range.end() - range.start();
        let delta = -resp.drag_delta().y * range_span / 150.0;
        let new_val = (*value + delta).clamp(*range.start(), *range.end());
        if (new_val - *value).abs() > f32::EPSILON {
            *value = new_val;
            resp.mark_changed();
        }
    }

    if ui.is_rect_visible(rect) {
        let p = ui.painter();

        if resp.hovered() || resp.dragged() {
            p.circle_stroke(center, radius + 3.5, Stroke::new(1.0, ACCENT.linear_multiply(0.35)));
        }

        p.circle_filled(center, radius, KNOB_BODY);
        p.circle_stroke(center, radius, Stroke::new(1.5, KNOB_RIM));

        let start_a = std::f32::consts::PI * 0.75;
        let sweep   = std::f32::consts::PI * 1.5;
        let norm    = (*value - range.start()) / (range.end() - range.start());
        let fill_a  = start_a + norm * sweep;
        let track_r = radius - 5.0;

        let arc_steps = 32usize;
        let bg_pts: Vec<Pos2> = (0..=arc_steps)
            .map(|i| {
                let a = start_a + (i as f32 / arc_steps as f32) * sweep;
                Pos2::new(center.x + a.cos() * track_r, center.y + a.sin() * track_r)
            })
            .collect();
        p.add(Shape::line(bg_pts, Stroke::new(2.5, Color32::from_rgb(38, 38, 45))));

        if norm > 0.005 {
            let fill_steps = ((arc_steps as f32 * norm) as usize).max(1);
            let fill_pts: Vec<Pos2> = (0..=fill_steps)
                .map(|i| {
                    let a = start_a + (i as f32 / arc_steps as f32) * sweep;
                    Pos2::new(center.x + a.cos() * track_r, center.y + a.sin() * track_r)
                })
                .collect();
            p.add(Shape::line(fill_pts, Stroke::new(2.5, ACCENT)));
        }

        p.circle_filled(
            Pos2::new(center.x + fill_a.cos() * (radius - 6.0), center.y + fill_a.sin() * (radius - 6.0)),
            2.5,
            Color32::WHITE,
        );

        p.text(Pos2::new(center.x, center.y + radius + 6.0), Align2::CENTER_TOP, fmt(*value), FontId::proportional(9.5), TEXT);
        p.text(Pos2::new(rect.center().x, rect.max.y - 1.0), Align2::CENTER_BOTTOM, name, FontId::proportional(9.0), TEXT_DIM);
    }

    resp
}

fn mode_button(ui: &mut egui::Ui, is_limit: bool) -> Option<bool> {
    let size = Vec2::new(72.0, 76.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());

    let clicked = resp.clicked();

    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        let btn = Rect::from_center_size(rect.center(), Vec2::new(62.0, 34.0));

        let (bg, rim, label_color) = if is_limit {
            (ACCENT, ACCENT.linear_multiply(1.3), Color32::WHITE)
        } else {
            (Color32::from_rgb(40, 40, 48), KNOB_RIM, TEXT_DIM)
        };

        p.rect_filled(btn, CornerRadius::same(5), bg);
        p.rect_stroke(btn, CornerRadius::same(5), Stroke::new(1.0, rim), egui::StrokeKind::Outside);
        p.text(btn.center(), Align2::CENTER_CENTER, if is_limit { "LIMIT" } else { "COMP" }, FontId::proportional(11.0), label_color);
        p.text(Pos2::new(rect.center().x, btn.max.y + 6.0), Align2::CENTER_TOP, "MODE", FontId::proportional(9.0), TEXT_DIM);
    }

    if clicked { Some(!is_limit) } else { None }
}
