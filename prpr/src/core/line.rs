use super::{chart::ChartSettings, object::CtrlObject, Anim, AnimFloat, BpmList, Matrix, Note, Object, Point, RenderConfig, Resource, Tweenable, Vector};
use crate::{
    config::Mods,
    ext::{draw_text_aligned, get_viewport, NotNanExt, SafeTexture},
    judge::{JudgeStatus, LIMIT_BAD},
    ui::Ui,
    info::ChartFormat,
};
use macroquad::prelude::*;
use macroquad::miniquad::{RenderPass, Texture, TextureParams, TextureWrap, FilterMode, MipmapFilter, TextureFormat, PassAction};
use nalgebra::Rotation2;
use serde::Deserialize;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

thread_local! {
    static ENHANCED_NOTES_BUFFER: RefCell<Vec<Vector>> = RefCell::new(Vec::with_capacity(256));
}

struct AnimFloatView<'a> {
    anim: &'a AnimFloat,
    time: f32,
    cursor: usize,
}

impl<'a> AnimFloatView<'a> {
    fn new(anim: &'a AnimFloat) -> Self {
        Self { anim, time: 0.0, cursor: 0 }
    }

    fn set_time(&mut self, time: f32) {
        let keyframes = &self.anim.keyframes;
        if keyframes.is_empty() || time == self.time {
            self.time = time;
            return;
        }
        // Notes are not visited in time order, so the cursor frequently jumps
        // backwards. Walk when we are already close, binary search otherwise,
        // otherwise a line of unsorted notes degrades to O(notes * keyframes).
        let target = keyframes.partition_point(|kf| kf.time <= time).saturating_sub(1);
        if self.cursor.abs_diff(target) > 8 {
            self.cursor = target;
        } else {
            while let Some(kf) = keyframes.get(self.cursor + 1) {
                if kf.time > time {
                    break;
                }
                self.cursor += 1;
            }
            while self.cursor != 0 && keyframes[self.cursor].time > time {
                self.cursor -= 1;
            }
        }
        self.time = time;
    }

    fn now(&self) -> f32 {
        if self.anim.keyframes.is_empty() {
            return 0.0;
        }
        let value = if self.cursor == self.anim.keyframes.len() - 1 {
            self.anim.keyframes[self.cursor].value
        } else {
            let kf1 = &self.anim.keyframes[self.cursor];
            let kf2 = &self.anim.keyframes[self.cursor + 1];
            let t = (self.time - kf1.time) / (kf2.time - kf1.time);
            f32::tween(&kf1.value, &kf2.value, kf1.tween.y(t))
        };
        if self.anim.next.is_some() {
            self.anim.now_opt().unwrap_or(0.0)
        } else {
            value
        }
    }
}

static FLIP_Y_MATRIX: Lazy<Matrix> = Lazy::new(|| {
    Matrix::identity().append_nonuniform_scaling(&Vector::new(1.0, -1.0))
});

static RENDER_CONSTANTS: Lazy<RenderConstants> = Lazy::new(|| RenderConstants {
    duration: 4.03,
    threshold: 0.2,
    inv_threshold: 1.0 / 0.2,
    inv_one_minus_threshold: 1.0 / (1.0 - 0.2),
    line_width_normal: 0.01,
    line_width_loading: 0.0075,
});

struct RenderConstants {
    duration: f32,
    threshold: f32,
    inv_threshold: f32,
    inv_one_minus_threshold: f32,
    line_width_normal: f32,
    line_width_loading: f32,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum UIElement {
    Pause = 1,
    ComboNumber = 2,
    Combo = 3,
    Score = 4,
    Bar = 5,
    Name = 6,
    Level = 7,
}

impl UIElement {
    pub fn from_u8(val: u8) -> Option<Self> {
        Some(match val {
            1 => Self::Bar,
            2 => Self::Pause,
            3 => Self::ComboNumber,
            4 => Self::Combo,
            5 => Self::Score,
            6 => Self::Name,
            7 => Self::Level,
            _ => return None,
        })
    }
}

#[derive(Default)]
pub enum JudgeLineKind {
    #[default]
    Normal,
    Texture(SafeTexture, String),
    Text(Anim<String>),
    Paint(Anim<f32>, Arc<Mutex<(Option<RenderPass>, bool)>>),
    TextureGif(Anim<f32>, GifFrames, String),
}

#[derive(Clone)]
pub struct JudgeLineCache {
    update_order: Vec<u32>,
    not_plain_count: usize,
    above_indices: Vec<usize>,
    below_indices: Vec<usize>,
}

impl JudgeLineCache {
    pub fn new(notes: &mut Vec<Note>) -> Self {
        notes.sort_by_key(|it| (it.plain(), !it.above, it.speed.not_nan(), ((it.height + it.object.translation.1.now()) * it.speed).not_nan()));
        let mut res = Self {
            update_order: Vec::new(),
            not_plain_count: 0,
            above_indices: Vec::new(),
            below_indices: Vec::new(),
        };
        res.reset(notes);
        res
    }

    pub(crate) fn reset(&mut self, notes: &mut Vec<Note>) {
        self.update_order = (0..notes.len() as u32).collect();
        self.above_indices.clear();
        self.below_indices.clear();
        let mut index = notes.iter().position(|it| it.plain()).unwrap_or(notes.len());
        self.not_plain_count = index;
        while notes.get(index).map_or(false, |it| it.above) {
            self.above_indices.push(index);
            let speed = notes[index].speed;
            loop {
                index += 1;
                if !notes.get(index).map_or(false, |it| it.above && it.speed == speed) {
                    break;
                }
            }
        }
        while index != notes.len() {
            self.below_indices.push(index);
            let speed = notes[index].speed;
            loop {
                index += 1;
                if !notes.get(index).map_or(false, |it| it.speed == speed) {
                    break;
                }
            }
        }
    }
}

pub struct GifFrames {
    /// cumulative end time (in milliseconds) of each frame, paired with it
    frames: Vec<(u128, SafeTexture)>,
    /// milliseconds
    total_time: u128,
}

impl GifFrames {
    pub fn new(frames: Vec<(u128, SafeTexture)>) -> Self {
        let mut total_time: u128 = 0;
        let frames = frames
            .into_iter()
            .map(|(duration, texture)| {
                total_time += duration;
                (total_time, texture)
            })
            .collect();
        Self { frames, total_time }
    }

    pub fn get_time_frame(&self, time: u128) -> &SafeTexture {
        let fallback = &self.frames.last().expect("GifFrames has no frames").1;
        if self.total_time == 0 {
            return fallback;
        }
        let time = time % self.total_time;
        // Prefix sums are kept in `frames`, so a linear scan is no longer needed.
        let idx = self.frames.partition_point(|(end, _)| *end <= time);
        &self.frames[idx].1
    }

    pub fn get_prog_frame(&self, prog: f32) -> &SafeTexture {
        let time = (prog * self.total_time as f32) as u128;
        self.get_time_frame(time)
    }

    pub fn total_time(&self) -> u128 {
        self.total_time
    }
}


pub struct JudgeLine {
    pub object: Object,
    pub ctrl_obj: Rc<RefCell<CtrlObject>>,
    pub kind: JudgeLineKind,
    pub height: AnimFloat,
    pub incline: AnimFloat,
    pub notes: Vec<Note>,
    pub color: Anim<Color>,
    pub parent: Option<usize>,
    pub z_index: i32,
    pub show_below: bool,
    pub attach_ui: Option<UIElement>,

    pub cache: JudgeLineCache,
    pub cached_world_pos: Option<Vector>,
}

impl JudgeLine {
    pub fn update(&mut self, res: &mut Resource, tr: Matrix, bpm_list: &BpmList, index: usize) {
        let rot = self.object.rotation.now();
        self.height.set_time(res.time);
        let line_height = self.height.now();
        let mut ctrl_obj = self.ctrl_obj.borrow_mut();
        self.cache.update_order.retain(|id| {
            let note = &mut self.notes[*id as usize];
            note.update(res, rot, &tr, &mut ctrl_obj, line_height, bpm_list, index);
            !note.dead()
        });
        drop(ctrl_obj);
        match &mut self.kind {
            JudgeLineKind::Text(anim) => {
                anim.set_time(res.time);
            }
            JudgeLineKind::Paint(anim, ..) => {
                anim.set_time(res.time);
            }
            JudgeLineKind::TextureGif(anim, ..) => {
                anim.set_time(res.time);
            }
            _ => {}
        }
        self.color.set_time(res.time);
        self.cache.above_indices.retain_mut(|index| {
            while matches!(self.notes[*index].judge, JudgeStatus::Judged) {
                if self
                    .notes
                    .get(*index + 1)
                    .map_or(false, |it| it.above && it.speed == self.notes[*index].speed)
                {
                    *index += 1;
                } else {
                    return false;
                }
            }
            true
        });
        self.cache.below_indices.retain_mut(|index| {
            while matches!(self.notes[*index].judge, JudgeStatus::Judged) {
                if self.notes.get(*index + 1).map_or(false, |it| it.speed == self.notes[*index].speed) {
                    *index += 1;
                } else {
                    return false;
                }
            }
            true
        });
    }

    pub fn update_hand_assign_with_world_pos(&mut self, res: &mut Resource, world_pos: Vector, index: usize) {
        if !res.config.hand_split || self.notes.is_empty() {
            return;
        }

        let config = &res.config;
        let rot = self.object.rotation.now();
        let time = res.time;
        let line_translation = world_pos;

        let chart_ratio_inv = res.chart_ratio_inv;
        let vw = 1.2 * chart_ratio_inv;
        let vh = chart_ratio_inv;

        ENHANCED_NOTES_BUFFER.with(|buffer| {
            let mut positions = buffer.borrow_mut();
            positions.clear();
            positions.reserve(self.notes.len());

            for note in &self.notes {
                let local_x = note.object.translation.0.now();
                let local_y = note.object.translation.1.now();
                positions.push(Vector::new(
                    local_x * vw + line_translation.x,
                    local_y * vh + line_translation.y,
                ));
            }

            crate::hand::assign_hands_unified_perspective(
                &mut self.notes,
                config,
                index,
                rot,
                time,
                &positions,
                || {
                    res.chart_target.as_ref()
                        .map(|t| t.read_pixels_resized(crate::hand::AI_IMAGE_W, crate::hand::AI_IMAGE_H))
                        .unwrap_or_else(|| vec![0.0; crate::hand::AI_IMAGE_SIZE])
                },
            );
        });
    }

    pub fn fetch_pos(line: &JudgeLine, res: &Resource, lines: &[JudgeLine]) -> Vector {
        // Positions computed during `Chart::update` this frame are reused, so a deep
        // parent chain costs O(depth) only on the first hit instead of per line.
        if let Some(parent) = line.parent {
            let parent = &lines[parent];
            let mut parent_translation = match parent.cached_world_pos {
                Some(pos) => pos,
                None => Self::fetch_pos(parent, res, lines),
            };
            parent_translation += Rotation2::new(parent.object.rotation.now().to_radians()) * line.object.now_translation(res);
            return parent_translation;
        }
        match line.cached_world_pos {
            Some(pos) => pos,
            None => line.object.now_translation(res),
        }
    }


    pub fn now_transform(&self, res: &Resource, lines: &[JudgeLine]) -> Matrix {
        self.object.now_rotation().append_translation(&Self::fetch_pos(self, res, lines))
    }

    pub fn now_transform_with_pos(&self, world_pos: Vector) -> Matrix {
        self.object.now_rotation().append_translation(&world_pos)
    }

    pub fn render(&self, ui: &mut Ui, res: &mut Resource, lines: &[JudgeLine], bpm_list: &BpmList, settings: &ChartSettings, id: usize) {
        let alpha = self.object.alpha.now_opt().unwrap_or(1.0) * res.alpha;
        let color = self.color.now_opt();
        let final_alpha = alpha.max(0.0);
        let is_debug = res.config.chart_debug;
        let is_fade_out = res.config.has_mod(Mods::FADE_OUT);
        // Use cached world position if available (set by chart.rs update())
        // to avoid redundant fetch_pos recursive traversal
        let transform = match self.cached_world_pos {
            Some(pos) => self.now_transform_with_pos(pos),
            None => self.now_transform(res, lines),
        };
        res.with_model(transform, |res| {
            res.with_model(self.object.now_scale(), |res| {
                res.apply_model(|res| {
                    match &self.kind {
                        JudgeLineKind::Normal => {
                            if !res.config.ui_line { return; }
                            let mut line_color = color.unwrap_or(res.judge_line_color);
                            line_color.a *= final_alpha;
                            if line_color.a == 0.0 && !is_debug { return; }
                            if is_debug { line_color.a = 0.10 + 0.90 * line_color.a; }
                            let len = res.info.line_length;
                            if res.config.disable_loading {
                                draw_line(-len, 0., len, 0., RENDER_CONSTANTS.line_width_normal, line_color);
                            } else {
                                let t_norm = res.loading_progress;
                                let current_len = len * t_norm;
                                draw_line(-current_len, 0., current_len, 0., RENDER_CONSTANTS.line_width_loading, line_color);
                            }
                        }
                        JudgeLineKind::Texture(texture, _) => {
                            if final_alpha == 0.0 && !is_debug { return; }
                            let mut tex_color = color.unwrap_or(WHITE);
                            if res.time <= 0. && tex_color == WHITE { tex_color = BLACK; }
                            tex_color.a = if is_debug {
                                0.10 + 0.90 * final_alpha
                            } else {
                                final_alpha
                            };
                            // `SafeTexture::get_tex()` is just an `Arc` deref and the
                            // handle is `Copy`, so a lock + hashmap lookup would only
                            // add overhead here.
                            let texture_2d = *texture.get_tex();

                            let hf = vec2(texture_2d.width(), texture_2d.height());
                            draw_texture_ex(
                                texture_2d,
                                -hf.x * 0.5,
                                -hf.y * 0.5,
                                tex_color,
                                DrawTextureParams {
                                    dest_size: Some(hf),
                                    flip_y: true,
                                    ..Default::default()
                                },
                            );
                        }
                        JudgeLineKind::TextureGif(anim, frames, _) => {
                            let t = anim.now_opt().unwrap_or(0.0);
                            let frame = frames.get_prog_frame(t);
                            let mut gif_color = color.unwrap_or(WHITE);
                            gif_color.a = final_alpha;

                            let hf = vec2(frame.width(), frame.height());
                            draw_texture_ex(
                                **frame,
                                -hf.x * 0.5,
                                -hf.y * 0.5,
                                gif_color,
                                DrawTextureParams {
                                    dest_size: Some(hf),
                                    flip_y: true,
                                    ..Default::default()
                                },
                            );
                        }
                        JudgeLineKind::Text(anim) => {
                            let mut base_color = color.unwrap_or(WHITE);
                            if base_color.r == 0.0 && base_color.g == 0.0 && base_color.b == 0.0 && base_color.a == 0.0 {
                                base_color = WHITE;
                            }

                            let mut final_color = base_color;
                            final_color.a = final_alpha;

                            if is_debug {
                                final_color.a = 0.10 + 0.90 * final_color.a;
                            } else if final_color.a == 0.0 {
                                return;
                            }

                            let now = anim.now();
                            res.apply_model_of(&FLIP_Y_MATRIX, |_| {
                                draw_text_aligned(ui, &now, 0., 0., (0.5, 0.5), 1., final_color);
                            });
                        }
                        JudgeLineKind::Paint(anim, state) => {
                            let size = anim.now();
                            if size <= 0.0 || final_alpha == 0.0 {
                                // Nothing visible this frame; the blit below is skipped too.
                                state.lock().unwrap().1 = false;
                            } else {
                                let mut paint_color = color.unwrap_or(WHITE);
                                paint_color.a = final_alpha * 2.55;
                                let vp = get_viewport();
                                let pass = {
                                    let mut guard = state.lock().unwrap();
                                    if guard.0.is_none() {
                                        let gl = unsafe { get_internal_gl() };
                                        let tex = Texture::new_render_texture(
                                            gl.quad_context,
                                            TextureParams {
                                                width: vp.2 as _,
                                                height: vp.3 as _,
                                                format: TextureFormat::RGBA8,
                                                filter: FilterMode::Linear,
                                                wrap: TextureWrap::Clamp,
                                                mipmap_filter: MipmapFilter::None,
                                            },
                                        );
                                        guard.0 = Some(RenderPass::new(gl.quad_context, tex, None));
                                    }
                                    guard.0.unwrap()
                                };
                                let (old_pass, old_vp) = {
                                    let gl = unsafe { get_internal_gl() };
                                    // Clear the offscreen target through a raw pass:
                                    // `clear_background` would additionally throw away every
                                    // vertex QuadGl has batched for this frame.
                                    gl.quad_context.begin_pass(pass, PassAction::clear_color(0., 0., 0., 0.));
                                    gl.quad_context.end_render_pass();
                                    let old_pass = gl.quad_gl.get_active_render_pass();
                                    let old_vp = gl.quad_gl.get_viewport();
                                    gl.quad_gl.render_pass(Some(pass));
                                    gl.quad_gl.viewport(None);
                                    (old_pass, old_vp)
                                };
                                ui.fill_circle(0., 0., size / vp.2 as f32 * 2., paint_color);
                                {
                                    let gl = unsafe { get_internal_gl() };
                                    gl.quad_gl.render_pass(old_pass);
                                    gl.quad_gl.viewport(old_vp);
                                }
                                state.lock().unwrap().1 = true;
                            }
                        }
                    }
                })
            });
            if let JudgeLineKind::Paint(_, state) = &self.kind {
                let guard = state.lock().unwrap();
                let ready = guard.1;
                let tex = ready
                    .then(|| guard.0.as_ref().map(|it| it.texture(unsafe { get_internal_gl() }.quad_context)))
                    .flatten();
                drop(guard);
                if let Some(tex) = tex {
                    let top = res.inv_aspect_ratio;
                    draw_texture_ex(
                        Texture2D::from_miniquad_texture(tex),
                        -1.,
                        -top,
                        WHITE,
                        DrawTextureParams {
                            dest_size: Some(vec2(2., top * 2.)),
                            ..Default::default()
                        },
                    );
                }
            }

            let mut ctrl_obj = self.ctrl_obj.borrow_mut();
            let line_height = self.height.now();

            let mut config = RenderConfig {
                settings,
                ctrl_obj: &mut *ctrl_obj,
                line_height,
                appear_before: f32::INFINITY,
                invisible_time: f32::INFINITY,
                draw_below: self.show_below,
                incline_sin: self.incline.now_opt().map(|it| it.to_radians().sin()).unwrap_or_default(),
                global_speed_factor: res.config.note_speed_factor,
            };
            if is_fade_out { config.invisible_time = LIMIT_BAD; }
            if alpha < 0.0 && settings.pe_alpha_extension {
                let w = (-alpha).floor() as u32;
                match w {
                    1 => return,
                    2 => config.draw_below = false,
                    100..=999 => config.appear_before = (w as f32 - 100.) * 0.1,
                    1000..=1999 => config.invisible_time = (w as f32 - 1000.) * 0.1,
                    _ => {}
                }
            } else if alpha < 0.0 {
                return;
            }

            let chart_ratio_inv = res.chart_ratio_inv; // 使用缓存的值
            let vw = 1.2 * chart_ratio_inv;
            let vh = chart_ratio_inv;

            let inv_aspect_ratio = res.inv_aspect_ratio; // 使用缓存的值
            let agg = res.config.aggressive;
            let chart_format_matches = matches!(res.chart_format, ChartFormat::Pgr | ChartFormat::Rpe);
            let note_scale_positive = res.config.note_scale > 0.;

            if !note_scale_positive { return; }

            // Viewport corners expressed in *this* judge line's local space, because
            // the aggressive culling below compares them against line-local note
            // heights. The corners therefore depend on this line's transform and must
            // not be cached globally across judge lines (the old cache was keyed only
            // by viewport, so every line except the first reused another line's bounds).
            // Only the aggressive path reads them, so skip the matrix inverse entirely
            // when aggressive culling is off.
            let (height_above, height_below) = if agg {
                let inv = (res.model() * transform)
                    .try_inverse()
                    .unwrap_or_else(Matrix::identity);
                let viewport_points = [
                    inv.transform_point(&Point::new(-vw, -vh)),
                    inv.transform_point(&Point::new(-vw, vh)),
                    inv.transform_point(&Point::new(vw, -vh)),
                    inv.transform_point(&Point::new(vw, vh)),
                ];
                let above = viewport_points.iter()
                    .map(|p| p.y)
                    .fold(f32::NEG_INFINITY, f32::max) * (1.0 / inv_aspect_ratio);
                let below = viewport_points.iter()
                    .map(|p| p.y)
                    .fold(f32::INFINITY, f32::min) * (1.0 / inv_aspect_ratio);
                (above, below)
            } else {
                (0.0, 0.0)
            };

            // Use a lightweight view that borrows keyframes instead of cloning the entire AnimFloat
            let mut height = AnimFloatView::new(&self.height);
            let not_plain_count = self.cache.not_plain_count;

            for note in self.notes[..not_plain_count].iter().filter(|n| n.above) {
                if agg && chart_format_matches {
                    height.set_time(note.time.min(res.time));
                    let line_height_at_note = height.now();
                    let note_height = note.height - line_height_at_note + note.object.translation.1.now();
                    let inv_speed = 1.0 / note.speed;
                    if note_height < height_below * inv_speed { continue; }
                    if note_height > height_above * inv_speed { break; }
                }
                note.render(res, &mut config, bpm_list);
            }

            for &index in &self.cache.above_indices {
                let speed = self.notes[index].speed;
                let inv_speed = 1.0 / speed;
                let height_below_scaled = height_below * inv_speed;
                let height_above_scaled = height_above * inv_speed;

                for note in &self.notes[index..] {
                    if !note.above || speed != note.speed { break; }
                    if agg {
                        let note_height = note.height - config.line_height + note.object.translation.1.now();
                        if note_height < height_below_scaled { continue; }
                        if note_height > height_above_scaled { break; }
                    }
                    note.render(res, &mut config, bpm_list);
                }
            }

            res.with_model(*FLIP_Y_MATRIX, |res| {
                for note in self.notes[..not_plain_count].iter().filter(|n| !n.above) {
                    if agg && chart_format_matches {
                        height.set_time(note.time.min(res.time));
                        let line_height_at_note = height.now();
                        let note_height = note.height - line_height_at_note + note.object.translation.1.now();
                        let inv_speed = 1.0 / note.speed;
                        if note_height < -height_above * inv_speed {
                            continue;
                        }
                        if note_height > -height_below * inv_speed {
                            break;
                        }
                    }

                    note.render(res, &mut config, bpm_list);
                }

                // 渲染下方音符（索引）
                for &index in &self.cache.below_indices {
                    let speed = self.notes[index].speed;
                    let inv_speed = 1.0 / speed;
                    let neg_height_above_scaled = -height_above * inv_speed;
                    let neg_height_below_scaled = -height_below * inv_speed;

                    for note in &self.notes[index..] {
                        if speed != note.speed {
                            break;
                        }

                        if agg {
                            let note_height = note.height - config.line_height + note.object.translation.1.now();
                            if note_height < neg_height_above_scaled {
                                continue;
                            }
                            if note_height > neg_height_below_scaled {
                                break;
                            }
                        }

                        note.render(res, &mut config, bpm_list);
                    }
                }
            });
            if res.config.chart_debug {
                let pos = self.cached_world_pos.unwrap_or_else(|| Self::fetch_pos(self, res, lines));
                let rotation = self.object.rotation.now();
                let text_alpha = {
                    let mut alpha_val = alpha;
                    if res.config.chart_debug {
                        alpha_val = 0.10 + 0.90 * alpha_val;
                    }
                    alpha_val.max(0.4)
                };

                let text_color = Color::new(1.0, 1.0, 1.0, text_alpha);
                let judged_count = self.notes.iter().filter(|n| matches!(n.judge, JudgeStatus::Judged)).count();
                let total_notes = self.notes.len();

                let mut parent_info = String::new();
                let mut current_parent = self.parent;
                let mut valid_parents = Vec::new();

                while let Some(parent_index) = current_parent {
                    if parent_index < lines.len() {
                        valid_parents.push(parent_index);
                        current_parent = lines[parent_index].parent;
                    } else {
                        break;
                    }
                }

                if !valid_parents.is_empty() {
                    parent_info = "Parents: ".to_string();
                    parent_info += &valid_parents
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(" -> ");
                }
                res.with_model(*FLIP_Y_MATRIX, |res| {
                    res.apply_model(|_| {
                        ui.text(id.to_string()).pos(0., -0.01).anchor(0.5, 1.).color(text_color).size(0.5).draw();
                        let state_str = format!(
                            "P({:.3},{:.3})   R{:.1}°  N{}/{}   {}",
                            pos.x, pos.y,
                            rotation,
                            judged_count,
                            total_notes,
                            parent_info
                        );

                        ui.text(&state_str)
                            .pos(0., -0.05)
                            .anchor(0.5, 1.)
                            .size(0.35)
                            .color(text_color)
                            .draw();
                    });
                });
            }
        });
    }
}