use super::{chart::ChartSettings, object::CtrlObject, Anim, AnimFloat, BpmList, Matrix, Note, Object, Point, RenderConfig, Resource, Tweenable, Vector};
use crate::{
    config::Mods,
    ext::{get_viewport, NotNanExt, SafeTexture},
    judge::{JudgeStatus, LIMIT_BAD},
    ui::Ui,
};
use macroquad::prelude::*;
use macroquad::miniquad::{RenderPass, Texture, TextureParams, TextureWrap, FilterMode, MipmapFilter, TextureFormat, PassAction};
use nalgebra::Rotation2;
use serde::Deserialize;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use super::{note::{NoteRenderContext, RenderFrame}, resource::NoteBuffer};
use crate::ui::{FontArc, TextRenderBuffer};
use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};

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

pub struct JudgeLineCache {
    state: NoteUpdateState,
    render: RefCell<(LineRenderBuffer, LineRenderGpu)>,
}

/// CPU-only note state can be borrowed by a worker without the line's GPU
/// cache or shared control curves.
#[derive(Clone, Default, Debug, PartialEq)]
struct NoteUpdateState {
    update_order: Vec<u32>,
    effect_order: Vec<u32>,
    not_plain_count: usize,
    above_indices: Vec<usize>,
    below_indices: Vec<usize>,
    render_order: Vec<usize>,
}

impl Clone for JudgeLineCache {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            render: RefCell::default(),
        }
    }
}

impl JudgeLineCache {
    pub fn new(notes: &mut Vec<Note>) -> Self {
        notes.sort_by_key(|it| (it.plain(), !it.above, it.speed.not_nan(), ((it.height + it.object.translation.1.now()) * it.speed).not_nan()));
        let mut res = Self {
            state: NoteUpdateState::default(),
            render: RefCell::default(),
        };
        res.reset(notes);
        res
    }

    pub(crate) fn reset(&mut self, notes: &mut Vec<Note>) {
        self.state.reset(notes);
    }
}

impl NoteUpdateState {
    fn reset(&mut self, notes: &[Note]) {
        self.update_order = (0..notes.len() as u32).collect();
        self.effect_order.clear();
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
        self.rebuild_render_order(notes);
    }

    fn update(&mut self, notes: &mut [Note], time: f32, parallel: bool) {
        if parallel && self.update_order.len() >= 1024 {
            // IDs remain sorted after retain. Disjoint spans let Rayon update
            // large lines without sharing mutable notes between workers.
            let mut tasks = Vec::with_capacity(self.update_order.len().div_ceil(128));
            let mut remaining = &mut *notes;
            let mut offset = 0;
            for ids in self.update_order.chunks(128) {
                let end = *ids.last().unwrap() as usize + 1;
                let (span, rest) = remaining.split_at_mut(end - offset);
                tasks.push((span, ids, offset));
                remaining = rest;
                offset = end;
            }
            tasks.par_iter_mut().with_max_len(1).for_each(|(span, ids, offset)| {
                for &id in *ids {
                    span[id as usize - *offset].object.set_time(time);
                }
            });
        } else {
            for &id in &self.update_order {
                notes[id as usize].object.set_time(time);
            }
        }

        self.effect_order.clear();
        self.update_order.retain(|&id| {
            let note = &notes[id as usize];
            if note.has_pending_effect(time) {
                self.effect_order.push(id);
            }
            !note.dead()
        });
        self.advance_render_order(notes);
    }

    fn advance_render_order(&mut self, notes: &[Note]) {
        let mut changed = false;
        self.above_indices.retain_mut(|index| {
            while matches!(notes[*index].judge, JudgeStatus::Judged) {
                changed = true;
                if notes.get(*index + 1).map_or(false, |it| it.above && it.speed == notes[*index].speed) {
                    *index += 1;
                } else {
                    return false;
                }
            }
            true
        });
        self.below_indices.retain_mut(|index| {
            while matches!(notes[*index].judge, JudgeStatus::Judged) {
                changed = true;
                if notes.get(*index + 1).map_or(false, |it| it.speed == notes[*index].speed) {
                    *index += 1;
                } else {
                    return false;
                }
            }
            true
        });
        if changed {
            self.rebuild_render_order(notes);
        }
    }

    fn rebuild_render_order(&mut self, notes: &[Note]) {
        self.render_order.clear();
        for above in [true, false] {
            self.render_order.extend(notes[..self.not_plain_count].iter().enumerate().filter_map(|(id, note)| (note.above == above).then_some(id)));
            let indices = if above { &self.above_indices } else { &self.below_indices };
            for &start in indices {
                let speed = notes[start].speed;
                for (offset, note) in notes[start..].iter().enumerate() {
                    if (above && !note.above) || note.speed != speed { break; }
                    self.render_order.push(start + offset);
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

#[derive(Clone, Copy)]
pub(crate) struct LineLocalState {
    pub translation: Vector,
    pub rotation: Rotation2<f32>,
}

impl Default for LineLocalState {
    fn default() -> Self {
        Self { translation: Vector::zeros(), rotation: Rotation2::identity() }
    }
}

enum LineAnimation<'a> {
    None,
    Text(&'a mut Anim<String>),
    Float(&'a mut AnimFloat),
}

pub(crate) struct LineUpdateTask<'a> {
    object: &'a mut Object,
    height: &'a mut AnimFloat,
    color: &'a mut Anim<Color>,
    animation: LineAnimation<'a>,
    notes: &'a mut [Note],
    state: &'a mut NoteUpdateState,
}

impl LineUpdateTask<'_> {
    fn update_state(&mut self, time: f32, parallel: bool) {
        self.height.set_time(time);
        self.state.update(self.notes, time, parallel);
        match &mut self.animation {
            LineAnimation::Text(anim) => anim.set_time(time),
            LineAnimation::Float(anim) => anim.set_time(time),
            LineAnimation::None => {}
        }
        self.color.set_time(time);
    }

    pub(crate) fn prepare(&mut self, time: f32, aspect_ratio: f32, parallel: bool) -> LineLocalState {
        self.object.set_time(time);
        self.update_state(time, parallel);
        let mut translation = self.object.translation.now();
        translation.y /= aspect_ratio;
        LineLocalState {
            translation,
            rotation: Rotation2::new(self.object.rotation.now().to_radians()),
        }
    }
}

impl JudgeLine {
    pub(crate) fn pending_updates(&self) -> usize {
        self.cache.state.update_order.len()
    }

    pub fn update(&mut self, res: &mut Resource, tr: Matrix, bpm_list: &BpmList, index: usize) {
        self.update_task().update_state(res.time, false);
        self.finish_update(res, tr, bpm_list, index);
    }

    pub(crate) fn update_task(&mut self) -> LineUpdateTask<'_> {
        let animation = match &mut self.kind {
            JudgeLineKind::Text(anim) => LineAnimation::Text(anim),
            JudgeLineKind::Paint(anim, ..) | JudgeLineKind::TextureGif(anim, ..) => LineAnimation::Float(anim),
            _ => LineAnimation::None,
        };
        LineUpdateTask {
            object: &mut self.object,
            height: &mut self.height,
            color: &mut self.color,
            animation,
            notes: &mut self.notes,
            state: &mut self.cache.state,
        }
    }

    /// Only Resource-dependent effects run here. Tasks retain their original
    /// line/note order, including control-curve updates and particle RNG calls.
    pub(crate) fn finish_update(&mut self, res: &mut Resource, tr: Matrix, bpm_list: &BpmList, index: usize) {
        if self.cache.state.effect_order.is_empty() {
            return;
        }
        let rot = self.object.rotation.now();
        let line_height = self.height.now();
        let mut ctrl_obj = self.ctrl_obj.borrow_mut();
        for &id in &self.cache.state.effect_order {
            self.notes[id as usize].update_effects(res, rot, &tr, &mut ctrl_obj, line_height, bpm_list, index);
        }
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
        if let Some(pos) = line.cached_world_pos {
            return pos;
        }
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

    pub(crate) fn render_task<'a>(&'a self, res: &Resource, lines: &[JudgeLine], ctrl: &'a CtrlObject) -> LineRenderTask<'a> {
        let transform = self.cached_world_pos.map_or_else(|| self.now_transform(res, lines), |pos| self.now_transform_with_pos(pos));
        let kind = match &self.kind {
            JudgeLineKind::Normal => LineRenderKind::Normal,
            JudgeLineKind::Texture(texture, _) => LineRenderKind::Texture(**texture),
            JudgeLineKind::TextureGif(anim, frames, _) => LineRenderKind::TextureGif(anim, frames),
            JudgeLineKind::Text(anim) => LineRenderKind::Text(anim),
            JudgeLineKind::Paint(anim, _) => LineRenderKind::Paint(anim),
        };
        LineRenderTask {
            object: &self.object,
            height: &self.height,
            incline: &self.incline,
            color: &self.color,
            kind,
            ctrl,
            notes: &self.notes,
            cache: &self.cache.state,
            show_below: self.show_below,
            transform: res.model() * transform,
            culling_transform: if res.config.aggressive { res.model() * transform * transform } else { Matrix::identity() },
        }
    }

    pub fn render(&self, ui: &mut Ui, res: &mut Resource, lines: &[JudgeLine], bpm_list: &BpmList, settings: &ChartSettings, id: usize) {
        let mut state = self.cache.render.borrow_mut();
        let (buffer, gpu) = &mut *state;
        let frame = super::note::RenderFrame::new(res);
        let ctrl = self.ctrl_obj.borrow();
        let task = self.render_task(res, lines, &ctrl);
        let vp = get_viewport();
        let ui_model = ui.get_matrix();
        let font = ui.text_painter.font();
        let limit = ui.text_painter.texture_limit();
        let parallel = super::chart::RENDER_POOL.current_num_threads() > 1;
        let prepare = || task.prepare(buffer, &frame, bpm_list, settings, vp, ui_model, &font, limit, parallel);
        super::chart::RENDER_POOL.install(prepare);
        drop(task);
        drop(ctrl);
        drop(frame);
        self.render_prepared(ui, res, lines, buffer, gpu, id);
    }

    pub(crate) fn render_prepared(&self, ui: &mut Ui, res: &mut Resource, lines: &[JudgeLine], buffer: &LineRenderBuffer, gpu: &mut LineRenderGpu, id: usize) {
        let mut texture = buffer.texture;
        if buffer.text_visible {
            let text = buffer.text.as_ref().unwrap();
            if gpu.dimensions != text.dimensions || gpu.texture.is_none() || !text.updates.is_empty() {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                if gpu.dimensions != text.dimensions || gpu.texture.is_none() {
                    gpu.texture = Some(Texture2D::from_miniquad_texture(Texture::new_render_texture(gl.quad_context, TextureParams {
                        width: text.dimensions.0,
                        height: text.dimensions.1,
                        filter: FilterMode::Linear,
                        wrap: TextureWrap::Clamp,
                        format: TextureFormat::RGBA8,
                        mipmap_filter: MipmapFilter::None,
                    })).into());
                    gpu.dimensions = text.dimensions;
                }
                let handle = gpu.texture.as_ref().unwrap().raw_miniquad_texture_handle();
                for (x, y, w, h, data) in &text.updates {
                    handle.update_texture_part(gl.quad_context, *x as _, *y as _, *w as _, *h as _, data);
                }
            }
            texture = gpu.texture.as_ref().map(|it| **it);
        }
        if let JudgeLineKind::Paint(_, state) = &self.kind {
            let mut guard = state.lock().unwrap();
            guard.1 = !buffer.vertices.is_empty();
            if guard.1 {
                let mut gl = unsafe { get_internal_gl() };
                gl.flush();
                let vp = get_viewport();
                if guard.0.is_none() || guard.0.map_or(false, |pass| {
                    let tex = pass.texture(gl.quad_context);
                    tex.width != vp.2 as u32 || tex.height != vp.3 as u32
                }) {
                    let tex = Texture::new_render_texture(gl.quad_context, TextureParams {
                        width: vp.2 as _, height: vp.3 as _, format: TextureFormat::RGBA8,
                        filter: FilterMode::Linear, wrap: TextureWrap::Clamp, mipmap_filter: MipmapFilter::None,
                    });
                    let pass = RenderPass::new(gl.quad_context, tex, None);
                    if let Some(old_pass) = guard.0.replace(pass) {
                        old_pass.delete(gl.quad_context);
                    }
                }
                let pass = guard.0.unwrap();
                gl.quad_context.begin_pass(pass, PassAction::clear_color(0., 0., 0., 0.));
                gl.quad_context.end_render_pass();
                let old_pass = gl.quad_gl.get_active_render_pass();
                let old_vp = gl.quad_gl.get_viewport();
                gl.quad_gl.render_pass(Some(pass));
                gl.quad_gl.viewport(None);
                gl.quad_gl.texture(None);
                gl.quad_gl.draw_mode(DrawMode::Triangles);
                gl.quad_gl.geometry(&buffer.vertices, &buffer.indices);
                gl.quad_gl.render_pass(old_pass);
                gl.quad_gl.viewport(old_vp);
                let tex = pass.texture(gl.quad_context);
                let top = res.inv_aspect_ratio;
                draw_texture_ex(Texture2D::from_miniquad_texture(tex), -1., -top, WHITE, DrawTextureParams { dest_size: Some(vec2(2., top * 2.)), ..Default::default() });
            }
        } else if buffer.text_visible {
            crate::ui::draw_text_vertices(texture, &buffer.vertices);
        } else if !buffer.vertices.is_empty() {
            let gl = unsafe { get_internal_gl() }.quad_gl;
            gl.texture(texture);
            gl.draw_mode(DrawMode::Triangles);
            gl.geometry(&buffer.vertices, &buffer.indices);
        }
        for notes in &buffer.note_buffers[..buffer.note_buffer_count] {
            res.note_buffer.borrow_mut().append(&notes.buffer.borrow());
        }
        if buffer.note_buffer_count != 0 && !self.ctrl_obj.borrow().y.is_default() {
            *self.ctrl_obj.borrow_mut() = buffer.note_buffers[0].ctrl.clone();
        }
        if buffer.draw_notes && res.config.chart_debug {
            let alpha = self.object.alpha.now_opt().unwrap_or(1.0) * res.alpha;
            let transform = self.cached_world_pos.map_or_else(|| self.now_transform(res, lines), |pos| self.now_transform_with_pos(pos));
            res.with_model(transform, |res| {
                let pos = self.cached_world_pos.unwrap_or_else(|| Self::fetch_pos(self, res, lines));
                let rotation = self.object.rotation.now();
                let text_alpha = (0.10 + 0.90 * alpha).max(0.4);
                let text_color = Color::new(1.0, 1.0, 1.0, text_alpha);
                let judged_count = self.notes.iter().filter(|note| matches!(note.judge, JudgeStatus::Judged)).count();
                let mut parents = Vec::new();
                let mut parent = self.parent;
                while let Some(index) = parent {
                    if index >= lines.len() || parents.contains(&index) { break; }
                    parents.push(index);
                    parent = lines[index].parent;
                }
                let parent_info = if parents.is_empty() { String::new() } else {
                    format!("Parents: {}", parents.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" -> "))
                };
                res.with_model(*FLIP_Y_MATRIX, |res| {
                    res.apply_model(|_| {
                        ui.text(id.to_string()).pos(0., -0.01).anchor(0.5, 1.).color(text_color).size(0.5).draw();
                        let state = format!("P({:.3},{:.3})   R{:.1}°  N{}/{}   {}", pos.x, pos.y, rotation, judged_count, self.notes.len(), parent_info);
                        ui.text(state).pos(0., -0.05).anchor(0.5, 1.).size(0.35).color(text_color).draw();
                    });
                });
            });
        }
    }
}

enum LineRenderKind<'a> {
    Normal,
    Texture(Texture2D),
    TextureGif(&'a AnimFloat, &'a GifFrames),
    Text(&'a Anim<String>),
    Paint(&'a AnimFloat),
}

pub(crate) struct LineRenderTask<'a> {
    object: &'a Object,
    height: &'a AnimFloat,
    incline: &'a AnimFloat,
    color: &'a Anim<Color>,
    kind: LineRenderKind<'a>,
    ctrl: &'a CtrlObject,
    notes: &'a [Note],
    cache: &'a NoteUpdateState,
    show_below: bool,
    transform: Matrix,
    culling_transform: Matrix,
}

struct NoteTaskBuffer {
    buffer: RefCell<NoteBuffer>,
    ctrl: CtrlObject,
}

#[derive(Default)]
pub(crate) struct LineRenderBuffer {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    texture: Option<Texture2D>,
    text: Option<TextRenderBuffer>,
    text_visible: bool,
    draw_notes: bool,
    candidates: Vec<usize>,
    note_buffers: Vec<NoteTaskBuffer>,
    note_buffer_count: usize,
    paint: VertexBuffers<Vertex, u16>,
    fill_tess: FillTessellator,
}

#[derive(Default)]
pub(crate) struct LineRenderGpu {
    texture: Option<SafeTexture>,
    dimensions: (u32, u32),
}

impl LineRenderBuffer {
    fn quad(&mut self, model: &Matrix, points: [Point; 4], uv: [(f32, f32); 4], color: Color, indices: [u16; 6]) {
        self.vertices.extend(points.into_iter().zip(uv).map(|(pt, (u, v))| {
            let pt = model.transform_point(&pt);
            Vertex::new(pt.x, pt.y, 0., u, v, color)
        }));
        self.indices.extend_from_slice(&indices);
    }

    fn texture(&mut self, model: &Matrix, texture: Texture2D, color: Color) {
        let w = texture.width() * 0.5;
        let h = texture.height() * 0.5;
        self.texture = Some(texture);
        self.quad(model, [Point::new(-w, h), Point::new(w, h), Point::new(w, -h), Point::new(-w, -h)], [(0., 0.), (1., 0.), (1., 1.), (0., 1.)], color, [0, 1, 2, 0, 2, 3]);
    }
}

impl LineRenderTask<'_> {
    pub fn prepare(&self, buffer: &mut LineRenderBuffer, res: &RenderFrame, bpm_list: &BpmList, settings: &ChartSettings, vp: (i32, i32, i32, i32), ui_model: Matrix, font: &FontArc, texture_limit: u32, parallel: bool) {
        buffer.vertices.clear();
        buffer.indices.clear();
        buffer.texture = None;
        buffer.text_visible = false;
        buffer.draw_notes = false;
        buffer.candidates.clear();
        buffer.note_buffer_count = 0;
        let alpha = self.object.alpha.now_opt().unwrap_or(1.0) * res.alpha;
        let final_alpha = alpha.max(0.0);
        let color = self.color.now_opt();
        let is_debug = res.config.chart_debug;
        let model = self.transform * self.object.now_scale();
        match self.kind {
            LineRenderKind::Normal => {
                let mut color = color.unwrap_or(res.judge_line_color);
                color.a *= final_alpha;
                if res.config.ui_line && (color.a != 0.0 || is_debug) {
                    if is_debug { color.a = 0.10 + 0.90 * color.a; }
                    let (len, width) = if res.config.disable_loading {
                        (res.line_length, RENDER_CONSTANTS.line_width_normal)
                    } else {
                        (res.line_length * res.loading_progress, RENDER_CONSTANTS.line_width_loading)
                    };
                    if len.abs() > f32::EPSILON {
                        let h = width * 0.5 * len.signum();
                        buffer.quad(&model, [Point::new(-len, h), Point::new(-len, -h), Point::new(len, h), Point::new(len, -h)], [(0., 0.); 4], color, [0, 1, 2, 2, 1, 3]);
                    }
                }
            }
            LineRenderKind::Texture(texture) => {
                if final_alpha != 0.0 || is_debug {
                    let mut color = color.unwrap_or(WHITE);
                    if res.time <= 0.0 && color == WHITE { color = BLACK; }
                    color.a = if is_debug { 0.10 + 0.90 * final_alpha } else { final_alpha };
                    buffer.texture(&model, texture, color);
                }
            }
            LineRenderKind::TextureGif(anim, frames) => {
                let texture = **frames.get_prog_frame(anim.now_opt().unwrap_or(0.0));
                let mut color = color.unwrap_or(WHITE);
                color.a = final_alpha;
                buffer.texture(&model, texture, color);
            }
            LineRenderKind::Text(anim) => {
                let mut color = color.unwrap_or(WHITE);
                if color == Color::new(0., 0., 0., 0.) { color = WHITE; }
                color.a = if is_debug { 0.10 + 0.90 * final_alpha } else { final_alpha };
                if color.a != 0.0 {
                    let text = buffer.text.get_or_insert_with(|| TextRenderBuffer::new(font.clone(), texture_limit));
                    if !text.matches_font(font) {
                        *text = TextRenderBuffer::new(font.clone(), texture_limit);
                    }
                    text.prepare(&anim.now(), vp.2, texture_limit);
                    let model = model * *FLIP_Y_MATRIX * ui_model * Matrix::new_translation(&text.offset) * Matrix::new_scaling(text.pixel_scale);
                    for quad in text.vertices.chunks_exact(4) {
                        buffer.vertices.extend(quad.iter().map(|vertex| {
                            let (pos, uv, _): ([f32; 3], [f32; 2], [f32; 4]) = (*vertex).into();
                            let pt = model.transform_point(&Point::new(pos[0], pos[1]));
                            Vertex::new(pt.x, pt.y, 0., uv[0], uv[1], color)
                        }));
                    }
                    buffer.text_visible = true;
                }
            }
            LineRenderKind::Paint(anim) => {
                let size = anim.now();
                if size > 0.0 && final_alpha != 0.0 {
                    let mut color = color.unwrap_or(WHITE);
                    color.a = final_alpha * 2.55;
                    let model = model * ui_model;
                    buffer.paint.vertices.clear();
                    buffer.paint.indices.clear();
                    let tolerance = 0.3 / (ui_model.transform_vector(&Vector::new(1., 0.)).norm() * vp.2 as f32);
                    buffer.fill_tess.tessellate_circle(lyon::math::point(0., 0.), size / vp.2 as f32 * 2., &FillOptions::default().with_tolerance(tolerance), &mut BuffersBuilder::new(&mut buffer.paint, |vertex: FillVertex| {
                        let pos = vertex.position();
                        let pt = model.transform_point(&Point::new(pos.x, pos.y));
                        Vertex::new(pt.x, pt.y, 0., 0., 0., color)
                    })).unwrap();
                    buffer.vertices.extend_from_slice(&buffer.paint.vertices);
                    buffer.indices.extend_from_slice(&buffer.paint.indices);
                }
            }
        }
        let mut draw_below = self.show_below;
        let mut appear_before = f32::INFINITY;
        let mut invisible_time = if res.config.has_mod(Mods::FADE_OUT) { LIMIT_BAD } else { f32::INFINITY };
        if alpha < 0.0 {
            if !settings.pe_alpha_extension { return; }
            match (-alpha).floor() as u32 {
                1 => return,
                2 => draw_below = false,
                100..=999 => appear_before = ((-alpha).floor() - 100.) * 0.1,
                1000..=1999 => invisible_time = ((-alpha).floor() - 1000.) * 0.1,
                _ => {}
            }
        }
        if res.config.note_scale <= 0.0 { return; }
        buffer.draw_notes = true;
        let line_height = self.height.now();
        let agg = res.config.aggressive;
        let (height_above, height_below) = if agg {
            let inv = self.culling_transform.try_inverse().unwrap_or_else(Matrix::identity);
            let vw = 1.2 * res.chart_ratio_inv;
            let vh = res.chart_ratio_inv;
            let points = [Point::new(-vw, -vh), Point::new(-vw, vh), Point::new(vw, -vh), Point::new(vw, vh)].map(|pt| inv.transform_point(&pt));
            (points.iter().map(|pt| pt.y).fold(f32::NEG_INFINITY, f32::max) * res.aspect_ratio, points.iter().map(|pt| pt.y).fold(f32::INFINITY, f32::min) * res.aspect_ratio)
        } else { (0., 0.) };
        let mut height = AnimFloatView::new(self.height);
        let not_plain_count = self.cache.not_plain_count;
        if agg {
            for above in [true, false] {
                for (index, note) in self.notes[..not_plain_count].iter().enumerate().filter(|(_, note)| note.above == above) {
                    if agg && res.chart_format_matches {
                        height.set_time(note.time.min(res.time));
                        let note_height = note.height - height.now() + note.object.translation.1.now();
                        let (low, high) = if above { (height_below, height_above) } else { (-height_above, -height_below) };
                        if note_height < low / note.speed { continue; }
                        if note_height > high / note.speed { break; }
                    }
                    buffer.candidates.push(index);
                }
                let indices = if above { &self.cache.above_indices } else { &self.cache.below_indices };
                for &start in indices {
                    let speed = self.notes[start].speed;
                    let (low, high) = if above { (height_below / speed, height_above / speed) } else { (-height_above / speed, -height_below / speed) };
                    for (offset, note) in self.notes[start..].iter().enumerate() {
                        if (above && !note.above) || note.speed != speed { break; }
                        if agg {
                            let note_height = note.height - line_height + note.object.translation.1.now();
                            if note_height < low { continue; }
                            if note_height > high { break; }
                        }
                        buffer.candidates.push(start + offset);
                    }
                }
            }
        }
        // The default order is already cached; only aggressive culling needs
        // a frame-local candidate list.
        let candidates = if agg { buffer.candidates.as_slice() } else { &self.cache.render_order };
        let parallel_notes = parallel && self.ctrl.y.is_default();
        let chunk_size = if parallel_notes && candidates.len() >= 512 { *super::chart::RENDER_CHUNK_SIZE } else { candidates.len().max(1) };
        let count = candidates.len().div_ceil(chunk_size);
        buffer.note_buffer_count = count;
        while buffer.note_buffers.len() < count {
            buffer.note_buffers.push(NoteTaskBuffer { buffer: RefCell::new(NoteBuffer::default()), ctrl: self.ctrl.clone() });
        }
        if !self.ctrl.y.is_default() && count != 0 {
            buffer.note_buffers[0].ctrl = self.ctrl.clone();
        }
        let below_transform = self.transform * *FLIP_Y_MATRIX;
        let incline_sin = self.incline.now_opt().map(|it| it.to_radians().sin()).unwrap_or_default();
        let render = |(notes, candidates): (&mut NoteTaskBuffer, &[usize])| {
            notes.buffer.get_mut().clear();
            let mut config = RenderConfig {
                settings, ctrl_obj: &mut notes.ctrl, line_height, appear_before, invisible_time, draw_below,
                incline_sin,
                global_speed_factor: res.config.note_speed_factor,
            };
            for &index in candidates {
                let note = &self.notes[index];
                let model = if note.above { self.transform } else { below_transform };
                note.render_prepared(&NoteRenderContext { frame: res, model, note_buffer: &notes.buffer }, &mut config, bpm_list);
            }
        };
        if parallel_notes && count > 1 {
            buffer.note_buffers[..count].par_iter_mut().zip(candidates.par_chunks(chunk_size)).with_max_len(1).for_each(render);
        } else {
            buffer.note_buffers[..count].iter_mut().zip(candidates.chunks(chunk_size)).for_each(render);
        }
    }
}

#[cfg(test)]
mod update_tests {
    use super::*;
    use crate::core::{note::Hand, AnimVector, Keyframe, NoteKind};

    fn note(id: usize) -> Note {
        let hold = id % 5 == 0;
        Note {
            time: id as f32 * 0.01,
            kind: if hold { NoteKind::Hold { end_time: 8.0, end_height: 4.0 } } else { NoteKind::Click },
            height: id as f32 * 0.02,
            object: Object {
                alpha: if id % 7 == 0 { Anim::fixed(1.0) } else {
                    Anim::new(vec![Keyframe::new(0.0, 1.0, 2), Keyframe::new(4.0, 0.0, 2)])
                },
                translation: AnimVector::fixed(Vector::new(id as f32 * 0.001, 0.0)),
                ..Object::default()
            },
            speed: (id % 3 + 1) as f32,
            end_speed: 1.0,
            start_height: 0.0,
            hand: Hand::Left,
            above: id % 2 == 0,
            multiple_hint: false,
            fake: id % 11 == 0,
            judge: if hold { JudgeStatus::Hold(true, (id % 4) as f32, 0.0, false, 0.0) } else { JudgeStatus::NotJudged },
            format: false,
        }
    }

    // Reference the original serial visitation: update one object, visit its
    // effect, then retain it. The worker implementation separates those passes.
    fn serial_visitation(notes: &mut [Note], active: &mut Vec<u32>, time: f32) -> Vec<u32> {
        let mut effects = Vec::new();
        active.retain(|&id| {
            let note = &mut notes[id as usize];
            note.object.set_time(time);
            if matches!(note.judge, JudgeStatus::Hold(_, at, ..) if time >= at) {
                effects.push(id);
            }
            !note.dead()
        });
        effects
    }

    #[test]
    fn parallel_note_updates_preserve_serial_state_and_effect_order() {
        let mut notes: Vec<_> = (0..4096).map(note).collect();
        let mut state = JudgeLineCache::new(&mut notes).state;
        let mut serial = notes.clone();
        let mut active = state.update_order.clone();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        for time in [0.0, 0.5, 1.0, 2.5, 4.5] {
            let effects = serial_visitation(&mut serial, &mut active, time);
            pool.install(|| state.update(&mut notes, time, true));
            assert_eq!(notes, serial, "note state at {time}");
            assert_eq!(state.update_order, active, "sparse active spans at {time}");
            assert_eq!(state.effect_order, effects, "particle visitation order at {time}");
            for &id in &effects {
                // Model the ordered effect consumer advancing each hold timer.
                for set in [&mut notes, &mut serial] {
                    if let JudgeStatus::Hold(_, at, ..) = &mut set[id as usize].judge {
                        *at = time + 0.3;
                    }
                }
            }
        }
    }

    #[test]
    fn render_groups_advance_past_judged_prefixes_in_original_order() {
        let mut notes: Vec<_> = (0..24).map(|id| {
            let mut note = note(id);
            note.kind = NoteKind::Click;
            note.fake = false;
            note.judge = JudgeStatus::NotJudged;
            note
        }).collect();
        let mut state = JudgeLineCache::new(&mut notes).state;
        let groups: Vec<_> = state.above_indices.iter().chain(&state.below_indices).copied().collect();
        for &start in &groups {
            notes[start].judge = JudgeStatus::Judged;
            notes[start + 1].judge = JudgeStatus::Judged;
        }
        state.update(&mut notes, 0.5, false);
        let expected: Vec<_> = notes.iter().enumerate().filter_map(|(id, note)| {
            (!matches!(note.judge, JudgeStatus::Judged)).then_some(id)
        }).collect();
        assert_eq!(state.render_order, expected);
        assert_eq!(state.above_indices, groups[..3].iter().map(|id| id + 2).collect::<Vec<_>>());
        assert_eq!(state.below_indices, groups[3..].iter().map(|id| id + 2).collect::<Vec<_>>());
    }

    fn line(kind: JudgeLineKind) -> JudgeLine {
        let mut notes = vec![note(0), note(1)];
        let cache = JudgeLineCache::new(&mut notes);
        JudgeLine {
            object: Object {
                rotation: Anim::new(vec![Keyframe::new(0.0, 0.0, 2), Keyframe::new(4.0, 90.0, 2)]),
                translation: AnimVector::fixed(Vector::new(0.3, 0.7)),
                ..Object::default()
            },
            ctrl_obj: Rc::new(RefCell::new(CtrlObject::default())),
            kind,
            height: Anim::fixed(2.0),
            incline: Anim::default(),
            notes,
            color: Anim::fixed(WHITE),
            parent: None,
            z_index: 0,
            show_below: true,
            attach_ui: None,
            cache,
            cached_world_pos: None,
        }
    }

    #[test]
    fn worker_tasks_update_line_animations_without_moving_shared_state() {
        let mut lines = [
            line(JudgeLineKind::Normal),
            line(JudgeLineKind::Text(Anim::fixed(String::from("text")))),
            line(JudgeLineKind::Paint(Anim::fixed(0.5), Arc::new(Mutex::new((None, false))))),
            line(JudgeLineKind::TextureGif(Anim::fixed(0.5), GifFrames::new(Vec::new()), String::new())),
        ];
        let controls: Vec<_> = lines.iter().map(|line| Rc::clone(&line.ctrl_obj)).collect();
        let mut locals = [LineLocalState::default(); 4];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        {
            let mut tasks: Vec<_> = lines.iter_mut().map(JudgeLine::update_task).collect();
            pool.install(|| tasks.par_iter_mut().zip(locals.par_iter_mut()).for_each(|(task, local)| {
                *local = task.prepare(2.0, 16.0 / 9.0, true);
            }));
        }
        for ((line, local), control) in lines.iter().zip(locals).zip(controls) {
            assert!(Rc::ptr_eq(&line.ctrl_obj, &control));
            assert_eq!(local.translation, Vector::new(0.3, 0.7 / (16.0 / 9.0)));
            assert_eq!(local.rotation.to_homogeneous(), line.object.now_rotation());
            assert_eq!(line.height.time, 2.0);
            assert_eq!(line.color.time, 2.0);
            match &line.kind {
                JudgeLineKind::Text(anim) => assert_eq!(anim.time, 2.0),
                JudgeLineKind::Paint(anim, ..) | JudgeLineKind::TextureGif(anim, ..) => assert_eq!(anim.time, 2.0),
                _ => {}
            }
        }
    }
}
