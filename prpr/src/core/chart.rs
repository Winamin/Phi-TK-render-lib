use super::{BpmList, Effect, JudgeLine, JudgeLineKind, Matrix, Resource, UIElement, Vector, Video};
use crate::{fs::FileSystem, judge::JudgeStatus, ui::Ui};
use anyhow::{Context, Result};
use macroquad::prelude::*;
use std::{cell::RefCell, collections::{HashMap, HashSet}};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use futures_util::{stream::FuturesUnordered, StreamExt};
use super::{line::{LineLocalState, LineRenderBuffer, LineRenderGpu}, note::RenderFrame};
use crate::ext::{get_viewport, spawn_task, SafeTexture};
use tracing::warn;

pub(crate) static RENDER_POOL: Lazy<rayon::ThreadPool> = Lazy::new(|| {
    let threads = std::env::var("PRPR_RENDER_THREADS").ok().and_then(|it| it.parse::<usize>().ok()).filter(|it| *it > 0).unwrap_or_else(|| num_cpus::get_physical().min(8).max(1));
    let builder = rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|id| format!("chart-render-{id}"));
    // While waiting for this frame's data, main can help the same pool without
    // another wakeup. Disable participation when caller CPU usage takes priority.
    let participate = std::env::var("PRPR_RENDER_MAIN_THREAD")
        .map_or(true, |value| !matches!(value.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off"));
    if participate { builder.use_current_thread() } else { builder }.build().unwrap()
});

static PARALLEL_UPDATE: Lazy<bool> = Lazy::new(|| {
    std::env::var("PRPR_PARALLEL_UPDATE").map_or(true, |value| !matches!(value.as_str(), "0" | "false" | "off"))
});

pub(crate) static RENDER_CHUNK_SIZE: Lazy<usize> = Lazy::new(|| {
    std::env::var("PRPR_RENDER_CHUNK_SIZE").ok().and_then(|it| it.parse::<usize>().ok()).filter(|it| *it > 0).unwrap_or(128)
});

#[derive(Default)]
pub struct ChartExtra {
    pub effects: Vec<Effect>,
    pub global_effects: Vec<Effect>,
    pub videos: Vec<Video>,
}

#[derive(Default)]
pub struct ChartSettings {
    pub pe_alpha_extension: bool,
    pub hold_partial_cover: bool,
}

pub struct Chart {
    pub offset: f32,
    pub lines: Vec<JudgeLine>,
    pub bpm_list: RefCell<BpmList>,
    pub settings: ChartSettings,
    pub extra: ChartExtra,
    pub order: Vec<usize>,
    pub attach_ui: [Option<usize>; 7],
    world_positions: Vec<Vector>,
    local_states: Vec<LineLocalState>,
    trs: Vec<Matrix>,
    position_order: Vec<usize>,
    parents: Vec<Option<usize>>,
    render_buffers: RefCell<Vec<LineRenderBuffer>>,
    render_gpu: RefCell<Vec<LineRenderGpu>>,
}

impl Chart {
    pub fn new(offset: f32, lines: Vec<JudgeLine>, bpm_list: BpmList, settings: ChartSettings, extra: ChartExtra) -> Self {
        let mut attach_ui = [None; 7];
        let mut order = (0..lines.len())
            .filter(|it| {
                if let Some(element) = lines[*it].attach_ui {
                    attach_ui[element as usize - 1] = Some(*it);
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        order.sort_by_key(|it| (lines[*it].z_index, *it));
        let capacity = lines.len();
        let parents = lines.iter().map(|it| it.parent).collect();
        let position_order = Self::position_order(&lines);
        Self {
            offset,
            lines,
            bpm_list: RefCell::new(bpm_list),
            settings,
            extra,

            order,
            attach_ui,
            world_positions: Vec::with_capacity(capacity),
            local_states: Vec::with_capacity(capacity),
            trs: Vec::with_capacity(capacity),
            position_order,
            parents,
            render_buffers: RefCell::new(Vec::new()),
            render_gpu: RefCell::new(Vec::new()),
        }
    }

    #[inline]
    pub fn with_element<R>(&self, ui: &mut Ui, res: &Resource, element: UIElement, ct: Option<(f32, f32)>, pt: Option<(f32, f32)>, f: impl FnOnce(&mut Ui, Color) -> R) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let lines = &self.lines;
            let line = &lines[id];
            let obj = &line.object;
            let mut tr = JudgeLine::fetch_pos(line, res, lines);
            tr.y = -tr.y;
            let mut color = self.lines[id].color.now_opt().unwrap_or(WHITE);
            color.a *= obj.now_alpha().max(0.);
            let scale = obj.now_scale_fix(ct.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x, y)));
            let ro = obj.new_rotation_wrt_point(-obj.rotation.now().to_radians(), pt.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x, y)));
            ui.with(Matrix::new_translation(&tr) * ro * scale, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub fn with_element_noscale<R>(&self, ui: &mut Ui, res: &Resource, element: UIElement, ct: Option<(f32, f32)>, f: impl FnOnce(&mut Ui, Color) -> R) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let obj = &self.lines[id].object;
            let mut tr = obj.now_translation(res);
            tr.y = -tr.y;
            let mut color = self.lines[id].color.now_opt().unwrap_or(WHITE);
            color.a *= obj.now_alpha().max(0.);
            let mut scale = obj.now_scale_fix(ct.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x , y)));
            scale.m11 = 1.0;
            ui.with(obj.now_rotation().append_translation(&tr) * scale, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub async fn load_textures(&mut self, fs: &mut dyn FileSystem) -> Result<()> {
        let mut textures: HashMap<String, SafeTexture> = HashMap::new();
        for line in &self.lines {
            if let JudgeLineKind::Texture(texture, path) = &line.kind {
                if texture.width() > 0.0 && texture.height() > 0.0 {
                    textures.entry(path.clone()).or_insert_with(|| texture.clone());
                }
            }
        }
        let mut seen = HashSet::new();
        let paths: Vec<_> = self.lines.iter().filter_map(|line| {
            if let JudgeLineKind::Texture(_, path) = &line.kind {
                (!textures.contains_key(path) && seen.insert(path.clone())).then(|| path.clone())
            } else { None }
        }).collect();
        let mut pending = FuturesUnordered::new();
        let limit = num_cpus::get_physical().clamp(1, 8);
        for path in paths {
            let data = fs.load_file(&path).await.with_context(|| format!("failed to load illustration {path}"))?;
            pending.push(async move {
                let context = path.clone();
                let image = spawn_task(move || {
                    let image = image::load_from_memory(&data)?.into_rgba8();
                    anyhow::ensure!(image.width() <= u16::MAX as u32 && image.height() <= u16::MAX as u32, "texture dimensions exceed u16");
                    Ok(image)
                }).await.with_context(|| format!("failed to decode illustration {context}"))?;
                Ok::<_, anyhow::Error>((path, image))
            });
            if pending.len() >= limit {
                let (path, image) = pending.next().await.unwrap()?;
                textures.insert(path, Texture2D::from_rgba8(image.width() as _, image.height() as _, image.as_raw()).into());
            }
        }
        while let Some(result) = pending.next().await {
            let (path, image) = result?;
            textures.insert(path, Texture2D::from_rgba8(image.width() as _, image.height() as _, image.as_raw()).into());
        }
        for line in &mut self.lines {
            if let JudgeLineKind::Texture(texture, path) = &mut line.kind {
                *texture = textures[path].clone();
            }
        }
        Ok(())
    }

    fn position_order(lines: &[JudgeLine]) -> Vec<usize> {
        let mut children = vec![Vec::new(); lines.len()];
        let mut pending = vec![false; lines.len()];
        let mut order = Vec::with_capacity(lines.len());
        for (id, line) in lines.iter().enumerate() {
            if let Some(parent) = line.parent.filter(|it| *it < lines.len() && *it != id) {
                children[parent].push(id);
                pending[id] = true;
            } else {
                order.push(id);
            }
        }
        let mut index = 0;
        while index < order.len() {
            for &child in &children[order[index]] {
                pending[child] = false;
                order.push(child);
            }
            index += 1;
        }
        order.extend(pending.iter().enumerate().filter_map(|(id, pending)| pending.then_some(id)));
        order
    }

    pub fn reset(&mut self) {
        self.lines
            .iter_mut()
            .flat_map(|it| it.notes.iter_mut())
            .for_each(|note| note.judge = JudgeStatus::NotJudged);
        for line in &mut self.lines {
            line.cache.reset(&mut line.notes);
        }
        for video in &mut self.extra.videos {
            video.next_frame = 0;
        }
    }

    pub fn update(&mut self, res: &mut Resource) {
        crate::core::begin_readback_cycle();
        let count = self.lines.len();
        if self.parents.len() != count || self.lines.iter().zip(&self.parents).any(|(line, parent)| line.parent != *parent) {
            self.parents = self.lines.iter().map(|it| it.parent).collect();
            self.position_order = Self::position_order(&self.lines);
        }
        self.local_states.resize(count, LineLocalState::default());
        let parallel = *PARALLEL_UPDATE && RENDER_POOL.current_num_threads() > 1
            && (count >= 16 || self.lines.iter().map(JudgeLine::pending_updates).sum::<usize>() >= 1024);
        let time = res.time;
        let aspect_ratio = res.aspect_ratio;
        if parallel {
            let mut tasks: Vec<_> = self.lines.iter_mut().map(JudgeLine::update_task).collect();
            let local_states = &mut self.local_states;
            RENDER_POOL.install(|| {
                tasks.par_iter_mut().zip(local_states.par_iter_mut()).with_max_len(1).for_each(|(task, local)| {
                    *local = task.prepare(time, aspect_ratio, true);
                });
            });
        } else {
            for (line, local) in self.lines.iter_mut().zip(&mut self.local_states) {
                *local = line.update_task().prepare(time, aspect_ratio, false);
            }
        }
        self.world_positions.clear();
        self.world_positions.resize(count, Vector::zeros());
        self.trs.resize(count, Matrix::identity());
        for &id in &self.position_order {
            let local = self.local_states[id];
            let pos = self.parents[id].filter(|parent| *parent < count && *parent != id).map_or(local.translation, |parent| {
                self.world_positions[parent] + self.local_states[parent].rotation * local.translation
            });
            self.world_positions[id] = pos;
            self.trs[id] = local.rotation.to_homogeneous().append_translation(&pos);
            self.lines[id].cached_world_pos = Some(pos);
        }
        let guard = self.bpm_list.borrow();
        for (index, line) in self.lines.iter_mut().enumerate() {
            line.finish_update(res, self.trs[index], &guard, index);

            if res.config.hand_split {
                line.update_hand_assign_with_world_pos(res, self.world_positions[index], index);
            }
        }
        drop(guard);

        for effect in &mut self.extra.effects {
            effect.update(res);
        }
        for video in &mut self.extra.videos {
            if let Err(err) = video.update(res.time) {
                warn!("video error: {err:?}");
            }
        }
    }

    pub fn render(&self, ui: &mut Ui, res: &mut Resource) {
        res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(if res.config.flip_x() { -1. } else { 1. }, 1.)), |res| {
            for video in &self.extra.videos {
                video.render(res);
            }
        });
        let guard = self.bpm_list.borrow();
        let settings = &self.settings;
        let bpm_list: &BpmList = &guard;
        let frame = RenderFrame::new(res);
        let controls: Vec<_> = self.lines.iter().map(|line| line.ctrl_obj.borrow()).collect();
        let tasks: Vec<_> = self.lines.iter().zip(&controls).map(|(line, ctrl)| line.attach_ui.is_none().then(|| line.render_task(res, &self.lines, ctrl))).collect();
        let vp = get_viewport();
        let ui_model = ui.get_matrix();
        let font = ui.text_painter.font();
        let texture_limit = ui.text_painter.texture_limit();
        let mut buffers = self.render_buffers.borrow_mut();
        buffers.resize_with(self.lines.len(), LineRenderBuffer::default);
        let parallel = RENDER_POOL.current_num_threads() > 1;
        let prepare = |(task, buffer): (&Option<super::line::LineRenderTask>, &mut LineRenderBuffer)| {
            if let Some(task) = task {
                task.prepare(buffer, &frame, bpm_list, settings, vp, ui_model, &font, texture_limit, parallel);
            }
        };
        if parallel {
            let buffer_slice = buffers.as_mut_slice();
            RENDER_POOL.install(|| tasks.par_iter().zip(buffer_slice.par_iter_mut()).with_max_len(1).for_each(prepare));
        } else {
            tasks.iter().zip(buffers.iter_mut()).for_each(prepare);
        }
        drop(tasks);
        drop(controls);
        drop(frame);
        drop(guard);
        let mut gpu = self.render_gpu.borrow_mut();
        gpu.resize_with(self.lines.len(), LineRenderGpu::default);
        res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(if res.config.flip_x() { -1. } else { 1. }, -1.)), |res| {
            for &id in &self.order {
                self.lines[id].render_prepared(ui, res, &self.lines, &buffers[id], &mut gpu[id], id);
            }
            res.note_buffer.borrow_mut().draw_all();
        });
    }
}
