//! 噪域 / ba (`blockAreaList`) effect — strict port of `ziheng18/noisepainter`.
//!
//! The reference is the WebGL tool "噪域画笔" (<https://noisepainter.sling.cc.cd/>),
//! whose fragment shader is the decompiled official Phigros block-area shader
//! (GLSL ES 100, `docs/block-area/build_full_fragment.py`). Everything below
//! mirrors it operation by operation:
//!
//!   * the mask is rasterised on the CPU over a grid of `screen / 8` cells
//!     (official `blockPx = 8`), each cell is displaced by the noise texture
//!     (`BlockCompose` pass, amplitude `0.1`), then upscaled 2x with nearest
//!     sampling; six 9-grid dilation passes (`GLOW_W`) produce the edge ring
//!     (`.g`) and the outer glow (`.b`), the fill mask is `.r`;
//!   * the **disabled layer** (`.a`) is the fill of block areas that are not in
//!     their active window: the reference's `disabledColor()` keeps the inner
//!     fill (`uDisabledFillColor * uDisabledFillOpacity`) and drops the edge /
//!     glow ring. Unlike the reference (which picks either layer per pixel),
//!     this port always draws the disabled fill as the **base** and composites
//!     the active layer over it, so the margin around an active block does not
//!     punch a hole into the inactive blocks underneath;
//!   * `displace.png` / `spark.png` / `noise.png` are the exact textures the
//!     reference ships;
//!   * the fragment shader keeps the official uniforms (`_EdgeColor`,
//!     `_FillColor`, `_GlowColor`, `_SparkTint`, opacities, displace/spark
//!     parameters) and the `One / OneMinusSrcAlpha` blend mode;
//!   * the touch-hover / ready layers are skipped (`_TouchPosCount` is 0 for an
//!     offline render), exactly like the reference does.
//!
//! Chart time is uploaded in a 1x1 RGBA texture (bound to `uNoiseTex`, unused by
//! the active layer) because macroquad overwrites `_Time` with wall-clock time,
//! which would desynchronise the effect from the chart during an offline render.
use super::{BlockArea, BlockPhase, Vector};
use macroquad::prelude::*;
use macroquad::miniquad::{BlendFactor, BlendState, BlendValue, Equation, PipelineParams, TextureWrap};
use once_cell::sync::Lazy;
use std::sync::Mutex;

// ---- `BASE_COLORS` / `BASE_FLOATS` from the reference ----------------------
const EDGE_COLOR: [f32; 3] = [1., 0.33018857, 0.33018857];
const FILL_COLOR: [f32; 3] = [0.7132075, 0.23549296, 0.23549296];
const GLOW_COLOR: [f32; 3] = [1., 0.17924517, 0.17924517];
const SPARK_TINT: [f32; 3] = [1., 0.28490567, 0.28490567];
const EDGE_OPACITY: f32 = 0.8;
const FILL_STRENGTH: f32 = 0.667;
const FILL_OPACITY: f32 = 0.667;
const GLOW_INTENSITY: f32 = 0.8;
const SPARK_MAP_OPACITY: f32 = 5.69;
const SPARK_HUE_SHIFT: f32 = 0.2;
const SPARK_DISPLACE_INTENSITY: f32 = 2.39;
const DISPLACE_BLEND_INTENSITY: f32 = 0.411;
const DISPLACE_SPEED: f32 = 1.5;
const DISPLACE_STRENGTH: f32 = 0.15;
const BACKGROUND_PIXEL_SCALE: f32 = 6.;
/// `blockPx` of the official game (one mask cell = 8 screen pixels).
const BLOCK_PX: f32 = 8.;

/// `GLOW_W[p] = (6 - p)^2.65 / sum(k^2.65)`, exactly as the reference computes.
const GLOW_W: [f32; 6] = [
    0.45856797, 0.28286122, 0.15658923, 0.07305908, 0.02494781, 0.00397469,
];

/// A block area resolved at one instant, in chart space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zone {
    pub center: Vector,
    pub half: Vector,
    pub angle: f32,
    pub invert: bool,
    pub active: bool,
}

impl Zone {
    pub fn from_area(area: &BlockArea, time: f64, aspect: f32) -> Option<Self> {
        let phase = area.phase(time);
        if phase == BlockPhase::Hidden {
            return None;
        }
        let tr = area.transform(time, aspect);
        let half = tr.size.map(|v| v.abs() * 0.5);
        if half.x == 0. || half.y == 0. {
            return None;
        }
        Some(Self {
            center: tr.center,
            half,
            angle: tr.rotation.to_radians(),
            invert: area.is_subtract,
            active: phase == BlockPhase::Active,
        })
    }
}

struct Effect {
    material: Material,
    mask: Texture2D,
    displace: Texture2D,
    spark: Texture2D,
    /// 1x1 RGBA holding the chart time as an `f32` (little-endian bytes in RGBA).
    clock: Texture2D,
    white: Texture2D,
    grid_w: usize,
    grid_h: usize,
    tex_w: usize,
    tex_h: usize,
    /// Upload buffer; written in place so a frame does not need a 0.5 MB clone.
    image: Image,
    /// Copy of the bytes currently uploaded to the GPU (skips redundant uploads).
    last_upload: Vec<u8>,
    /// bitset of the active fill (used by the dilation).
    bits: Vec<u64>,
    scratch: Vec<u64>,
    merge: Vec<u64>,
    flags_active: Vec<u8>,
    flags_disabled: Vec<u8>,
    compose_active: Vec<u8>,
    compose_disabled: Vec<u8>,
    fm_active: Vec<u8>,
    fm_disabled: Vec<u8>,
    noise_r: Vec<u8>,
    noise_w: usize,
    noise_h: usize,
    centered: [f32; 256],
    seen: u64,
    dumped: u32,
    build_ms: f64,
    build_n: u64,
    last_hash: u64,
    last_build_time: f32,
    stage_ms: (f64, f64, f64, f64, f64),
}

static EFFECT: Lazy<Mutex<Option<Effect>>> = Lazy::new(|| Mutex::new(None));

fn tex(bytes: &[u8], wrap: TextureWrap) -> Texture2D {
    let image = image::load_from_memory(bytes).expect("block effect texture").to_rgba8();
    let texture = Texture2D::from_rgba8(image.width() as u16, image.height() as u16, image.as_raw());
    texture.set_filter(FilterMode::Nearest);
    texture.raw_miniquad_texture_handle().set_wrap(unsafe { get_internal_gl() }.quad_context, wrap);
    texture
}

/// `medium(v)` from the reference: round-to-nearest-even at half-float
/// precision (10 mantissa bits), used to build the `CENTERED` lookup table.
fn medium(v: f32) -> f32 {
    let bits = v.to_bits();
    let bits = bits.wrapping_add(0xfff + ((bits >> 13) & 1)) & !0x1fff;
    f32::from_bits(bits)
}

/// `noiseIndex(v, size)` from the reference: mirrored addressing.
fn noise_index(v: f32, size: usize) -> usize {
    let mut v = v % 2.;
    if v < 0. {
        v += 2.;
    }
    if v > 1. {
        v = 2. - v;
    }
    let i = (v * size as f32).floor() as i64;
    i.clamp(0, size as i64 - 1) as usize
}

fn create() -> Effect {
    let material = load_material(
        shader::VERTEX,
        shader::FRAGMENT,
        MaterialParams {
            pipeline_params: PipelineParams {
                // The official active layer blends with One / OneMinusSrcAlpha.
                // macroquad picks `color_blend` or `alpha_blend` depending on
                // whether the drawn texture has an alpha channel; both need the
                // same state.
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                alpha_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
            uniforms: Vec::new(),
            textures: vec![
                "uMasks".to_owned(),
                "uDisplaceTex".to_owned(),
                "uSparkTex".to_owned(),
                "uNoiseTex".to_owned(),
                "uScene".to_owned(),
            ],
        },
    )
    .expect("block effect material");

    let displace = tex(include_bytes!("../../assets/blockarea/displace.png"), TextureWrap::Mirror);
    let spark = tex(include_bytes!("../../assets/blockarea/spark.png"), TextureWrap::Repeat);
    let noise = image::load_from_memory(include_bytes!("../../assets/blockarea/noise.png"))
        .expect("noise.png")
        .to_rgba8();
    let noise_w = noise.width() as usize;
    let noise_h = noise.height() as usize;
    let noise_r: Vec<u8> = noise.pixels().map(|p| p[0]).collect();

    let mut centered = [0f32; 256];
    for (i, slot) in centered.iter_mut().enumerate() {
        *slot = medium(medium(i as f32 / 255.) - 0.5);
    }

    let mask = Texture2D::from_rgba8(1, 1, &[0, 0, 0, 0]);
    let clock = Texture2D::from_rgba8(1, 1, &[0, 0, 0, 0]);
    let white = Texture2D::from_rgba8(1, 1, &[255, 255, 255, 255]);
    material.set_texture("uDisplaceTex", displace);
    material.set_texture("uSparkTex", spark);
    material.set_texture("uNoiseTex", clock);
    material.set_texture("uMasks", mask);

    Effect {
        material,
        mask,
        displace,
        spark,
        clock,
        white,
        grid_w: 1,
        grid_h: 1,
        tex_w: 1,
        tex_h: 1,
        image: Image {
            width: 1,
            height: 1,
            bytes: vec![0; 4],
        },
        last_upload: Vec::new(),
        bits: Vec::new(),
        scratch: Vec::new(),
        merge: Vec::new(),
        flags_active: Vec::new(),
        flags_disabled: Vec::new(),
        compose_active: Vec::new(),
        compose_disabled: Vec::new(),
        fm_active: Vec::new(),
        fm_disabled: Vec::new(),
        noise_r,
        noise_w,
        noise_h,
        centered,
        seen: 0,
        dumped: 0,
        build_ms: 0.,
        build_n: 0,
        last_hash: 0,
        last_build_time: f32::NEG_INFINITY,
        stage_ms: (0., 0., 0., 0., 0.),
    }
}

/// Rasterise one zone into `flags` (bit 0 = normal coverage, bit 1 = subtract).
fn rasterize(flags: &mut [u8], z: &Zone, aspect: f32, grid_w: usize, grid_h: usize) {
    let to_cell = |p: Vector| {
        let u = (p.x + 1.) * 0.5 * grid_w as f32;
        let v = (p.y * aspect + 1.) * 0.5 * grid_h as f32;
        (u, v)
    };
    let (s, c) = z.angle.sin_cos();
    let corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)].map(|(x, y)| {
        let (x, y) = (x * z.half.x, y * z.half.y);
        Vector::new(z.center.x + c * x - s * y, z.center.y + s * x + c * y)
    });
    let cells: Vec<(f32, f32)> = corners.iter().map(|p| to_cell(*p)).collect();
    let min_y = cells.iter().map(|c| c.1).fold(f32::INFINITY, f32::min).floor().max(0.) as usize;
    let max_y = cells
        .iter()
        .map(|c| c.1)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(grid_h as f32) as usize;
    let mask_bit: u8 = if z.invert { 2 } else { 1 };
    for gy in min_y..max_y {
        let y = gy as f32 + 0.5;
        // Even-odd scanline: collect x where the polygon crosses this row.
        let mut xs = [0f32; 4];
        let mut n = 0;
        for i in 0..4 {
            let (x0, y0) = cells[i];
            let (x1, y1) = cells[(i + 1) % 4];
            if (y0 <= y) != (y1 <= y) {
                xs[n] = x0 + (y - y0) / (y1 - y0) * (x1 - x0);
                n += 1;
            }
        }
        if n < 2 {
            continue;
        }
        xs[..n].sort_by(f32::total_cmp);
        for pair in xs[..n].chunks_exact(2) {
            let x0 = pair[0].max(0.).floor() as usize;
            let x1 = pair[1].min(grid_w as f32).ceil() as usize;
            for gx in x0..x1 {
                flags[gy * grid_w + gx] |= mask_bit;
            }
        }
    }
}

/// `compose = |norm - (subtract inside [0.09, 0.12))|`, i.e. a plain XOR.
fn compose(src: &[u8], dst: &mut [u8]) {
    for (i, slot) in dst.iter_mut().enumerate() {
        let f = src[i];
        *slot = if (f & 1 != 0) ^ (f & 2 != 0) { 255 } else { 0 };
    }
}

/// Nine-grid dilation in 64-bit words: three vertical ORs then a 3-bit
/// horizontal spread (much cheaper than the per-pixel 3x3 loop).
fn dilate(src: &[u64], merge: &mut [u64], dst: &mut [u64], w: usize, h: usize) {
    let stride = w.div_ceil(64);
    let tail = w % 64;
    let last_mask = if tail == 0 { u64::MAX } else { (1u64 << tail) - 1 };

    for y in 0..h {
        let row = y * stride;
        for wi in 0..stride {
            let mut v = src[row + wi];
            if y > 0 {
                v |= src[row - stride + wi];
            }
            if y + 1 < h {
                v |= src[row + stride + wi];
            }
            merge[row + wi] = v;
        }
    }
    for y in 0..h {
        let row = y * stride;
        for wi in 0..stride {
            let v = merge[row + wi];
            let mut o = v | (v << 1) | (v >> 1);
            if wi > 0 {
                o |= merge[row + wi - 1] >> 63;
            }
            if wi + 1 < stride {
                o |= merge[row + wi + 1] << 63;
            }
            dst[row + wi] = o;
        }
        dst[row + stride - 1] &= last_mask;
    }
}

/// Rebuild the RGBA mask for the given zones.
///
/// `r` = active fill, `g` = edge ring, `b` = glow, `a` = disabled fill. The
/// active and disabled layers are composited from different zone sets; only the
/// active one is dilated (the disabled layer has no ring or glow).
fn rebuild(effect: &mut Effect, zones: &[Zone], aspect: f32, grid_w: usize, grid_h: usize, time: f32) {
    let tex_w = grid_w * 2;
    let tex_h = grid_h * 2;
    let n = grid_w * grid_h;
    let stride = tex_w.div_ceil(64);

    if effect.tex_w != tex_w || effect.tex_h != tex_h {
        effect.tex_w = tex_w;
        effect.tex_h = tex_h;
        effect.image = Image {
            width: tex_w as u16,
            height: tex_h as u16,
            bytes: vec![0; tex_w * tex_h * 4],
        };
        effect.bits = vec![0u64; stride * tex_h];
        effect.scratch = vec![0u64; stride * tex_h];
        effect.merge = vec![0u64; stride * tex_h];
        effect.mask = Texture2D::from_rgba8(tex_w as u16, tex_h as u16, &effect.image.bytes);
        effect.material.set_texture("uMasks", effect.mask);
    } else {
        effect.image.bytes.fill(0);
        effect.bits.fill(0);
    }
    effect.grid_w = grid_w;
    effect.grid_h = grid_h;

    if effect.flags_active.len() != n {
        effect.flags_active = vec![0u8; n];
        effect.flags_disabled = vec![0u8; n];
        effect.compose_active = vec![0u8; n];
        effect.compose_disabled = vec![0u8; n];
        effect.fm_active = vec![0u8; n];
        effect.fm_disabled = vec![0u8; n];
    }
    let _t_raster = std::time::Instant::now();
    effect.flags_active.fill(0);
    effect.flags_disabled.fill(0);

    for z in zones {
        let flags = if z.active {
            &mut effect.flags_active
        } else {
            &mut effect.flags_disabled
        };
        rasterize(flags, z, aspect, grid_w, grid_h);
    }
    let raster_ms = _t_raster.elapsed().as_secs_f64() * 1000.;
    let _t_noise = std::time::Instant::now();
    compose(&effect.flags_active, &mut effect.compose_active);
    compose(&effect.flags_disabled, &mut effect.compose_disabled);

    // Per-cell noise displacement (official BlockCompose pass 0). Both layers
    // use the same displacement field, exactly like the reference.
    let d = 0.70703125f32;
    let dt = d * (time / 20. * 2.59);
    let amp = 0.1; // "噪声强度" 1 = official
    let nw = effect.noise_w.max(1);
    let nh = effect.noise_h.max(1);
    {
        let mut wx_a = vec![0usize; grid_w];
        let mut wx_b = vec![0usize; grid_w];
        let mut wy_row = vec![0usize; grid_h];
        for x in 0..grid_w {
            let u = (x as f32 + 0.5) / grid_w as f32;
            wx_a[x] = noise_index(dt + u * 2.13, nw);
            wx_b[x] = noise_index(-dt + u * 2.13, nw);
        }
        for y in 0..grid_h {
            wy_row[y] = noise_index(dt + ((y as f32 + 0.5) / grid_h as f32) * 1.02, nh) * nw;
        }
        // The per-cell displacement is the hottest CPU loop (1.6 ms at 1080p),
        // so split the rows across threads. Results are written per row, so the
        // split is deterministic.
        let mut fm_active = std::mem::take(&mut effect.fm_active);
        let mut fm_disabled = std::mem::take(&mut effect.fm_disabled);
        {
            let compose_active = &effect.compose_active;
            let compose_disabled = &effect.compose_disabled;
            let noise_r = &effect.noise_r;
            let centered = &effect.centered;
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(1, 8);
            let rows_per = grid_h.div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                for ((chunk, disabled_chunk), y0) in fm_active
                    .chunks_mut(rows_per * grid_w)
                    .zip(fm_disabled.chunks_mut(rows_per * grid_w))
                    .zip((0..grid_h).step_by(rows_per))
                {
                    let wx_a = &wx_a;
                    let wx_b = &wx_b;
                    let wy_row = &wy_row;
                    scope.spawn(move || {
                        for (dy, (row_out, disabled_row)) in chunk
                            .chunks_mut(grid_w)
                            .zip(disabled_chunk.chunks_mut(grid_w))
                            .enumerate()
                        {
                            let y = y0 + dy;
                            let row = wy_row[y];
                            for x in 0..grid_w {
                                let a = centered[noise_r[row + wx_a[x]] as usize];
                                let b = centered[noise_r[row + wx_b[x]] as usize];
                                let u = (d * a + b * -d) * amp + (x as f32 + 0.5) / grid_w as f32;
                                let v = (d * a + b * d) * amp + (y as f32 + 0.5) / grid_h as f32;
                                let sxi = ((u * grid_w as f32).floor() as i64)
                                    .clamp(0, grid_w as i64 - 1) as usize;
                                let syi = ((v * grid_h as f32).floor() as i64)
                                    .clamp(0, grid_h as i64 - 1) as usize;
                                let idx = syi * grid_w + sxi;
                                row_out[x] = compose_active[idx];
                                // The disabled layer is drawn as the raw block
                                // rectangle: no noise displacement, no textures.
                                disabled_row[x] = compose_disabled[y * grid_w + x];
                            }
                        }
                    });
                }
            });
        }
        effect.fm_active = fm_active;
        effect.fm_disabled = fm_disabled;
    }

    let noise_ms = _t_noise.elapsed().as_secs_f64() * 1000.;
    let _t_up = std::time::Instant::now();
    // 2x nearest upscale: r = active fill (and the bitset), a = disabled fill.
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let active = effect.fm_active[idx];
            let disabled = effect.fm_disabled[idx];
            if active == 0 && disabled == 0 {
                continue;
            }
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let x = gx * 2 + dx;
                let y = gy * 2 + dy;
                let o = (y * tex_w + x) * 4;
                if active != 0 {
                    effect.image.bytes[o] = active;
                    effect.bits[y * stride + (x >> 6)] |= 1u64 << (x & 63);
                }
                if disabled != 0 {
                    effect.image.bytes[o + 3] = disabled;
                }
            }
        }
    }

    let upscale_ms = _t_up.elapsed().as_secs_f64() * 1000.;
    let _t_dil = std::time::Instant::now();
    // Edge / glow: nine-grid dilation + ring extraction. `edgePassesOf()` with
    // the official edgeStrength = 6 gives one edge ring; `w < 0.01` stops early
    // (the official loop has the same break, so only five passes run).
    let edge_passes = 1;
    for pass in 0..6 {
        let w = GLOW_W[pass];
        if w < 0.01 {
            break;
        }
        dilate(&effect.bits, &mut effect.merge, &mut effect.scratch, tex_w, tex_h);
        for y in 0..tex_h {
            let base = y * stride;
            for wi in 0..stride {
                let mut ring = effect.scratch[base + wi] & !effect.bits[base + wi];
                while ring != 0 {
                    let bit = ring.trailing_zeros() as usize;
                    let x = wi * 64 + bit;
                    if x < tex_w {
                        let dst = (y * tex_w + x) * 4;
                        if pass < edge_passes {
                            effect.image.bytes[dst + 1] = 255;
                        }
                        effect.image.bytes[dst + 2] = (w * 255.) as u8;
                    }
                    ring &= ring - 1;
                }
            }
        }
        std::mem::swap(&mut effect.bits, &mut effect.scratch);
    }

    let dilate_ms = _t_dil.elapsed().as_secs_f64() * 1000.;
    let _t_up2 = std::time::Instant::now();
    // The GPU upload is the single most expensive step on software Vulkan
    // (2.4 ms for 0.5 MB at 1080p). Static block areas produce identical bytes,
    // so skip the upload in that case.
    if effect.last_upload != effect.image.bytes {
        effect.mask.update(&effect.image);
        effect.last_upload.clear();
        effect.last_upload.extend_from_slice(&effect.image.bytes);
    }
    effect.stage_ms = (raster_ms, noise_ms, upscale_ms, dilate_ms, _t_up2.elapsed().as_secs_f64() * 1000.);
}

/// Upload the chart time as an `f32` in a 1x1 RGBA texture.
fn upload_clock(effect: &Effect, time: f32) {
    let b = time.to_le_bytes();
    effect.clock.update(&Image {
        width: 1,
        height: 1,
        bytes: vec![b[0], b[1], b[2], b[3]],
    });
    effect.material.set_texture("uNoiseTex", effect.clock);
}

/// Draw the block-area effect over the already-composited chart.
///
/// `viewport` is the chart rectangle in screen pixels; `scene` is the chart
/// texture the effect samples for its refraction.
pub fn draw(
    zones: &[Zone],
    aspect: f32,
    viewport: (i32, i32, i32, i32),
    scene: Texture2D,
    time: f32,
    dst: (f32, f32, f32, f32),
) {
    if zones.is_empty() || viewport.2 <= 0 || viewport.3 <= 0 {
        return;
    }
    // Debug switch: render the same chart with and without 噪域 for A/B frames.
    if std::env::var_os("PRPR_BLOCK_OFF").is_some() {
        return;
    }
    let mut guard = EFFECT.lock().unwrap();
    let effect = guard.get_or_insert_with(create);

    let grid_w = ((viewport.2 as f32 / BLOCK_PX).floor() as usize).max(8).min(1024);
    let grid_h = ((viewport.3 as f32 / BLOCK_PX).floor() as usize).max(8).min(1024);
    // The mask only drifts with the noise field (0.13 cells/second) and with the
    // zone transforms, so rebuilding it at 30 Hz is visually identical while
    // halving the CPU cost (and the GPU upload) on 60 fps output.
    let mut hash = 0xcbf29ce484222325u64;
    for z in zones {
        for v in [z.center.x, z.center.y, z.half.x, z.half.y, z.angle] {
            hash ^= v.to_bits() as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= z.active as u64 | (z.invert as u64) << 1;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let stale = effect.last_hash != hash || time - effect.last_build_time >= 1.0 / 30.0 - 1e-6;
    let t0 = std::time::Instant::now();
    if stale {
        rebuild(effect, zones, aspect, grid_w, grid_h, time);
        effect.last_hash = hash;
        effect.last_build_time = time;
    }
    let rebuild_ms = if stale { t0.elapsed().as_secs_f64() * 1000. } else { 0. };
    upload_clock(effect, time);
    if std::env::var_os("PRPR_BLOCK_TIME").is_some() {
        effect.build_ms += rebuild_ms;
        effect.build_n += 1;
        if effect.build_n % 120 == 0 {
            let (r, n, u, d, up) = effect.stage_ms;
            eprintln!(
                "[block-time] rebuild avg {:.2}ms over {} frames (last: raster {r:.2} noise {n:.2} upscale {u:.2} dilate {d:.2} upload {up:.2})",
                effect.build_ms / effect.build_n as f64,
                effect.build_n
            );
        }
    }
    if std::env::var_os("PRPR_BLOCK_DEBUG").is_some() {
        let active = effect.image.bytes.chunks_exact(4).filter(|p| p[0] > 0).count();
        let disabled = effect.image.bytes.chunks_exact(4).filter(|p| p[3] > 0).count();
        eprintln!(
            "[block] zones={} active={} disabled={} grid={grid_w}x{grid_h} tex={}x{}",
            zones.len(),
            active,
            disabled,
            effect.tex_w,
            effect.tex_h
        );
        effect.seen += 1;
        if effect.dumped < 3 && effect.seen % 400 == 0 {
            effect.dumped += 1;
            let path = format!("/tmp/ba-mask-{}-{}.raw", effect.seen, effect.dumped);
            let mut bytes = Vec::with_capacity(8 + effect.image.bytes.len());
            bytes.extend_from_slice(&(effect.tex_w as u32).to_le_bytes());
            bytes.extend_from_slice(&(effect.tex_h as u32).to_le_bytes());
            bytes.extend_from_slice(&effect.image.bytes);
            let _ = std::fs::write(&path, bytes);
        }
    }

    let m = effect.material;
    m.set_texture("uScene", scene);
    gl_use_material(m);
    draw_texture_ex(
        effect.white,
        dst.0,
        dst.1,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(dst.2, dst.3)),
            ..Default::default()
        },
    );
    gl_use_default_material();
}

mod shader {
    pub const VERTEX: &str = r#"#version 450 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in uvec4 color0;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 Projection;
    mat4 Model;
    vec4 _Time;
};

layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 color;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    uv = texcoord;
    color = color0 / 255.0;
}"#;

    // Translated from the reference's `FRAG_SRC` (decompiled official shader).
    // Every constant mirrors `BASE_COLORS` / `BASE_FLOATS`; the active layer
    // runs `fullActive()` and the inactive one `disabledColor()` with
    // `_TouchPosCount == 0`.
    pub const FRAGMENT: &str = r#"#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 Projection;
    mat4 Model;
    vec4 _Time;
};

layout(set = 0, binding = 3) uniform sampler2D uMasks;
layout(set = 0, binding = 4) uniform sampler2D uDisplaceTex;
layout(set = 0, binding = 5) uniform sampler2D uSparkTex;
layout(set = 0, binding = 6) uniform sampler2D uNoiseTex;
layout(set = 0, binding = 7) uniform sampler2D uScene;

layout(location = 0) out vec4 fragColor;

const vec3 EDGE_COLOR = vec3(1.0, 0.33018857, 0.33018857);
const float EDGE_OPACITY = 0.8;
const vec3 FILL_COLOR = vec3(0.7132075, 0.23549296, 0.23549296);
const float FILL_STRENGTH = 0.667;
const float FILL_OPACITY = 0.667;
const vec3 GLOW_COLOR = vec3(1.0, 0.17924517, 0.17924517);
const float GLOW_INTENSITY = 0.8;
const vec3 SPARK_TINT = vec3(1.0, 0.28490567, 0.28490567);
const float SPARK_MAP_OPACITY = 5.69;
const float SPARK_HUE_SHIFT = 0.2;
const float SPARK_DISPLACE_INTENSITY = 2.39;
const float DISPLACE_BLEND_INTENSITY = 0.411;
const float DISPLACE_SPEED = 1.5;
const float DISPLACE_STRENGTH = 0.15;
const vec2 DISPLACE_DIRECTION = vec2(1.0, 1.0);
const float BACKGROUND_PIXEL_SCALE = 6.0;

// Disabled layer (`BASE_COLORS` / `BASE_FLOATS` of the official shader).
const vec3 DISABLED_FILL_COLOR = vec3(0.497, 0.13766898, 0.13766898);
const float DISABLED_FILL_OPACITY = 0.4;
const vec3 DISABLED_SPARK_TINT = vec3(0.31132078, 0.077830195, 0.077830195);
const float DISABLED_SPARK_OPACITY = 3.5;
const float DISABLED_SPARK_INTENSITY = 2.29;
const float DISABLED_SPEED = 0.3;

// `_EffectRT_TexelSize.zw * 0.5` — the mask *grid* size (the texture is 2x).
vec2 basePixelUV(vec2 uvv, vec2 effectSize) {
    vec2 size = effectSize * 0.5;
    return (floor(uvv * size) + 0.5) / size;
}

// `snapshotSample()`: the scene is sampled on a screen/3 pixel grid.
vec4 snapshotSample(vec2 uvv, vec2 screenSize) {
    vec2 size = max(floor(screenSize / 3.0), vec2(1.0));
    vec2 p = (floor(clamp(uvv, vec2(0.0), vec2(1.0)) * size) + 0.5) / size;
    return texture(uScene, p);
}

// Chart time, uploaded in the 1x1 clock texture (see `upload_clock`).
float chartTime() {
    vec4 b = texture(uNoiseTex, vec2(0.5));
    uint bits = uint(floor(b.r * 255.0 + 0.5))
              | (uint(floor(b.g * 255.0 + 0.5)) << 8)
              | (uint(floor(b.b * 255.0 + 0.5)) << 16)
              | (uint(floor(b.a * 255.0 + 0.5)) << 24);
    return uintBitsToFloat(bits);
}

// Inactive block area: the plain rectangle, no displacement and no textures.
// It is the base layer, so an active block can be composited on top of it.
vec4 disabledColor(float mask) {
    if (mask < 0.0001) { return vec4(0.0); }
    vec3 col = DISABLED_FILL_COLOR * DISABLED_FILL_OPACITY;
    // Slightly washed out: the disabled block reads as a faded version of the
    // active one (+9% white).
    col = mix(col, vec3(1.0), 0.09);
    // Same opacity as the active fill layer.
    return vec4(mask * col, mask * FILL_OPACITY);
}

vec4 fullActive(vec2 fieldUV, float time, vec2 screenSize, vec2 effectSize) {
    vec2 sceneUV = fieldUV;
    vec2 displaceUV = fieldUV * vec2(0.8, 0.3);
    vec2 sparkUV = fieldUV * vec2(3.0, 1.2);

    // composeSample() / effectSample(): fill mask on the cell grid, edge and
    // glow sampled from the mask itself.
    float fill = texture(uMasks, basePixelUV(fieldUV, effectSize)).r;
    vec2 pxUV = (floor(fieldUV * effectSize) + 0.5) / effectSize;
    float edge = texture(uMasks, pxUV).g;
    float glow = texture(uMasks, fieldUV).b;

    float edgePlusFill = edge + fill;
    float total = glow + edgePlusFill;
    if (total <= 0.0001) { discard; }

    float edgeA = edge * EDGE_OPACITY;
    float glowW = glow * (1.0 - edgePlusFill);
    float glowTerm = glowW * GLOW_INTENSITY;

    // --- displace: two mirrored samples snapped to `backgroundPixelScale` ---
    vec2 dir = normalize(DISPLACE_DIRECTION);
    float dt = (time / 20.0) * DISPLACE_SPEED;
    float scale = max(BACKGROUND_PIXEL_SCALE, 1.0);
    vec2 dispA = displaceUV + dir * dt;
    vec2 dispB = displaceUV + vec2(-dir.y, dir.x) * dt;
    vec2 snapA = (floor(dispA * screenSize / scale) * scale + scale * 0.5) / screenSize;
    vec2 snapB = (floor(dispB * screenSize / scale) * scale + scale * 0.5) / screenSize;
    float dA = texture(uDisplaceTex, snapA).x;
    float dB = texture(uDisplaceTex, snapB).x;
    float avgD = (dA + dB) * 0.5;
    vec2 off = dir * (dA - 0.5) + vec2(-dir.y, dir.x) * (dB - 0.5);

    vec2 scenePx = (floor(sceneUV * screenSize / scale) * scale + scale * 0.5) / screenSize;
    vec2 sceneUv = off * DISPLACE_STRENGTH + scenePx;
    vec2 sparkUv = off * SPARK_DISPLACE_INTENSITY + sparkUV;

    float sparkV = texture(uSparkTex, sparkUv).x;
    vec3 sparkCol = vec3(sparkV) * SPARK_TINT;
    vec3 sceneCol = snapshotSample(sceneUv, screenSize).xyz;

    vec3 sparkTerm = avgD * sparkCol;
    vec3 sparkScaled = sparkTerm * SPARK_MAP_OPACITY;

    // --- RGB -> HSV of the scene colour (official ActiveBlock hue shift) ---
    float pick = (sceneCol.y >= sceneCol.z) ? 1.0 : 0.0;
    vec2 t2 = pick * (sceneCol.yz + vec2(-sceneCol.z, -sceneCol.y)) + sceneCol.zy;
    vec2 t3 = pick * vec2(1.0, -1.0) + vec2(-1.0, 0.666666687);
    float pick2 = (sceneCol.x >= t2.x) ? 1.0 : 0.0;
    vec3 t7 = vec3(-t2.x, -t2.y, -t3.y);
    float t8x = sceneCol.x + t7.x;
    vec3 t8yzw = vec3(t2.y, t3.x, t2.x) + vec3(t7.y, t7.z, -sceneCol.x);
    vec3 t21 = pick2 * vec3(t8x, t8yzw.x, t8yzw.y) + vec3(t2.x, t2.y, t3.y);
    float t5x = pick2 * t8yzw.z + sceneCol.x;
    float mn = min(t21.y, t5x);
    float delta = t21.x - mn;
    float maxc = t5x - t21.y;
    float hue = sparkScaled.x * SPARK_HUE_SHIFT + abs(maxc / (delta * 6.0 + 1.00000001e-10) + t21.z);
    float sat = sparkScaled.y * SPARK_HUE_SHIFT + delta / (t21.x + 1.00000001e-10);
    float val = sparkScaled.z * SPARK_HUE_SHIFT + t21.x;

    // --- HSV -> RGB ---
    vec3 hsv = fract(vec3(hue) + vec3(1.0, 0.666666687, 0.333333343));
    hsv = hsv * 6.0 - 3.0;
    hsv = clamp(abs(hsv) - 1.0, 0.0, 1.0) - 1.0;
    vec3 c1 = vec3(sat) * hsv + 1.0;
    vec3 c2 = c1 * val;
    vec3 c3 = c2 * 2.0;
    vec3 c4 = sparkScaled * c3;
    vec3 c5 = (-val) * c1 + 1.0;
    c5 = c5 * 2.0;
    vec3 c6 = (-sparkTerm) * SPARK_MAP_OPACITY + 1.0;
    c6 = (-c5) * c6 + 1.0;
    vec3 sel = step(vec3(0.5), c2);
    vec3 sparkOut = clamp(sel * ((-c3) * sparkScaled + c6) + c4, 0.0, 1.0);

    // --- final compose ---
    vec3 fillBase = (-avgD) * DISPLACE_BLEND_INTENSITY + FILL_COLOR;
    vec3 fillCol = FILL_STRENGTH * (sparkOut - fillBase) + fillBase;
    vec3 edgeGlow = glowTerm * GLOW_COLOR + EDGE_COLOR * edgeA;
    vec3 outCol = fillCol * fill + edgeGlow;
    // The reference relies on GL clamping gl_FragColor to [0, 1]; without that,
    // fill + edge can exceed 1 and `1 - alpha` goes negative, which crushes the
    // destination to black.
    float outA = min(glowW * GLOW_INTENSITY + (fill * FILL_OPACITY + edgeA), 1.0);
    return vec4(outCol, outA);
}

void main() {
    vec2 effectSize = vec2(textureSize(uMasks, 0));
    vec2 screenSize = vec2(textureSize(uScene, 0));
    vec2 pxUV = (floor(uv * effectSize) + 0.5) / effectSize;
    vec4 mask = texture(uMasks, basePixelUV(uv, effectSize));
    float edge = texture(uMasks, pxUV).g;
    float glow = texture(uMasks, uv).b;
    float activeMask = mask.r + edge + glow;

    // The inactive fill is the base layer: an active block (including its edge
    // and glow margin) is drawn *over* it instead of replacing it, so the
    // margin around an active area no longer punches a hole into the inactive
    // areas underneath.
    vec4 result = vec4(0.0);
    if (mask.a > 0.0001) {
        result = disabledColor(mask.a);
    }
    if (activeMask > 0.0001) {
        vec4 act = fullActive(uv, chartTime(), screenSize, effectSize);
        result = act + result * (1.0 - act.a);
    }
    if (result.a <= 0.0001) { discard; }
    fragColor = vec4(result.rgb, clamp(result.a, 0.0, 1.0)) * color;
}"#;
}
