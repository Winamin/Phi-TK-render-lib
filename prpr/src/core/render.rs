use macroquad::{
    texture::{RenderTarget, Texture2D},
    window::get_internal_gl,
    miniquad::{RenderPass, Texture, TextureFormat, TextureParams},
};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static C: AtomicU64 = AtomicU64::new(0);

pub fn begin_readback_cycle() {
    C.fetch_add(1, Ordering::Relaxed);
}

struct Rb {
    cache: Option<(u64, usize, Vec<f32>)>,
    target: Option<RenderTarget>,
    staging: Vec<u8>,
}

static RB: Lazy<Mutex<Rb>> = Lazy::new(|| Mutex::new(Rb { cache: None, target: None, staging: Vec::new() }));

pub struct MSRenderTarget {
    dim: (u32, u32),
    /// Multisampled pass resolving into `out[0]`; absent when the backend
    /// cannot multisample, in which case drawing goes to `out[0]` directly.
    inp: Option<RenderTarget>,
    out: [RenderTarget; 2],
}

fn new_rt(w: u32, h: u32) -> RenderTarget {
    let gl = unsafe { get_internal_gl() };
    let t = Texture::new_render_texture(
        gl.quad_context,
        TextureParams {
            width: w,
            height: h,
            format: TextureFormat::RGB8,
            ..Default::default()
        },
    );
    let p = RenderPass::new(gl.quad_context, t, None);
    RenderTarget { texture: Texture2D::from_miniquad_texture(t), render_pass: p }
}

impl MSRenderTarget {
    pub fn new(dim: (u32, u32), samples: u32) -> Self {
        let gl = unsafe { get_internal_gl() };
        let o0 = new_rt(dim.0, dim.1);
        let o1 = new_rt(dim.0, dim.1);

        let inp = if samples > 1 {
            let handle = o0.texture.raw_miniquad_texture_handle();
            let pass = RenderPass::new_multisampled(gl.quad_context, handle, samples);
            Some(RenderTarget { texture: o0.texture, render_pass: pass })
        } else {
            None
        };

        Self { dim, inp, out: [o0, o1] }
    }

    pub fn swap(&mut self) {
        self.out.swap(0, 1);
    }

    pub fn input(&self) -> RenderTarget {
        self.inp.unwrap_or(self.out[0])
    }

    pub fn output(&self) -> RenderTarget {
        self.out[0]
    }

    pub fn old(&self) -> RenderTarget {
        self.out[1]
    }

    pub fn read_pixels_resized(&self, target_w: usize, target_h: usize) -> Vec<f32> {
        let cycle = C.load(Ordering::Relaxed);
        let len = target_w * target_h * 3;

        let mut st = RB.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((cc, cl, d)) = st.cache.as_ref() {
            if *cc == cycle && *cl == len {
                return d.clone();
            }
        }
        let fresh = self.raw(&mut st, target_w, target_h);
        st.cache = Some((cycle, len, fresh.clone()));
        fresh
    }

    fn raw(&self, st: &mut Rb, tw: usize, th: usize) -> Vec<f32> {
        let gl = unsafe { get_internal_gl() };
        let rgba = tw * th * 4;

        let size_matches = st
            .target
            .as_ref()
            .map_or(false, |target| (target.texture.width(), target.texture.height()) == (tw as f32, th as f32));
        if !size_matches {
            if let Some(target) = st.target.take() {
                target.delete();
            }
            let texture = Texture::new_render_texture(
                gl.quad_context,
                TextureParams {
                    width: tw as u32,
                    height: th as u32,
                    format: TextureFormat::RGBA8,
                    ..Default::default()
                },
            );
            let render_pass = RenderPass::new(gl.quad_context, texture, None);
            st.target = Some(RenderTarget {
                texture: Texture2D::from_miniquad_texture(texture),
                render_pass,
            });
        }

        let target = st.target.as_ref().unwrap();
        gl.quad_context.copy_render_target(self.output().render_pass, target.render_pass);

        st.staging.clear();
        st.staging.resize(rgba, 0);
        target.texture.raw_miniquad_texture_handle().read_pixels(gl.quad_context, &mut st.staging);

        st.staging
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0])
            .collect()
    }
}

impl Drop for MSRenderTarget {
    fn drop(&mut self) {
        self.out[0].delete();
        self.out[1].delete();
    }
}