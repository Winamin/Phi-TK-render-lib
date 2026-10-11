use macroquad::{
    texture::{RenderTarget, Texture2D},
    window::get_internal_gl,
    miniquad::{gl::GLuint, RenderPass, Texture, TextureFormat, TextureParams},
};
use once_cell::sync::Lazy;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static C: AtomicU64 = AtomicU64::new(0);

pub fn begin_readback_cycle() {
    C.fetch_add(1, Ordering::Relaxed);
}

struct Rb {
    cache: Option<(u64, GLuint, usize, Vec<f32>)>,
    staging: Vec<u8>,
    fbo: GLuint,
    tex: GLuint,
    sz: (u32, u32),
}

static RB: Lazy<Mutex<Rb>> = Lazy::new(|| {
    Mutex::new(Rb { cache: None, staging: Vec::new(), fbo: 0, tex: 0, sz: (0, 0) })
});

pub struct MSRenderTarget {
    dim: (u32, u32),
    fbo: GLuint,
    rbo: GLuint,
    inp: Option<RenderTarget>,
    out: [RenderTarget; 2],
}

pub fn copy_fbo(src: GLuint, dst: GLuint, dim: (u32, u32)) -> bool {
    unsafe {
        use macroquad::miniquad::gl::*;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
        let (w, h) = (dim.0 as i32, dim.1 as i32);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glGetError() == GL_NO_ERROR
    }
}

pub fn internal_id(target: &RenderTarget) -> GLuint {
    target.render_pass.gl_internal_id(unsafe { get_internal_gl() }.quad_context)
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

fn mk_ms(dim: (u32, u32), mut s: u32) -> (GLuint, GLuint) {
    unsafe {
        use macroquad::miniquad::gl::*;
        let (mut fbo, mut rbo) = (0, 0);
        while s > 1 {
            while glGetError() != GL_NO_ERROR {}
            let mut r = 0;
            glGenRenderbuffers(1, &mut r);
            glBindRenderbuffer(GL_RENDERBUFFER, r);
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, s as i32, GL_RGB8, dim.0 as i32, dim.1 as i32);
            let mut f = 0;
            glGenFramebuffers(1, &mut f);
            glBindFramebuffer(GL_FRAMEBUFFER, f);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, r);
            let k = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindRenderbuffer(GL_RENDERBUFFER, 0);
            if k == GL_FRAMEBUFFER_COMPLETE {
                fbo = f;
                rbo = r;
                break;
            }
            glDeleteFramebuffers(1, &f);
            glDeleteRenderbuffers(1, &r);
            s = s.div_ceil(2);
        }
        (fbo, rbo)
    }
}

impl MSRenderTarget {
    pub fn new(dim: (u32, u32), samples: u32) -> Self {
        let (fbo, rbo) = mk_ms(dim, samples);
        let o0 = new_rt(dim.0, dim.1);
        let o1 = new_rt(dim.0, dim.1);

        let inp = if fbo != 0 {
            let gl = unsafe { get_internal_gl() };
            let t = o0.texture.raw_miniquad_texture_handle();
            let p = RenderPass::from_raw(gl.quad_context, fbo, t);
            Some(RenderTarget { texture: o0.texture, render_pass: p })
        } else {
            None
        };

        Self { dim, fbo, rbo, inp, out: [o0, o1] }
    }

    pub fn blit(&self) {
        if self.fbo == 0 {
            return;
        }
        copy_fbo(self.fbo, internal_id(&self.out[0]), self.dim);
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
        let fbo = internal_id(&self.out[0]);
        let len = target_w * target_h * 3;

        let mut st = RB.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((cc, cf, cl, d)) = st.cache.as_ref() {
            if *cc == cycle && *cf == fbo && *cl == len {
                return d.clone();
            }
        }
        let fresh = self.raw(&mut st, target_w, target_h);
        st.cache = Some((cycle, fbo, len, fresh.clone()));
        fresh
    }

    fn raw(&self, st: &mut Rb, tw: usize, th: usize) -> Vec<f32> {
        let (w, h) = self.dim;
        let n = tw * th * 3;

        unsafe {
            use macroquad::miniquad::gl::*;
            if st.sz != (tw as u32, th as u32) {
                if st.tex != 0 {
                    glDeleteTextures(1, &st.tex);
                    st.tex = 0;
                }
                if st.fbo != 0 {
                    glDeleteFramebuffers(1, &st.fbo);
                    st.fbo = 0;
                }
                let mut t = 0;
                glGenTextures(1, &mut t);
                glBindTexture(GL_TEXTURE_2D, t);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as i32);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as i32);
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGB8 as i32,
                    tw as i32,
                    th as i32,
                    0,
                    GL_RGB,
                    GL_UNSIGNED_BYTE,
                    ptr::null(),
                );
                let mut f = 0;
                glGenFramebuffers(1, &mut f);
                glBindFramebuffer(GL_FRAMEBUFFER, f);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
                st.tex = t;
                st.fbo = f;
                st.sz = (tw as u32, th as u32);
            }
            glBindFramebuffer(GL_READ_FRAMEBUFFER, internal_id(&self.out[0]));
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, st.fbo);
            glBlitFramebuffer(
                0,
                0,
                w as i32,
                h as i32,
                0,
                0,
                tw as i32,
                th as i32,
                GL_COLOR_BUFFER_BIT,
                GL_LINEAR,
            );
            glBindFramebuffer(GL_READ_FRAMEBUFFER, st.fbo);
            st.staging.clear();
            st.staging.resize(n, 0);
            glReadPixels(0, 0, tw as i32, th as i32, GL_RGB, GL_UNSIGNED_BYTE, st.staging.as_mut_ptr() as *mut _);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        }

        st.staging.iter().map(|&v| v as f32 / 255.0).collect()
    }
}

impl Drop for MSRenderTarget {
    fn drop(&mut self) {
        unsafe {
            use macroquad::miniquad::gl::*;
            if self.rbo != 0 {
                glDeleteRenderbuffers(1, &self.rbo);
            }
            if self.fbo != 0 {
                glDeleteFramebuffers(1, &self.fbo);
            }
        }
        self.out[0].delete();
        self.out[1].delete();
    }
}