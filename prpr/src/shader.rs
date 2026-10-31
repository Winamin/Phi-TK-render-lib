//! Compile the library's Vulkan-style shaders for the active graphics backend.
//! GLES uses named uniforms (the miniquad GL uniform API), not descriptor sets.
use macroquad::{material::{Material, MaterialParams}, miniquad::{conf::RenderingBackend, Context}};
use once_cell::sync::Lazy;
use regex::Regex;
use std::borrow::Cow;

pub(crate) fn source<'a>(ctx: &Context, source: &'a str) -> Cow<'a, str> {
    if ctx.rendering_backend() == RenderingBackend::Vulkan {
        return Cow::Borrowed(source);
    }
    opengl_source(source).into()
}

fn opengl_source(source: &str) -> String {
    static LAYOUT: Lazy<Regex> = Lazy::new(|| Regex::new(r"layout\s*\([^)]*\)\s*").unwrap());
    static BLOCK: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)uniform\s+Uniforms\s*\{([^}]*)\}\s*;").unwrap());
    static PRECISION: Lazy<Regex> = Lazy::new(|| Regex::new(r"precision\s+(lowp|mediump|highp)\s+(float|int)\s*;").unwrap());
    let source = LAYOUT.replace_all(source, "");
    let source = BLOCK.replace_all(&source, |caps: &regex::Captures| {
        caps[1].split(';').filter_map(|member| {
            let member = member.trim();
            (!member.is_empty()).then(|| format!("uniform {member};\n"))
        }).collect::<String>()
    });
    // GL's UByte4 vertex attribute uses glVertexAttribPointer, which delivers floats.
    let source = PRECISION.replace_all(&source, "").replace("in uvec4 color0", "in vec4 color0");
    let source = source.trim_start();
    let body = if source.starts_with("#version") {
        source.split_once('\n').map_or("", |(_, body)| body)
    } else { source };
    // Chart packs also contain GLSL ES 1.00 fragment effects. Match their
    // interface to the ES 3.00 vertex stage used by the renderer.
    static LEGACY: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(varying|attribute|texture2D|gl_FragColor)\b").unwrap());
    let fragment_output = if body.contains("gl_FragColor") { "out vec4 prpr_FragColor;\n" } else { "" };
    let body = LEGACY.replace_all(body, |caps: &regex::Captures| match &caps[1] {
        "varying" | "attribute" => "in",
        "texture2D" => "texture",
        _ => "prpr_FragColor",
    });
    let version = if cfg!(any(target_os = "android", target_os = "ios", target_arch = "wasm32")) {
        "#version 300 es\nprecision highp float;\nprecision highp int;\n"
    } else {
        "#version 330 core\n"
    };
    format!("{version}{fragment_output}{body}")
}

pub(crate) fn load_material(vertex: &str, fragment: &str, params: MaterialParams) -> Result<Material, macroquad::miniquad::ShaderError> {
    let ctx = unsafe { macroquad::window::get_internal_gl() }.quad_context;
    let vertex = source(ctx, vertex);
    let fragment = source(ctx, fragment);
    macroquad::material::load_material(&vertex, &fragment, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gles_uniforms_match_the_named_gl_api() {
        let gl = opengl_source("#version 450 core\nlayout(location = 2) in uvec4 color0;\nlayout(set = 0, binding = 0) uniform Uniforms { mat4 Projection; vec4 _Time; };\nlayout(set = 0, binding = 1) uniform sampler2D Texture;\nvoid main() {}\n");
        assert!(!gl.contains("layout("));
        assert!(!gl.contains("uniform Uniforms"));
        assert!(gl.contains("uniform mat4 Projection;"));
        assert!(gl.contains("uniform vec4 _Time;"));
        assert!(gl.contains("uniform sampler2D Texture;"));
        assert!(gl.contains("in vec4 color0;"));
    }

    #[test]
    fn legacy_chart_effects_use_the_same_interface_as_the_vertex_stage() {
        let gl = opengl_source("#version 100\nprecision mediump float;\nvarying vec2 uv;\nuniform sampler2D screenTexture;\nvoid main() { gl_FragColor = texture2D(screenTexture, uv); }");
        assert!(gl.contains("in vec2 uv;"));
        assert!(gl.contains("out vec4 prpr_FragColor;"));
        assert!(gl.contains("prpr_FragColor = texture(screenTexture, uv)"));
        assert!(!gl.contains("mediump"));
    }
}
