pub mod bin;
pub mod config;
pub mod core;
pub mod dir;
pub mod ext;
pub mod fs;
pub mod info;
pub mod judge;
pub mod l10n;
pub mod parse;
pub mod particle;
pub mod scene;
mod shader;
pub mod task;
pub mod time;
pub mod ui;

pub mod hand;
pub mod hand_model;
pub mod loss;

pub mod gpu_vit;
#[cfg(feature = "log")]
pub mod log;

#[cfg(feature = "closed")]
pub mod inner;

#[cfg(target_os = "ios")]
pub mod objc;

pub use scene::Main;

/// Own this before constructing scenes and drop it while the graphics context
/// is still current. Android may create another preview in the same process.
pub struct GraphicsSession;

impl Drop for GraphicsSession {
    fn drop(&mut self) {
        scene::DIALOG.with(|dialog| { dialog.borrow_mut().take(); });
        scene::BILLBOARD.with(|billboard| {
            *billboard.borrow_mut() = (ui::BillBoard::new(), time::TimeManager::default());
        });
        ui::release_graphics();
        core::release_graphics();
        ext::release_graphics();
    }
}

pub fn build_conf() -> macroquad::window::Conf {
    macroquad::window::Conf {
        window_title: "Phi TK".to_string(),
        window_width: 600,
        window_height: 800,
        ..Default::default()
    }
}
