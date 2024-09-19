//#![windows_subsystem = "windows"]

use clap::Parser;
use eframe::egui;

mod app;
mod metadata;

#[derive(Parser)]
struct Args {
    path: Option<String>,
}

fn main() -> eframe::Result {
    let args = Args::parse();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 600.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };
    eframe::run_native(
        "LoRA Explorer",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, args.path)))),
    )
}
