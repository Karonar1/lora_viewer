use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    str::FromStr,
};

use eframe::egui;
use egui_file::FileDialog;
use serde::{Deserialize, Serialize};

use crate::metadata::LoraData;

#[derive(Default, Serialize, Deserialize)]
pub struct App {
    lora_file: Option<PathBuf>,
    #[serde(skip)]
    open_dialog: Option<FileDialog>,
    #[serde(skip)]
    metadata: Option<LoraData>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<String>) -> App {
        if let Some(path) = path {
            App {
                lora_file: PathBuf::from_str(&path).ok(),
                ..Default::default()
            }
        } else {
            if let Some(storage) = cc.storage {
                eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
            } else {
                Default::default()
            }
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open LoRA").clicked() {
                        let filter = Box::new({
                            let ext = Some(OsStr::new("safetensors"));
                            move |path: &Path| -> bool { path.extension() == ext }
                        });
                        let mut dialog = FileDialog::open_file(
                            self.lora_file
                                .as_ref()
                                .and_then(|path| path.parent().map(|p| p.to_path_buf())),
                        )
                        .show_files_filter(filter);
                        dialog.open();
                        self.open_dialog = Some(dialog);
                    }
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.add_space(16.0);
            });

            if let Some(dialog) = &mut self.open_dialog {
                if dialog.show(ctx).selected() {
                    if let Some(path) = dialog.path() {
                        self.lora_file = Some(path.to_path_buf());
                        self.metadata = None;
                    }
                }
            }
        });

        if self.metadata.is_none() {
            if let Some(lora) = &self.lora_file {
                if let Ok(buffer) = std::fs::read(lora) {
                    self.metadata = Some(LoraData::from_buffer(&buffer).unwrap_or_default());
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("LoRA Metadata Viewer");

            ui.horizontal(|ui| {
                ui.label("LoRA name: ");
                if let Some(file) = &self.lora_file {
                    ui.label(file.file_stem().unwrap().to_string_lossy());
                }
            });

            ui.horizontal(|ui| {
                ui.label("Base model: ");
                if let Some(metadata) = &self.metadata {
                    ui.label(
                        metadata
                            .base_model
                            .as_ref()
                            .unwrap_or(&"Unknown".to_string()),
                    );
                }
            });

            ui.separator();

            if let Some(metadata) = &self.metadata {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Grid::new("tags")
                            .num_columns(3)
                            .striped(true)
                            .show(ui, |ui| {
                                for (tag, freq) in &metadata.tag_frequencies {
                                    ui.label(tag);
                                    ui.label(format!("{freq}"));
                                    ui.allocate_space(egui::vec2(ui.available_width(), 0.0));
                                    ui.end_row();
                                }
                            });
                    });
            }
        });
    }
}
