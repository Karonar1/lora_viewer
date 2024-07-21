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
    selected: usize,
    #[serde(skip)]
    open_dialog: Option<FileDialog>,
    #[serde(skip)]
    metadata: Option<Vec<(PathBuf, Option<LoraData>)>>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<String>) -> App {
        if let Some(path) = path {
            App {
                lora_file: PathBuf::from_str(&path).ok(),
                ..Default::default()
            }
        } else if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // Menu bar
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open LoRA").clicked() {
                        let filter = Box::new({
                            let ext = Some(OsStr::new("safetensors"));
                            move |path: &Path| -> bool { path.extension() == ext }
                        });
                        let mut dialog =
                            FileDialog::open_file(self.lora_file.as_ref().and_then(|path| {
                                if path.is_file() {
                                    path.parent().map(|p| p.to_path_buf())
                                } else {
                                    Some(path.to_path_buf())
                                }
                            }))
                            .show_files_filter(filter);
                        dialog.open();
                        self.open_dialog = Some(dialog);
                    }
                    if ui.button("Scan directory").clicked() {
                        let filter = Box::new(|path: &Path| -> bool { path.is_dir() });
                        let mut dialog =
                            FileDialog::select_folder(self.lora_file.as_ref().and_then(|path| {
                                if path.is_file() {
                                    path.parent().map(|p| p.to_path_buf())
                                } else {
                                    Some(path.to_path_buf())
                                }
                            }))
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

            // Set path and clear metadata if open dialog has finished
            if let Some(dialog) = &mut self.open_dialog {
                if dialog.show(ctx).selected() {
                    if let Some(path) = dialog.path() {
                        self.lora_file = Some(path.to_path_buf());
                        self.metadata = None;
                    }
                }
            }
        });

        // Populate the metadata record if it's empty and we have a path defined
        if self.metadata.is_none() {
            if let Some(lora) = &self.lora_file {
                if lora.is_file() {
                    // If the path is a single file, we just have one record
                    self.metadata = Some(vec![(lora.clone(), None)]);
                } else if lora.is_dir() {
                    // Otherwise scan the directory and add all safetensors files
                    if let Ok(files) = std::fs::read_dir(lora) {
                        let ext = Some(OsStr::new("safetensors"));
                        self.metadata = Some(
                            files
                                .into_iter()
                                .filter_map(|f| {
                                    f.ok().and_then(|f| {
                                        let path = f.path();
                                        if path.is_file() && path.extension() == ext {
                                            Some((path, None))
                                        } else {
                                            None
                                        }
                                    })
                                })
                                .collect(),
                        );
                    }
                }
                self.selected = 0;
            }
        }

        // We may now assume that if a path is defined, metadata is valid

        // If our path is to a directory, add a side panel to select LoRAs
        if let Some(path) = &self.lora_file {
            if path.is_dir() {
                egui::SidePanel::left("left_panel").show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let selected = self.selected;

                            if let Some(metadata) = &self.metadata {
                                for (index, (path, _)) in metadata.iter().enumerate() {
                                    if ui
                                        .add(egui::widgets::SelectableLabel::new(
                                            index == selected,
                                            path.file_name().unwrap().to_string_lossy(),
                                        ))
                                        .clicked()
                                    {
                                        self.selected = index;
                                    };
                                }
                            }
                        });
                });
            }
        }

        // Ensure that the metadata for the selected LoRA is actually loaded
        if let Some(ref mut metadata) = &mut self.metadata {
            if self.selected < metadata.len() && metadata[self.selected].1.is_none() {
                let path = &metadata[self.selected].0;
                if let Ok(buffer) = std::fs::read(path) {
                    metadata[self.selected].1 =
                        Some(LoraData::from_buffer(&buffer).unwrap_or_default());
                } else {
                    metadata[self.selected].1 = Some(Default::default());
                }
            }
        }

        // Get a reference to the selected entry, if it exists. The metadata is guaranteed to be
        // defined, even if the file couldn't be loaded
        let selected = self.metadata.as_ref().and_then(|m| m.get(self.selected));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("LoRA Metadata Viewer");

            ui.horizontal(|ui| {
                ui.label("LoRA name: ");
                if let Some(metadata) = selected {
                    ui.label(metadata.0.file_stem().unwrap().to_string_lossy());
                }
            });

            ui.horizontal(|ui| {
                ui.label("Base model: ");
                if let Some(metadata) = selected {
                    let metadata = metadata.1.as_ref().unwrap();
                    ui.label(
                        metadata
                            .base_model
                            .as_ref()
                            .unwrap_or(&"Unknown".to_string()),
                    );
                }
            });

            ui.separator();

            if let Some(metadata) = selected {
                let metadata = metadata.1.as_ref().unwrap();
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
