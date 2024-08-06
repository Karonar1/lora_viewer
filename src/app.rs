use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        mpsc::{channel, Sender},
        Arc, LazyLock, Mutex,
    },
    thread,
};

use eframe::egui::{self, TextEdit};
use egui_file::FileDialog;
use serde::{Deserialize, Serialize};

use crate::metadata::{read_header, LoraData};

type MetadataRecord = (
    PathBuf,
    LazyLock<LoraData, Box<dyn FnOnce() -> LoraData + Send + Sync + 'static>>,
);
type MetadataStore = Arc<Vec<MetadataRecord>>;

fn metadata_record(path: &Path) -> MetadataRecord {
    let path = path.to_path_buf();
    (
        path.clone(),
        LazyLock::new(Box::new(move || {
            if let Ok(buffer) = read_header(&path) {
                LoraData::from_buffer(&buffer).unwrap_or_default()
            } else {
                Default::default()
            }
        })),
    )
}

#[derive(Eq, PartialEq)]
enum SearchResult {
    NoMatch,
    Name,
    Tag,
}

#[derive(Default, Serialize, Deserialize)]
pub struct App {
    lora_file: Option<PathBuf>,
    selected: usize,
    #[serde(skip)]
    open_dialog: Option<FileDialog>,
    #[serde(skip)]
    metadata: Option<MetadataStore>,
    metadata_dialog: bool,
    tensors_dialog: bool,
    #[serde(skip)]
    background_loader: Option<Sender<MetadataStore>>,
    #[serde(skip)]
    loader_state: Arc<Mutex<(usize, usize)>>,
    search_text: String,
    #[serde(skip)]
    search_results: Option<Vec<SearchResult>>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<String>) -> App {
        let mut app = if let Some(path) = path {
            App {
                lora_file: PathBuf::from_str(&path).ok(),
                ..Default::default()
            }
        } else if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };
        let (send, recv) = channel();
        app.background_loader = Some(send);

        let ctx = cc.egui_ctx.clone();
        let state = app.loader_state.clone();
        thread::spawn(move || loop {
            let mut store = recv.recv().unwrap();
            let mut i = 0;
            while i < store.len() {
                LazyLock::force(&store[i].1);
                i += 1;
                *state.lock().unwrap() = (i, store.len());
                ctx.request_repaint();
                match recv.try_recv() {
                    Ok(new_store) => {
                        store = new_store;
                        i = 0;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => panic!(),
                    Err(_) => (),
                }
            }
        });
        app
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
                    if ui.button("Open model").clicked() {
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
                        ui.close_menu();
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
                        ui.close_menu();
                    }
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        ui.close_menu();
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
                        self.search_results = None;
                        *self.loader_state.lock().unwrap() = (0, 0);
                        self.selected = 0;
                        self.metadata_dialog = false;
                        self.tensors_dialog = false;
                    }
                }
            }
        });

        // Populate the metadata record if it's empty and we have a path defined
        if self.metadata.is_none() {
            if let Some(lora) = &self.lora_file {
                if lora.is_file() {
                    // If the path is a single file, we just have one record
                    let metadata = Arc::new(vec![metadata_record(lora)]);
                    if let Some(loader) = &self.background_loader {
                        loader.send(metadata.clone()).ok();
                    }
                    self.metadata = Some(metadata);
                } else if lora.is_dir() {
                    // Otherwise scan the directory and add all safetensors files
                    if let Ok(files) = std::fs::read_dir(lora) {
                        let ext = Some(OsStr::new("safetensors"));
                        let metadata = Arc::new({
                            let mut files: Vec<_> = files
                                .into_iter()
                                .filter_map(|f| {
                                    f.ok().and_then(|f| {
                                        let path = f.path();
                                        if path.is_file() && path.extension() == ext {
                                            Some(metadata_record(&path))
                                        } else {
                                            None
                                        }
                                    })
                                })
                                .collect();
                            files.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            files
                        });
                        if let Some(loader) = &self.background_loader {
                            loader.send(metadata.clone()).ok();
                        }
                        self.metadata = Some(metadata);
                    }
                }
            }
        }

        // If our path is to a directory, add a side panel to select LoRAs
        if let Some(path) = &self.lora_file {
            if path.is_dir() {
                egui::SidePanel::left("left_panel").show(ctx, |ui| {
                    let (loaded, total) = *self.loader_state.lock().unwrap();
                    if loaded < total {
                        ui.label(format!("Scanning {loaded} / {total}"));
                        ui.separator();
                    } else if ui
                        .add(TextEdit::singleline(&mut self.search_text))
                        .changed()
                    {
                        self.search_results = None;
                    }

                    if let Some(metadata) = &self.metadata {
                        if loaded < total || total == 0 {
                            self.search_results = None;
                        } else if self.search_results.is_none() {
                            self.search_results = Some(
                                metadata
                                    .iter()
                                    .map(|model| {
                                        let name_match = model
                                            .0
                                            .file_name()
                                            .and_then(|s| {
                                                s.to_str().map(|s| {
                                                    s.to_ascii_lowercase().contains(
                                                        &self.search_text.to_ascii_lowercase(),
                                                    )
                                                })
                                            })
                                            .unwrap_or(false);
                                        let tag_match =
                                            model.1.tag_frequencies.iter().any(|(tag, _)| {
                                                tag.to_ascii_lowercase().contains(
                                                    &self.search_text.to_ascii_lowercase(),
                                                )
                                            });
                                        if name_match {
                                            SearchResult::Name
                                        } else if tag_match {
                                            SearchResult::Tag
                                        } else {
                                            SearchResult::NoMatch
                                        }
                                    })
                                    .collect(),
                            );
                        }
                    }

                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let selected = self.selected;

                            if let Some(metadata) = &self.metadata {
                                for (index, (path, _)) in metadata.iter().enumerate() {
                                    if self.search_results.is_none()
                                        || self.search_results.as_ref().unwrap()[index]
                                            != SearchResult::NoMatch
                                    {
                                        if ui
                                            .add(egui::widgets::SelectableLabel::new(
                                                index == selected,
                                                path.file_name().unwrap().to_string_lossy(),
                                            ))
                                            .clicked()
                                        {
                                            self.selected = index;
                                            self.metadata_dialog = false;
                                            self.tensors_dialog = false;
                                        };
                                    }
                                }
                            }
                        });
                });
            }
        }

        // Get a reference to the selected entry, if it exists. The metadata is guaranteed to be
        // defined, even if the file couldn't be loaded
        let selected = self.metadata.as_ref().and_then(|m| m.get(self.selected));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("LoRA Metadata Viewer");

            ui.horizontal(|ui| {
                ui.label("Model name: ");
                if let Some(metadata) = selected {
                    ui.label(metadata.0.file_stem().unwrap().to_string_lossy());
                    if ui.button("Full metadata").clicked() {
                        self.metadata_dialog = true;
                    }
                    if ui.button("Tensors").clicked() {
                        self.tensors_dialog = true;
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Model type: ");
                if let Some(metadata) = selected {
                    for model in &metadata.1.model_types {
                        ui.label(model.to_string());
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Base checkpoint: ");
                if let Some((_, metadata)) = selected {
                    ui.label(
                        metadata
                            .base_model
                            .as_ref()
                            .unwrap_or(&"Unknown".to_string()),
                    );
                }
            });

            ui.separator();

            if let Some((_, metadata)) = selected {
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

        if let Some((_, metadata)) = selected {
            if self.metadata_dialog {
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("metadata_window"),
                    egui::ViewportBuilder::default()
                        .with_title("LoRA metadata")
                        .with_inner_size([600.0, 300.0]),
                    |ctx, _class| {
                        if ctx.input(|i| i.viewport().close_requested()) {
                            self.metadata_dialog = false;
                        }
                        let mut metadata: Vec<_> = metadata.raw_metadata.iter().collect();
                        metadata.sort();
                        egui::CentralPanel::default().show(ctx, |ui| {
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    egui::Grid::new("metadata")
                                        .num_columns(2)
                                        .striped(true)
                                        .show(ui, |ui| {
                                            for (tag, value) in metadata {
                                                ui.label(tag);
                                                ui.add(egui::Label::new(value).wrap());
                                                ui.end_row();
                                            }
                                        })
                                })
                        });
                    },
                );
            }
            if self.tensors_dialog {
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("tensors_window"),
                    egui::ViewportBuilder::default()
                        .with_title("Tensor list")
                        .with_inner_size([600.0, 300.0]),
                    |ctx, _class| {
                        if ctx.input(|i| i.viewport().close_requested()) {
                            self.tensors_dialog = false;
                        }
                        egui::CentralPanel::default().show(ctx, |ui| {
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    egui::Grid::new("tensors")
                                        .num_columns(3)
                                        .striped(true)
                                        .show(ui, |ui| {
                                            for (name, shape) in &metadata.tensors {
                                                ui.label(name);
                                                let shape: Vec<_> =
                                                    shape.iter().map(|v| format!("{v}")).collect();
                                                ui.label(shape.join(", "));
                                                ui.allocate_space(egui::vec2(
                                                    ui.available_width(),
                                                    0.0,
                                                ));
                                                ui.end_row();
                                            }
                                        })
                                })
                        })
                    },
                );
            }
        } else {
            self.metadata_dialog = false;
            self.tensors_dialog = false;
        }
    }
}
