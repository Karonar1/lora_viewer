# LoRA Metadata Viewer

Basic tool for viewing essential metadata for LoRA models. Shows the base model and tags (and frequencies) the model was trained with, which are essential if you've forgotten or can't find the original documentation. Has an additional window showing all model metadata, allowing basic inspection of other safetensors files.

Works with most, but not all, LoRAs in safetensors format.

## Requirements

This is a native application, initially developed for Windows. It does not depend on Python, CUDA, Automatic1111, ComfyUI or having a functioning Stable Diffusion installation at all. If you've got a model file, it'll work. If you want to build from source, you will need a working Rust installation.

Releases will include (unsigned) executable binaries for Windows only. If you don't trust unsigned binaries, build from source - this should be as simple as ```cargo build --release```.

The tool can load any model in safetensors format, but is designed to work with LoRAs with normal metadata. Some models have different metadata tags or no metadata at all - while these can still be loaded, the tool won't show any useful information.

For the current version, model loading runs on the UI thread, which means the tool will freeze briefly when loading a model. For typical LoRA size models on an NVME drive, this is not an issue, but it can take a few seconds if you try to load a checkpoint model instead (not recommended). Future versions should load metadata on a separate thread.
