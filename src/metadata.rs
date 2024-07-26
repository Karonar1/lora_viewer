use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use anyhow::{anyhow, bail, ensure, Result};
use safetensors::SafeTensors;
use tinyjson::JsonValue;

#[derive(Debug, Eq, Hash, Ord, PartialOrd, PartialEq)]
pub enum LoraSubtype {
    LoRA,
    LoHa,
    LoKr,
}
impl Display for LoraSubtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LoraSubtype::LoRA => "LoRA",
            LoraSubtype::LoHa => "LoHa",
            LoraSubtype::LoKr => "LoKr",
        })
    }
}

#[derive(Debug, Eq, Hash, Ord, PartialOrd, PartialEq)]
pub enum ModelType {
    SdCheckpoint,
    SdxlCheckpoint,
    UnetLora(LoraSubtype),
    SdClipLora(LoraSubtype),
    SdxlClipLora(LoraSubtype),
    BakedVae,
    StandaloneVae,
}
impl ModelType {
    /// Attempt to infer model type from a tensor name
    ///
    /// Given a tensor name, this function _may_ return the type of model it belongs to. In
    /// general, it only returns a non-None value for tensor names that are unique to a model type.
    fn from_tensor_name(name: &str, shape: &[usize]) -> Option<ModelType> {
        // SDXL has two text models, and SD only has one.
        if name.starts_with("conditioner.embedders.0.") {
            return Some(ModelType::SdxlCheckpoint);
        }
        if name.starts_with("cond_stage_model.") {
            return Some(ModelType::SdCheckpoint);
        }

        // Standalone and baked-in VAEs have easily recognized model names
        if name.starts_with("encoder.") {
            return Some(ModelType::StandaloneVae);
        }
        if name.starts_with("first_stage_model.") {
            return Some(ModelType::BakedVae);
        }

        // All remaining model types we recognize are some kind of LoRA, so find the subtype first
        let subtype = if name.ends_with("lora_down.weight") || name.ends_with("lora_up.weight") {
            LoraSubtype::LoRA
        } else if name.ends_with("hada_w1_a") {
            LoraSubtype::LoHa
        } else if name.ends_with("lokr_w1") {
            LoraSubtype::LoKr
        } else {
            return None;
        };

        if name.starts_with("lora_te_") {
            return Some(ModelType::SdClipLora(subtype));
        }
        if name.starts_with("lora_te1_") {
            return Some(ModelType::SdxlClipLora(subtype));
        }

        if (name.starts_with("lora_unet_down_") || name.starts_with("lora_unet_input_"))
            && shape.len() == 2
        {
            return Some(ModelType::UnetLora(subtype));
        }

        None
    }
}
impl Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self {
            ModelType::SdCheckpoint => "SD Checkpoint".to_string(),
            ModelType::SdxlCheckpoint => "SDXL Checkpoint".to_string(),
            ModelType::SdClipLora(subtype) => format!("SD CLIP {subtype}"),
            ModelType::UnetLora(subtype) => format!("UNet {subtype}"),
            ModelType::SdxlClipLora(subtype) => format!("SDXL CLIP {subtype}"),
            ModelType::BakedVae => "Baked-in VAE".to_string(),
            ModelType::StandaloneVae => "Standalone VAE".to_string(),
        })
    }
}

#[derive(Default)]
pub struct LoraData {
    pub raw_metadata: HashMap<String, String>,
    pub tag_frequencies: Vec<(String, f64)>,
    pub base_model: Option<String>,
    pub tensors: Vec<(String, Vec<usize>)>,
    pub model_types: Vec<ModelType>,
}

/// Read the header of a safetensors file, padding to the full model size
///
/// This is a convenience function for reading the header of a safetensors file when we know we
/// aren't going to need the tensor weights themselves. It only reads the header (as defined by the
/// first 8 bytes of the file), returning a buffer where all remaining data is zero. As with the
/// real safetensors implementation, the header size is restricted to 100MB, although we're a
/// little more generous in allowing 100*2^20 instead of 100*10^6.
///
/// This is clearly not very memory efficient, but allows us to use the proper safetensors
/// validation, which needs the buffer to be the correct size. Its main purpose is to speed up
/// processing of very large model files, for example when accidentally trying to load a checkpoint
/// instead of a LoRA.
pub fn read_header(path: &Path) -> Result<Vec<u8>> {
    let mut read = std::fs::File::open(path)?;
    let model_size: usize = read.metadata()?.len().try_into()?;

    let mut size: [u8; 8] = [0; 8];
    read.read_exact(&mut size)?;
    let size: usize = u64::from_le_bytes(size).try_into()?;
    let size = size.checked_add(8).ok_or(anyhow!("Invalid header size"))?;
    ensure!(size < 100 * 1048576);

    let mut buffer = Vec::with_capacity(model_size);
    buffer.resize(size, 0);
    read.seek(SeekFrom::Start(0))?;
    read.read_exact(&mut buffer)?;
    buffer.resize(model_size, 0);
    Ok(buffer)
}

impl LoraData {
    pub fn from_buffer(buffer: &[u8]) -> Result<LoraData> {
        let metadata = SafeTensors::read_metadata(buffer)?.1;
        let metadata: HashMap<_, _> = metadata
            .metadata()
            .as_ref()
            .map(|m| m.iter().collect())
            .unwrap_or_default();

        let all_tags = tag_frequencies(&metadata).ok().unwrap_or_default();

        let tensors = SafeTensors::deserialize(buffer)?;
        let mut names = tensors.names();
        names.sort();
        let tensors: Result<Vec<_>> = names
            .iter()
            .map(|name| Ok((name.to_string(), tensors.tensor(name)?.shape().to_vec())))
            .collect();
        let tensors = tensors.unwrap_or_default();

        let model_types: HashSet<_> = tensors
            .iter()
            .filter_map(|(name, shape)| ModelType::from_tensor_name(name, shape))
            .collect();
        let mut model_types: Vec<_> = model_types.into_iter().collect();
        model_types.sort();

        Ok(LoraData {
            raw_metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            tag_frequencies: all_tags,
            base_model: metadata
                .get(&"ss_sd_model_name".to_string())
                .map(|s| s.to_string()),
            tensors,
            model_types,
        })
    }
}

fn tag_frequencies(metadata: &HashMap<&String, &String>) -> Result<Vec<(String, f64)>> {
    let frequencies = metadata
        .get(&"ss_tag_frequency".to_string())
        .ok_or(anyhow!("Could not get tag frequencies"))?;
    let mut all_tags = HashMap::new();
    let frequencies: JsonValue = frequencies.parse().unwrap();
    let JsonValue::Object(dirs) = frequencies else {
        bail!("Unexpected json structure")
    };
    for dir in dirs.iter() {
        let JsonValue::Object(tags) = dir.1 else {
            bail!("Unexpected json structure")
        };
        for tag in tags {
            all_tags
                .entry(tag.0.to_string())
                .and_modify(|v| *v += tag.1.get::<f64>().unwrap())
                .or_insert(*tag.1.get::<f64>().unwrap());
        }
    }
    let mut all_tags: Vec<_> = all_tags.into_iter().collect();
    all_tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(all_tags)
}
