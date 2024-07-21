use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use anyhow::{anyhow, bail, ensure, Result};
use safetensors::SafeTensors;
use tinyjson::JsonValue;

#[derive(Default)]
pub struct LoraData {
    pub raw_metadata: HashMap<String, String>,
    pub tag_frequencies: Vec<(String, f64)>,
    pub base_model: Option<String>,
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
            .ok_or(anyhow!("Could not read metadata"))?
            .iter()
            .collect();

        let all_tags = tag_frequencies(&metadata).ok().unwrap_or_default();

        Ok(LoraData {
            raw_metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            tag_frequencies: all_tags,
            base_model: metadata
                .get(&"ss_sd_model_name".to_string())
                .map(|s| s.to_string()),
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
