use crate::ipc::MemoryChunk;
use serde::{Deserialize, Serialize};
use candle_core::{Device, Tensor};
use std::collections::{VecDeque, HashMap};
use std::sync::Arc;
use std::fs;
use std::path::Path;

// This works across ALL models (Llama, Qwen, Mistral).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContextMessage {
    pub role: String,
    pub content: String,
}

// THE OS PAGEFILE MANAGER (SSD Swap)
pub struct Pager;

impl Pager {
    const SWAP_DIR: &'static str = "../swap";

    pub fn ensure_swap_drive() {
        if !Path::new(Self::SWAP_DIR).exists() {
            fs::create_dir_all(Self::SWAP_DIR).expect("Failed to create SSD Swap directory");
        }
    }

    /// Tier 1 Paging, Freeze the Agent's Chat History to the SSD
    pub fn page_out_history(app_id: &str, history: &Vec<ContextMessage>) {
        Self::ensure_swap_drive();
        let path = format!("{}/{}.json", Self::SWAP_DIR, app_id);

        if let Ok(data) = serde_json::to_string_pretty(history) {
            let _ = fs::write(&path, data);
            kprintln!("-> [PAGER] Agent '{}' history paged OUT to SSD.", app_id);
        }
    }

    /// Stream the Agent's Chat History from the SSD back into RAM
    pub fn page_in_history(app_id: &str) -> Vec<ContextMessage> {
        let path = format!("{}/{}.json", Self::SWAP_DIR, app_id);

        if Path::new(&path).exists()
            && let Ok(data) = fs::read_to_string(&path)
            && let Ok(history) = serde_json::from_str::<Vec<ContextMessage>>(&data)
        {
            kprintln!("-> [PAGER] Agent '{}' history paged IN from SSD.", app_id);
            return history;
        }
        Vec::new()
    }

    pub fn page_out_semantic(pipe_name: &str, chunks: &VecDeque<Arc<MemoryChunk>>) {
        Self::ensure_swap_drive();
        let path = format!("{}/{}.pipe", Self::SWAP_DIR, pipe_name);

        // Bincode freezes the RAM structure into pure 1s and 0s instantly
        if let Ok(data) = bincode::serialize(chunks) {
            let _ = fs::write(&path, data);
            kprintln!("-> [PAGER] Semantic Pipe '{}' flushed to SSD (.pipe).", pipe_name);
        }
    }

    pub fn page_in_semantic(pipe_name: &str) -> Option<VecDeque<Arc<MemoryChunk>>> {
        let path = format!("{}/{}.pipe", Self::SWAP_DIR, pipe_name);

        if Path::new(&path).exists() {
            // Read raw bytes instead of strings
            if let Ok(data) = fs::read(&path) {
                if let Ok(chunks) = bincode::deserialize::<VecDeque<Arc<MemoryChunk>>>(&data) {
                    kprintln!("-> [PAGER] Semantic Pipe '{}' mapped IN from SSD.", pipe_name);
                    return Some(chunks);
                } else {
                    kprintln!("-> [PAGER] [ERROR] Failed to deserialize pipe '{}'. The binary file might be corrupt or from an older version.", pipe_name);
                }
            }
        }
        None
    }

    pub fn page_out_kv_cache(app_id: &str, model_name: &str, tensors: &HashMap<String, Tensor>) {
        Self::ensure_swap_drive();
        let safe_model = model_name.replace(":", "-");
        let path = format!("{}/{}_{}.safetensors", Self::SWAP_DIR, app_id, safe_model);
        
        // Save the raw math matrices directly to the SSD
        if let Err(e) = candle_core::safetensors::save(tensors, &path) {
            kprintln!("-> [PAGER] [ERROR] Failed to save KV-Cache to SSD: {}", e);
        } else {
            kprintln!("-> [PAGER] Agent '{}' KV-Cache ({} Tensors) paged OUT to SSD.", app_id, tensors.len());
        }
    }

    pub fn page_in_kv_cache(app_id: &str, model_name: &str, device: &Device) -> Option<HashMap<String, Tensor>> {
        let safe_model = model_name.replace(":", "-");
        let path = format!("{}/{}_{}.safetensors", Self::SWAP_DIR, app_id, safe_model);

        if Path::new(&path).exists() {
            match candle_core::safetensors::load(&path, device) {
                Ok(tensors) => {
                    kprintln!("-> [PAGER] Agent '{}' KV-Cache paged IN from SSD.", app_id);
                    return Some(tensors);
                }
                Err(e) => {
                    kprintln!("-> [PAGER] [WARN] Failed to load KV-Cache: {}. Falling back to JSON History.", e);
                }
            }
        }
        None
    }

    pub fn get_kv_cache_size_mb(app_id: &str, model_name: &str) -> u32 {
        let safe_model = model_name.replace(":", "-");
        let path = format!("{}/{}_{}.safetensors", Self::SWAP_DIR, app_id, safe_model);

        if let Ok(metadata) = fs::metadata(&path) {
            (metadata.len() / (1024 * 1024)) as u32
        } else {
            0
        }
    }

    /// Wipe the memory clean
    pub fn clear_page(app_id: &str) {
        let _ = fs::remove_file(format!("{}/{}.json", Self::SWAP_DIR, app_id));
        let _ = fs::remove_file(format!("{}/{}.pipe", Self::SWAP_DIR, app_id));

        // Sweep for any Model-Specific Safetensor KV-Caches
        if let Ok(entries) = fs::read_dir(Self::SWAP_DIR) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with(&format!("{}_", app_id)) && file_name.ends_with(".safetensors") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        
        kprintln!("-> [PAGER] Completely wiped all swap files for Agent '{}'", app_id);
    }

    pub fn delete_kv_cache(app_id: &str) {
        if let Ok(entries) = fs::read_dir(Self::SWAP_DIR) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with(&format!("{}_", app_id)) && file_name.ends_with(".safetensors") {
                    let _ = fs::remove_file(entry.path());
                    kprintln!("-> [PAGER] Deleted stale KV-Cache for '{}' (Memory Compaction).", app_id);
                }
            }
        }
    }
}
