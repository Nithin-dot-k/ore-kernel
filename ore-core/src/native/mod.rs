pub mod engine;
pub mod gguf_tokenizer;
pub mod models;
pub mod kv_manager;

use crate::driver::{DriverError, InferenceDriver, LocalModel, VramProcess};
use crate::swap::ContextMessage;
use anyhow::{Result};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use engine::ActiveEngine;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use time::OffsetDateTime;
use time::macros::format_description;
use tokio::sync::mpsc::UnboundedSender;

pub struct NativeDriver {
    engine: Arc<StdMutex<Option<ActiveEngine>>>,
    device: Device,
}

impl Default for NativeDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeDriver {
    pub fn new() -> Self {
        println!("-> [CANDLE] Probing Motherboard for Hardware Compute...");
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        Self {
            engine: Arc::new(StdMutex::new(None)),
            device,
        }
    }

    fn load_weights_into_memory(model_name: &str, device: &Device) -> Result<ActiveEngine> {
        ActiveEngine::load(model_name, device)
    }
}

#[async_trait]
impl InferenceDriver for NativeDriver {
    fn engine_name(&self) -> &'static str {
        "Native Candle Engine"
    }

    async fn is_online(&self) -> bool {
        true
    }

    async fn get_running_models(&self) -> Result<Vec<VramProcess>, DriverError> {
        let state = self.engine.lock().unwrap();
        if let Some(active) = &*state {
            Ok(vec![VramProcess {
                model_name: active.model_name.clone(),
                size_bytes: 1024 * 1024 * 1024,
                size_vram_bytes: 0,
            }])
        } else {
            Ok(vec![])
        }
    }

    async fn preload_model(&self, model: &str) -> Result<(), DriverError> {
        let model = model.trim().replace(":", "-");
        let mut state = self.engine.lock().unwrap();
        if state.is_none() || state.as_ref().unwrap().model_name != model {
            *state = Some(
                Self::load_weights_into_memory(&model, &self.device)
                    .map_err(|e| DriverError::ExecutionFailed(e.to_string()))?,
            );
        }
        Ok(())
    }

    async fn unload_model(&self, _model: &str) -> Result<(), DriverError> {
        let mut state = self.engine.lock().unwrap();
        *state = None;
        Ok(())
    }

    async fn generate_text(
        &self,
        model: &str,
        app_id: &str,
        stateful_paging: bool,
        prompt: &str,
        _history: Option<Vec<ContextMessage>>,
        tx: UnboundedSender<String>,
    ) -> Result<(), DriverError> {
        let model = model.trim().replace(':', "-");
        {
            let mut state = self.engine.lock().unwrap();
            if state.is_none() || state.as_ref().unwrap().model_name != model {
                *state = Some(
                    Self::load_weights_into_memory(&model, &self.device)
                        .map_err(|e| DriverError::ExecutionFailed(e.to_string()))?,
                );
            }
        }

        let engine_arc = Arc::clone(&self.engine);
        let safe_prompt = prompt.to_string();
        let device_clone = self.device.clone();

        let a_id = app_id.to_string();

        let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
            let mut state_guard = engine_arc.lock().unwrap();
            let active = state_guard.as_mut().unwrap();

            let mut current_cache_len = 0;
            if stateful_paging {
                if let Some(frozen_tensors) = crate::swap::Pager::page_in_kv_cache(&a_id, &model, &device_clone) {
                    // Unflatten the SSD file back into 3D Neural Tensors
                    let cache = crate::native::kv_manager::KvManager::unflatten_cache(&frozen_tensors, active.model.num_layers());
                    
                    // Inject directly into the Engine's brain!
                    active.model.set_kv_cache(cache);
                    current_cache_len = active.model.get_kv_cache_len(); 
                    
                    println!("-> [NATIVE DRIVER] KV-Cache injected ({} tokens). Bypassing Prefill.", current_cache_len);
                }
            } else {
                // If paging is off, ensure we start with a clean brain!
                active.model.clear_kv_cache();
            }

            let formatted_prompt = (active.config.formatter)(&safe_prompt);
            let mut tokens = active
                .tokenizer
                .encode(formatted_prompt, true)
                .unwrap()
                .get_ids()
                .to_vec();

            let mut start_pos = current_cache_len;

            for index in 0..8192 {
                let context_size = if index > 0 { 1 } else { tokens.len() };

                let input_tensor = Tensor::new(&tokens[tokens.len() - context_size..], &device_clone)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap();
                let logits = active.model.forward(&input_tensor, start_pos).unwrap();
                
                let logits = logits
                    .squeeze(0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();

                let next_token_id = active.logits_processor.sample(&logits).unwrap();

                if active.config.stop_tokens.contains(&next_token_id) {
                    break;
                }

                let word = active.tokenizer.decode(&[next_token_id], true).unwrap();

                if tx.send(word).is_err() {
                    break;
                }

                tokens.push(next_token_id);
                start_pos += context_size;
            }

            if stateful_paging {
                println!("-> [NATIVE DRIVER] Freezing Brain State to SSD...");
                
                // Extract the raw electricity from the Engine
                let raw_cache = active.model.get_kv_cache();
                let flat_tensors = crate::native::kv_manager::KvManager::flatten_cache(&raw_cache);
                
                let out_id = a_id.clone();
                let out_model = model.clone();
                
                // Blast it to the NVMe drive in the background!
                tokio::spawn(async move {
                    crate::swap::Pager::page_out_kv_cache(&out_id, &out_model, &flat_tensors);
                });
            }
            
            Ok(())
        })
        .await
        .map_err(|e| DriverError::ExecutionFailed(e.to_string()))?;

        result.map_err(DriverError::ExecutionFailed)
    }

    async fn list_local_models(&self) -> Result<Vec<LocalModel>, DriverError> {
        let mut models = Vec::new();
        let models_dir = Path::new("../models");

        if !models_dir.exists() {
            return Ok(models);
        }

        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata()
                    && metadata.is_dir()
                {
                    let folder_name = entry.file_name().to_string_lossy().to_string();
                    let gguf_path = entry.path().join("model.gguf");

                    let mut size_bytes = 0;
                    let mut modified_at = "UNKNOWN".to_string();

                    if let Ok(gguf_meta) = fs::metadata(&gguf_path) {
                        size_bytes = gguf_meta.len();

                        if let Ok(sys_time) = gguf_meta.modified() {
                            let dt: OffsetDateTime = sys_time.into();

                            let local_offset = time::UtcOffset::current_local_offset()
                                .unwrap_or(time::UtcOffset::UTC);
                            let local_dt = dt.to_offset(local_offset);

                            // Compile-time macro format! (Zero runtime parsing cost)
                            let format = format_description!(
                                "[day]-[month]-[year] [hour]:[minute]:[second]"
                            );
                            modified_at = local_dt
                                .format(&format)
                                .unwrap_or_else(|_| "UNKNOWN".to_string());
                        }
                    }

                    let display_name = folder_name.replace("-", ":");

                    models.push(LocalModel {
                        name: display_name,
                        size_bytes,
                        modified_at,
                    });
                }
            }
        }
        Ok(models)
    }

    async fn generate_embeddings(
        &self,
        model_name: &str,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, DriverError> {
        let safe_model = model_name.replace(":", "-");
        let device = self.device.clone();

        // Spawn a blocking thread
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, String> {
            let model_dir = format!("../models/{}", safe_model);
            let config_path = format!("{}/config.json", model_dir);

            if !Path::new(&config_path).exists() {
                return Err(format!(
                    "Embedder config missing. Run 'ore pull {}'",
                    safe_model
                ));
            }

            // Detemine the architecture
            let config_str = fs::read_to_string(&config_path).map_err(|e| e.to_string())?;
            let config_val: serde_json::Value =
                serde_json::from_str(&config_str).map_err(|e| e.to_string())?;

            let arch = config_val
                .get("architectures")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .unwrap_or("BertModel"); // Default to standard BERT if missing

            println!("-> [NATIVE] Detected Embedder Architecture: '{}'", arch);

            let vectors = match arch {
                "NomicBertModel" => {
                    // Route to custom Nomic RoPE/SwiGLU implementation
                    let embedder = models::nomic::SystemEmbedder::load(&model_dir, &device)
                        .map_err(|e| format!("Failed to load Nomic embedder: {}", e))?;
                    embedder
                        .embed_batch(inputs)
                        .map_err(|e| format!("Nomic math failed: {}", e))?
                }
                "BertModel" => {
                    // Route to the ultra-fast standard MiniLM implementation
                    let embedder = models::bert::SystemEmbedder::load(&model_dir, &device)
                        .map_err(|e| format!("Failed to load BERT embedder: {}", e))?;
                    embedder
                        .embed_batch(inputs)
                        .map_err(|e| format!("BERT math failed: {}", e))?
                }
                _ => return Err(format!("Unsupported embedding architecture: {}", arch)),
            };

            // The moment this thread finishes, `embedder` goes out of scope.
            // Rust's memory safety automatically drops the model and flushes the RAM to 0MB.

            Ok(vectors)
        })
        .await
        .map_err(|e| DriverError::ExecutionFailed(e.to_string()))?;

        result.map_err(DriverError::ExecutionFailed)
    }

    // just for the sake of trait implementation, taken care by CLI
    async fn pull_model(&self, _model: &str) -> Result<(), DriverError> {
        Ok(())
    }
}
