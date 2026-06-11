use anyhow::{Error as E, Result};
use crate::native::gguf_tokenizer::TokenizerFromGguf;
use crate::native::models;
use crate::swap::ContextMessage;
use std::path::Path;
use std::fs::File;
use std::io::Cursor;
use memmap2::Mmap;
use candle_core::{Tensor, Device};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;
use crate::native::models::llama::ModelWeights as LlamaModel;
use crate::native::models::qwen2::ModelWeights as Qwen2Model;
use tokenizers::Tokenizer;

// Supports multiple architectures
pub enum OreEngine {
    Qwen2(Qwen2Model),
    Llama(LlamaModel),
}

impl OreEngine {
    pub fn forward(&mut self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        match self {
            OreEngine::Qwen2(m) => m.forward(input, start_pos).map_err(E::msg),
            OreEngine::Llama(m) => m.forward(input, start_pos).map_err(E::msg),
        }

        // --- TO BE EVALUATED FOR NORMALIZATION ---
        // let logits = match self {
        //     OreEngine::Qwen2(m) => m.forward(input, start_pos).map_err(E::msg),
        //     OreEngine::Llama(m) => m.forward(input, start_pos).map_err(E::msg),
        // }
        // let dims = logits.dims().len();
        // let normalized_logits = if dims == 3 {
        //     logits.squeeze(0)?.squeeze(0)? // If Qwen (3D), squeeze twice
        // } else if dims == 2 {
        //     logits.squeeze(0)?             // If Llama (2D), squeeze once
        // } else {
        //     logits
        // };
        // Ok(normalized_logits)
    }

    pub fn num_layers(&self) -> usize {
        match self {
            OreEngine::Llama(m) => m.layers.len(),
            OreEngine::Qwen2(m) => m.layers.len(),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            OreEngine::Llama(m) => {
                for layer in m.layers.iter_mut() {
                    layer.kv_cache = None;
                }
            }
            OreEngine::Qwen2(m) => {
                for layer in m.layers.iter_mut() {
                    layer.kv_cache = None;
                }
            }
        }
    }

    pub fn get_kv_cache_len(&self) -> usize {
        // Look at Layer 0's Key tensor. The 3rd dimension is usually the sequence length.
        let first_layer_cache = match self {
            OreEngine::Llama(m) => m.layers[0].kv_cache.as_ref(),
            OreEngine::Qwen2(m) => m.layers[0].kv_cache.as_ref(),
        };

        if let Some((k, _v)) = first_layer_cache {
            k.dim(2).unwrap_or(0) 
        } else {
            0
        }
    }

    /// Rips the physical brain state out of the GPU
    pub fn get_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        match self {
            OreEngine::Llama(m) => m.layers.iter().map(|l| l.kv_cache.clone()).collect(),
            OreEngine::Qwen2(m) => m.layers.iter().map(|l| l.kv_cache.clone()).collect(),
        }
    }

    /// Injects a frozen brain state back into the AI
    pub fn set_kv_cache(&mut self, cache: Vec<Option<(Tensor, Tensor)>>) {
        match self {
            OreEngine::Llama(m) => {
                for (layer, saved_cache) in m.layers.iter_mut().zip(cache.into_iter()) {
                    layer.kv_cache = saved_cache;
                }
            }
            OreEngine::Qwen2(m) => {
                for (layer, saved_cache) in m.layers.iter_mut().zip(cache.into_iter()) {
                    layer.kv_cache = saved_cache;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub stop_tokens: Vec<u32>,
    pub formatter: fn(&[ContextMessage], &str, bool) -> String, 
}

pub struct ActiveEngine {
    pub model: OreEngine,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub model_name: String,
    pub config: ModelConfig,
    pub _mmap: memmap2::Mmap,
}

impl ActiveEngine {
    /// The ultra-fast, zero-copy GGUF loader using OS-level memory mapping (mmap)
    pub fn load(model_name: &str, device: &Device) -> Result<Self> {
        let safe_folder_name = model_name.replace(":", "-");
        let model_dir = Path::new("../models").join(&safe_folder_name);
        let gguf_path = model_dir.join("model.gguf");
        let local_tokenizer_path = model_dir.join("tokenizer.json");

        if !Path::new(&gguf_path).exists() {
            return Err(E::msg(format!(
                "Files not found. Run 'ore pull {}'",
                model_name
            )));
        }
        
        // 1. Memory Map the Weights
        println!("-> [CANDLE] Allocating Virtual Memory Pointer via mmap...");
        let file = File::open(&gguf_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        // 2. Extract Metadata
        let model_content = gguf_file::Content::read(&mut cursor).map_err(E::msg)?;
        let arch_name = match model_content.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(arch)) => arch.clone(),
            _ => "unknown".to_string(),
        };
        println!("-> [CANDLE] Detected Architecture: '{}'", arch_name);

        // 3. Tokenizer Resolution
        let global_tokenizer_name = if model_name.to_lowercase().contains("qwen2.5") {
            "qwen2.5"
        } else if model_name.to_lowercase().contains("llama4")
            || model_name.to_lowercase().contains("llama-4")
        {
            "llama4"
        } else if model_name.to_lowercase().contains("llama3.3")
            || model_name.to_lowercase().contains("llama-3.3")
        {
            "llama3.3"
        } else if model_name.to_lowercase().contains("llama3.2")
            || model_name.to_lowercase().contains("llama3")
            || model_name.to_lowercase().contains("llama-3.2")
            || model_name.to_lowercase().contains("llama-3")
        {
            "llama3.2"
        } else if model_name.to_lowercase().contains("llama2")
            || model_name.to_lowercase().contains("llama-2")
        {
            "llama2"
        } else if model_name.to_lowercase().contains("codellama") {
            "codellama"
        } else if model_name.to_lowercase().contains("gemma") {
            "gemma"
        } else {
            arch_name.as_str()
        };

        let global_path = format!("../tokenizers/{}.json", global_tokenizer_name);
        
        // universal tokenizer fallback
        let tokenizer = if Path::new(&local_tokenizer_path).exists() {
            println!("-> [CANDLE] Using Local Dictionary...");
            Tokenizer::from_file(&local_tokenizer_path).map_err(E::msg)?
        } else if Path::new(&global_path).exists() {
            println!(
                "-> [CANDLE] Local dictionary not found. Using Universal OS Dictionary for '{}'...",
                arch_name
            );
            Tokenizer::from_file(&global_path).map_err(E::msg)?
        } else {
            // THE RAW GGUF EXTRACTOR
            println!(
                "-> [CANDLE] [WARN] No JSON found. Extracting Tokenizer directly from GGUF metadata..."
            );
            let tok_file = File::open(&gguf_path)?;
            let mut reader = std::io::BufReader::new(tok_file);
            let content = gguf_file::Content::read(&mut reader).map_err(E::msg)?;

            let extracted_tokenizer = Tokenizer::from_gguf(&content).map_err(E::msg)?;

            // SAVE IT TO DISK
            println!(
                "-> [CANDLE] JIT Cache: Saving extracted dictionary to {}...",
                local_tokenizer_path.display()
            );
            if let Err(e) = extracted_tokenizer.save(&local_tokenizer_path, true) {
                println!("-> [CANDLE] [WARN] Could not save cached tokenizer: {}", e);
            } else {
                println!("-> [CANDLE] [SUCCESS] Dictionary permanently cached.");
            }

            extracted_tokenizer
        };

        // 4. Load Neural Weights (Architecture Router)
        let (model, config) = match arch_name.as_str() {
            "llama" => models::llama::load(model_name, model_content, &mut cursor, device, &tokenizer)?,
            "qwen2" => models::qwen2::load(model_name, model_content, &mut cursor, device, &tokenizer)?,
            _ => return Err(E::msg(format!("Architecture not supported natively: {}", arch_name))),
        };

        let logits_processor = LogitsProcessor::new(299792458, Some(0.7), None);

        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            model_name: model_name.to_string(),
            config,
            _mmap: mmap,
        })
    }
}