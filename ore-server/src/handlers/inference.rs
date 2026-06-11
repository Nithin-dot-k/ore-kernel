use crate::payloads::RunRequest;
use crate::state::KernelState;
use axum::body::Body;
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use ore_core::firewall::ContextFirewall;
use ore_core::swap::Pager;
use std::sync::Arc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

// inference engine (The Proxy & Firewall)
pub async fn ask_ai(State(state): State<Arc<KernelState>>, Path(prompt): Path<String>) -> String {
    let clean_prompt = prompt.replace("_", " ");

    println!("\n-> Incoming App Request: {}", clean_prompt);

    let app_id = "openclaw"; // In the future, this comes from an API Key/Token
    let manifest = match state.registry.get_app(app_id) {
        Some(m) => m.clone(),
        None => return format!("ORE KERNEL ALERT: Unregistered Agent '{}'.", app_id),
    };

    let secured_prompt = match ContextFirewall::secure_request(&manifest, &clean_prompt) {
        Ok(safe_text) => {
            println!("-> Security Check Passed.");
            if safe_text != clean_prompt {
                println!("-> [NOTICE] PII Redacted from prompt.");
            }
            safe_text
        }
        Err(e) => {
            println!("-> [BLOCKED] {}", e);
            return format!("ORE KERNEL ALERT: {}", e);
        }
    };

    let mut current_context = None;
    if manifest.resources.json_history {
        current_context = Some(Pager::page_in_history(app_id));
    }

    println!("-> Waiting for GPU Scheduler...");

    // If the agent lists allowed_models, pick the first one. Default to "llama3.2:1b"
    let target_model = manifest
        .resources
        .allowed_models
        .first()
        .map(|s| s.as_str())
        .unwrap_or("llama3.2:1b");

    // the GPU scheduler
    let lease = state.scheduler.request_gpu(target_model).await;
    println!(
        "-> GPU Lease Granted for '{}'. Routing to Driver...",
        lease.model
    );

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let driver = Arc::clone(&state.driver);
    let model_name = lease.model.clone();
    let prompt_clone = secured_prompt.clone();
    let context_clone = current_context.clone();

    tokio::spawn(async move {
        // production update required: app_id -> &manifest.app_id on function signature
        if let Err(e) = driver.generate_text(&model_name, app_id, manifest.resources.stateful_paging, &prompt_clone, context_clone, tx).await {
            println!("-> [KERNEL ERROR] Inference execution failed: {}", e);
        }

        println!("-> Agent Execution complete. Releasing GPU Lock.");
        drop(lease);
    });

    let mut full_response = String::new();
    while let Some(word) = rx.recv().await {
        full_response.push_str(&word);
    }

    if manifest.resources.json_history {
        let mut new_history = current_context.unwrap_or_default();
        new_history.push(ore_core::swap::ContextMessage {
            role: "user".to_string(),
            content: secured_prompt,
        });
        new_history.push(ore_core::swap::ContextMessage {
            role: "assistant".to_string(),
            content: full_response.clone(),
        });

        Pager::page_out_history(app_id, &new_history);

        let total_chars: usize = new_history.iter().map(|m| m.content.len()).sum();
        let estimated_tokens = (total_chars / 4) as u32;
        let token_limit = manifest.memory_limits.max_json_tokens;
        let token_cap_hit = estimated_tokens > token_limit;

        // CALCULATE PHYSICAL VRAM/SSD USAGE
        let mut kv_cap_hit = false;
        if manifest.resources.stateful_paging && manifest.memory_limits.max_kv_cache_mb > 0 {
            let current_kv_mb = Pager::get_kv_cache_size_mb(app_id, target_model);
            kv_cap_hit = current_kv_mb > manifest.memory_limits.max_kv_cache_mb;
        }

        if token_cap_hit || kv_cap_hit {
            if manifest.memory_limits.auto_summarize_on_cap {
                println!("-> [KERNEL] Agent '{}' memory cap reached ({} > {} tokens). Triggering Background Compaction...", app_id, estimated_tokens, token_limit);

                // Clone variables for the background thread so the user gets their response instantly
                let history_to_compress = new_history.clone();
                let m_id = app_id.to_string();
                let driver_clone = Arc::clone(&state.driver);
                let scheduler_clone = Arc::clone(&state.scheduler);
                let model_to_use = target_model.to_string();

                tokio::spawn(async move {
                    let text_to_summarize = history_to_compress.iter()
                        .map(|m| format!("{}: {}", m.role, m.content))
                        .collect::<Vec<String>>()
                        .join("\n");
                    
                    let summary_prompt = format!(
                        "Summarize the following conversation history densely, preserving key facts, decisions, and context. Be concise:\n\n{}", 
                        text_to_summarize
                    );

                    // Grab the GPU Lock to do the heavy compression
                    let lease = scheduler_clone.request_gpu(&model_to_use).await;
                    println!("-> [COMPACTION] GPU Lease acquired for background summarization.");
                    
                    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                    let m_clone = model_to_use.clone();
                    let app_clone = m_id.clone();

                    tokio::spawn(async move {
                        // We set stateful_paging = false here so the summarizer does not enter an infinite loop!
                        let _ = driver_clone.generate_text(&m_clone, &app_clone, false, &summary_prompt, None, tx).await;
                    });
                    
                    let mut summary = String::new();
                    while let Some(word) = rx.recv().await {
                        summary.push_str(&word);
                    }
                    
                    drop(lease); // Release the GPU
                    
                    // REWRITE THE BRAIN
                    let mut compacted_history = Vec::new();
                    compacted_history.push(ore_core::swap::ContextMessage {
                        role: "system".to_string(),
                        content: format!("You are a helpful AI assistant. Previous context summary: {}", summary),
                    });
                    
                    // Keep the last 2 messages so the conversation flow isn't jarring to the user
                    if history_to_compress.len() >= 2 {
                        let len = history_to_compress.len();
                        compacted_history.push(history_to_compress[len - 2].clone());
                        compacted_history.push(history_to_compress[len - 1].clone());
                    }

                    // Save the tiny, highly-dense memory back to the SSD
                    Pager::page_out_history(&m_id, &compacted_history);
                    
                    // CRITICAL: We must delete the old .safetensors KV-cache because the sequence of tokens just fundamentally changed!
                    if manifest.resources.stateful_paging {
                        Pager::delete_kv_cache(&m_id);
                        println!("-> [COMPACTION] KV-Cache invalidated and erased from disk.");
                    }
                    println!("-> [COMPACTION] Memory compressed successfully. VRAM footprint reset to 0.");
                });
                
            } else {
                // If auto_summarize is OFF, we use brutal FIFO pruning
                println!("-> [KERNEL] Agent '{}' memory cap reached. Pruning oldest messages (FIFO)...", app_id);
                while new_history.iter().map(|m| m.content.len()).sum::<usize>() / 4 > token_limit as usize && new_history.len() > 2 {
                    new_history.remove(0);
                }
                Pager::page_out_history(app_id, &new_history);
                if manifest.resources.stateful_paging {
                    Pager::delete_kv_cache(app_id);
                }
            }
        }
    }

    full_response
}

pub async fn run_process(
    State(state): State<Arc<KernelState>>,
    Json(payload): Json<RunRequest>,
) -> Response {
    println!(
        "-> [EXEC] Model: {} | Prompt: {}",
        payload.model, payload.prompt
    );

    let app_id = "terminal_user";
    let manifest = match state.registry.get_app(app_id) {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                format!("ORE KERNEL ALERT: Unregistered User '{}'.", app_id),
            )
                .into_response()
        }
    };

    // future update: calculate tokens based on prompt length or use a more dynamic approach
    let limit = manifest.resources.max_tokens_per_minute;
    if !state.rate_limiter.check_and_add(app_id, limit, 1000) {
        println!("-> [BLOCKED] Agent '{}' exceeded GPU rate limit.", app_id);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            format!("ORE KERNEL ALERT: Rate Limit Exceeded ({} t/min).", limit),
        )
            .into_response();
    }

    let secured_prompt = match ContextFirewall::secure_request(&manifest, &payload.prompt) {
        Ok(safe_text) => {
            println!("-> Security Check Passed.");
            safe_text
        }
        Err(e) => {
            println!("-> [BLOCKED] {}", e);
            return (StatusCode::FORBIDDEN, format!("ORE KERNEL ALERT: {}", e)).into_response();
        }
    };

    println!("-> Waiting for GPU Scheduler...");

    // request a GPU lease for the specified model
    let lease = state.scheduler.request_gpu(&payload.model).await;
    println!(
        "-> GPU Lease Granted. Executing natively via {}...",
        state.driver.engine_name()
    );

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let driver = Arc::clone(&state.driver);
    let model_name = lease.model.clone();
    let prompt = secured_prompt.clone();

    tokio::spawn(async move {
        if let Err(e) = driver.generate_text(&model_name, app_id, manifest.resources.stateful_paging, &prompt, None, tx).await {
            println!("-> [KERNEL ERROR] Inference execution failed: {}", e);
        }
        println!("-> Execution complete. Releasing GPU Lock.");

        drop(lease);
    });

    let stream = UnboundedReceiverStream::new(rx)
        .map(|chunk| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(chunk)));

    Body::from_stream(stream).into_response()
}
