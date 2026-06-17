use std::sync::Arc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

pub struct GpuScheduler {
    execution_lock: Arc<Semaphore>,
    state: Mutex<GpuState>,
}

/// Tracks what is currently physically loaded in VRAM.
struct GpuState {
    active_model: Option<String>,
    active_app_id: Option<String>,
    active_users: u32,
}

impl Default for GpuScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuScheduler {
    pub fn new() -> Self {
        Self {
            execution_lock: Arc::new(Semaphore::new(1)),
            state: Mutex::new(GpuState {
                active_model: None,
                active_app_id: None,
                active_users: 0,
            }),
        }
    }

    pub async fn request_gpu(&self, requested_model: &str, app_id: &str) -> GpuLease {
        let permit = Arc::clone(&self.execution_lock)
            .acquire_owned()
            .await
            .unwrap();

        // 2. Check the Memory Map (What's in VRAM?)
        let mut state = self.state.lock().await;

        let is_same_model = state.active_model.as_deref() == Some(requested_model);
        let is_same_agent = state.active_app_id.as_deref() == Some(app_id);

        if is_same_model && is_same_agent {
            // [TIER 1] PERFECT HIT
            println!(
                "-> [SCHEDULER] Hot Hit! '{}' for Agent '{}' is already active.",
                requested_model, app_id
            );
            state.active_users += 1;
        } else if is_same_model && !is_same_agent {
            // [TIER 2] AGENT SWAP (The Massive Optimization)
            let old_agent = state
                .active_app_id
                .clone()
                .unwrap_or_else(|| "Unknown".to_string());
            println!(
                "-> [SCHEDULER] Agent Swap: Retaining '{}' weights. Evicting '{}' KV-Cache -> Injecting '{}'.",
                requested_model, old_agent, app_id
            );
            state.active_app_id = Some(app_id.to_string());
            state.active_users = 1;
        } else {
            // [TIER 3] MODEL SWAP (Cold Start)
            if let Some(old_model) = &state.active_model {
                println!(
                    "-> [SCHEDULER] Model Swap: Evicting '{}' -> Loading '{}' for '{}'",
                    old_model, requested_model, app_id
                );
            } else {
                println!(
                    "-> [SCHEDULER] Cold Start: Loading '{}' into VRAM for '{}'.",
                    requested_model, app_id
                );
            }
            state.active_model = Some(requested_model.to_string());
            state.active_app_id = Some(app_id.to_string());
            state.active_users = 1;
        }

        GpuLease {
            _permit: permit,
            model: requested_model.to_string(),
        }
    }

    /// Helper to see what's currently running
    pub async fn get_status(&self) -> String {
        let state = self.state.lock().await;
        match &state.active_model {
            Some(m) => format!("ACTIVE (Model: {}, Users: {})", m, state.active_users),
            None => "IDLE (VRAM Empty)".to_string(),
        }
    }
}

// When this struct drops (variable goes out of scope), the GPU is unlocked.
pub struct GpuLease {
    _permit: OwnedSemaphorePermit,
    pub model: String,
}
