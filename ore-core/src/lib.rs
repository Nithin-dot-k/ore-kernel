#[macro_export]
macro_rules! kprintln {
    ($($arg:tt)*) => {{
        // Format & Clean the Thread ID using the stable Debug trait
        let thread_desc = format!("{:?}", std::thread::current().id());
        let clean_thread = thread_desc.replace("ThreadId(", "T").replace(")", "");

        // Lock stdout manually to prevent thread write contention
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        
        // Tokio's Task ID implements Display natively! We print it directly.
        if let Some(task_id) = tokio::task::try_id() {
            let _ = write!(handle, "[{:<4} | Task:{:<3}] ", clean_thread, task_id);
        } else {
            let _ = write!(handle, "[{:<4} | Task:main] ", clean_thread);
        }
        let _ = writeln!(handle, $($arg)*);
    }};
}

pub mod driver;
pub mod external;
pub mod firewall;
pub mod ipc;
pub mod native;
pub mod registry;
pub mod scheduler;
pub mod swap;
