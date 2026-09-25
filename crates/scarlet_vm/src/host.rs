//! What a program sees of the world outside it: its arguments, its
//! environment, its clock and its GPU. The driver makes one and hands it to
//! [`crate::run`], so a test can hand in a world of its own.

use std::sync::Arc;
use std::time::Instant;

use crate::platform::Gpu;

/// The world a run sees.
pub struct Host {
    /// `os.argv`: the path of the entrypoint that was run, then the arguments
    /// after it.
    argv: Vec<String>,
    /// `os.env`: each variable whose name and value are both UTF-8, in the
    /// order the OS lists them.
    env: Vec<(String, String)>,
    /// What `time.monotonic` counts from.
    started: Instant,
    /// What `scarlet/metal` reaches, shared by every run on this host. With
    /// none, the machine has no GPU the run can use, and `metal.device` says
    /// so.
    gpu: Option<Arc<Gpu>>,
}

impl Host {
    pub fn new(argv: Vec<String>, env: Vec<(String, String)>) -> Host {
        Host {
            argv,
            env,
            started: Instant::now(),
            gpu: None,
        }
    }

    /// This world, with `gpu` as its GPU instead of none.
    pub fn with_gpu(self, gpu: Arc<Gpu>) -> Host {
        Host {
            gpu: Some(gpu),
            ..self
        }
    }

    /// This OS process's world: its environment as it is now, and `argv`.
    pub fn of_this_process(argv: Vec<String>) -> Host {
        let env = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        Host::new(argv, env)
    }

    pub(crate) fn argv(&self) -> &[String] {
        &self.argv
    }

    pub(crate) fn env(&self) -> &[(String, String)] {
        &self.env
    }

    pub(crate) fn gpu(&self) -> Option<&Gpu> {
        self.gpu.as_deref()
    }

    /// Milliseconds since the run began, on a clock that only goes forward.
    pub(crate) fn monotonic_ms(&self) -> u128 {
        self.started.elapsed().as_millis()
    }
}
