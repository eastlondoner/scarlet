//! A [`Platform`] made of plain memory, standing in for Metal in tests: the
//! VM's own, and the suite `scarlet_metal` runs against both it and Metal,
//! so the two cannot drift apart.
//!
//! It checks the VM as it goes. Making an object under an id it already
//! used, naming one it does not hold, or releasing one twice is written down,
//! and a fake dropped with any of that written down, or with an object still
//! held, fails the test that made it. A test written later gets both checks
//! without asking.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::platform::{
    Buffer, BufferBytes, BufferError, Device, DeviceError, DeviceInfo, Fault, Handle, Id, Platform,
    ReadInto,
};

/// A GPU in memory. [`Fake::new`] has one device, whose largest buffer is
/// larger than any binary, as Metal's is for any binary a test can make.
pub struct Fake {
    answer: Answer,
    max_buffer_bytes: u64,
    state: Mutex<State>,
}

/// What the fake says to a request it could meet. Only the VM's own tests
/// ask it to refuse; another crate's tests see the fake that meets them all.
#[cfg_attr(not(test), allow(dead_code))]
enum Answer {
    Meet,
    NoDevice,
    OutOfMemory,
}

#[derive(Default)]
struct State {
    devices: HashSet<Id<Device>>,
    buffers: HashMap<Id<Buffer>, Vec<u8>>,
    released: HashSet<Handle>,
    wrong: Vec<String>,
}

impl Default for Fake {
    fn default() -> Fake {
        Fake::new()
    }
}

impl Fake {
    pub fn new() -> Fake {
        Fake {
            answer: Answer::Meet,
            max_buffer_bytes: u64::MAX,
            state: Mutex::default(),
        }
    }

    /// A fake whose device's largest buffer holds `n` bytes.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_max_buffer_bytes(mut self, n: u64) -> Fake {
        self.max_buffer_bytes = n;
        self
    }

    /// A fake with a GPU API and no GPU, like a Mac in some virtual machines.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn without_device(mut self) -> Fake {
        self.answer = Answer::NoDevice;
        self
    }

    /// A fake whose GPU has no memory for any buffer.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn out_of_memory(mut self) -> Fake {
        self.answer = Answer::OutOfMemory;
        self
    }

    /// How many objects it holds.
    pub fn held(&self) -> usize {
        let state = self.state();
        state.devices.len() + state.buffers.len()
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl State {
    fn taken(&mut self, handle: Handle) -> bool {
        let taken = match handle {
            Handle::Device(id) => self.devices.contains(&id),
            Handle::Buffer(id) => self.buffers.contains_key(&id),
        } || self.released.contains(&handle);
        if taken {
            self.wrong.push(format!("{handle:?} made twice"));
        }
        taken
    }
}

impl Platform for Fake {
    fn device(&self, device: Id<Device>) -> Result<DeviceInfo, DeviceError> {
        match self.answer {
            Answer::Meet | Answer::OutOfMemory => {}
            Answer::NoDevice => return Err(DeviceError::NoDevice),
        }
        let mut state = self.state();
        if state.taken(Handle::Device(device)) {
            return Err(DeviceError::Fault(Fault::new("an id made twice")));
        }
        state.devices.insert(device);
        Ok(DeviceInfo::new("Fake GPU".into(), self.max_buffer_bytes))
    }

    fn buffer(&self, buffer: Id<Buffer>, bytes: BufferBytes<'_>) -> Result<(), BufferError> {
        let mut state = self.state();
        if !state.devices.contains(&bytes.device()) {
            let wrong = format!("a buffer on {:?}, which it does not hold", bytes.device());
            state.wrong.push(wrong.clone());
            return Err(BufferError::Fault(Fault::new(wrong)));
        }
        match self.answer {
            Answer::Meet | Answer::NoDevice => {}
            Answer::OutOfMemory => return Err(BufferError::OutOfMemory),
        }
        if state.taken(Handle::Buffer(buffer)) {
            return Err(BufferError::Fault(Fault::new("an id made twice")));
        }
        state.buffers.insert(buffer, bytes.bytes().to_vec());
        Ok(())
    }

    fn read(&self, to: ReadInto<'_>) -> Result<(), Fault> {
        let mut state = self.state();
        let id = to.buffer();
        let into = to.into_slice();
        match state.buffers.get(&id) {
            Some(bytes) if bytes.len() == into.len() => {
                into.copy_from_slice(bytes);
                Ok(())
            }
            Some(_) | None => {
                let wrong = format!("a read of {id:?} into {} bytes", into.len());
                state.wrong.push(wrong.clone());
                Err(Fault::new(wrong))
            }
        }
    }

    fn release(&self, handle: Handle) {
        let mut state = self.state();
        let held = match handle {
            Handle::Device(id) => state.devices.remove(&id),
            Handle::Buffer(id) => state.buffers.remove(&id).is_some(),
        };
        if !held {
            let wrong = if state.released.contains(&handle) {
                format!("{handle:?} released twice")
            } else {
                format!("{handle:?} released, never made")
            };
            state.wrong.push(wrong);
        }
        state.released.insert(handle);
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        let state = self.state();
        assert_eq!(state.wrong, Vec::<String>::new(), "the VM misused the fake");
        assert!(
            state.devices.is_empty() && state.buffers.is_empty(),
            "the fake still holds {:?} and {:?} once the runs using it are over",
            state.devices,
            state.buffers.keys().collect::<Vec<_>>()
        );
    }
}
