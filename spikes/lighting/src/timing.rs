//! GPU timestamps at encoder / stage boundaries, through an MTLCounterSampleBuffer.
//! Unlike `GPUStartTime`/`GPUEndTime` this times passes inside one command
//! buffer, so there is no idle gap between them for the GPU to clock down in.

use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLStorageMode,
};

use crate::gpu::Gpu;

/// Sample slots: cull start/end, vertex start/end, fragment start/end.
pub const SAMPLES: usize = 6;

pub struct Timestamps {
    pub buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    ns_per_tick: f64,
}

impl Timestamps {
    /// `None` when the device cannot sample at stage boundaries.
    pub fn new(gpu: &Gpu) -> Option<Timestamps> {
        let device = &gpu.device;
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return None;
        }
        // SAFETY: an extern NSString constant.
        let wanted = unsafe { MTLCommonCounterSetTimestamp }.to_string();
        let sets = device.counterSets()?;
        let set = sets
            .iter()
            .find(|s: &Retained<ProtocolObject<dyn MTLCounterSet>>| {
                s.name().to_string() == wanted
            })?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a small positive count.
        unsafe { desc.setSampleCount(SAMPLES) };
        let buffer = device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .ok()?;

        // GPU timestamp units are not documented as nanoseconds; calibrate
        // them against the wall clock.
        let sample = || {
            let (mut cpu, mut gpu_t) = (0u64, 0u64);
            // SAFETY: two valid out-pointers.
            unsafe {
                device.sampleTimestamps_gpuTimestamp(
                    std::ptr::NonNull::from(&mut cpu),
                    std::ptr::NonNull::from(&mut gpu_t),
                )
            };
            (Instant::now(), gpu_t)
        };
        let (t0, g0) = sample();
        std::thread::sleep(Duration::from_millis(50));
        let (t1, g1) = sample();
        let ns_per_tick = (t1 - t0).as_nanos() as f64 / (g1 - g0) as f64;
        Some(Timestamps {
            buffer,
            ns_per_tick,
        })
    }

    /// Milliseconds between sample pairs (0,1), (2,3), (4,5), and (0,5).
    pub fn resolve(&self) -> Option<[f64; 4]> {
        // SAFETY: the range is within the buffer's sample count.
        let data = unsafe { self.buffer.resolveCounterRange(NSRange::new(0, SAMPLES)) }?;
        let bytes = data.to_vec();
        let t: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if t.iter().any(|&v| v == 0 || v == u64::MAX) {
            return None;
        }
        let ms = |a: usize, b: usize| t[b].saturating_sub(t[a]) as f64 * self.ns_per_tick / 1e6;
        Some([ms(0, 1), ms(2, 3), ms(4, 5), ms(0, 5)])
    }
}
