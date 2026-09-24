//! A thin layer over objc2-metal: device, queue, runtime shader compilation,
//! shared buffers, and command-buffer timing.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLResourceOptions,
};

pub type Device = ProtocolObject<dyn MTLDevice>;
pub type Queue = ProtocolObject<dyn MTLCommandQueue>;
pub type Buffer = ProtocolObject<dyn MTLBuffer>;
pub type Library = ProtocolObject<dyn MTLLibrary>;
pub type Function = ProtocolObject<dyn MTLFunction>;
pub type ComputePipeline = ProtocolObject<dyn MTLComputePipelineState>;
pub type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

const COMMON: &str = include_str!("../shaders/common.metal");
pub const SHADE_SRC: &str = include_str!("../shaders/shade.metal");
pub const MESH_SRC: &str = include_str!("../shaders/mesh.metal");
pub const CULL_SRC: &str = include_str!("../shaders/cull.metal");
pub const LIGHT_SRC: &str = include_str!("../shaders/light.metal");
pub const DRAW_SRC: &str = include_str!("../shaders/draw.metal");
pub const PROBE_SRC: &str = include_str!("../shaders/probe.metal");

pub struct Gpu {
    pub device: Retained<Device>,
    pub queue: Retained<Queue>,
}

impl Gpu {
    /// Panics without a Metal device: on macOS that is a failure, not a skip.
    pub fn new() -> Gpu {
        let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = device.newCommandQueue().expect("no command queue");
        Gpu { device, queue }
    }

    /// Compiles `common.metal` followed by each of `srcs`.
    pub fn library(&self, srcs: &[&str]) -> Retained<Library> {
        let mut text = COMMON.to_string();
        for s in srcs {
            text.push('\n');
            text.push_str(s);
        }
        let source = NSString::from_str(&text);
        match self
            .device
            .newLibraryWithSource_options_error(&source, None)
        {
            Ok(lib) => lib,
            Err(e) => panic!("shader compile failed: {}", e.localizedDescription()),
        }
    }

    pub fn function(&self, lib: &Library, name: &str) -> Retained<Function> {
        lib.newFunctionWithName(&NSString::from_str(name))
            .unwrap_or_else(|| panic!("no function {name}"))
    }

    pub fn compute_pipeline(&self, lib: &Library, name: &str) -> Retained<ComputePipeline> {
        let f = self.function(lib, name);
        match self.device.newComputePipelineStateWithFunction_error(&f) {
            Ok(p) => p,
            Err(e) => panic!("pipeline {name} failed: {}", e.localizedDescription()),
        }
    }

    /// A zero-filled shared buffer. Apple GPUs have unified memory, so shared
    /// storage costs no copy; the CPU writes in place.
    pub fn buffer(&self, bytes: usize) -> Retained<Buffer> {
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("could not allocate {bytes} bytes"));
        // SAFETY: the buffer is at least `bytes` long and nothing else uses it yet.
        unsafe { std::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>(), 0, bytes) };
        buf
    }

    pub fn buffer_with<T: Copy>(&self, data: &[T]) -> Retained<Buffer> {
        let buf = self.buffer(size_of_val(data));
        write(&buf, 0, data);
        buf
    }

    pub fn command_buffer(&self) -> Retained<CommandBuffer> {
        self.queue.commandBuffer().expect("no command buffer")
    }
}

impl Default for Gpu {
    fn default() -> Self {
        Gpu::new()
    }
}

/// Copies `data` into `buf` starting at element `at` (in units of `T`).
pub fn write<T: Copy>(buf: &Buffer, at: usize, data: &[T]) {
    let end = (at + data.len()) * size_of::<T>();
    assert!(end <= buf.length(), "write past the end of a buffer");
    // SAFETY: bounds checked above; `T: Copy` has no drop, and the GPU is not
    // reading this range (callers only write between command buffers).
    unsafe {
        let dst = buf.contents().as_ptr().cast::<T>().add(at);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

/// Copies one `T` into `buf` at a byte offset.
pub fn write_at<T: Copy>(buf: &Buffer, byte_at: usize, v: &T) {
    assert!(
        byte_at + size_of::<T>() <= buf.length(),
        "write past the end of a buffer"
    );
    // SAFETY: bounds checked above; `T: Copy` has no drop, and callers only
    // write between command buffers.
    unsafe {
        let dst = buf
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(byte_at)
            .cast::<T>();
        std::ptr::write_unaligned(dst, *v);
    }
}

/// Reads one `T` from `buf` at a byte offset.
pub fn read_at<T: Copy>(buf: &Buffer, byte_at: usize) -> T {
    assert!(
        byte_at + size_of::<T>() <= buf.length(),
        "read past the end of a buffer"
    );
    // SAFETY: bounds checked above, and callers only read after the GPU work
    // writing this buffer has completed.
    unsafe {
        std::ptr::read_unaligned(
            buf.contents()
                .as_ptr()
                .cast::<u8>()
                .add(byte_at)
                .cast::<T>(),
        )
    }
}

/// Reads `n` elements of `T` from the start of `buf`.
pub fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    assert!(
        n * size_of::<T>() <= buf.length(),
        "read past the end of a buffer"
    );
    // SAFETY: bounds checked above, and callers only read after the GPU work
    // writing this buffer has completed.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<T>(), n).to_vec() }
}

pub fn bytes_of<T>(v: &T) -> NonNull<std::ffi::c_void> {
    NonNull::from(v).cast()
}

/// Commits, waits, and panics if the GPU reported an error. Returns the GPU
/// time in milliseconds, from `GPUStartTime` to `GPUEndTime`.
pub fn submit(cb: &CommandBuffer) -> f64 {
    cb.commit();
    wait(cb)
}

/// Waits for a committed command buffer; see `submit`.
pub fn wait(cb: &CommandBuffer) -> f64 {
    cb.waitUntilCompleted();
    if cb.status() != MTLCommandBufferStatus::Completed {
        let err = cb.error().map(|e| e.localizedDescription().to_string());
        panic!("command buffer failed: {err:?}");
    }
    (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
}
