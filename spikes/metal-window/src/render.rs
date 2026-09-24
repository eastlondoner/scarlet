//! Metal on the VM thread: a clear colour that changes with time and a
//! rotating triangle, drawn into the CAMetalLayer's next drawable.

use std::ptr::NonNull;
use std::time::{Duration, Instant};

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDrawable, MTLLibrary, MTLLoadAction,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction, MTLTexture,
};
use objc2_quartz_core::CAMetalDrawable;

use crate::app::DrawableSource;

use crate::model::AppError;

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color; };
vertex VOut vs(uint vid [[vertex_id]], constant float &angle [[buffer(0)]]) {
    const float2 p[3] = { float2(0.0, 0.6), float2(-0.52, -0.3), float2(0.52, -0.3) };
    const float3 c[3] = { float3(1, 0.2, 0.2), float3(0.2, 1, 0.2), float3(0.2, 0.2, 1) };
    float s = sin(angle), k = cos(angle);
    VOut o;
    o.pos = float4(p[vid].x * k - p[vid].y * s, p[vid].x * s + p[vid].y * k, 0, 1);
    o.color = float4(c[vid], 1);
    return o;
}
fragment float4 fs(VOut in [[stage_in]]) { return in.color; }
"#;

pub struct Renderer {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    started: Instant,
}

pub struct Drawn {
    /// Time blocked in `nextDrawable`.
    pub drawable_wait: Duration,
    /// The drawable's texture size, to compare with the frame's size.
    pub texture_size: (u32, u32),
}

impl Renderer {
    pub fn new() -> Result<Renderer, AppError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(AppError::NoMetalDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(AppError::Unsupported("newCommandQueue returned nil".into()))?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADER), None)
            .map_err(|e| AppError::Unsupported(e.localizedDescription().to_string()))?;
        let vs = library
            .newFunctionWithName(&NSString::from_str("vs"))
            .ok_or(AppError::Unsupported("no vs".into()))?;
        let fs = library
            .newFunctionWithName(&NSString::from_str("fs"))
            .ok_or(AppError::Unsupported("no fs".into()))?;
        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(&vs));
        desc.setFragmentFunction(Some(&fs));
        // SAFETY: index 0 is within the colour attachment array (8 slots).
        let att = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
        // The layer's format; a mismatch here is the SIGABRT metal-design.md measured.
        att.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .map_err(|e| AppError::Unsupported(e.localizedDescription().to_string()))?;
        Ok(Renderer {
            device,
            queue,
            pipeline,
            started: Instant::now(),
        })
    }

    /// Encodes and presents one frame. Returns `Ok(None)` when no drawable was
    /// available (nextDrawable timed out after ~1 s, or the layer has no size).
    pub fn draw(&self, source: DrawableSource<'_>) -> Result<Option<Drawn>, AppError> {
        // The VM thread has no autorelease pool of its own; without one per
        // frame, autoreleased Metal/CA objects pile up until the thread exits.
        autoreleasepool(|_| {
            let t = self.started.elapsed().as_secs_f64();
            let before = Instant::now();
            let drawable = match source {
                DrawableSource::Next(layer) => layer.nextDrawable(),
                DrawableSource::Delivered(d) => d,
            };
            let Some(drawable) = drawable else {
                return Ok(None);
            };
            let drawable_wait = before.elapsed();
            let texture = drawable.texture();
            let texture_size = (texture.width() as u32, texture.height() as u32);

            let pass = MTLRenderPassDescriptor::new();
            // SAFETY: index 0 is within the colour attachment array.
            let ca = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            ca.setTexture(Some(&texture));
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setStoreAction(MTLStoreAction::Store);
            ca.setClearColor(MTLClearColor {
                red: 0.5 + 0.5 * (t * 0.7).sin(),
                green: 0.5 + 0.5 * (t * 1.1 + 2.0).sin(),
                blue: 0.5 + 0.5 * (t * 1.3 + 4.0).sin(),
                alpha: 1.0,
            });

            let cmd = self
                .queue
                .commandBuffer()
                .ok_or(AppError::Unsupported("commandBuffer returned nil".into()))?;
            let enc = cmd
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or(AppError::Unsupported("encoder returned nil".into()))?;
            enc.setRenderPipelineState(&self.pipeline);
            let angle = t as f32;
            // SAFETY: 4 bytes read from a live f32, bound at the index the shader reads.
            unsafe {
                enc.setVertexBytes_length_atIndex(NonNull::from(&angle).cast(), 4, 0);
            }
            // SAFETY: the pipeline's vertex function reads vertex_id 0..3 only.
            unsafe {
                enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            enc.endEncoding();
            let as_drawable: &ProtocolObject<dyn MTLDrawable> =
                ProtocolObject::from_ref(&*drawable);
            match crate::appkit::knob("SPIKE_PRESENT_MIN_MS") {
                Some(ms) => cmd.presentDrawable_afterMinimumDuration(as_drawable, ms / 1000.0),
                None => cmd.presentDrawable(as_drawable),
            }
            cmd.commit();
            Ok(Some(Drawn {
                drawable_wait,
                texture_size,
            }))
        })
    }
}
