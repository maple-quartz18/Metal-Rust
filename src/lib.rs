#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Safe Rust interfaces for Apple Metal.
//!
//! Raw Objective-C objects and unsafe operations are confined to the audited
//! `metal-rust-ffi` implementation crate. This public crate never exposes
//! object pointers or requires callers to uphold Objective-C runtime
//! invariants.
//!
//! The following visibility fixture must fail to compile: an application can
//! own a device, but cannot extract its implementation object or bypass the
//! safe boundary.
//!
//! ```compile_fail
//! let device = metal_rust::Device::system_default().unwrap();
//! let _implementation_object = device.inner;
//! ```
//!
//! Buffer readback is available for CPU-visible shared buffers and remains
//! range-checked. Callers are responsible for synchronizing the relevant
//! producer work before reading resource contents.
//!
//! ```compile_fail
//! let device = metal_rust::Device::system_default().unwrap();
//! let descriptor = metal_rust::TextureDescriptor::new_2d(
//!     metal_rust::PixelFormat::RGBA8_UNORM,
//!     1,
//!     1,
//!     metal_rust::ResourceOptions::SHARED,
//!     metal_rust::TextureUsage::SHADER_READ,
//! )
//! .unwrap();
//! let texture = device.new_texture(&descriptor).unwrap();
//! let region = metal_rust::Region::new(
//!     metal_rust::Origin::new(0, 0, 0),
//!     metal_rust::Size::new(1, 1, 1),
//! );
//! let _bytes = texture.read_region(region, 4);
//! ```
//!
//! ```compile_fail
//! let device = metal_rust::Device::system_default().unwrap();
//! let heap = device.new_timestamp_counter_heap(1).unwrap();
//! let _timestamps = heap.resolve(0..1);
//! ```
//!
//! Submission consumes the recording-state command buffer, so it cannot be
//! committed twice:
//!
//! ```compile_fail
//! let device = metal_rust::Device::system_default().unwrap();
//! let queue = device.new_command_queue(None).unwrap();
//! let command_buffer = queue.command_buffer().unwrap();
//! let submitted = command_buffer.commit();
//! let _ = command_buffer.commit();
//! let _ = submitted;
//! ```
//!
//! An encoder's exclusive borrow also prevents submission while encoding is
//! active:
//!
//! ```compile_fail
//! let device = metal_rust::Device::system_default().unwrap();
//! let queue = device.new_command_queue(None).unwrap();
//! let mut command_buffer = queue.command_buffer().unwrap();
//! let encoder = command_buffer.blit_encoder().unwrap();
//! let _submitted = command_buffer.commit();
//! encoder.end_encoding();
//! ```

#[path = "Foundation/Foundation.rs"]
pub mod foundation;
#[path = "Private/GeneratedFacade.rs"]
mod generated_facade;
#[path = "Metal/Metal.rs"]
pub mod metal;
#[path = "Metal4/Metal4.rs"]
pub mod metal4;
#[path = "Metal4FX/Metal4FX.rs"]
pub mod metal4_fx;
#[path = "MetalFX/MetalFX.rs"]
pub mod metal_fx;
#[path = "QuartzCore/QuartzCore.rs"]
pub mod quartz_core;
pub mod metal_io;

pub use foundation::{Error, ErrorKind};
pub use metal::{
    BlitCommandEncoder, Buffer, BufferReadback, CaptureSession, ClearColor, CommandBuffer,
    CommandBufferStatus, CommandQueue, CompileOptions, CompletedCommandBuffer,
    ComputeCommandEncoder, ComputePassDescriptor, ComputePipelineState, Device, DeviceCapability,
    DeviceNumericProperty, Function, Library, MAX_TIMESTAMP_COUNTERS, Origin, PixelFormat,
    PrimitiveType, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPassOptions,
    RenderPipelineDescriptor, RenderPipelineOptions, RenderPipelineState, ResourceOptions, Size,
    StorageMode, SubmittedCommandBuffer, Texture, TextureDescriptor, TextureReadback,
    TextureReadbackData, TextureType, TextureUsage, TimestampCounterHeap, Viewport,
};
pub use metal_fx::{
    Matrix4x4, MatrixValidationError, SpatialScaler, SpatialScalerColorProcessingMode,
    SpatialScalerDescriptor, TemporalScaler, TemporalScalerDescriptor,
};
pub use quartz_core::{Drawable, Layer};
