//! Minimal runtime probe for Metal IO compression and source-offset semantics.
//!
//! Run on a physical macOS system:
//! `cargo run --example metal_io_zlib_probe --release`

use metal_rust::metal::{
    Device, IOCommandQueueDescriptor, IOCompressionMethod, IoCompressionContext, ResourceOptions,
};
use std::env;
use std::fs;
use std::path::Path;

const HEADER_LEN: usize = 48;
const VERTEX_LEN: usize = 4096;
const INDEX_LEN: usize = 3072;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::system_default().ok_or("no Metal device")?;
    eprintln!("device: {}", device.name());

    let root = env::temp_dir().join(format!("metal-io-zlib-probe-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let path = root.join("payload.gpuio");

    let header = vec![0x48; HEADER_LEN];
    let vertex: Vec<u8> = (0..VERTEX_LEN)
        .map(|i| (i as u8).wrapping_mul(17))
        .collect();
    let index: Vec<u8> = (0..INDEX_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(3))
        .collect();

    write_compressed(&path, &header, &vertex, &index)?;
    let compressed_len = fs::metadata(&path)?.len();
    eprintln!(
        "logical bytes: {}, compressed bytes: {}",
        HEADER_LEN + VERTEX_LEN + INDEX_LEN,
        compressed_len
    );

    let descriptor = IOCommandQueueDescriptor::new()?;
    let io_queue = device.new_io_command_queue(&descriptor)?;
    let handle =
        device.new_io_file_handle(&path, Some(IOCompressionMethod::IOCompressionMethodZlib))?;

    let logical_result = load_and_read_back(
        &device, &io_queue, &handle, "logical", HEADER_LEN, &vertex, &index,
    )?;
    report_case("logical-offset", &logical_result, &vertex, &index);

    // The raw-file-offset hypothesis is intentionally tested separately. A
    // successful command is not enough: compare the bytes to distinguish a
    // valid read from a valid-but-wrong source offset.
    let raw_result = load_and_read_back(&device, &io_queue, &handle, "raw", 0, &vertex, &index)?;
    report_case("raw-offset", &raw_result, &vertex, &index);

    if logical_result.0 != vertex || logical_result.1 != index {
        return Err("logical offsets completed but returned incorrect bytes".into());
    }
    if raw_result.0 == vertex && raw_result.1 == index {
        return Err("raw offset unexpectedly matched the logical payload".into());
    }
    println!("PASS: source offsets address the decompressed logical stream");

    let _ = fs::remove_dir_all(&root);
    Ok(())
}

fn load_and_read_back(
    device: &Device,
    io_queue: &metal_rust::metal::IoCommandQueue,
    handle: &metal_rust::metal::IoFileHandle,
    label: &str,
    vertex_offset: usize,
    expected_vertex: &[u8],
    expected_index: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
    let vertex_buffer = device.new_buffer(VERTEX_LEN, ResourceOptions::PRIVATE)?;
    let index_buffer = device.new_buffer(INDEX_LEN, ResourceOptions::PRIVATE)?;

    let mut io_commands = io_queue.command_buffer()?;
    io_commands.set_label(Some(&format!("Metal IO zlib {label} offset probe")));
    io_commands.load_buffer(&vertex_buffer, 0, VERTEX_LEN, handle, vertex_offset)?;
    io_commands.load_buffer(
        &index_buffer,
        0,
        INDEX_LEN,
        handle,
        vertex_offset + VERTEX_LEN,
    )?;
    io_commands.commit().wait()?;

    // Use the regular Metal command queue only for a readback of the PRIVATE
    // buffers. This does not alter the IO load path being measured.
    let queue = device.new_command_queue(None)?;
    let mut commands = queue.command_buffer()?;
    let blit = commands.blit_encoder()?;
    let vertex_readback = blit.read_buffer(&vertex_buffer, 0, VERTEX_LEN)?;
    let index_readback = blit.read_buffer(&index_buffer, 0, INDEX_LEN)?;
    blit.end_encoding();
    let completed = commands.commit().wait()?;
    let actual_vertex = completed.resolve_buffer(vertex_readback)?;
    let actual_index = completed.resolve_buffer(index_readback)?;

    if actual_vertex.len() != expected_vertex.len() || actual_index.len() != expected_index.len() {
        return Err("readback length mismatch".into());
    }
    Ok((actual_vertex, actual_index))
}

fn report_case(name: &str, result: &(Vec<u8>, Vec<u8>), vertex: &[u8], index: &[u8]) {
    let vertex_ok = result.0 == vertex;
    let index_ok = result.1 == index;
    eprintln!("{name}: vertex_match={vertex_ok}, index_match={index_ok}");
}

fn write_compressed(
    path: &Path,
    header: &[u8],
    vertex: &[u8],
    index: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let chunk_size = IoCompressionContext::default_chunk_size()?;
    eprintln!("Metal preferred compression chunk: {chunk_size} bytes");
    let mut context = IoCompressionContext::new(
        path,
        IOCompressionMethod::IOCompressionMethodZlib,
        chunk_size,
    )?;
    context.append(header)?;
    context.append(vertex)?;
    context.append(index)?;
    context.finish()?;
    Ok(())
}
