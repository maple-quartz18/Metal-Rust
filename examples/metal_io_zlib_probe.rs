//! Runtime probe for the standalone Metal IO zlib cache module.
//!
//! Run on a physical macOS system:
//! `cargo run --example metal_io_zlib_probe --release`

use metal_rust::metal::Device;
use metal_rust::metal_io::{self, CachePaths, HEADER_LEN, ModelPayload};
use std::env;
use std::fs;

const VERTEX_LEN: usize = 4096;
const INDEX_LEN: usize = 3072;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::system_default().ok_or("no Metal device")?;
    eprintln!("device: {}", device.name());

    let root = env::temp_dir().join(format!("metal-io-zlib-probe-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let paths = CachePaths {
        payload: root.join("payload.gpuio"),
        metadata: root.join("payload.meta"),
    };
    let vertex: Vec<u8> = (0..VERTEX_LEN)
        .map(|i| (i as u8).wrapping_mul(17))
        .collect();
    let index: Vec<u8> = (0..INDEX_LEN)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(3))
        .collect();
    let payload = ModelPayload {
        identity: "metal-io-zlib-probe",
        vertex: &vertex,
        index: &index,
    };

    metal_io::write(&paths, payload)?;
    let compressed_len = fs::metadata(&paths.payload)?.len();
    eprintln!(
        "logical bytes: {}, compressed bytes: {}",
        HEADER_LEN + VERTEX_LEN + INDEX_LEN,
        compressed_len
    );
    metal_io::validate(&paths, payload.identity, VERTEX_LEN, INDEX_LEN)?;

    let io_queue = metal_io::new_io_queue(&device)?;
    let (vertex_buffer, index_buffer) = metal_io::read(
        &device,
        &io_queue,
        &paths,
        payload.identity,
        VERTEX_LEN,
        INDEX_LEN,
    )?;

    let queue = device.new_command_queue(None)?;
    let mut commands = queue.command_buffer()?;
    let blit = commands.blit_encoder()?;
    let vertex_readback = blit.read_buffer(&vertex_buffer, 0, VERTEX_LEN)?;
    let index_readback = blit.read_buffer(&index_buffer, 0, INDEX_LEN)?;
    blit.end_encoding();
    let completed = commands.commit().wait()?;
    let actual_vertex = completed.resolve_buffer(vertex_readback)?;
    let actual_index = completed.resolve_buffer(index_readback)?;

    let vertex_match = actual_vertex == vertex;
    let index_match = actual_index == index;
    eprintln!("vertex_match={vertex_match}, index_match={index_match}");
    if !vertex_match || !index_match {
        return Err("Metal IO cache readback did not match the source payload".into());
    }

    println!("PASS: compressed write, sidecar validation, PRIVATE load, and readback");
    let _ = fs::remove_dir_all(root);
    Ok(())
}
