//! Metal-native compressed model cache files.
//!
//! The container is a Metal IO compression stream containing:
//! `[header][vertex bytes][index bytes]`.
//! The header is duplicated in a small plaintext sidecar so cache validation
//! never needs to read or decompress the payload on the CPU.

use crate::Error;
use crate::metal::{
    Buffer, Device, IOCommandQueueDescriptor, IOCompressionMethod, IoCommandQueue,
    IoCompressionContext, ResourceOptions,
};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Directory names reserved for the uncompressed legacy and zlib formats.
/// Legacy `SXMTLIO2` cache directory.
pub const LEGACY_CACHE_DIR: &str = "metal_io_models_v2";
/// Metal-native zlib cache directory.
pub const ZLIB_CACHE_DIR: &str = "metal_io_models_zlib";
/// Number of bytes in the duplicated plaintext metadata header.
pub const HEADER_LEN: usize = 48;
/// Magic stored at the start of every compressed-cache header.
pub const MAGIC: [u8; 8] = *b"SXMTLIOZ";
/// Current compressed-cache header version.
pub const VERSION: u32 = 1;

/// Selects which model-cache directory a caller should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheFormat {
    /// Existing `SXMTLIO2` cache, owned by the caller.
    Legacy,
    /// Metal-native zlib cache written by this module.
    Zlib,
}

impl CacheFormat {
    /// Returns the stable directory name for this format.
    #[must_use]
    pub const fn directory_name(self) -> &'static str {
        match self {
            Self::Legacy => LEGACY_CACHE_DIR,
            Self::Zlib => ZLIB_CACHE_DIR,
        }
    }
}

/// Returns the directory for a selected cache format below `cache_root`.
#[must_use]
pub fn cache_directory(cache_root: &Path, format: CacheFormat) -> PathBuf {
    cache_root.join(format.directory_name())
}

/// Input data for one compressed model cache entry.
#[derive(Clone, Copy, Debug)]
pub struct ModelPayload<'a> {
    /// Stable identity used to reject stale cache entries.
    pub identity: &'a str,
    /// Vertex payload written after the header.
    pub vertex: &'a [u8],
    /// Index payload written after the vertex payload.
    pub index: &'a [u8],
}

/// Metadata stored in the plaintext sidecar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheMetadata {
    /// Vertex payload length in bytes.
    pub vertex_len: u64,
    /// Index payload length in bytes.
    pub index_len: u64,
    /// CRC32 of the vertex payload.
    pub vertex_crc32: u32,
    /// CRC32 of the index payload.
    pub index_crc32: u32,
    /// CRC32 of the identity string.
    pub identity_crc32: u32,
}

/// Compressed payload and sidecar paths for one cache entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachePaths {
    /// Metal IO compression container path.
    pub payload: PathBuf,
    /// Plaintext metadata sidecar path.
    pub metadata: PathBuf,
}

/// Builds zlib cache paths below a format-specific cache directory.
#[must_use]
pub fn cache_paths(cache_root: &Path, key: &str) -> CachePaths {
    let directory = cache_directory(cache_root, CacheFormat::Zlib);
    CachePaths {
        payload: directory.join(format!("{key}.gpuio")),
        metadata: directory.join(format!("{key}.meta")),
    }
}

/// Successful result of sidecar validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedCache {
    /// Metadata read from the sidecar.
    pub metadata: CacheMetadata,
}

/// Writes a compressed cache entry.
///
/// Returns `false` when either destination already exists.
pub fn write(
    paths: &CachePaths,
    payload: ModelPayload<'_>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let payload_exists = paths.payload.exists();
    let metadata_exists = paths.metadata.exists();
    if payload_exists && metadata_exists {
        return Ok(false);
    }
    if payload_exists || metadata_exists {
        return Err(
            "Metal IO cache has an incomplete payload/metadata pair; remove both files before rewriting"
                .into(),
        );
    }
    if let Some(parent) = paths.payload.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = paths.metadata.parent() {
        fs::create_dir_all(parent)?;
    }

    let header = encode_header(payload)?;
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let payload_tmp = temporary_path(&paths.payload, &nonce);
    let metadata_tmp = temporary_path(&paths.metadata, &nonce);

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        cleanup_temporary_files(paths)?;
        write_new_file(&metadata_tmp, &header)?;
        let chunk_size = IoCompressionContext::default_chunk_size()?;
        let mut context = IoCompressionContext::new(
            &payload_tmp,
            IOCompressionMethod::IOCompressionMethodZlib,
            chunk_size,
        )?;
        context.append(&header)?;
        context.append(payload.vertex)?;
        context.append(payload.index)?;
        context.finish()?;
        sync_parent(paths.payload.parent())?;
        fs::rename(&payload_tmp, &paths.payload)?;
        fs::rename(&metadata_tmp, &paths.metadata)?;
        sync_parent(paths.metadata.parent())?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&payload_tmp);
        let _ = fs::remove_file(&metadata_tmp);
    }
    result.map(|()| true)
}

fn temporary_path(path: &Path, nonce: &str) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    path.with_file_name(format!(".{name}.{nonce}.tmp"))
}

fn cleanup_temporary_files(paths: &CachePaths) -> Result<(), Box<dyn std::error::Error>> {
    for path in [&paths.payload, &paths.metadata] {
        let Some(directory) = path.parent() else {
            continue;
        };
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let prefix = format!(".{file_name}.");
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".tmp") {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    Ok(())
}

fn sync_parent(parent: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = parent {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

/// Validates the sidecar, identity, lengths, and payload presence.
pub fn validate(
    paths: &CachePaths,
    identity: &str,
    vertex_len: usize,
    index_len: usize,
) -> Result<ValidatedCache, Box<dyn std::error::Error>> {
    let metadata = read_metadata(&paths.metadata)?;
    if metadata.identity_crc32 != crc32(identity.as_bytes()) {
        return Err("Metal IO cache identity mismatch".into());
    }
    if metadata.vertex_len != vertex_len as u64 || metadata.index_len != index_len as u64 {
        return Err("Metal IO cache segment length mismatch".into());
    }
    if !paths.payload.is_file() {
        return Err("Metal IO cache payload is missing".into());
    }
    Ok(ValidatedCache { metadata })
}

/// Loads vertex and index payloads into PRIVATE Metal buffers.
pub fn read(
    device: &Device,
    io_queue: &IoCommandQueue,
    paths: &CachePaths,
    identity: &str,
    vertex_len: usize,
    index_len: usize,
) -> Result<(Buffer, Buffer), Box<dyn std::error::Error>> {
    validate(paths, identity, vertex_len, index_len)?;
    let handle = device.new_io_file_handle(
        &paths.payload,
        Some(IOCompressionMethod::IOCompressionMethodZlib),
    )?;
    let vertex_buffer = device.new_buffer(vertex_len, ResourceOptions::PRIVATE)?;
    let index_buffer = device.new_buffer(index_len, ResourceOptions::PRIVATE)?;
    let mut commands = io_queue.command_buffer()?;
    commands.load_buffer(&vertex_buffer, 0, vertex_len, &handle, HEADER_LEN)?;
    commands.load_buffer(
        &index_buffer,
        0,
        index_len,
        &handle,
        HEADER_LEN
            .checked_add(vertex_len)
            .ok_or("Metal IO cache source offset overflow")?,
    )?;
    commands.commit().wait()?;
    Ok((vertex_buffer, index_buffer))
}

/// Creates a default Metal IO command queue.
pub fn new_io_queue(device: &Device) -> Result<IoCommandQueue, Error> {
    let descriptor = IOCommandQueueDescriptor::new()?;
    device.new_io_command_queue(&descriptor)
}

fn encode_header(
    payload: ModelPayload<'_>,
) -> Result<[u8; HEADER_LEN], Box<dyn std::error::Error>> {
    let vertex_len = u64::try_from(payload.vertex.len())?;
    let index_len = u64::try_from(payload.index.len())?;
    let mut header = [0u8; HEADER_LEN];
    header[..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..20].copy_from_slice(&vertex_len.to_le_bytes());
    header[20..28].copy_from_slice(&index_len.to_le_bytes());
    header[28..32].copy_from_slice(&crc32(payload.vertex).to_le_bytes());
    header[32..36].copy_from_slice(&crc32(payload.index).to_le_bytes());
    header[36..40].copy_from_slice(&crc32(payload.identity.as_bytes()).to_le_bytes());
    Ok(header)
}

fn read_metadata(path: &Path) -> Result<CacheMetadata, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut header = [0u8; HEADER_LEN];
    file.read_exact(&mut header)?;
    if file.metadata()?.len() != HEADER_LEN as u64 {
        return Err("Metal IO cache metadata has an invalid size".into());
    }
    if header[..8] != MAGIC {
        return Err("Metal IO cache magic mismatch".into());
    }
    if u32::from_le_bytes(header[8..12].try_into()?) != VERSION {
        return Err("Metal IO cache version mismatch".into());
    }
    Ok(CacheMetadata {
        vertex_len: u64::from_le_bytes(header[12..20].try_into()?),
        index_len: u64::from_le_bytes(header[20..28].try_into()?),
        vertex_crc32: u32::from_le_bytes(header[28..32].try_into()?),
        index_crc32: u32::from_le_bytes(header[32..36].try_into()?),
        identity_crc32: u32::from_le_bytes(header[36..40].try_into()?),
    })
}

fn write_new_file(path: &Path, bytes: &[u8; HEADER_LEN]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn header_contains_lengths_and_checksums() {
        let payload = ModelPayload {
            identity: "model|v=1|i=2",
            vertex: b"vertex",
            index: b"index",
        };
        let header = encode_header(payload).unwrap();
        assert_eq!(&header[..8], b"SXMTLIOZ");
        assert_eq!(
            u64::from_le_bytes(header[12..20].try_into().unwrap()),
            payload.vertex.len() as u64
        );
        assert_eq!(
            u32::from_le_bytes(header[36..40].try_into().unwrap()),
            crc32(payload.identity.as_bytes())
        );
    }

    #[test]
    fn metadata_rejects_wrong_magic() {
        let path =
            std::env::temp_dir().join(format!("metal-io-metadata-test-{}", std::process::id()));
        let mut header = [0u8; HEADER_LEN];
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        write_new_file(&path, &header).unwrap();
        assert!(read_metadata(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn format_directories_are_disjoint() {
        let root = Path::new("/cache");
        assert_eq!(
            cache_directory(root, CacheFormat::Legacy),
            root.join("metal_io_models_v2")
        );
        assert_eq!(
            cache_directory(root, CacheFormat::Zlib),
            root.join("metal_io_models_zlib")
        );
        assert_ne!(
            cache_paths(root, "abc").payload,
            root.join("metal_io_models_v2/abc.buf")
        );
    }

    #[test]
    fn stale_temporary_files_are_removed() {
        let root =
            std::env::temp_dir().join(format!("metal-io-cleanup-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let paths = CachePaths {
            payload: root.join("model.gpuio"),
            metadata: root.join("model.meta"),
        };
        let stale_payload = root.join(".model.gpuio.old.tmp");
        let stale_metadata = root.join(".model.meta.old.tmp");
        File::create(&stale_payload).unwrap();
        File::create(&stale_metadata).unwrap();
        cleanup_temporary_files(&paths).unwrap();
        assert!(!stale_payload.exists());
        assert!(!stale_metadata.exists());
        let _ = fs::remove_dir_all(root);
    }
}
