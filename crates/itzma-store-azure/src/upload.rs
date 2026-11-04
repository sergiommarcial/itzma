//! Multipart uploads as Azure block blobs: each part is staged as a block and
//! `complete` commits the list in part order.

use std::sync::Arc;

use async_trait::async_trait;
use azure_core::http::{NoFormat, RequestContent};
use azure_storage_blob::BlobContainerClient;
use azure_storage_blob::models::{BlockBlobClientCommitBlockListOptions, BlockLookupList};
use bytes::Bytes;
use object_store::{MultipartUpload, PutPayload, PutResult, Result, UploadPart};

use crate::errors::{self, Write};

pub(crate) struct Staged {
    pub(crate) container: Arc<BlobContainerClient>,
    pub(crate) name: String,
    /// Random per upload, so two uploads to one key never stage blocks under
    /// the same id (Azure would let the later one replace the earlier's).
    pub(crate) nonce: [u8; 8],
    /// Block ids in part order: the nonce, then the part number, 16 bytes,
    /// the same length the SDK's own staged uploads use. Azure refuses blocks
    /// of another length on a blob that still holds uncommitted ones.
    pub(crate) blocks: Vec<Vec<u8>>,
}

/// Eight bytes no other upload in any process is likely to draw.
pub(crate) fn nonce() -> [u8; 8] {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    // RandomState is seeded randomly per process; the counter and the clock
    // separate uploads within one.
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(NEXT.fetch_add(1, Ordering::Relaxed));
    hasher.write_u32(std::process::id());
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    );
    hasher.finish().to_be_bytes()
}

impl std::fmt::Debug for Staged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Staged")
            .field("name", &self.name)
            .field("blocks", &self.blocks.len())
            .finish()
    }
}

#[async_trait]
impl MultipartUpload for Staged {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let mut id = self.nonce.to_vec();
        id.extend_from_slice(&(self.blocks.len() as u64).to_be_bytes());
        self.blocks.push(id.clone());
        let (container, name) = (self.container.clone(), self.name.clone());
        Box::pin(async move {
            let body = Bytes::from(data);
            container
                .blob_client(&name)
                .block_blob_client()
                .stage_block(
                    &id,
                    body.len() as u64,
                    <RequestContent<Bytes, NoFormat> as From<Bytes>>::from(body),
                    None,
                )
                .await
                .map_err(|e| errors::map(e, &name, Write::None))?;
            Ok(())
        })
    }

    async fn complete(&mut self) -> Result<PutResult> {
        // Kept until the commit succeeds, so a retried `complete` commits the
        // same parts instead of an empty list.
        let list = BlockLookupList {
            latest: Some(self.blocks.clone()),
            ..Default::default()
        };
        let content = RequestContent::try_from(list).map_err(|e| errors::generic(e.to_string()))?;
        let options = BlockBlobClientCommitBlockListOptions {
            blob_content_type: Some("application/octet-stream".into()),
            ..Default::default()
        };
        let response = self
            .container
            .blob_client(&self.name)
            .block_blob_client()
            .commit_block_list(content, Some(options))
            .await
            .map_err(|e| errors::map(e, &self.name, Write::None))?;
        let e_tag = {
            use azure_storage_blob::models::BlockBlobClientCommitBlockListResultHeaders;
            response
                .etag()
                .map_err(|e| errors::map(e, &self.name, Write::None))?
                .map(|e| e.to_string())
        };
        self.blocks.clear();
        Ok(PutResult {
            e_tag,
            version: None,
            extensions: Default::default(),
        })
    }

    /// Staged blocks can't be removed: Azure discards uncommitted blocks after a
    /// week. Their ids match the SDK's length, so they never block a later put.
    async fn abort(&mut self) -> Result<()> {
        self.blocks.clear();
        Ok(())
    }
}
