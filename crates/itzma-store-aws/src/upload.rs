//! Multipart uploads as S3 multipart uploads: parts go up as they come, and
//! `complete` sends their ETags in part order. S3 needs 5 MiB in every part
//! but the last, so smaller parts are held until they add up to that.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use bytes::{Bytes, BytesMut};
use object_store::{MultipartUpload, PutPayload, PutResult, Result, UploadPart};

use crate::errors::{self, Write};

/// S3's minimum size for every part but the last.
const MIN_PART: usize = 5 * 1024 * 1024;

pub(crate) struct Upload {
    pub(crate) client: Arc<Client>,
    pub(crate) bucket: String,
    pub(crate) key: String,
    pub(crate) upload_id: String,
    /// Each part's ETag by index, filled in as its upload finishes.
    pub(crate) parts: Arc<Mutex<Vec<Option<String>>>>,
    /// Parts too small to send yet, joined.
    pub(crate) pending: BytesMut,
    /// The part `complete` adds: what is pending, or the one empty part an
    /// upload with none needs. Kept so a retried `complete` can resend it.
    pub(crate) last: Option<Bytes>,
}

impl std::fmt::Debug for Upload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upload")
            .field("key", &self.key)
            .field("upload_id", &self.upload_id)
            .finish()
    }
}

impl Upload {
    /// Uploads `data` as the part at `index`, recording its ETag there.
    fn send(&self, index: usize, data: Bytes) -> UploadPart {
        let (client, bucket, key, upload_id, parts) = (
            self.client.clone(),
            self.bucket.clone(),
            self.key.clone(),
            self.upload_id.clone(),
            self.parts.clone(),
        );
        Box::pin(async move {
            let number = i32::try_from(index + 1)
                .map_err(|_| errors::generic("too many parts for one upload"))?;
            let response = client
                .upload_part()
                .bucket(bucket)
                .key(&key)
                .upload_id(upload_id)
                .part_number(number)
                .body(ByteStream::from(data))
                .send()
                .await
                .map_err(|e| errors::map(e, &key, Write::None))?;
            let e_tag = response
                .e_tag()
                .ok_or_else(|| errors::generic(format!("part {number} of {key} has no ETag")))?;
            parts.lock().unwrap_or_else(|e| e.into_inner())[index] = Some(e_tag.to_string());
            Ok(())
        })
    }
}

#[async_trait]
impl MultipartUpload for Upload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let data = Bytes::from(data);
        let data = if self.pending.is_empty() && data.len() >= MIN_PART {
            data
        } else {
            self.pending.extend_from_slice(&data);
            if self.pending.len() < MIN_PART {
                return Box::pin(async { Ok(()) });
            }
            std::mem::take(&mut self.pending).freeze()
        };
        let index = {
            let mut parts = self.parts.lock().unwrap_or_else(|e| e.into_inner());
            parts.push(None);
            parts.len() - 1
        };
        self.send(index, data)
    }

    async fn complete(&mut self) -> Result<PutResult> {
        // The pending bytes go up as the last part. S3 refuses to complete
        // an upload with no parts, so an empty object is one empty part. A
        // retried `complete` sends that part again if it failed.
        let last = {
            let mut parts = self.parts.lock().unwrap_or_else(|e| e.into_inner());
            if self.last.is_none() && (parts.is_empty() || !self.pending.is_empty()) {
                parts.push(None);
                self.last = Some(std::mem::take(&mut self.pending).freeze());
            }
            let index = parts.len() - 1;
            match &self.last {
                Some(data) if parts[index].is_none() => Some((index, data.clone())),
                _ => None,
            }
        };
        if let Some((index, data)) = last {
            self.send(index, data).await?;
        }
        let parts = self
            .parts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .enumerate()
            .map(|(i, e_tag)| {
                let e_tag = e_tag
                    .clone()
                    .ok_or_else(|| errors::generic(format!("part {} never finished", i + 1)))?;
                Ok(CompletedPart::builder()
                    .part_number(i32::try_from(i + 1).unwrap_or(i32::MAX))
                    .e_tag(e_tag)
                    .build())
            })
            .collect::<Result<Vec<_>>>()?;
        let response = self
            .client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(parts))
                    .build(),
            )
            .send()
            .await
            .map_err(|e| errors::map(e, &self.key, Write::None))?;
        Ok(PutResult {
            e_tag: response.e_tag().map(str::to_string),
            version: response.version_id().map(str::to_string),
            extensions: Default::default(),
        })
    }

    async fn abort(&mut self) -> Result<()> {
        self.client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .send()
            .await
            .map_err(|e| errors::map(e, &self.key, Write::None))?;
        Ok(())
    }
}
