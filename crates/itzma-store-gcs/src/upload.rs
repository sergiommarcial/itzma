//! Multipart uploads.

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use google_cloud_storage::client::Storage;
use object_store::{MultipartUpload, PutPayload, PutResult, Result, UploadPart};

use crate::errors::{self, Write};

/// Keeps every part in memory and writes the object in one upload on
/// `complete`. Correct, but memory grows with the object; streaming the parts
/// through a resumable upload waits for a caller that writes objects too big
/// to buffer.
#[derive(Debug)]
pub(crate) struct Buffered {
    pub(crate) storage: Storage,
    pub(crate) resource: String,
    pub(crate) name: String,
    pub(crate) parts: Vec<Bytes>,
}

#[async_trait]
impl MultipartUpload for Buffered {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.parts.push(Bytes::from(data));
        Box::pin(futures::future::ready(Ok(())))
    }

    async fn complete(&mut self) -> Result<PutResult> {
        let size = self.parts.iter().map(Bytes::len).sum();
        let mut body = BytesMut::with_capacity(size);
        for part in self.parts.drain(..) {
            body.extend_from_slice(&part);
        }
        let object = self
            .storage
            .write_object(&self.resource, &self.name, body.freeze())
            .send_buffered()
            .await
            .map_err(|e| errors::map(e, &self.name, Write::None))?;
        Ok(PutResult {
            e_tag: (!object.etag.is_empty()).then(|| object.etag.clone()),
            version: Some(object.generation.to_string()),
            extensions: Default::default(),
        })
    }

    async fn abort(&mut self) -> Result<()> {
        self.parts.clear();
        Ok(())
    }
}
