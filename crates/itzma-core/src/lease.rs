//! The maintenance lease: which of the processes sharing a namespace runs its
//! maintenance.
//!
//! One object, `<namespace>/maintenance/lease.json`, names the holder. A loop
//! calls [`MaintenanceLease::tick`] every [`LeaseOptions::renew`] and runs
//! [`Namespace::maintain`](crate::Namespace::maintain) only while the tick says
//! [`Role::Leader`]. The lease is claimed with put-if-absent and renewed or
//! taken over with update-if-version, the same two primitives the rest of the
//! crate relies on. Like `maintain`, this module starts no task: the caller
//! owns the loop.
//!
//! A follower decides the leader is gone when the lease object's version has
//! not changed for [`LeaseOptions::ttl`], measured on its own monotonic clock,
//! so clocks on different machines are never compared. The lease saves work
//! and decides nothing about correctness: two leaders at once is what two
//! unleased loops would do, and compaction and gc are safe under that.

use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutResult, UpdateVersion};
use serde::{Deserialize, Serialize};

use crate::error::{ItzmaError, Result};

/// The default for [`LeaseOptions::renew`]: 15 seconds.
pub const DEFAULT_LEASE_RENEW: Duration = Duration::from_secs(15);

/// The default for [`LeaseOptions::ttl`]: 60 seconds, four missed renews.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(60);

/// How often a lease is ticked and how long a silent leader keeps it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseOptions {
    /// How often the loop ticks: the leader renews, a follower looks. Default
    /// [`DEFAULT_LEASE_RENEW`].
    pub renew: Duration,
    /// How long the lease object may stay unchanged before a follower takes
    /// it over. At least twice `renew`. Default [`DEFAULT_LEASE_TTL`].
    pub ttl: Duration,
}

impl Default for LeaseOptions {
    fn default() -> Self {
        Self {
            renew: DEFAULT_LEASE_RENEW,
            ttl: DEFAULT_LEASE_TTL,
        }
    }
}

impl LeaseOptions {
    pub(crate) fn validate(self) -> Result<Self> {
        if self.renew.is_zero() {
            return Err(ItzmaError::InvalidConfig(
                "lease renew must be more than zero",
            ));
        }
        if self
            .renew
            .checked_mul(2)
            .is_none_or(|twice| self.ttl < twice)
        {
            return Err(ItzmaError::InvalidConfig(
                "lease ttl must be at least twice renew",
            ));
        }
        Ok(self)
    }
}

/// What a loop is after a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// This loop holds the lease and should maintain.
    Leader,
    /// Another loop holds it, or nobody could be confirmed. Do no work.
    Follower,
}

/// The lease object's body.
#[derive(Debug, Serialize, Deserialize)]
struct Body {
    holder: String,
    pid: u32,
    beat: u64,
    released: bool,
}

#[derive(Debug)]
enum State {
    Leader {
        version: UpdateVersion,
        beat: u64,
        renewed: Instant,
    },
    /// The version last seen and when it was first seen.
    Follower {
        seen: Option<(UpdateVersion, Instant)>,
    },
}

/// One loop's view of a namespace's maintenance lease. Build it with
/// [`Namespace::maintenance_lease`](crate::Namespace::maintenance_lease).
#[derive(Debug)]
pub struct MaintenanceLease {
    store: Arc<dyn ObjectStore>,
    key: Path,
    holder: String,
    options: LeaseOptions,
    state: State,
}

fn holder_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u32(std::process::id());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    hasher.write_u128(nanos);
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", hasher.finish())
}

fn version_of(put: PutResult) -> UpdateVersion {
    UpdateVersion {
        e_tag: put.e_tag,
        version: put.version,
    }
}

/// A conditional write lost: someone else changed or removed the object.
fn lost(e: &object_store::Error) -> bool {
    matches!(
        e,
        object_store::Error::Precondition { .. } | object_store::Error::NotFound { .. }
    )
}

impl MaintenanceLease {
    pub(crate) fn new(
        store: Arc<dyn ObjectStore>,
        namespace: &str,
        options: LeaseOptions,
    ) -> Result<Self> {
        Ok(Self {
            store,
            key: Path::from(format!("{namespace}/maintenance/lease.json")),
            holder: holder_id(),
            options: options.validate()?,
            state: State::Follower { seen: None },
        })
    }

    /// This loop's holder id, written into the lease object while it leads.
    pub fn holder(&self) -> &str {
        &self.holder
    }

    /// The options the lease was built with.
    pub fn options(&self) -> LeaseOptions {
        self.options
    }

    /// The role after the last tick or release.
    pub fn role(&self) -> Role {
        match self.state {
            State::Leader { .. } => Role::Leader,
            State::Follower { .. } => Role::Follower,
        }
    }

    /// Renew, claim, take over or watch, and return the role.
    ///
    /// A leader renews conditional on its last version, and a lost race makes
    /// it a follower. A follower claims a missing lease, takes over one that is
    /// released or has not changed for `ttl` since `now` first saw it, and
    /// otherwise watches. `now` must come from a monotonic clock.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::Storage`] for a store failure other than a lost race,
    /// including `NotImplemented` on a store without conditional update. A
    /// leader that fails to renew is a follower afterwards, and gets the lease
    /// back after `ttl` like anyone else.
    pub async fn tick(&mut self, now: Instant) -> Result<Role> {
        match std::mem::replace(&mut self.state, State::Follower { seen: None }) {
            State::Leader {
                version,
                beat,
                renewed,
            } => self.renew(version, beat, renewed, now).await,
            State::Follower { seen } => {
                self.state = State::Follower { seen };
                self.watch(now).await
            }
        }
    }

    async fn renew(
        &mut self,
        version: UpdateVersion,
        beat: u64,
        renewed: Instant,
        now: Instant,
    ) -> Result<Role> {
        match self
            .write(PutMode::Update(version.clone()), beat + 1, false)
            .await
        {
            Ok(put) => {
                self.state = State::Leader {
                    version: version_of(put),
                    beat: beat + 1,
                    renewed: now,
                };
                Ok(Role::Leader)
            }
            Err(e) if lost(&e) => Ok(Role::Follower),
            Err(e) => {
                self.state = State::Follower {
                    seen: Some((version, renewed)),
                };
                Err(e.into())
            }
        }
    }

    async fn watch(&mut self, now: Instant) -> Result<Role> {
        let result = match self.store.get(&self.key).await {
            Ok(r) => r,
            Err(object_store::Error::NotFound { .. }) => return self.claim(now).await,
            Err(e) => return Err(e.into()),
        };
        let version = UpdateVersion {
            e_tag: result.meta.e_tag.clone(),
            version: result.meta.version.clone(),
        };
        let held: Option<Body> = serde_json::from_slice(&result.bytes().await?).ok();
        let released = held.as_ref().is_some_and(|b| b.released);
        let stale = match &self.state {
            State::Follower {
                seen: Some((seen, at)),
            } => *seen == version && now.saturating_duration_since(*at) >= self.options.ttl,
            _ => false,
        };
        if !released && !stale {
            if !matches!(&self.state, State::Follower { seen: Some((seen, _)) } if *seen == version)
            {
                self.state = State::Follower {
                    seen: Some((version, now)),
                };
            }
            return Ok(Role::Follower);
        }
        let beat = held.map_or(0, |b| b.beat + 1);
        match self.write(PutMode::Update(version), beat, false).await {
            Ok(put) => self.lead(put, beat, now),
            Err(e) if lost(&e) => {
                self.state = State::Follower { seen: None };
                Ok(Role::Follower)
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn claim(&mut self, now: Instant) -> Result<Role> {
        match self.write(PutMode::Create, 0, false).await {
            Ok(put) => self.lead(put, 0, now),
            Err(object_store::Error::AlreadyExists { .. }) => {
                self.state = State::Follower { seen: None };
                Ok(Role::Follower)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn lead(&mut self, put: PutResult, beat: u64, now: Instant) -> Result<Role> {
        self.state = State::Leader {
            version: version_of(put),
            beat,
            renewed: now,
        };
        Ok(Role::Leader)
    }

    /// Give the lease up so a follower can claim it on its next tick instead of
    /// waiting out `ttl`. Writes `released: true` conditional on this loop's
    /// version. Does nothing when not the leader. Best effort: if the process
    /// dies first, `ttl` covers it.
    ///
    /// # Errors
    ///
    /// [`ItzmaError::Storage`] for a store failure other than a lost race. The
    /// lease is a follower afterwards either way.
    pub async fn release(&mut self) -> Result<()> {
        let State::Leader { version, beat, .. } =
            std::mem::replace(&mut self.state, State::Follower { seen: None })
        else {
            return Ok(());
        };
        match self.write(PutMode::Update(version), beat + 1, true).await {
            Ok(_) => Ok(()),
            Err(e) if lost(&e) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    async fn write(
        &self,
        mode: PutMode,
        beat: u64,
        released: bool,
    ) -> std::result::Result<PutResult, object_store::Error> {
        let body = Body {
            holder: self.holder.clone(),
            pid: std::process::id(),
            beat,
            released,
        };
        let bytes = serde_json::to_vec(&body).map_err(|e| object_store::Error::Generic {
            store: "lease",
            source: Box::new(e),
        })?;
        self.store
            .put_opts(&self.key, bytes.into(), PutOptions::from(mode))
            .await
    }
}
