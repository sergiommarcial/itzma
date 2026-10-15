//! The shared fetch planner: one policy, one place, that decides
//! how many requests a ranged read sends and how big each one is.
//!
//! Every reader in this crate used to make this call on its own —
//! `RangedSegment::vectors_of`'s fixed 8 KiB coalescing, `sq8::scan_object`'s
//! "only merge touching partitions", `index::open`'s fixed 32 KiB guess —
//! each tuned for `InMemory`/`LocalFileSystem`, where a request costs about
//! as much as reading its bytes. On real object storage a request has a
//! time to first byte of tens of milliseconds, so merging nearby ranges and
//! splitting huge ones into parallel parts is worth far more bytes of slack
//! than that. `plan` decides the shape; `execute` sends it.

use std::ops::Range;

use bytes::Bytes;
use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

use crate::budget::ReadBudget;
use crate::error::{ItzmaError, Result};

/// How a ranged read turns "these pieces of this object" into requests.
/// Chosen per store kind (see [`FetchPolicy::local`], [`FetchPolicy::object_store`])
/// and set on a [`Namespace`](crate::Namespace) with
/// [`with_fetch_policy`](crate::Namespace::with_fetch_policy).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchPolicy {
    /// Merge two ranges when the gap between them is at most
    /// `clamp(wanted_so_far / gap_divisor, min_gap, max_gap)`. The floor of
    /// that clamp.
    pub min_gap: u64,
    /// The ceiling of the gap clamp (see [`min_gap`](Self::min_gap)).
    pub max_gap: u64,
    /// How much of what has been merged so far the gap limit scales down
    /// by (see [`min_gap`](Self::min_gap)).
    pub gap_divisor: u64,
    /// A merged range longer than this is split into `part_size`-sized grid
    /// cells (aligned to multiples of `part_size` from the object start),
    /// fetched concurrently.
    pub split_above: u64,
    /// The width of a split grid cell (see [`split_above`](Self::split_above)).
    pub part_size: u64,
    /// Round a merged, unsplit range out to a multiple of this, so nearby
    /// reads from different queries land on the same cache keys.
    pub align: u64,
}

impl FetchPolicy {
    /// Tuned for `InMemory` and `LocalFileSystem`, where a request costs
    /// about as much as the bytes it reads: merge within 8 KiB, never
    /// split, no extra alignment.
    pub fn local() -> Self {
        Self {
            min_gap: 8 * 1024,
            max_gap: 64 * 1024,
            gap_divisor: 8,
            split_above: u64::MAX,
            part_size: 1,
            align: 1,
        }
    }

    /// The default: tuned for S3 and other real object stores, where a
    /// request has tens of milliseconds of time to first byte. `min_gap`
    /// was measured against MinIO: with a decoded cache it beat 256 KiB on
    /// requests and bytes at once. The other numbers are starting points that
    /// still need a run with real latency.
    pub fn object_store() -> Self {
        Self {
            min_gap: 64 * 1024,
            max_gap: 4 * 1024 * 1024,
            gap_divisor: 8,
            split_above: 16 * 1024 * 1024,
            part_size: 8 * 1024 * 1024,
            align: 64 * 1024,
        }
    }

    /// Set [`min_gap`](Self::min_gap). `#[non_exhaustive]` means no crate
    /// outside this one can build a `FetchPolicy` with a struct literal
    /// (even `..base` update syntax), so a caller that wants to tweak one
    /// field of a default — the Python bridge's `fetch=` kwargs, for
    /// instance — goes through setters like this one instead.
    #[must_use]
    pub fn min_gap(mut self, min_gap: u64) -> Self {
        self.min_gap = min_gap;
        self
    }

    /// Set [`max_gap`](Self::max_gap).
    #[must_use]
    pub fn max_gap(mut self, max_gap: u64) -> Self {
        self.max_gap = max_gap;
        self
    }

    /// Set [`gap_divisor`](Self::gap_divisor).
    #[must_use]
    pub fn gap_divisor(mut self, gap_divisor: u64) -> Self {
        self.gap_divisor = gap_divisor;
        self
    }

    /// Set [`split_above`](Self::split_above).
    #[must_use]
    pub fn split_above(mut self, split_above: u64) -> Self {
        self.split_above = split_above;
        self
    }

    /// Set [`part_size`](Self::part_size).
    #[must_use]
    pub fn part_size(mut self, part_size: u64) -> Self {
        self.part_size = part_size;
        self
    }

    /// Set [`align`](Self::align).
    #[must_use]
    pub fn align(mut self, align: u64) -> Self {
        self.align = align;
        self
    }
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self::object_store()
    }
}

/// One physical request this plan will send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Req {
    pub(crate) range: Range<u64>,
}

/// One candidate shape for a plan: the requests it would send, for each
/// wanted range (same order, same length as the `Plan`'s own `wanted`)
/// which request(s) in `requests` cover it, and the total bytes it would
/// cost. Almost always one request per wanted range; more than one only
/// when that wanted range's own group was split because it was longer
/// than `split_above` (the whole-object read case, see `whole_object`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Candidate {
    /// In object order.
    pub(crate) requests: Vec<Req>,
    /// `owners[i]` is the (in-order) list of indices into `requests` that
    /// together cover `wanted[i]`.
    pub(crate) owners: Vec<Vec<usize>>,
    pub(crate) total_bytes: u64,
}

/// The merged way to fetch `wanted` (nearby ranges share a request), and
/// the means to build the unmerged way on demand. Pure, deterministic, no
/// I/O — the choice between the two needs a real, atomic reservation
/// against the budget to be race-free under concurrent callers sharing one
/// (see `choose`), so it isn't made here. The unmerged candidate is built
/// only when the merged one doesn't fit, which is the rare case: building
/// both up front cost every small read a second candidate it almost never
/// used (the copy and overhead work after the first measurement).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) merged: Candidate,
    pub(crate) unmerged: Unmerged,
    /// The original wanted ranges, same order both candidates' `owners`
    /// index by. A physical request is almost always bigger than the
    /// wanted range it covers (that's the whole point of merging), so
    /// `execute` needs this to know where inside the fetched bytes the
    /// wanted slice starts.
    pub(crate) wanted: Vec<Range<u64>>,
}

/// What [`Plan::unmerged`] needs to build the unmerged candidate later:
/// the policy, the wanted ranges' sorted visiting order, and the object's
/// size. Owned, so it can outlive the borrow `plan` was built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unmerged {
    policy: FetchPolicy,
    order: Vec<usize>,
    object_size: u64,
}

impl Unmerged {
    /// The unmerged candidate for `wanted`, the same ranges the `Plan` was
    /// built from.
    pub(crate) fn build(&self, wanted: &[Range<u64>]) -> Candidate {
        build_candidate(&self.policy, wanted, &self.order, self.object_size, false)
    }
}

fn gap_limit(policy: &FetchPolicy, wanted_so_far: u64) -> u64 {
    // Order-tolerant: `u64::clamp` panics if its own min exceeds its max,
    // and a misconfigured policy (min_gap above max_gap) is reachable from
    // outside this crate — FetchPolicy's setters are public, and the
    // Python bridge builds one from six independent kwargs with no
    // cross-field validation. Degrade to the narrower of the two instead
    // of panicking.
    let (lo, hi) = (
        policy.min_gap.min(policy.max_gap),
        policy.min_gap.max(policy.max_gap),
    );
    (wanted_so_far / policy.gap_divisor.max(1)).clamp(lo, hi)
}

/// Merge `wanted` (visited in `order`) into non-overlapping groups,
/// proportionally, per `gap_limit`. Returns the groups and, for each
/// position in `order`, which group it landed in.
fn merge(
    policy: &FetchPolicy,
    wanted: &[Range<u64>],
    order: &[usize],
) -> (Vec<Range<u64>>, Vec<usize>) {
    let mut groups: Vec<Range<u64>> = Vec::new();
    let mut owner_of_sorted = Vec::with_capacity(order.len());
    let mut wanted_so_far = 0u64;
    for &i in order {
        let r = wanted[i].clone();
        wanted_so_far += r.end.saturating_sub(r.start);
        let limit = gap_limit(policy, wanted_so_far);
        match groups.last_mut() {
            Some(last) if r.start.saturating_sub(last.end) <= limit => {
                last.end = last.end.max(r.end);
                owner_of_sorted.push(groups.len() - 1);
            }
            _ => {
                groups.push(r);
                owner_of_sorted.push(groups.len() - 1);
            }
        }
    }
    (groups, owner_of_sorted)
}

/// Grid-aligned parts of `range`, each `part_size` wide and aligned to a
/// multiple of `part_size` measured from object offset 0, clipped to
/// `range`. The grid (not the range's own start) decides the boundaries, so
/// two different reads of overlapping regions land on the same parts.
fn split(range: &Range<u64>, part_size: u64) -> Vec<Range<u64>> {
    if part_size == 0 || range.end <= range.start {
        return vec![range.clone()];
    }
    let first_cell = range.start / part_size;
    let last_cell = (range.end - 1) / part_size;
    (first_cell..=last_cell)
        .map(|cell| {
            let start = (cell * part_size).max(range.start);
            let end = ((cell + 1) * part_size).min(range.end);
            start..end
        })
        .collect()
}

fn align_out(range: &Range<u64>, align: u64, object_size: u64) -> Range<u64> {
    if align <= 1 {
        return range.clone();
    }
    let start = (range.start / align) * align;
    let end = (range.end.div_ceil(align) * align).min(object_size);
    start..end
}

/// Coalesce already-sorted, aligned groups that now touch or overlap: each
/// is widened independently by [`align_out`], so when `align` is bigger
/// than the gap [`merge`] used to keep two groups apart, their aligned
/// extents can collide into two physical requests for the same or
/// overlapping bytes. Returns the coalesced groups and, for each input
/// group's position, which output group it landed in — the same shape
/// [`merge`] returns, so a caller remaps through it the same way.
fn coalesce_aligned(aligned: &[Range<u64>]) -> (Vec<Range<u64>>, Vec<usize>) {
    let mut groups: Vec<Range<u64>> = Vec::with_capacity(aligned.len());
    let mut remap = Vec::with_capacity(aligned.len());
    for g in aligned {
        match groups.last_mut() {
            Some(last) if g.start <= last.end => {
                last.end = last.end.max(g.end);
                remap.push(groups.len() - 1);
            }
            _ => {
                groups.push(g.clone());
                remap.push(groups.len() - 1);
            }
        }
    }
    (groups, remap)
}

/// Both candidate shapes for fetching `wanted` (any order, may overlap):
/// merged (nearby ranges share a request) and unmerged (only exactly
/// touching ones do). Never merges across `object_size`. Pure,
/// deterministic, no I/O, no budget involved — `execute` chooses between
/// the two candidates atomically against the real budget.
///
/// # Panics
///
/// Debug builds panic if a wanted range runs past `object_size`: that is a
/// caller bug (every reader must clip to its own object's size first), not
/// something the planner should paper over.
pub(crate) fn plan(policy: &FetchPolicy, wanted: &[Range<u64>], object_size: u64) -> Plan {
    if wanted.is_empty() {
        let empty = Candidate {
            requests: Vec::new(),
            owners: Vec::new(),
            total_bytes: 0,
        };
        return Plan {
            merged: empty,
            unmerged: Unmerged {
                policy: *policy,
                order: Vec::new(),
                object_size,
            },
            wanted: Vec::new(),
        };
    }
    debug_assert!(
        wanted.iter().all(|r| r.end <= object_size),
        "a wanted range runs past the object's own size"
    );

    let mut order: Vec<usize> = (0..wanted.len()).collect();
    order.sort_unstable_by_key(|&i| (wanted[i].start, wanted[i].end));

    Plan {
        merged: build_candidate(policy, wanted, &order, object_size, true),
        unmerged: Unmerged {
            policy: *policy,
            order,
            object_size,
        },
        wanted: wanted.to_vec(),
    }
}

/// One candidate shape for `wanted`, visited in `order`: merged (nearby
/// ranges share a request, aligned out, split when long) or unmerged (each
/// wanted range its own group, only exactly touching ones combined).
fn build_candidate(
    policy: &FetchPolicy,
    wanted: &[Range<u64>],
    order: &[usize],
    object_size: u64,
    merge_groups: bool,
) -> Candidate {
    let (groups, owner_of_sorted) = if merge_groups {
        merge(policy, wanted, order)
    } else {
        // "Unmerged": each wanted range is its own group, only exactly
        // touching ones combined (gap 0), no proportional growth.
        let flat = FetchPolicy {
            min_gap: 0,
            max_gap: 0,
            gap_divisor: 1,
            ..*policy
        };
        merge(&flat, wanted, order)
    };
    // `group_remap[g]` is where `merge`'s original group `g` ended up
    // after alignment: identity when nothing was aligned, but when it
    // was, two originally separate groups can collide once widened
    // (Finding 9) and `coalesce_aligned` folds them into one, which
    // `owners` below must follow through `group_remap`, not the raw
    // `owner_of_sorted` index.
    let (aligned, group_remap): (Vec<Range<u64>>, Vec<usize>) = if merge_groups {
        let raw: Vec<Range<u64>> = groups
            .iter()
            .map(|g| align_out(g, policy.align, object_size))
            .collect();
        coalesce_aligned(&raw)
    } else {
        let identity = (0..groups.len()).collect();
        (groups, identity)
    };
    let mut requests = Vec::new();
    // Each group's requests are a contiguous run of `requests`, so a
    // range of indices says which ones, with no `Vec` per group.
    let mut group_to_requests: Vec<Range<usize>> = Vec::with_capacity(aligned.len());
    for g in &aligned {
        let parts = if g.end - g.start > policy.split_above {
            split(g, policy.part_size)
        } else {
            vec![g.clone()]
        };
        let start_idx = requests.len();
        requests.extend(parts.into_iter().map(|range| Req { range }));
        group_to_requests.push(start_idx..requests.len());
    }
    // Each wanted range's owner is only the part(s) of its group it
    // actually overlaps, not the whole group: the common case is one
    // request (no split), and a wanted range whose own group was split
    // owns just the one or two parts its own bytes fall into — not
    // every part of that group, which execute() would otherwise
    // concatenate per wanted range regardless of how little of it that
    // range needed.
    let mut owners = vec![Vec::new(); wanted.len()];
    for (pos, &i) in order.iter().enumerate() {
        let group = group_remap[owner_of_sorted[pos]];
        let w = &wanted[i];
        // Strict overlap for a real (non-empty) wanted range — `<` on
        // both sides, so a range merely touching a part's boundary
        // isn't wrongly counted as overlapping it. A zero-row SQ8
        // partition's byte range is legitimately empty (`start ==
        // end`), and no part strictly overlaps a point, so that case
        // falls back to inclusive point containment instead — every
        // such group collapses to exactly one part covering it (an
        // empty range can't be split), so this never picks more than
        // what's already there.
        let overlaps = |req: &Range<u64>| {
            if w.start == w.end {
                req.start <= w.start && w.start <= req.end
            } else {
                req.start < w.end && w.start < req.end
            }
        };
        owners[i] = group_to_requests[group]
            .clone()
            .filter(|&r| overlaps(&requests[r].range))
            .collect();
    }
    let total_bytes = requests.iter().map(|r| r.range.end - r.range.start).sum();
    Candidate {
        requests,
        owners,
        total_bytes,
    }
}

const FETCH_CONCURRENCY: usize = 16;

/// Pick between `plan`'s two candidates, send the winner's requests
/// (concurrently), and return one slice per originally wanted range, in
/// wanted order — zero-copy when that range's own group was not split,
/// one concatenation when it was (see `whole_object`).
///
/// The choice is one atomic reservation against `budget`
/// ([`ReadBudget::try_reserve`]), not a snapshot-then-decide: merging
/// nearby ranges is tried first (fewer requests), falling back to the
/// unmerged shape if that alone doesn't fit — a read never fails for the
/// planner's sake — and if *neither* fits, a real [`ReadBudget::charge`]
/// for the (smaller) unmerged total reports an honest, accurate refusal.
/// Deciding from a `remaining()` snapshot and charging later would leave a
/// window where a concurrent caller sharing the same budget charges in
/// between, making the snapshot the decision was made on already stale by
/// the time it is acted on — reserving atomically closes that window: at
/// most one concurrent decision can ever win a given byte of budget.
///
/// A missing object is [`ItzmaError::Storage`] here; callers that need it
/// reported as damage (`segment::missing`, `index::missing`, ...) do that
/// translation themselves, as they do today for every other fetch.
pub(crate) async fn execute(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    path: &Path,
    plan: Plan,
) -> Result<Vec<Bytes>> {
    if plan.wanted.is_empty() {
        return Ok(Vec::new());
    }
    let Plan {
        merged,
        unmerged,
        wanted,
    } = plan;
    let chosen = choose(budget, path, merged, || unmerged.build(&wanted))?;
    if chosen.requests.is_empty() {
        return Ok(Vec::new());
    }
    let fetched = fetch_parts(store, path, &chosen.requests).await?;
    chosen
        .owners
        .iter()
        .zip(&wanted)
        .map(|(parts, w)| {
            let base = chosen.requests[parts[0]].range.start;
            let local_start = (w.start - base) as usize;
            let local_len = (w.end - w.start) as usize;
            let whole = if parts.len() == 1 {
                fetched[parts[0]].clone()
            } else {
                let pieces: Vec<Bytes> = parts.iter().map(|&p| fetched[p].clone()).collect();
                concat(&pieces)
            };
            // Checked, not a direct `Bytes::slice` (which panics out of
            // bounds): a store that returns fewer bytes than its own
            // response claimed is a store bug, and library code never
            // panics on one (library code does not `unwrap`, decoders
            // never panic).
            if local_start + local_len > whole.len() {
                return Err(short_read(path, whole.len(), local_start + local_len));
            }
            Ok(whole.slice(local_start..local_start + local_len))
        })
        .collect()
}

/// Pick between a merged and an unmerged candidate and reserve its bytes.
/// The choice is one atomic reservation against `budget`
/// ([`ReadBudget::try_reserve`]), not a snapshot-then-decide: merging is
/// tried first (fewer requests), falling back to the unmerged shape if that
/// alone doesn't fit — a read never fails for the planner's sake — and if
/// *neither* fits, a real [`ReadBudget::charge`] for the (smaller) unmerged
/// total reports an honest, accurate refusal. Deciding from a `remaining()`
/// snapshot and charging later would leave a window where a concurrent
/// caller sharing the same budget charges in between, making the snapshot
/// the decision was made on already stale by the time it is acted on.
fn choose(
    budget: &ReadBudget,
    path: &Path,
    merged: Candidate,
    unmerged: impl FnOnce() -> Candidate,
) -> Result<Candidate> {
    if budget.try_reserve(merged.total_bytes) {
        return Ok(merged);
    }
    let unmerged = unmerged();
    if budget.try_reserve(unmerged.total_bytes) {
        return Ok(unmerged);
    }
    // Neither fits on its own: charge the true minimum (the unmerged total
    // can never exceed the merged one) for an honest refusal, sticky like
    // every other charge in this crate. This can only succeed if budget
    // freed up between the two `try_reserve` calls above and this one, in
    // which case the unmerged plan genuinely does fit now and proceeding
    // with it is correct.
    budget.charge(path.as_ref(), unmerged.total_bytes)?;
    Ok(unmerged)
}

/// Send `requests` concurrently and return their bytes in request order.
/// No per-request `budget.charge` here: the whole total was already
/// reserved by [`choose`], specifically so this loop can run concurrently
/// without each request racing a sibling task's charges.
pub(crate) async fn fetch_parts(
    store: &dyn ObjectStore,
    path: &Path,
    requests: &[Req],
) -> Result<Vec<Bytes>> {
    // Collected into an owned Vec before the stream, not `.iter().map(...)`
    // taken straight into `stream::iter`: a closure whose parameter borrows
    // from `requests` inline like that is inferred higher-ranked, which
    // then fails "Send is not general enough" at every caller that spawns
    // the future this builds (see `segment::read_all`'s own comment on the
    // same shape of fix).
    let ranges: Vec<Range<u64>> = requests.iter().map(|r| r.range.clone()).collect();
    stream::iter(ranges)
        .map(|range| async move {
            if range.end <= range.start {
                return Ok(Bytes::new());
            }
            store.get_range(path, range).await.map_err(ItzmaError::from)
        })
        .buffered(FETCH_CONCURRENCY)
        .try_collect()
        .await
}

/// One copy of `parts`, concatenated in order. A single part is returned
/// as is, without a copy.
fn concat(parts: &[Bytes]) -> Bytes {
    if let [only] = parts {
        return only.clone();
    }
    let mut out = Vec::with_capacity(parts.iter().map(Bytes::len).sum());
    for part in parts {
        out.extend_from_slice(part);
    }
    Bytes::from(out)
}

/// The error for a store that returned fewer bytes than its own response
/// claimed. Library code never panics on a store bug, so this is an error.
fn short_read(path: &Path, have: usize, need: usize) -> ItzmaError {
    ItzmaError::Storage(object_store::Error::Generic {
        store: "fetch",
        source: format!(
            "{}: the store returned {have} bytes, fewer than the {need} a ranged read needed",
            path.as_ref()
        )
        .into(),
    })
}

/// The discovery probe's floor, regardless of policy: `policy.part_size`
/// alone would make `local()` (`part_size: 1`) send a second request for
/// the head of any object bigger than a single byte, defeating `local()`'s
/// whole purpose of matching the earlier one-GET behavior. Matches
/// `RangedSegment::open`'s own `TAIL_GUESS` — the same trick, the same
/// reasoning: most test- and namespace-sized objects fit in one response.
const DISCOVERY_PROBE_FLOOR: u64 = 64 * 1024;

/// Fetch a whole object, learning its size from the first request instead
/// of a separate HEAD — the same trick `RangedSegment::open` already uses
/// for a segment's trailer: a suffix request returns the object's real
/// size in its response even when the object is smaller than asked, in
/// which case that one response already holds everything. Otherwise the
/// rest (`0..size - probe`) is planned and executed like any other wanted
/// range, splitting into parallel parts if it is big enough, and the two
/// pieces are concatenated in object order.
pub(crate) async fn whole_object(
    store: &dyn ObjectStore,
    budget: &ReadBudget,
    policy: &FetchPolicy,
    path: &Path,
) -> Result<Bytes> {
    use object_store::{GetOptions, GetRange};

    let probe = policy.part_size.max(DISCOVERY_PROBE_FLOOR);
    let options = GetOptions {
        range: Some(GetRange::Suffix(probe)),
        ..GetOptions::default()
    };
    // Charged after the request, not before: its real cost (the response's
    // own range, which can be smaller than `probe` for a small object) is
    // only known once the headers are in hand, and a missing object — the
    // common, unexceptional case for an optional object — must cost nothing, the same way every other fetch in this crate
    // charges nothing for an object that was never there to read.
    let got = store
        .get_opts(path, options)
        .await
        .map_err(ItzmaError::from)?;
    let size = got.meta.size;
    let tail_start = got.range.start;
    budget.charge(path.as_ref(), got.range.end - tail_start)?;
    // The tail's chunks are kept as they arrive, not collected into one
    // buffer first: the final object is assembled below with one copy of
    // each byte, instead of one copy into a tail buffer and another into
    // the object.
    let mut pieces: Vec<Bytes> = got
        .into_stream()
        .try_collect()
        .await
        .map_err(ItzmaError::from)?;

    if tail_start == 0 {
        // The probe already covered the whole object.
        return assemble(path, pieces, size);
    }

    #[allow(clippy::single_range_in_vec_init)] // one wanted range: the head before the tail
    let wanted = [0..tail_start];
    // Bounded by `tail_start`, not the object's real `size`: the merged
    // candidate's own alignment (`align_out`) clips a group's end to
    // whatever "object size" it is given, so planning against the real,
    // bigger `size` here let an aligned head group round its end straight
    // into the tail region the probe above already fetched — re-fetching
    // and re-charging the overlap even though the answer stayed correct
    // (Finding 8). The head's own address space ends at `tail_start` by
    // construction, so that is its true bound.
    let planned = plan(policy, &wanted, tail_start);
    let chosen = choose(budget, path, planned.merged, || {
        planned.unmerged.build(&wanted)
    })?;
    // The chosen requests cover exactly `0..tail_start`: the head is
    // planned against `tail_start` (see above), so every part is inside it
    // and in object order, and concatenating them is the head.
    let head = fetch_parts(store, path, &chosen.requests).await?;
    let mut all = head;
    all.append(&mut pieces);
    assemble(path, all, size)
}

/// Concatenate `parts` (object order) into one object of `size` bytes,
/// copying each byte once. A single part is returned without a copy. A
/// store that delivers a different number of bytes than it declared is
/// [`short_read`], not a panic.
fn assemble(path: &Path, parts: Vec<Bytes>, size: u64) -> Result<Bytes> {
    let have: usize = parts.iter().map(Bytes::len).sum();
    if have as u64 != size {
        return Err(short_read(path, have, size as usize));
    }
    Ok(concat(&parts))
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)] // a one-range `wanted` slice is a normal, valid test case here, not a mistake
mod tests {
    use super::*;

    fn policy(
        min_gap: u64,
        max_gap: u64,
        gap_divisor: u64,
        split_above: u64,
        part_size: u64,
        align: u64,
    ) -> FetchPolicy {
        FetchPolicy {
            min_gap,
            max_gap,
            gap_divisor,
            split_above,
            part_size,
            align,
        }
    }

    #[test]
    fn object_store_merges_within_64_kib() {
        // Measured against MinIO: with a
        // decoded cache and fresh queries, a 64 KiB floor beat 256 KiB on
        // requests and bytes at once in every configuration measured.
        assert_eq!(
            FetchPolicy::object_store(),
            policy(64 << 10, 4 << 20, 8, 16 << 20, 8 << 20, 64 << 10)
        );
    }

    #[test]
    fn two_ranges_within_the_gap_merge_into_one_request() {
        // min_gap 8, so a gap of exactly 8 merges, one more byte does not.
        let p = policy(8, 1024, 8, u64::MAX, 1, 1);
        let touching = plan(&p, &[0..4, 12..16], 1000);
        assert_eq!(touching.merged.requests.len(), 1);
        assert_eq!(touching.merged.requests[0].range, 0..16);

        let apart = plan(&p, &[0..4, 13..16], 1000);
        assert_eq!(apart.merged.requests.len(), 2);
    }

    #[test]
    fn a_request_never_crosses_the_object_end() {
        let p = policy(1024, 1024, 1, u64::MAX, 1, 1);
        let plan = plan(&p, &[0..10], 10);
        assert_eq!(plan.merged.requests[0].range, 0..10);
    }

    #[test]
    fn a_misconfigured_policy_with_min_gap_above_max_gap_never_panics() {
        // Reachable from outside this crate (FetchPolicy's setters are
        // public, and the Python bridge builds one from six independent
        // optional kwargs with no cross-field validation), so this must
        // degrade gracefully, not panic — `u64::clamp` panics if its own
        // min exceeds its max.
        let p = policy(1 << 20, 8, 1, u64::MAX, 1, 1); // min_gap way above max_gap
        let plan = plan(&p, &[0..4, 1_000_000..1_000_004], 2_000_000);
        assert!(!plan.merged.requests.is_empty());
    }

    #[test]
    fn an_empty_wanted_range_still_gets_an_owner_not_an_empty_list() {
        // A zero-row SQ8 partition's byte range is legitimately empty
        // (start == end). A strict overlap test (`start < end`) never
        // matches an empty range against anything, including its own
        // single-part group, leaving `owners[i]` empty — which panics in
        // `execute` on `parts[0]`. Caught by a real integration test
        // (tests/index.rs's `a_segment_with_no_documents_is_indexed_when_asked`)
        // going through sq8::fetch_wave with a genuinely empty partition.
        let p = policy(8, 1024, 8, u64::MAX, 1, 1);
        let plan = plan(&p, &[5..5], 10);
        assert_eq!(plan.merged.owners, vec![vec![0]]);
    }

    #[test]
    fn a_small_wanted_range_owns_only_the_parts_it_overlaps_not_the_whole_group() {
        // Four tiny ranges, far enough apart that nothing would merge under
        // a tight policy, forced into one group by a huge gap limit. That
        // group is wider than split_above (100) with part_size 40, so it
        // splits into 4 grid cells: 0..40, 40..80, 80..120, 120..130. Each
        // tiny wanted range sits entirely inside exactly one cell and must
        // own only that cell — not all 4, which is what execute() would
        // otherwise concatenate per wanted range.
        let p = policy(1 << 20, 1 << 20, 1, 100, 40, 1);
        let wanted = [5..6, 45..46, 85..86, 125..126];
        let plan = plan(&p, &wanted, 130);
        // The merged group spans only the wanted extent (5..126), grid-split
        // into cells 0..40, 40..80, 80..120, 120..160 clipped to that span.
        let cells: Vec<_> = plan
            .merged
            .requests
            .iter()
            .map(|r| r.range.clone())
            .collect();
        assert_eq!(cells, vec![5..40, 40..80, 80..120, 120..126]);
        assert_eq!(plan.merged.owners, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn a_range_above_split_above_is_cut_into_part_sized_grid_cells() {
        let p = policy(0, 0, 1, 100, 40, 1);
        let plan = plan(&p, &[0..130], 130);
        let ranges: Vec<_> = plan
            .merged
            .requests
            .iter()
            .map(|r| r.range.clone())
            .collect();
        assert_eq!(ranges, vec![0..40, 40..80, 80..120, 120..130]);
    }

    #[test]
    fn merged_ranges_are_rounded_out_to_align() {
        let p = policy(0, 0, 1, u64::MAX, 1, 64);
        let plan = plan(&p, &[10..20], 1000);
        assert_eq!(plan.merged.requests[0].range, 0..64);
    }

    #[test]
    fn alignment_never_crosses_the_object_end() {
        let p = policy(0, 0, 1, u64::MAX, 1, 64);
        let plan = plan(&p, &[10..20], 50);
        assert_eq!(plan.merged.requests[0].range, 0..50);
    }

    #[test]
    fn empty_wanted_makes_no_requests() {
        let p = FetchPolicy::local();
        let plan = plan(&p, &[], 1000);
        assert!(plan.merged.requests.is_empty());
        assert!(plan.merged.owners.is_empty());
        let unmerged = plan.unmerged.build(&[]);
        assert!(unmerged.requests.is_empty());
        assert!(unmerged.owners.is_empty());
    }

    #[test]
    fn overlapping_and_duplicate_wanted_ranges_map_to_the_right_request() {
        let p = policy(8, 1024, 8, u64::MAX, 1, 1);
        let wanted = [4..8, 0..4, 4..8];
        let plan = plan(&p, &wanted, 1000);
        assert_eq!(plan.merged.requests.len(), 1);
        assert_eq!(plan.merged.owners, vec![vec![0], vec![0], vec![0]]);
    }

    #[test]
    fn plan_returns_both_shapes_without_a_budget_and_the_unmerged_one_on_demand() {
        // `plan` takes no budget at all: it is pure, builds the merged shape
        // up front and the unmerged one only when asked, leaving the choice
        // to `choose`'s atomic reservation against a real `ReadBudget`
        // (see the `execute_*` tests below for that choice itself).
        let p = policy(1000, 1000, 1, u64::MAX, 1, 1);
        let wanted = [0..1, 999..1000];
        let planned = plan(&p, &wanted, 1000);
        assert_eq!(planned.merged.requests.len(), 1);
        assert_eq!(planned.merged.total_bytes, 1000);
        let unmerged = planned.unmerged.build(&wanted);
        assert_eq!(unmerged.requests.len(), 2);
        assert_eq!(unmerged.requests[0].range, 0..1);
        assert_eq!(unmerged.requests[1].range, 999..1000);
        assert_eq!(unmerged.total_bytes, 2);
    }

    #[tokio::test]
    async fn execute_falls_back_to_the_unmerged_plan_when_merging_would_exceed_the_budget() {
        // The scenario Finding 3 was about: the merged plan (1000 bytes)
        // does not fit a budget of 2, but the unmerged plan (2 bytes) does
        // — and the choice is made atomically inside `execute`, not from a
        // `remaining()` snapshot passed into `plan`.
        use object_store::ObjectStoreExt;
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let path = Path::from("obj.bin");
        store.put(&path, vec![0u8; 1000].into()).await.expect("put");

        let p = policy(1000, 1000, 1, u64::MAX, 1, 1);
        let wanted = [0..1, 999..1000];
        let planned = plan(&p, &wanted, 1000);
        assert_eq!(planned.merged.total_bytes, 1000);
        assert_eq!(planned.unmerged.build(&wanted).total_bytes, 2);

        let budget = ReadBudget::new(2); // exactly the unmerged floor, far under the merged total
        let got = execute(&store, &budget, &path, planned)
            .await
            .expect("execute falls back to the unmerged plan");
        assert_eq!(got.len(), 2);
        assert_eq!(budget.used(), 2);
    }

    #[tokio::test]
    async fn execute_refuses_honestly_when_neither_plan_fits() {
        use object_store::ObjectStoreExt;
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let path = Path::from("obj.bin");
        store.put(&path, vec![0u8; 1000].into()).await.expect("put");

        let p = policy(1000, 1000, 1, u64::MAX, 1, 1);
        let wanted = [0..1, 999..1000];
        let planned = plan(&p, &wanted, 1000);
        let budget = ReadBudget::new(1); // one byte short of even the unmerged floor (2)
        let err = execute(&store, &budget, &path, planned)
            .await
            .expect_err("neither plan fits a budget of 1");
        assert!(
            matches!(err, ItzmaError::ReadBudgetExceeded { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn aligned_groups_that_now_overlap_are_coalesced_into_one_request() {
        // Finding 9: align_out expands each merge() group independently of
        // its neighbors, so when `align` is bigger than the gap merge()
        // itself used to keep two groups apart, their aligned extents can
        // collide — two physical requests for the same or overlapping
        // bytes instead of one. object_store() reaches this (min_gap and
        // align are both 64 KiB, so two groups just over 64 KiB apart can
        // align onto the same boundary); local() cannot (align 1).
        let p = policy(0, 0, 1, u64::MAX, 1, 2048);
        let wanted = [0..1, 1000..1001]; // gap of 999, too wide for merge()'s min_gap of 0
        let plan = plan(&p, &wanted, 3000);
        assert_eq!(
            plan.merged.requests.len(),
            1,
            "two groups whose aligned extents collide must coalesce into one request"
        );
        assert_eq!(plan.merged.requests[0].range, 0..2048);
        assert_eq!(plan.merged.owners, vec![vec![0], vec![0]]);
    }

    #[test]
    fn requests_are_sent_in_object_order_and_results_come_back_in_wanted_order() {
        let p = policy(0, 0, 1, u64::MAX, 1, 1);
        let wanted = [50..60, 0..10];
        let plan = plan(&p, &wanted, 1000);
        let starts: Vec<u64> = plan.merged.requests.iter().map(|r| r.range.start).collect();
        assert_eq!(starts, vec![0, 50]);
        assert_eq!(plan.merged.owners.len(), 2);
    }

    // Property test: every wanted byte is inside the request(s) its owner
    // names, and total fetched is at most wanted + gaps + alignment.
    #[test]
    fn every_wanted_byte_is_covered_by_its_owner_and_waste_is_bounded() {
        let mut lcg = 12345u64;
        let mut next = |bound: u64| {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            (lcg >> 33) % bound.max(1)
        };
        for _ in 0..200 {
            let object_size = 1000 + next(100_000);
            let p = FetchPolicy::object_store();
            let n = 1 + next(20) as usize;
            let mut wanted: Vec<Range<u64>> = Vec::new();
            for _ in 0..n {
                let start = next(object_size.saturating_sub(1));
                let len = 1 + next((object_size - start).max(1));
                wanted.push(start..(start + len).min(object_size));
            }
            let plan = plan(&p, &wanted, object_size);
            let wanted_bytes: u64 = wanted.iter().map(|r| r.end - r.start).sum();
            let fetched_bytes: u64 = plan
                .merged
                .requests
                .iter()
                .map(|r| r.range.end - r.range.start)
                .sum();
            for (i, w) in wanted.iter().enumerate() {
                let parts = &plan.merged.owners[i];
                let covering_start = plan.merged.requests[parts[0]].range.start;
                let covering_end = plan.merged.requests[*parts.last().unwrap()].range.end;
                assert!(
                    covering_start <= w.start && w.end <= covering_end,
                    "{w:?} not covered by {parts:?}"
                );
            }
            let allowed_waste = wanted.len() as u64 * (p.max_gap + p.align) + p.part_size;
            assert!(
                fetched_bytes <= wanted_bytes + allowed_waste,
                "fetched {fetched_bytes} > wanted {wanted_bytes} + waste {allowed_waste}"
            );
        }
    }

    #[tokio::test]
    async fn execute_returns_one_slice_per_wanted_range_in_wanted_order() {
        use object_store::ObjectStoreExt;
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let path = Path::from("obj.bin");
        let bytes: Vec<u8> = (0u8..100).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        let p = FetchPolicy::local();
        let wanted = [50..60, 0..10, 50..60];
        let budget = ReadBudget::unlimited();
        let planned = plan(&p, &wanted, 100);
        let got = execute(&store, &budget, &path, planned)
            .await
            .expect("execute");
        assert_eq!(got.len(), 3);
        assert_eq!(&got[0][..], &bytes[50..60]);
        assert_eq!(&got[1][..], &bytes[0..10]);
        assert_eq!(&got[2][..], &bytes[50..60]);
    }

    #[tokio::test]
    async fn execute_on_a_split_group_returns_correct_bytes_for_each_small_wanted_range() {
        use object_store::ObjectStoreExt;
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let path = Path::from("obj.bin");
        let bytes: Vec<u8> = (0u32..130).map(|i| (i % 256) as u8).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        let p = policy(1 << 20, 1 << 20, 1, 100, 40, 1);
        let wanted = [5..6, 45..46, 85..86, 125..126];
        let budget = ReadBudget::unlimited();
        let planned = plan(&p, &wanted, 130);
        let got = execute(&store, &budget, &path, planned)
            .await
            .expect("execute");
        assert_eq!(got.len(), 4);
        for (i, w) in wanted.iter().enumerate() {
            assert_eq!(
                &got[i][..],
                &bytes[w.start as usize..w.end as usize],
                "wanted[{i}]"
            );
        }
    }

    #[tokio::test]
    async fn execute_charges_every_request_before_sending_it() {
        use object_store::ObjectStoreExt;
        use object_store::memory::InMemory;
        let store = InMemory::new();
        let path = Path::from("obj.bin");
        store.put(&path, vec![0u8; 100].into()).await.expect("put");

        let p = FetchPolicy::local();
        let planned = plan(&p, &[0..10], 100);
        let tiny = ReadBudget::new(5);
        let err = execute(&store, &tiny, &path, planned)
            .await
            .expect_err("refused");
        assert!(matches!(err, ItzmaError::ReadBudgetExceeded { .. }));
    }

    /// An in-memory store that counts every `get_opts` call, so a test can
    /// say how many requests `whole_object` made. Not shared with the
    /// similarly-named type in other modules (see `segment/ranged.rs`'s doc
    /// comment on why).
    #[derive(Debug, Default)]
    struct CountingStore {
        inner: object_store::memory::InMemory,
        served: std::sync::atomic::AtomicUsize,
    }

    impl CountingStore {
        fn requests(&self) -> usize {
            self.served.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl std::fmt::Display for CountingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CountingStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let got = self.inner.get_opts(location, options).await?;
            self.served
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(got)
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[tokio::test]
    async fn whole_object_of_a_missing_object_charges_nothing() {
        let budget = ReadBudget::new(0);
        let store = object_store::memory::InMemory::new();
        let p = FetchPolicy::object_store(); // part_size is 8 MiB — would blow a 0-byte budget if charged speculatively
        let err = whole_object(&store, &budget, &p, &Path::from("missing.bin"))
            .await
            .expect_err("missing");
        assert!(
            matches!(
                err,
                ItzmaError::Storage(object_store::Error::NotFound { .. })
            ),
            "{err:?}"
        );
        assert_eq!(budget.used(), 0, "a missing object must cost nothing");
    }

    #[tokio::test]
    async fn whole_object_under_local_is_still_one_request_for_a_small_object() {
        // local()'s part_size is 1 (it never splits), but that same number
        // must not become the discovery probe's size too — a 1-byte probe
        // would force a second request for the head of any object bigger
        // than 1 byte, contradicting local()'s whole purpose: matching the
        // earlier single-GET behavior on InMemory/LocalFileSystem.
        use object_store::ObjectStoreExt;
        let store = CountingStore::default();
        let path = Path::from("small.bin");
        let bytes: Vec<u8> = (0u8..200).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        let budget = ReadBudget::unlimited();
        let got = whole_object(&store, &budget, &FetchPolicy::local(), &path)
            .await
            .expect("whole_object");
        assert_eq!(&got[..], &bytes[..]);
        assert_eq!(store.requests(), 1);
    }

    #[tokio::test]
    async fn whole_object_smaller_than_the_probe_is_one_request() {
        use object_store::ObjectStoreExt;
        let store = CountingStore::default();
        let path = Path::from("small.bin");
        let bytes: Vec<u8> = (0u8..50).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        let budget = ReadBudget::unlimited();
        let p = FetchPolicy::object_store(); // part_size is 8 MiB, far bigger than this object
        let got = whole_object(&store, &budget, &p, &path)
            .await
            .expect("whole_object");
        assert_eq!(&got[..], &bytes[..]);
        assert_eq!(store.requests(), 1);
    }

    #[tokio::test]
    async fn whole_object_bigger_than_the_probe_is_head_plus_tail_reassembled_in_order() {
        use object_store::ObjectStoreExt;
        let store = CountingStore::default();
        let path = Path::from("big.bin");
        // Bigger than DISCOVERY_PROBE_FLOOR (64 KiB), so the probe here is
        // genuinely policy.part_size, not the floor stepping in.
        let bytes: Vec<u8> = (0u32..250_000).map(|i| (i % 256) as u8).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        let budget = ReadBudget::unlimited();
        let p = FetchPolicy {
            split_above: u64::MAX,
            part_size: 100_000,
            ..FetchPolicy::local()
        }; // probe = part_size
        let got = whole_object(&store, &budget, &p, &path)
            .await
            .expect("whole_object");
        assert_eq!(&got[..], &bytes[..]);
        assert!(store.requests() >= 2);
    }

    #[tokio::test]
    async fn whole_object_with_a_split_head_reassembles_every_part_in_order() {
        // The head (about 234 KB) is above split_above, so it goes out as
        // several parallel parts; the result must still be the exact bytes.
        use object_store::ObjectStoreExt;
        let store = CountingStore::default();
        let path = Path::from("split.bin");
        let size: u32 = 300_000;
        let bytes: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        let p = FetchPolicy {
            split_above: 100_000,
            part_size: 50_000,
            ..FetchPolicy::local()
        }; // probe = 64 KiB floor, so the head is 0..(300_000 - 65_536)
        let budget = ReadBudget::unlimited();
        let got = whole_object(&store, &budget, &p, &path)
            .await
            .expect("whole_object");
        assert_eq!(&got[..], &bytes[..]);
        assert!(store.requests() > 2, "the head should have been split");
    }

    #[tokio::test]
    async fn whole_object_never_double_charges_bytes_the_tail_probe_already_fetched() {
        // Finding 8: the head's own plan used to be built against the
        // object's full size, so `align_out` could round the head group's
        // end past `tail_start` — straight into bytes the tail probe
        // already fetched, re-fetching and re-charging them. The head must
        // be planned against `tail_start`, not `size`, so its own aligned
        // extent can never cross into the tail's region. A budget set to
        // exactly the real object size must suffice; under the bug it did
        // not, because the double-counted overlap pushed the real total
        // charged past that size.
        use object_store::ObjectStoreExt;
        let store = CountingStore::default();
        let path = Path::from("obj.bin");
        let size: u32 = 70_000;
        let bytes: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        store.put(&path, bytes.clone().into()).await.expect("put");

        // part_size doubles as the probe size here (DISCOVERY_PROBE_FLOOR is
        // 64 KiB, matched exactly so tail_start lands at size - 65536 =
        // 4464), and align (128 KiB) is far bigger than that head region —
        // big enough that an unclipped align_out would round the head's end
        // out past the whole object, deep into the tail. An unlimited
        // budget (not a tight one) is what forces execute() to pick the
        // merged, aligned head candidate instead of silently falling back
        // to the unaligned unmerged one — the fallback that would otherwise
        // paper over this bug without the budget even noticing.
        let p = policy(0, 0, 1, u64::MAX, 65_536, 131_072);
        let budget = ReadBudget::unlimited();
        let got = whole_object(&store, &budget, &p, &path)
            .await
            .expect("whole_object");
        assert_eq!(&got[..], &bytes[..]);
        assert_eq!(
            budget.used(),
            u64::from(size),
            "the head must never be planned past tail_start, or its \
             aligned-out extent re-fetches and re-charges bytes the tail \
             probe already covered"
        );
    }

    /// A store that always serves a truncated response, shorter than
    /// whatever range was requested — standing in for a buggy or
    /// misbehaving backend. Library code must never panic on this (CLAUDE.md's
    /// "decoders never panic"), it must error.
    #[derive(Debug, Default)]
    struct TruncatingStore {
        inner: object_store::memory::InMemory,
    }

    impl std::fmt::Display for TruncatingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TruncatingStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for TruncatingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let got = self.inner.get_opts(location, options).await?;
            let (meta, range, attributes) =
                (got.meta.clone(), got.range.clone(), got.attributes.clone());
            let full = got.bytes().await?;
            let short = full.slice(..full.len() / 2);
            Ok(object_store::GetResult {
                payload: object_store::GetResultPayload::Stream(Box::pin(futures::stream::once(
                    async move { Ok(short) },
                ))),
                meta,
                range,
                attributes,
                extensions: Default::default(),
            })
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[tokio::test]
    async fn a_short_response_is_an_error_not_a_panic() {
        use object_store::ObjectStoreExt;
        let store = TruncatingStore::default();
        let path = Path::from("obj.bin");
        store.put(&path, vec![0u8; 100].into()).await.expect("put");

        let p = FetchPolicy::local();
        let budget = ReadBudget::unlimited();
        let planned = plan(&p, &[0..100], 100);
        let err = execute(&store, &budget, &path, planned)
            .await
            .expect_err("a truncated response must error, not panic");
        assert!(matches!(err, ItzmaError::Storage(_)), "{err:?}");
    }
}
