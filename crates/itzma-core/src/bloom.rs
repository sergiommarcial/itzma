//! A Bloom filter over byte keys, used by segment format version 2.
//!
//! It answers "might this id be in the segment?" so a query can skip the
//! block reads that check whether a newer segment shadows a candidate. The
//! bytes go into the segment object, and segments are compared byte for byte,
//! so the filter must be a pure function of its keys: a fixed
//! hash written here, no library and no per-process seed.

/// Bits stored per key.
const BITS_PER_KEY: usize = 10;
/// Probes per key. Fixed by the format: about 0.8% false positives at 10 bits per key.
const PROBES: u32 = 7;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const BASIS_A: u64 = 0xcbf2_9ce4_8422_2325;
const BASIS_B: u64 = 0x8422_2325_cbf2_9ce4;
/// The word count is stored as a u32. Capping a filter only makes it less
/// selective, it never makes it wrong.
const MAX_WORDS: usize = u32::MAX as usize;

/// A fixed-size Bloom filter. Built from a key set, then only queried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Bloom {
    words: Vec<u64>,
}

impl Bloom {
    /// A filter holding every key in `keys`. The size depends only on how many
    /// there are, and the bits only on what they are, so the same set always
    /// gives the same bytes whatever order it arrives in.
    pub(crate) fn build<'a>(keys: impl ExactSizeIterator<Item = &'a [u8]>) -> Bloom {
        let words = (keys.len().saturating_mul(BITS_PER_KEY).div_ceil(64)).clamp(1, MAX_WORDS);
        let mut bloom = Bloom {
            words: vec![0; words],
        };
        for key in keys {
            bloom.insert(key);
        }
        bloom
    }

    /// False means the key was never added. True means it probably was.
    pub(crate) fn might_contain(&self, key: &[u8]) -> bool {
        self.bit_positions(key)
            .all(|bit| self.words[bit / 64] >> (bit % 64) & 1 == 1)
    }

    /// `PROBES` then the word count as little-endian u32s, then the words.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let count = u32::try_from(self.words.len()).unwrap_or(u32::MAX);
        let mut out = Vec::with_capacity(8 + self.words.len() * 8);
        out.extend_from_slice(&PROBES.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        for word in &self.words {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Parse what [`Bloom::to_bytes`] wrote. `None` unless the probe count is
    /// the format's, the word count is nonzero, and the length matches it
    /// exactly. Nothing is allocated before that check.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Bloom> {
        let probes = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
        let count = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?) as usize;
        let body = bytes.get(8..)?;
        if probes != PROBES || count == 0 || body.len() / 8 != count || body.len() % 8 != 0 {
            return None;
        }
        let words = body
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect();
        Some(Bloom { words })
    }

    fn insert(&mut self, key: &[u8]) {
        let bits: Vec<usize> = self.bit_positions(key).collect();
        for bit in bits {
            self.words[bit / 64] |= 1 << (bit % 64);
        }
    }

    /// Double hashing: probe `i` is `h1 + i * h2` modulo the bit count.
    fn bit_positions(&self, key: &[u8]) -> impl Iterator<Item = usize> {
        let bits = self.words.len() as u64 * 64;
        let h1 = fnv1a(key, BASIS_A);
        // Odd, so the stride never collapses to zero.
        let h2 = fnv1a(key, BASIS_B) | 1;
        (0..u64::from(PROBES)).map(move |i| (h1.wrapping_add(i.wrapping_mul(h2)) % bits) as usize)
    }
}

fn fnv1a(key: &[u8], basis: u64) -> u64 {
    key.iter().fold(basis, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cargo test -p itzma-core --release --lib bloom::tests::measure_build -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn measure_build() {
        let ids: Vec<String> = (0..100_000).map(|i| format!("doc-{i:08}")).collect();
        let keys: Vec<&[u8]> = ids.iter().map(|k| k.as_bytes()).collect();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(Bloom::build(keys.iter().copied()));
        }
        println!("bloom build, 100k keys: {:?}", start.elapsed() / 20);
        let mut layer = crate::layer::Layer::new();
        for (i, id) in ids.iter().enumerate() {
            layer.apply(crate::document::WalOp::Upsert(crate::document::Document {
                id: id.clone(),
                vector: (0..128).map(|d| ((i + d) % 97) as f32).collect(),
                attrs: None,
                sparse: None,
            }));
        }
        let start = std::time::Instant::now();
        for _ in 0..5 {
            std::hint::black_box(crate::segment::encode_as(&layer, 128, 2).expect("encode"));
        }
        println!("segment encode, 100k x 128: {:?}", start.elapsed() / 5);
    }

    fn keys(prefix: &str, n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| format!("{prefix}-{i}").into_bytes())
            .collect()
    }

    fn build(keys: &[Vec<u8>]) -> Bloom {
        Bloom::build(keys.iter().map(Vec::as_slice))
    }

    #[test]
    fn has_no_false_negatives() {
        let added = keys("in", 20_000);
        let bloom = build(&added);
        assert!(added.iter().all(|k| bloom.might_contain(k)));
    }

    #[test]
    fn false_positive_rate_stays_low() {
        let bloom = build(&keys("in", 20_000));
        let absent = keys("out", 20_000);
        let hits = absent.iter().filter(|k| bloom.might_contain(k)).count();
        // Theory says about 0.8%. The bound is loose so a fixed hash cannot flake.
        assert!(hits < 400, "{hits} false positives in 20000");
    }

    #[test]
    fn insertion_order_does_not_change_the_bytes() {
        let forward = keys("k", 1_000);
        let mut backward = forward.clone();
        backward.reverse();
        assert_eq!(build(&forward).to_bytes(), build(&backward).to_bytes());
    }

    #[test]
    fn bytes_are_fixed() {
        // Pins the hash and the layout: a change here breaks byte-for-byte
        // segment reuse, so it must be deliberate.
        let bloom = build(&[b"a".to_vec(), b"b".to_vec()]);
        let bytes = bloom.to_bytes();
        assert_eq!(&bytes[..8], &[7, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(bytes.len(), 16);
        // Worked out separately in Python from the FNV-1a definition.
        assert_eq!(
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            0x4002_0030_0180_1fc0
        );
    }

    #[test]
    fn empty_set_has_one_word_and_matches_nothing() {
        let bloom = build(&[]);
        assert_eq!(bloom.to_bytes().len(), 16);
        assert!(!bloom.might_contain(b"anything"));
        assert!(!bloom.might_contain(b""));
    }

    #[test]
    fn size_follows_the_key_count() {
        // ceil(100 * 10 / 64) = 16 words.
        assert_eq!(build(&keys("k", 100)).to_bytes().len(), 8 + 16 * 8);
        // 6 keys need 60 bits, one word. 7 need 70 bits, two.
        assert_eq!(build(&keys("k", 6)).to_bytes().len(), 8 + 8);
        assert_eq!(build(&keys("k", 7)).to_bytes().len(), 8 + 16);
    }

    #[test]
    fn round_trips_through_bytes() {
        let bloom = build(&keys("k", 500));
        let back = Bloom::from_bytes(&bloom.to_bytes()).unwrap();
        assert_eq!(back, bloom);
        assert!(back.might_contain(b"k-499"));
    }

    #[test]
    fn from_bytes_rejects_bad_input() {
        let good = build(&keys("k", 10)).to_bytes();
        assert!(Bloom::from_bytes(&good).is_some());
        assert!(Bloom::from_bytes(&[]).is_none());
        assert!(Bloom::from_bytes(&good[..7]).is_none());
        assert!(Bloom::from_bytes(&good[..good.len() - 1]).is_none());
        let mut longer = good.clone();
        longer.push(0);
        assert!(Bloom::from_bytes(&longer).is_none());
        let mut zero_probes = good.clone();
        zero_probes[..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(Bloom::from_bytes(&zero_probes).is_none());
        let mut other_probes = good.clone();
        other_probes[..4].copy_from_slice(&8u32.to_le_bytes());
        assert!(Bloom::from_bytes(&other_probes).is_none());
        let mut zero_words = good.clone();
        zero_words[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(Bloom::from_bytes(&zero_words[..8]).is_none());
        // A count far past the bytes must be refused, not allocated.
        let mut huge = good;
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Bloom::from_bytes(&huge).is_none());
    }

    #[test]
    fn keys_of_any_length_are_fine() {
        let long = vec![7u8; 100_000];
        let bloom = Bloom::build([b"".as_slice(), long.as_slice()].into_iter());
        assert!(bloom.might_contain(b""));
        assert!(bloom.might_contain(&long));
    }
}
