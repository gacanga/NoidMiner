// Dumps the constants and test vectors needed by the C/CUDA NOID miner.
use noid_core::hardware::{tower_to_flat_u128, flat_to_tower_u128};
use noid_core::Block128;
use noid_poseidon2b::batch::FixedFieldNonceBatch;
use noid_poseidon2b::native::compression::Poseidon2bSponge;
use noid_poseidon2b::native::domain::{capacity_iv, capacity_iv_flat, TAG_POWHDR};
use noid_poseidon2b::native::permutation::{ROUND_CONSTANTS, MDS_FULL, MDS_PARTIAL, N_ROUNDS, STATE_SIZE};

fn h(v: u128) -> String { format!("0x{:032x}", v) }

fn digest(fields: &[Block128; 16]) -> [u8; 32] {
    let mut s = Poseidon2bSponge::with_iv(capacity_iv(TAG_POWHDR));
    for c in fields.chunks_exact(2) { s.absorb_pair(c[0], c[1]); }
    s.finalize_no_pad()
}

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
           fn u128(&mut self) -> u128 { ((self.next() as u128) << 64) | self.next() as u128 } }

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "consts" {
        println!("T2F"); for i in 0..128 { println!("{}", h(tower_to_flat_u128(1u128 << i))); }
        println!("F2T"); for i in 0..128 { println!("{}", h(flat_to_tower_u128(1u128 << i))); }
        let iv = capacity_iv_flat(TAG_POWHDR); println!("IV {} {}", h(iv[0]), h(iv[1]));
        println!("RC"); for i in 0..STATE_SIZE { for r in 0..N_ROUNDS { println!("{}", h(tower_to_flat_u128(ROUND_CONSTANTS[i][r]))); } }
        println!("MDSF"); for i in 0..4 { for j in 0..4 { println!("{}", h(tower_to_flat_u128(MDS_FULL[i][j]))); } }
        println!("MDSP"); for i in 0..4 { for j in 0..4 { println!("{}", h(tower_to_flat_u128(MDS_PARTIAL[i][j]))); } }
        return;
    }
    // vectors: "fields(16 x 32hex LE-u128 in tower basis) nonce digest(64 hex bytes)"
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for t in 0..64 {
        let mut f = [Block128::from(0u128); 16];
        for i in 0..16 { f[i] = Block128::from(if t == 0 { 0 } else { rng.u128() }); }
        // realistic small scalars on some vectors
        if t % 2 == 1 { f[6] = Block128::from(1_760_000_000u128 + t as u128); f[7] = Block128::from(250_000u128 + t as u128);
                        f[13] = Block128::from(9u128); f[14] = Block128::from(300u128); f[15] = Block128::from(4000u128); }
        let d = digest(&f);
        // cross-check with production batch path, for 4 consecutive nonces
        let nonce0 = f[10].to_u128();
        let mut out = [[0u8; 32]; 4];
        FixedFieldNonceBatch::new(TAG_POWHDR, &f, 10).hash_into(nonce0, &mut out);
        assert_eq!(out[0], d, "batch path mismatch");
        for k in 1..4 { let mut g = f; g[10] = Block128::from(nonce0.wrapping_add(k as u128)); if nonce0.checked_add(k as u128).is_some() { assert_eq!(out[k], digest(&g)); } }
        let fs: Vec<String> = f.iter().map(|x| format!("{:032x}", x.to_u128())).collect();
        println!("{} {}", fs.join(","), hex::encode(d));
    }
}
