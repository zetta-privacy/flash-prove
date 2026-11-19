//! Merkle commitment over the encoded table and column openings
//! (paper Definition 4.2, Construction D.11, Proposition 6.5).
//!
//! Construction D.11: "the leaf at a position hashes the symbol there and a
//! leaf salt." Salts are part of the prover's coins (Definition 5.16) and are
//! revealed in each opening (the proof string carries "a leaf salt"), so the
//! verifier can recompute the leaf digest. We instantiate the coin map with a
//! per-leaf salt `H(seed ‖ index)`.

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as Fp;
use rayon::prelude::*;
use rs_merkle::{algorithms::Sha256 as MerkleSha, MerkleProof, MerkleTree};
use sha2::{Digest as _, Sha256};

/// A 32-byte SHA-256 digest.
pub type Digest = [u8; 32];

/// A 32-byte leaf salt.
pub type Salt = [u8; 32];

/// Hash a field element plus a leaf salt to a Merkle leaf (Construction D.11).
pub fn leaf_salted(elem: Fp, salt: &Salt) -> Digest {
    let mut h = Sha256::new();
    h.update(elem.as_canonical_u64().to_le_bytes());
    h.update(salt);
    h.finalize().into()
}

/// Derive the leaf salt for a flat leaf index from a secret seed (the prover's
/// coin map Γ, Definition 5.16). `index` is `row * cols + col`.
pub fn derive_salt(seed: &Salt, index: usize) -> Salt {
    let mut h = Sha256::new();
    h.update(seed);
    h.update((index as u64).to_le_bytes());
    h.finalize().into()
}

/// The encoded table, stored row-major (`data[row * cols + col]`).
///
/// Rows are the `2^v` table rows (Definition 4.2); columns are the `|D|`
/// evaluation positions. A "column" `T[ϕ]` is one evaluation position `ϕ`
/// across all rows.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub rows: usize,
    pub cols: usize,
    #[serde(with = "crate::field::fp_vec")]
    pub data: Vec<Fp>,
}

impl Table {
    pub fn new(rows: usize, cols: usize, data: Vec<Fp>) -> Self {
        assert_eq!(data.len(), rows * cols, "table data length");
        Self { rows, cols, data }
    }
}

/// Commit to a table: the Merkle root and the tree used to make proofs.
///
/// `salt_seed` is the prover's secret leaf-salt seed (a coin, Definition 5.16).
pub fn commit(table: &Table, salt_seed: &Salt) -> (Digest, MerkleTree<MerkleSha>) {
    let leaves: Vec<Digest> = table
        .data
        .par_iter()
        .enumerate()
        .map(|(i, &e)| leaf_salted(e, &derive_salt(salt_seed, i)))
        .collect();
    let tree = MerkleTree::<MerkleSha>::from_leaves(&leaves);
    let root = tree.root().expect("table is non-empty");
    (root, tree)
}

/// A column opening: sorted leaf indices, the revealed values and their salts
/// (all aligned with the indices), and a multi-leaf Merkle proof.
pub struct ColumnOpening {
    pub indices: Vec<usize>,
    pub values: Vec<Fp>,
    pub salts: Vec<Salt>,
    pub proof: MerkleProof<MerkleSha>,
}

/// Open a set of columns (evaluation positions `ϕ ∈ D`) against a committed
/// table. Each column yields one leaf per row (`T[ϕ] ∈ F^{2^v}`).
pub fn open_columns(
    tree: &MerkleTree<MerkleSha>,
    table: &Table,
    columns: &[usize],
    salt_seed: &Salt,
) -> ColumnOpening {
    // Collect (leaf index, value, salt) triples, then sort by index —
    // rs_merkle requires leaf indices in ascending order.
    let mut triples: Vec<(usize, Fp, Salt)> = Vec::new();
    for &col in columns {
        for row in 0..table.rows {
            let idx = row * table.cols + col;
            triples.push((idx, table.data[idx], derive_salt(salt_seed, idx)));
        }
    }
    triples.sort_unstable_by_key(|&(idx, _, _)| idx);

    let indices: Vec<usize> = triples.iter().map(|&(idx, _, _)| idx).collect();
    let values: Vec<Fp> = triples.iter().map(|&(_, v, _)| v).collect();
    let salts: Vec<Salt> = triples.iter().map(|&(_, _, s)| s).collect();

    ColumnOpening {
        indices: indices.clone(),
        values,
        salts,
        proof: tree.proof(&indices),
    }
}

impl ColumnOpening {
    /// Verify the opening against a root, given the total number of leaves.
    pub fn verify(&self, root: Digest, total_leaves: usize) -> bool {
        let leaves: Vec<Digest> = self
            .values
            .iter()
            .zip(&self.salts)
            .map(|(&v, s)| leaf_salted(v, s))
            .collect();
        self.proof
            .verify(root, &self.indices, &leaves, total_leaves)
    }
}

/// Tracks columns opened against one commitment, enforcing the `t`-column
/// opening budget (Definition 4.2, Proposition 6.5).
pub struct OpeningBudget {
    total: usize,
    opened: usize,
}

impl OpeningBudget {
    pub fn new(total: usize) -> Self {
        Self { total, opened: 0 }
    }

    /// Try to record `n` more opened columns. Returns `false` (and records
    /// nothing) if it would exceed the budget.
    pub fn try_open(&mut self, n: usize) -> bool {
        if self.opened + n > self.total {
            return false;
        }
        self.opened += n;
        true
    }

    pub fn opened(&self) -> usize {
        self.opened
    }

    pub fn remaining(&self) -> usize {
        self.total - self.opened
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_table() -> Table {
        let data = (0..32u64).map(Fp::new).collect();
        Table::new(4, 8, data)
    }

    #[test]
    fn commit_and_open_columns_verify() {
        let table = small_table();
        let seed = [7u8; 32];
        let (root, tree) = commit(&table, &seed);

        let opening = open_columns(&tree, &table, &[1, 3], &seed);
        assert_eq!(opening.values.len(), 8); // 2 cols × 4 rows
        assert!(opening.verify(root, table.rows * table.cols));
    }

    #[test]
    fn tampered_opening_fails() {
        let table = small_table();
        let seed = [7u8; 32];
        let (root, tree) = commit(&table, &seed);

        let mut opening = open_columns(&tree, &table, &[1], &seed);
        opening.values[0] = Fp::new(999); // corrupt row 0 of column 1
        assert!(!opening.verify(root, table.rows * table.cols));
    }

    #[test]
    fn wrong_salt_fails() {
        let table = small_table();
        let seed = [7u8; 32];
        let (root, tree) = commit(&table, &seed);

        let mut opening = open_columns(&tree, &table, &[1], &seed);
        opening.salts[0] = [0u8; 32]; // wrong salt for row 0 of column 1
        assert!(!opening.verify(root, table.rows * table.cols));
    }

    #[test]
    fn budget_is_enforced() {
        let mut budget = OpeningBudget::new(5);
        assert!(budget.try_open(3));
        assert!(budget.try_open(2));
        assert!(!budget.try_open(1)); // would exceed 5
        assert_eq!(budget.remaining(), 0);
    }
}
