//! The base field F = F_p with p = 2^64 − 2^32 + 1 (Goldilocks) and the
//! degree-3 extension E = F[x]/(x³ − x − 1), used for out-of-domain samples and
//! challenges (paper Table 2, Definition 4.3).

pub use p3_field::extension::CubicTrinomialExtensionField;
pub use p3_goldilocks::Goldilocks as Fp;

use p3_field::PrimeField64;

/// A base-field element from a canonical (already-reduced) integer.
pub fn from_u64(v: u64) -> Fp {
    Fp::new(v)
}

/// An element of E, Plonky3's degree-3 extension of Goldilocks (x³ = x + 1).
///
/// NOTE: the paper's E is F[ϑ]/(ϑ³ − 2), an isomorphic-but-different degree-3
/// extension. This uses Plonky3's native cubic extension so the OOD samples and
/// the FRI challenge field are the *same* field.
pub type Ext = CubicTrinomialExtensionField<Fp>;

/// `#[serde(with = "crate::field::fp_vec")]` for `Vec<Fp>`.
pub mod fp_vec {
    use super::Fp;
    use p3_field::PrimeField64;
    use serde::Deserialize;

    pub fn serialize<S: serde::Serializer>(v: &[Fp], s: S) -> Result<S::Ok, S::Error> {
        let u: Vec<u64> = v.iter().map(|x| x.as_canonical_u64()).collect();
        serde::Serialize::serialize(&u, s)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Fp>, D::Error> {
        let u = <Vec<u64>>::deserialize(d)?;
        Ok(u.into_iter().map(Fp::new).collect())
    }
}

/// `#[serde(with = "crate::field::fp_vec_vec")]` for `Vec<Vec<Fp>>`.
pub mod fp_vec_vec {
    use super::Fp;
    use p3_field::PrimeField64;
    use serde::Deserialize;

    pub fn serialize<S: serde::Serializer>(v: &[Vec<Fp>], s: S) -> Result<S::Ok, S::Error> {
        let u: Vec<Vec<u64>> = v
            .iter()
            .map(|row| row.iter().map(|x| x.as_canonical_u64()).collect())
            .collect();
        serde::Serialize::serialize(&u, s)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Vec<Fp>>, D::Error> {
        let u = <Vec<Vec<u64>>>::deserialize(d)?;
        Ok(u.into_iter()
            .map(|row| row.into_iter().map(Fp::new).collect())
            .collect())
    }
}
