//! Small field utilities shared by the GKR layer.

use crate::field::Ext;
use p3_field::PrimeCharacteristicRing;

/// Boolean point of `k` as `log` field elements (bit 0 lowest).
pub fn bool_point(k: usize, log: usize) -> Vec<Ext> {
    (0..log)
        .map(|b| {
            if (k >> b) & 1 == 1 {
                Ext::ONE
            } else {
                Ext::ZERO
            }
        })
        .collect()
}
