//! The base field F = F_p with p = 2^64 − 2^32 + 1 (Goldilocks) and the
//! degree-3 extension E = F[ϑ]/(ϑ³ − 2), used for out-of-domain samples and
//! challenges (paper Table 2, Definition 4.3).

pub use p3_goldilocks::Goldilocks as Fp;

/// A base-field element from a canonical (already-reduced) integer.
pub fn from_u64(v: u64) -> Fp {
    Fp::new(v)
}

/// An element of E, stored as `a + b·ϑ + c·ϑ²` (three base-field coefficients).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ext([Fp; 3]);

impl Ext {
    pub const ZERO: Ext = Ext([Fp::new(0), Fp::new(0), Fp::new(0)]);
    pub const ONE: Ext = Ext([Fp::new(1), Fp::new(0), Fp::new(0)]);

    pub fn new(a: Fp, b: Fp, c: Fp) -> Self {
        Self([a, b, c])
    }

    /// Embed a base-field element into E (b = c = 0).
    pub fn from_base(a: Fp) -> Self {
        Self([a, Fp::new(0), Fp::new(0)])
    }

    /// Coefficients `(a, b, c)` with `self = a + b·ϑ + c·ϑ²`.
    pub fn parts(&self) -> [Fp; 3] {
        self.0
    }

    /// Canonical base-field representative (the `c` coefficient if self ∈ F).
    pub fn as_base(&self) -> Option<Fp> {
        if self.0[1] == Fp::new(0) && self.0[2] == Fp::new(0) {
            Some(self.0[0])
        } else {
            None
        }
    }
}

impl std::ops::Add for Ext {
    type Output = Ext;
    fn add(self, rhs: Ext) -> Ext {
        Ext([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl std::ops::Sub for Ext {
    type Output = Ext;
    fn sub(self, rhs: Ext) -> Ext {
        Ext([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl std::ops::Neg for Ext {
    type Output = Ext;
    fn neg(self) -> Ext {
        Ext([-self.0[0], -self.0[1], -self.0[2]])
    }
}

/// Multiplication modulo ϑ³ = 2:
///
/// (a + bϑ + cϑ²)(a' + b'ϑ + c'ϑ²)
///   = aa' + 2(bc' + cb') + (ab' + ba' + 2cc')ϑ + (ac' + bb' + ca')ϑ².
impl std::ops::Mul for Ext {
    type Output = Ext;
    fn mul(self, rhs: Ext) -> Ext {
        let [a, b, c] = self.0;
        let [d, e, f] = rhs.0;
        let two = Fp::new(2);
        Ext([
            a * d + two * (b * f + c * e),
            a * e + b * d + two * (c * f),
            a * f + b * e + c * d,
        ])
    }
}
