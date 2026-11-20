# FlashProve (Rust)

A Rust reimplementation of **FlashProve** — hash-based proofs of LLM inference — built on the
**GKR + LogUp + FRI** stack. It lets a prover convince a verifier that a quantized
Mixture-of-Experts language model (e.g. OLMoE-1B-7B) emitted a particular token, without the
verifier ever running the model or seeing its weights.

This document is about *how it works under the hood*. For commands and layout, see the
`Makefile` and `python/`.

---

## The claim being proven

The model is quantized to small integers: `int8` weights, `int8` activations, and every matmul is
followed by a fixed-point requantization (a rounding right-shift). Inference is therefore a
*deterministic integer computation*, and — crucially — integer arithmetic coincides with
arithmetic in a finite field as long as every intermediate value stays inside the field range.
That coincidence is what makes the whole thing provable.

The prover runs the model once, recording every intermediate value (the **trace**): the
embedding, per-layer Q/K/V, attention scores and weights, MoE gate/up/down/act, the router's
top-`k` choice, residuals, and the final logits. The claim is a single bit of output:

> `head_token = argmax(logits[last])`

i.e. the last token the model would emit is the one the prover says it is.

---

## The core idea: arithmetize, commit, then reduce

There are three ingredients, layered on top of each other.

### 1. Arithmetization — turn the model into an op graph

The model is not a monolithic function; it's a graph of small, well-understood operations:

| op | constraint |
|---|---|
| `embed` | `out[b] = E[tok[b]]` (a table lookup) |
| `matmul` | `out = requant(W · x, shift)` |
| `norm` | `out = requant(x · γ · inv_q, Q)`, `inv_q = round(2^Q / √(mean_sq + ε))` |
| `rope` | rotate each head-dim pair by `pos`/`θ` |
| `scores` | `s = round(q · kᵀ / scale)` |
| `causal_softmax` | `w = exp(s − running_max) / Σ exp(·)`, masked to `t ≤ pos` |
| `weighted_v` | `out = w · v` |
| `silu` / `mul` / `add` | elementwise |
| `top_k` | `choice = indicator of the k largest router scores` |
| `weighted_sum` | `out = Σ_e choice[e] · term_e` |

The prover commits every cell and every weight to a polynomial commitment. Each op then becomes a
constraint *between committed values* that the verifier must check.

### 2. Commitment — FRI

Every cell and weight is committed with **FRI** (a fast Reed–Solomon-style IOP of proximity): the
data is laid out as evaluations of a polynomial over a finite field, and the commitment is a Merkle
tree root over those evaluations. FRI lets the prover *open* the polynomial at a point and prove
the opened value is consistent with the commitment, in `O(log n)`.

The base field is **Goldilocks** (`p = 2⁶⁴ − 2³² + 1`), and challenges live in its cubic extension.
There is **no trusted setup** — it's all Merkle trees and hashing.

### 3. Reduction — the three ways each op is checked

This is where the `O(log n)` verifier comes from. Each op reduces a large claim to a handful of
small field checks:

- **Matmul → GKR sumcheck.** To check `C = A·B` the verifier samples random points `(r, s)`,
  forms the single claim `C̃(r,s) = Σ_k Ã(r,k)·B̃(k,s)`, and runs a **sumcheck** over the
  contraction index `k`. `log n` rounds later the claim is a product of *two* MLE evaluations
  at one point — each opened through FRI. The verifier never touches the `n` entries of `C`.

- **Nonlinearity → LogUp lookup.** `out[i] = f(in[i])` for a public `f` (SiLU, `exp`, `top-k`
  sign). The prover proves each `(in[i], out[i])` is a row of a public table of `(x, f(x))` via a
  grand-product/LogUp argument. The table is committed once and reused.

- **Elementwise linear ops → one random combination.** `add`, `mul`, `requant`, RMSNorm's scale,
  and the RoPE rotation are checked at a single random point via Schwartz–Zippel, plus a
  **row-tie** that binds a per-(position, head) slice back to the whole-tensor commitment.

A subtle but important point: `requant`, the rounding shift after every matmul, is itself proven
(not assumed) — the verifier is convinced the value stayed in range, so integer arithmetic really
did equal field arithmetic.

### The head argmax

The final claim — `head_token` is the argmax of the logits — is proven without revealing the whole
logits row: a **range check** shows every score is `≤ head_max`, and a **point evaluation** shows
`scores[head_token] = head_max`. The sign table for this is over the *score* range, not the
vocab-scaled top-`k` delta, so it stays small.

---

## One FRI proof to rule them all

Every check above ends in an *opening* of some committed polynomial at a field point. The prover
doesn't produce a FRI proof per opening — it records them all into a transcript and emits a
**single shared FRI proof** at the end. The verifier replays the exact same transcript (same
challenge derivation order), accumulates the same openings, and checks the one FRI proof.

All challenges are derived by **Fiat–Shamir**: hashing the prover's transcript with Poseidon2. The
opening *order* is load-bearing — prove and verify must walk the op graph and sample challenges in
identical sequence, or the single batch check fails.

---

## Commit → prove → verify (the binding)

The three programs split the work the way the paper does:

1. **`cp_commit`** commits the model weights *once* (offline) and publishes the roots.
2. **`cp_prove`** re-derives the same (deterministic) commitments, commits the request's trace,
   and proves. The proof embeds `name → commitment` for every weight.
3. **`cp_verify`** checks that the proof's weight commitments **equal the pre-shared roots** —
   this is what *binds* the proof to the real model — then verifies.

That binding is the soundness core: a malicious prover who swaps in different weights produces a
different commitment, which won't match the published one, so the proof is rejected. The verifier
holds only the small weight commitment + one proof, and needs no GPU.

---

## Why the verifier is cheap

The verifier's work is, per op, a few sumcheck rounds or one lookup — each `O(log n)` — plus one
FRI proof check. It never materializes the `n`-sized matrices or activations, and never
recomputes a single matmul. Total: `O(depth · log n)`. The prover, meanwhile, does the full
inference plus the proof generation (committing and opening everything), which is the expensive
side — as intended.

---

## Status

Research prototype. FRI parameters (`log_blowup = 1`, `num_queries = 10`) are far below the
paper's 128-bit security target; the scheme is **not zero-knowledge** (openings leak linear
combinations); and there is no cross-request batching, segmentation, or link-proof amortization.
The Lean formalization from the paper is out of scope.
