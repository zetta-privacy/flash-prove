# FlashProve (Rust)

A Rust reimplementation of **FlashProve** — hash-based proofs of LLM inference — built on the
**GKR + LogUp + WHIR** stack. It lets a prover convince a verifier that a quantized
Mixture-of-Experts language model (e.g. OLMoE-1B-7B) emitted a particular token, without the
verifier ever running the model or seeing its weights.

This document is about *how it works under the hood*. For commands and layout, see the
`Makefile` and `python/`.

---

## Scale

The target is **OLMoE-1B-7B** in full: 16 MoE layers, hidden 2048, 64 experts with 8 active,
vocab 50304, and a real prompt window. Nothing in the Rust is model-specific — the architecture
comes from a spec JSON and the weights from a pre-committed file — but the configuration actually
exercised end to end so far is a reduced one (one layer, 32k vocab, a short window), because the
full model's activations do not fit in the 16 GB of prover memory available here without
streaming them segment by segment.

Cost figures for that reduced configuration, with the machine they were taken on, are collected
under [Development measurements](#development-measurements). They are not a claim about
full-scale cost.

---

## The claim being proven

The model is quantized to small integers: `int8` weights and activations, and every matmul is
followed by a fixed-point requantization (a rounding right-shift). Inference is therefore a
*deterministic integer computation*, and — crucially — integer arithmetic coincides with
arithmetic in a finite field as long as every intermediate value stays inside the field range.
That coincidence is what makes the whole thing provable.

The prover runs the model once, recording every intermediate value (the **trace**): the
embedding, per-layer Q/K/V, attention scores and weights, MoE gate/up/down/act, the router's
top-`k` choice, residuals, and the final logits. The claim is a single bit of output:

> `head_token = argmax(logits[last])`

i.e. the last token the model would emit is the one the prover says it is.

### Keeping the integers small (the shift table)

"Stays inside the field range" is not free: the network's values grow through the residual stream
and the MoE products. The engine therefore requantizes **every** op that produces a cell, and the
shift is a **per-role constant** carried in the spec (`layer.*.expert.*.gate_proj`,
`add:layer.*.residual_2`, …) — never per op instance, because the terms of a `weighted_sum` must
share a scale or the sum is meaningless.

`python/calibrate.py` produces that table: it runs the integer model, measures each role's largest
pre-shift value, and sets the shift that brings it to ~2¹¹ bits, iterating until the table stops
moving. The committed values then all fit comfortably inside `(−p/2, p/2)` — `FLASHPROVE_
CHECK_BITS=31` reports *every committed value fits* for every segment. This is quantization by
construction: the reference model is whatever the table says it is, and the spec is what both
sides prove about it.

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
| `silu` / `mul` / `add` | elementwise, optionally rescaling the result |
| `top_k` | `choice = indicator of the k largest router scores` |
| `weighted_sum` | `out = Σ_e choice[e] · term_e` |

The prover commits every cell and every weight to a polynomial commitment. Each op then becomes a
constraint *between committed values* that the verifier must check. The op graph is data
(`data/olmoe_small.json`), so nothing model-specific lives in the Rust.

### 2. Commitment — stacked WHIR

A whole **segment** (one slice of the op graph) is committed as **one** WHIR root. Every cell of
that segment is a table in a *stacked* commitment — one Merkle tree over all the tables — so a
segment costs exactly one root and one WHIR proof, no matter how many claims the ops inside it
make.

Cells are **packed**: all cells of the same arity share one table, one column each (up to 64
columns). This matters more than it sounds. The verifier's WHIR work is paid per *table*, not per
element, so packing a segment's 6,205 cells into 103 tables cut verification of that segment from
132 s to 7.5 s on its own. Both sides derive the packing from the cell arities, so nothing extra
travels in the proof.

WHIR is a BaseFold-style multilinear IOP of proximity: the tables are Reed–Solomon encoded over a
larger domain, Merkle-committed, and opened at a point with a folding argument. The verifier's
proximity check is `O(log n)` — it never reads the `n` encoded entries.

The base field is **Goldilocks** (`p = 2⁶⁴ − 2³² + 1`), challenges live in its cubic extension, the
commitments are BLAKE3 over SIMD-batched field leaves, and Fiat–Shamir is a Poseidon2 duplex
sponge. There is **no trusted setup** — it's all Merkle trees and hashing.

### 3. Reduction — the three ways each op is checked

Each op reduces a large claim to a handful of small field checks:

- **Matmul → GKR sumcheck.** To check `C = A·B` the verifier samples random points `(r, s)`,
  forms the single claim `C̃(r,s) = Σ_k Ã(r,k)·B̃(k,s)`, and runs a **sumcheck** over the
  contraction index `k`. `log n` rounds later the claim is a product of *two* MLE evaluations
  at one point — each opened against the segment's WHIR root. The verifier never touches the `n`
  entries of `C`.

- **Nonlinearity → LogUp lookup.** `out[i] = f(in[i])` for a public `f` (SiLU, `exp`, `top-k`
  sign). The prover proves each `(in[i], out[i])` is a row of a public table of `(x, f(x))` via a
  grand-product/LogUp argument. The table is committed once and reused.

- **Elementwise linear ops → one random combination.** `add`, `mul`, `requant`, RMSNorm's scale,
  and the RoPE rotation are checked at a single random point via Schwartz–Zippel, plus a
  **row-tie** that binds a per-(position, head) slice back to the whole-tensor commitment.

A subtle but important point: `requant`, the rounding shift after every matmul, is itself proven
(not assumed) — the verifier is convinced the value stayed in range, so integer arithmetic really
did equal field arithmetic. Each requant carries a sign/magnitude decomposition and two
digit/bit range checks, which is why the arithmetization is claim-heavy around matmuls.

### The random vector is never materialized

Every linear claim is `⟨r, v⟩ = b` for a "random" `r`. The obvious spelling — draw `n` field
elements — makes the *verifier* pay `O(n)` per claim, which dominated everything in an early
version of this code. Instead `r` is the **equality tensor** of `m = log n` coins,
`r_j = Π_k (j_k ? c_k : 1 − c_k)`, and the shapes the op graph needs each have a closed form:

| shape | what it is | evaluation |
|---|---|---|
| `Tensor` | the whole vector | `Π_k ((1−c_k)(1−z_k) + c_k·z_k)` |
| `Blocks` | blocks of it, scaled and offset (a tensor row inside a cell, a digit decomposition) | one product plus `O(blocks)` public terms |
| `Ones` | all ones (LogUp balance, argmax count) | the constant `1` |
| `Dense`, `Block`, `Sparse` | genuinely arbitrary public vectors | the explicit sum |

So the verifier's per-claim work is `O(m)` — a handful of multiplications — and the proof carries
the `m` coins rather than the `2^m`-entry vector. Soundness is the usual `m/|F|` Schwartz–Zippel
bound. The prover still expands the vector (`O(n)`, which it has the data for anyway).

### The head argmax

The final claim — `head_token` is the argmax of the logits — is proven without revealing the whole
logits row: a **range check** shows every score is `≤ head_max`, and a **point evaluation** shows
`scores[head_token] = head_max`. The sign table for this is over the *score* range, not the
vocab-scaled top-`k` delta, so it stays small.

---

## One WHIR proof per segment

Every check above ends in an *opening* of a committed table at a field point. The prover does not
produce a proof per opening: it records every claim against the segment's stacked root, then
emits a **single WHIR proof for the segment** at the end. The verifier replays the same transcript
(same challenge derivation order), accumulates the same claims, and checks the one WHIR proof.

All challenges are derived by **Fiat–Shamir**. The opening *order* is load-bearing — prove and
verify must walk the op graph and sample challenges in identical sequence, or the batch check
fails. That is also the reason a claim's stored random vector is `m` coins and not the expansion:
both sides derive it from the transcript, and the proof only pins it.

Segments are independent (each has its own root, transcript and proof), so they verify on one
thread each. Cross-segment copies are tied by opening both at a public point, and those ties are
checked in parallel with the segments.

---

## Commit → prove → verify (the binding)

The three programs split the work the way the paper does:

1. **`cp_commit`** commits every weight tensor to its own WHIR root *once* (offline, dropping each
   tensor's witness as soon as its root exists) and publishes the `name → root` map.
2. **`cp_prove`** commits the request's trace and proves. Weights are never copied into the
   activation segments: each weight claim is opened against its pre-shared root.
3. **`cp_verify`** checks that the roots the proof opens against **equal the pre-shared map** —
   this is what *binds* the proof to the real model — then verifies each segment. It materialises only the
   norm `γ` vectors and the embedding's shape; the remaining ~550 MB of `int8` values are
   skipped rather than parsed.

That binding is the soundness core: a malicious prover who swaps in different weights produces a
different commitment, which won't match the published one, so the proof is rejected. The verifier
holds only the small weight commitment + one proof, and needs no GPU.

---

## What the verifier actually costs

Per claim, a handful of sumcheck rounds and one `O(m)` evaluation — no `n`-sized vector, no
matmul. Per segment, one WHIR proof check, which is `O(log n)` in the segment's encoded size. What
is *not* polylog is the **number of claims**: one or a few per op, a per-row one for `norm`, and a
per-(row, expert) one for the MoE combine. That count, times the per-claim constant, is what
verification time is made of, and it scales with the model — 16 layers of 64 experts is 16× the
work of the development configuration, spread over more segments and therefore more threads.

So the honest summary is: polylog per claim and per segment, linear in the arithmetic's *claim
count*. Closing that last gap means either fewer claims per op or collapsing the per-op claims
into one GKR over the whole op graph — the latter also reduces the prover's hashing, but it is a
rewrite of the claim layer, not a tuning pass.

The remaining slack in the current implementation, in order: overlapping each segment's WHIR check
with its claim replay (they are independent and currently run one after the other); the few
remaining brute-force random vectors in `norm`, `softmax`, `rope` and the head claim; the
per-(row, expert) claim count of the MoE combine; and the size of the recorded openings in the
proof file.

---

## Development measurements

All of the following are one reduced configuration, run on an **Apple M1 Pro with 16 GB of RAM**,
macOS. It is a development stand-in for OLMoE-1B-7B, not the target model:

- `olmoe/config_small.json`: **1** MoE layer, hidden 2048, 64 experts (8 active), vocab 32768
- a 64-token window

| step | time | peak RSS |
|---|---|---|
| `cp_commit` (weights, pre-shared roots) | 36 s | 7.0 GB |
| `cp_prove` (Metal kernels on) | 2,593 s | 10.3 GB |
| `cp_verify` | 5.1 s wall, 9.0 s CPU | 0.9 GB |
| proof file | 63.7 MB | — |

Verification details at that scale: nine segments on nine threads, per segment 4.5 s / 1.4 s /
0.5 s ×7, plus ~0.6 s of loading. Of the 63.7 MB proof, the WHIR proofs themselves are 2.5 MB; the
rest is the per-op sub-proofs and the recorded openings. The prover is Metal-accelerated
(`FLASHPROVE_GPU=1`) for Merkle hashing, the WHIR NTT, MLE evaluations and the sumcheck;
verification is CPU-only.

Treat these as the shape of the cost and the development loop's timings. They say nothing about
the full model's numbers, which have not been produced here.

---

## Status

Working end to end: commit → prove → verify, `accept`, 75 library + 2 integration tests.

- **Succinct.** The verifier never reads the tables: `O(m)` per claim, one `O(log n)` WHIR check per
  segment. Not succinct in *claim count* — one per norm row, one per (row, expert) of the MoE
  combine — so verification is linear in it (5.1 s; of the 63.7 MB proof, 2.5 MB is WHIR).
- **Complete.** An honest prover for the spec the verifier holds is always accepted; the tests also
  cover the rejection side. The statement is about the *integer* model, not the float checkpoint.
- **Not zero-knowledge, deliberately.** OLMoE is open-weight, so hiding the weights buys nothing —
  the proof buys integrity, not confidentiality. The witness is the trace (prompt and activations),
  so request privacy would need a real ZK layer.
- Full OLMoE-1B-7B has not been proved end to end here; prover memory is the constraint.
- The quantization is ours, not a faithful OLMoE: per-tensor-max weights, calibrated power-of-two
  activation scales.
- WHIR parameters are a first working choice (`security_level = 100`), below the paper's 128-bit
  target. No cross-request batching. Lean formalization out of scope.
