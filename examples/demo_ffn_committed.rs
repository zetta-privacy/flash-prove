//! Committed demo FFN: run the demo FFN op graph
//! (`embed → matmul+requant → silu → matmul+requant → add → matmul+requant`)
//! end-to-end through the committed (succinct) verifier — all on FRI-committed
//! data, so the verifier holds only commitments + opening proofs.

use flashprove::committed::{prove_matmul_committed, verify_matmul_committed, MatmulProof};
use flashprove::field::Fp;
use flashprove::fri::{self, Challenger, Prover};
use flashprove::lookup_committed::{
    prove_lookup2_committed, verify_lookup2_committed, Lookup2Proof,
};
use flashprove::zk::{
    prove_inner_product_committed_given, verify_inner_product_committed, InnerProductProof,
    Openings,
};

fn f(x: i64) -> Fp {
    if x < 0 {
        -Fp::new(x.unsigned_abs())
    } else {
        Fp::new(x as u64)
    }
}

struct DemoFfnProof {
    // matmul + requant + silu + add proofs.
    mm1: MatmulProof,   // acc1 = w1·embedding
    req1: Lookup2Proof, // h1 = requant(acc1)
    silu: Lookup2Proof, // act = silu(h1)
    mm2: MatmulProof,   // acc2 = w2·act
    req2: Lookup2Proof, // h2 = requant(acc2)
    // add: residual = h2 + embedding.
    com_h2: fri::Commitment,
    dom_h2: fri::Domain,
    com_emb: fri::Commitment,
    dom_emb: fri::Domain,
    com_res: fri::Commitment,
    dom_res: fri::Domain,
    r: Vec<Fp>,
    b_h2: Fp,
    b_emb: Fp,
    b_res: Fp,
    claim_h2: InnerProductProof,
    claim_emb: InnerProductProof,
    claim_res: InnerProductProof,
    mm3: MatmulProof,   // acc3 = lm_head·residual
    req3: Lookup2Proof, // logits = requant(acc3)
    // public table parameters (shared requant/silu lookups are evaluated directly).
    req_lo: i64,
    silu_lo: i64,
}

#[allow(clippy::too_many_arguments)]
fn prove_demo_ffn_committed(
    ctx: &mut Prover,
    embedding: &[i64],
    acc1: &[i64],
    h1: &[i64],
    act: &[i64],
    acc2: &[i64],
    h2: &[i64],
    residual: &[i64],
    acc3: &[i64],
    logits: &[i64],
    w1: &[i8],
    w2: &[i8],
    lm_head: &[i8],
    out1: usize,
    in_: usize,
    out2: usize,
    vocab: usize,
    req_lo: i64,
    silu_lo: i64,
    alpha: Fp,
    beta: Fp,
) -> DemoFfnProof {
    let emb_fp: Vec<Fp> = embedding.iter().map(|&x| f(x)).collect();
    let acc1_fp: Vec<Fp> = acc1.iter().map(|&x| f(x)).collect();
    let act_fp: Vec<Fp> = act.iter().map(|&x| f(x)).collect();
    let acc2_fp: Vec<Fp> = acc2.iter().map(|&x| f(x)).collect();
    let h2_fp: Vec<Fp> = h2.iter().map(|&x| f(x)).collect();
    let res_fp: Vec<Fp> = residual.iter().map(|&x| f(x)).collect();
    let acc3_fp: Vec<Fp> = acc3.iter().map(|&x| f(x)).collect();
    let w1_fp: Vec<Fp> = w1.iter().map(|&x| f(x as i64)).collect();
    let w2_fp: Vec<Fp> = w2.iter().map(|&x| f(x as i64)).collect();
    let lm_fp: Vec<Fp> = lm_head.iter().map(|&x| f(x as i64)).collect();

    let mm1 = prove_matmul_committed(ctx, &w1_fp, &emb_fp, &acc1_fp, out1, in_);
    let req1 = prove_lookup2_committed(ctx, acc1, h1, req_lo, alpha, beta);
    let silu = prove_lookup2_committed(ctx, h1, act, silu_lo, alpha, beta);
    let mm2 = prove_matmul_committed(ctx, &w2_fp, &act_fp, &acc2_fp, out2, out1);
    let req2 = prove_lookup2_committed(ctx, acc2, h2, req_lo, alpha, beta);

    let (com_h2, pd_h2, dom_h2) = fri::commit_vec(&ctx.pcs, &h2_fp);
    let (com_emb, pd_emb, dom_emb) = fri::commit_vec(&ctx.pcs, &emb_fp);
    let (com_res, pd_res, dom_res) = fri::commit_vec(&ctx.pcs, &res_fp);
    let mut tr = fri::Transcript::new(&ctx.challenger);
    tr.observe(&com_h2);
    tr.observe(&com_emb);
    tr.observe(&com_res);
    let r: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();
    let b_h2: Fp = r.iter().zip(&h2_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_emb: Fp = r.iter().zip(&emb_fp).map(|(&ri, &vi)| ri * vi).sum();
    let b_res: Fp = r.iter().zip(&res_fp).map(|(&ri, &vi)| ri * vi).sum();
    let claim_h2 = prove_inner_product_committed_given(ctx, &com_h2, &pd_h2, dom_h2, &h2_fp, &r);
    let claim_emb =
        prove_inner_product_committed_given(ctx, &com_emb, &pd_emb, dom_emb, &emb_fp, &r);
    let claim_res =
        prove_inner_product_committed_given(ctx, &com_res, &pd_res, dom_res, &res_fp, &r);

    let mm3 = prove_matmul_committed(ctx, &lm_fp, &res_fp, &acc3_fp, vocab, in_);
    let req3 = prove_lookup2_committed(ctx, acc3, logits, req_lo, alpha, beta);

    DemoFfnProof {
        mm1,
        req1,
        silu,
        mm2,
        req2,
        com_h2,
        dom_h2,
        com_emb,
        dom_emb,
        com_res,
        dom_res,
        r,
        b_h2,
        b_emb,
        b_res,
        claim_h2,
        claim_emb,
        claim_res,
        mm3,
        req3,
        req_lo,
        silu_lo,
    }
}

fn verify_demo_ffn_committed(
    challenger: &Challenger,
    out1: usize,
    in_: usize,
    out2: usize,
    vocab: usize,
    proof: &DemoFfnProof,
    values: &[flashprove::field::Ext],
    acc: &mut Openings,
) -> bool {
    if !verify_matmul_committed(challenger, out1, in_, &proof.mm1, values, acc) {
        return false;
    }
    if !verify_lookup2_committed(
        challenger,
        |x| flashprove::spec::requant(x, 4),
        proof.req_lo,
        proof.req1.alpha,
        proof.req1.beta,
        &proof.req1,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_lookup2_committed(
        challenger,
        flashprove::spec::silu,
        proof.silu_lo,
        proof.silu.alpha,
        proof.silu.beta,
        &proof.silu,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_matmul_committed(challenger, out2, out1, &proof.mm2, values, acc) {
        return false;
    }
    if !verify_lookup2_committed(
        challenger,
        |x| flashprove::spec::requant(x, 4),
        proof.req_lo,
        proof.req2.alpha,
        proof.req2.beta,
        &proof.req2,
        values,
        acc,
    ) {
        return false;
    }
    let mut tr = fri::Transcript::new(challenger);
    tr.observe(&proof.com_h2);
    tr.observe(&proof.com_emb);
    tr.observe(&proof.com_res);
    let r: Vec<Fp> = (0..in_).map(|_| tr.sample_val()).collect();
    if !verify_inner_product_committed(
        challenger,
        &proof.com_h2,
        proof.dom_h2,
        &r,
        proof.b_h2,
        &proof.claim_h2,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_emb,
        proof.dom_emb,
        &r,
        proof.b_emb,
        &proof.claim_emb,
        values,
        acc,
    ) {
        return false;
    }
    if !verify_inner_product_committed(
        challenger,
        &proof.com_res,
        proof.dom_res,
        &r,
        proof.b_res,
        &proof.claim_res,
        values,
        acc,
    ) {
        return false;
    }
    if proof.b_res != proof.b_h2 + proof.b_emb {
        return false;
    }
    if !verify_matmul_committed(challenger, vocab, in_, &proof.mm3, values, acc) {
        return false;
    }
    verify_lookup2_committed(
        challenger,
        |x| flashprove::spec::requant(x, 4),
        proof.req_lo,
        proof.req3.alpha,
        proof.req3.beta,
        &proof.req3,
        values,
        acc,
    )
}

fn main() {
    let mut ctx = fri::Prover::new(1, 10);
    // Small dims: in=2, out1=2, out2=2, vocab=4.
    let (in_, out1, out2, vocab) = (2usize, 2usize, 2usize, 4usize);
    let embedding = [3i64, -2];
    let w1 = [1i8, 0, 0, 1];
    let acc1 = [3i64, -2];
    let h1 = [
        flashprove::spec::requant(3, 4),
        flashprove::spec::requant(-2, 4),
    ];
    let act: Vec<i64> = h1.iter().map(|&x| flashprove::spec::silu(x)).collect();
    let w2 = [1i8, 0, 0, 1];
    let acc2 = act.clone();
    let h2: Vec<i64> = act
        .iter()
        .map(|&x| flashprove::spec::requant(x, 4))
        .collect();
    let residual: Vec<i64> = h2.iter().zip(&embedding).map(|(&a, &b)| a + b).collect();
    let lm_head = [1i8, 0, 0, 1, 0, 0, 0, 0];
    let acc3 = [residual[0], residual[1], 0, 0];
    let logits: Vec<i64> = acc3
        .iter()
        .map(|&x| flashprove::spec::requant(x, 4))
        .collect();
    let alpha = f(11);
    let beta = f(5);

    let proof = prove_demo_ffn_committed(
        &mut ctx, &embedding, &acc1, &h1, &act, &acc2, &h2, &residual, &acc3, &logits, &w1, &w2,
        &lm_head, out1, in_, out2, vocab, -64, -64, alpha, beta,
    );
    let fri_proof = ctx.open_all();
    let values = ctx.values.clone();
    let mut acc = Openings::new();
    let ok = verify_demo_ffn_committed(
        &ctx.challenger,
        out1,
        in_,
        out2,
        vocab,
        &proof,
        &values,
        &mut acc,
    ) && fri::open_many_verify(&ctx.pcs, &ctx.challenger, &acc, &fri_proof);
    println!("committed demo FFN: {}", if ok { "OK" } else { "REJECTED" });
    assert!(ok);
}
