# **FlashProve: Hash-Based Proofs of LLM Inference at 400 Billion Parameters** 

This paper is based in part on a collaboration with Alessandro Chiesa, Giacomo Fenzi, and Guy Weissenberg, who provided ideas and direction towards the current protocol. 

##### **Abstract** 

FlashProve is a hash-based commit-and-prove argument for language-model inference: a provider commits to a model’s weights, then proves its greedy outputs correct. Three properties of the protocol are machine-checked in Lean, at exact match, in the random-oracle model, for a configuration fixed before the oracle is sampled, each under the conditions stated in Section 1.3: index-binding knowledge soundness (error about _q/_ 2<sup>124</sup> + 2<sup>_−_101</sup> after _q_ queries) and, for one request per proof, completeness on served requests (error at most 2<sup>_−_40</sup> ) and zero knowledge (error below 2<sup>_−_154</sup> , up to 688,127 replies and 2<sup>64</sup> hash queries); one bit per request, which depends on the weights, is revealed. C++ and CUDA programs produce proofs for a 400-billion-parameter mixture-of-experts integer model on public weights, on one H100 at a steady rate of about 110 prefill tokens per second, in batches of 24 prompts, with the masking of the weights on; each proof has 18.74 to 18.81 MB, checks one output token, and has been checked by the C++ verifier alone. The theorems concern the protocol and its prover in Lean, not these programs; that the programs likely hide the weights, at the default configuration, in honest runs, under conditions that the release’s documentation lists, is our judgement, not a proof, and does not cover a real model’s weights. 

#### **Contents** 

|**1**|**Introduction**|**2**|
|---|---|---|
|**2**|**Preliminaries and definitions**|**8**|
|**3**|**The relation: model, policies, configuration**|**11**|
|**4**|**The protocol**|**14**|
|**5**|**Soundness**|**20**|
|**6**|**Zero knowledge and completeness**|**26**|
|**7**|**Evaluation**|**40**|
|**8**|**Limitations and open problems**|**46**|
|**9**|**Related work**|**50**|
|**W**|**ho wrote and checked this**|**51**|
|**A **|**Proof of the soundness theorem**|**56**|
|**B **|**Further statements and proofs for Section 5**|**66**|
|**C **|**Proofs on zero knowledge**|**70**|
|**D **|**Sub-protocols, parameters and round errors**|**77**|
|**E **|**Sources and credit**|**82**|
|**F **|**Notation and Lean names**|**87**|
|**G **|**Further limitations**|**94**|
|**H **|**The artifact and the evaluation in detail**|**98**|



1 

## **1 Introduction** 

### **1.1 The problem** 

A client of a hosted language model has difficulty checking which model produced its text [CSZS25]: the weights are secret. 

In FlashProve, a hash-based commit-and-prove argument [CFQ19], the provider publishes a commitment to its weights _M_ and proves, for a prompt _x_ and an output _y_ , that _y_ is the greedy output of committed weights on _x_ (at exact match; Section 1.3 says which weights). The task resembles that of a function-hiding functional commitment [BNO21]. Since floating-point inference differs between machines [DiFR], the proofs concern an integer model with the shape of Llama 4 Maverick (Definition 3.1). _Exact match_ requires the greedy token at every token position; a _clipped-gap policy_ , which builds on the logit margin of Token-DiFR [DiFR], bounds the clipped logit gaps on average (Definition 3.4). 

The goals are soundness and zero knowledge, formally _index privacy_ (Definition 2.14); what is proved, after compilation, is zero knowledge with request leakage (Definition 6.34; Remark 6.40). By design, every published value that depends on the weights is masked or enters only through a salted hash, apart from a declared list (the release’s `docs/what-is-not-hidden.md` ; Definition H.1) and from what a client learns by design; in Lean zero knowledge is proved for the protocol with one request per proof, under exact match (Section 6.5, with its bounds), and nothing for the programs: that the programs reveal nothing more is the design’s aim, not an established property. 

### **1.2 Overview of the protocol** 

Informally, the protocol (Section 4) is as follows. 

- _Masked encoding._ The weights, written as a matrix of digits, are encoded row by row, over a 64-bit prime field, by a Reed–Solomon code with _t_ random lowest coefficients ([CFWpc]; [CY24, Section 32.5]; [CFW26]). Any _t_ columns of this _committed table_ are independent of the weights; _t_ is the _opening budget_ (Definition 4.2). 

- _Copies._ Every proof opens columns, hence the provider makes _copies_ with fresh randomness ([CFWpc]; [CFW26]). An _offline proof_ shows that the first table decodes to a matrix in range; a _link proof_ , that a copy decodes to the same matrix; an _online proof_ , that the output meets the policy at that matrix (Definition 4.6 and Definition 4.7). 

- _Out-of-domain samples._ The proximity tests [CFW26] run beyond unique decoding, strictly below the Johnson radius, where fewer opened columns suffice but a word can be close to several codewords. Masked linear forms in each table’s rows, published at a point derived from its root, determine one of these up to a small error (Definition 4.5; [CFWpc]; the masks: [CFW26, version of February 25, 2026, Section 9.2]; the technique: [BGKS20], as in [ACFY24b, Section 4.4]). 

- _Compilation._ Rounds that sample positions have proof of work (Construction 4.13; [CFWpc], with published uses in [Sta21, Section 3.11.3], [ACFY24a, Section 6.2] and [ACFY24b, Section 6.2]). Merkle trees and the Fiat–Shamir transformation make the proofs non-interactive [BCS16, COS20]. 

- _Cost._ The weights, not the tokens, dominate the cost [LY26]: the table is committed once, and a batch opens one copy once for all its requests, in at most _ν_ on columns (Table 2), as in Brakedown [GLSTW23] and TensorSwitch [BFRW25] (see also [CFW26]). The model’s run is proved segment by segment, so that one GPU’s memory can suffice (Appendix E.2). 

### **1.3 Results** 

Statements here are informal; Table 1 lists the theorems by object, and Section 1.4 gives their chief limits. The theorems, machine-checked in Lean, concern the protocol as Lean states it; the C++ and CUDA programs are tested and measured, not proved. Under human direction, AI systems wrote this paper and almost all of the programs and the Lean library, and made every test, calculation, reading and review that it reports; as far as we know, no person has checked these tests, calculations, readings and reviews (Who wrote and checked this). 

2 

**The protocol.** FlashProve _Z_ consists of three interactive oracle proofs (IOPs; Construction 4.13), with honest provers written in Lean as definitions, the _Lean provers_ (Definition 6.14); compilation turns them into the compiled Lean argument Π<sup>lp</sup> _Z_<sup>:acommitter,aproverandtheverifier</sup><sup>_VZ_,whichisalsotheverifierofΠ</sup><sup>_Z_</sup> (Construction 4.15 and Definition 6.15). Π _Z_ is the argument whose committer forms several copies and whose prover serves batches; in the Lean statements of Section 5 its committer and prover are constant functions (Remark 6.16). 

The _configuration Z_ , three byte strings called its files, states all parameters and the policy; an _admissible configuration_ equals the default one, _Z_ 0, up to its scales and eight numbers, within stated bounds (Definition 3.8). Its files are _in use_ if, moreover, the policy is exact match, which fixes three of the eight numbers, and the _parameter checks_ pass, without which _VZ_ rejects (Definition 6.11); those of _Z_ 0, as written out in Lean, are in use (Proposition 6.17). 

Three properties are machine-checked, in the random-oracle model: soundness, a property of the verifier alone, and completeness and zero knowledge, for the protocol whose verifier is proved sound: their provers are those of the IOPs from which _VZ_ is compiled, and _VZ_ itself judges the compiled prover, all of one compiled system, Π<sup>lp</sup> _Z_<sup>(Remark6.46).</sup> 

**Main theorem** (informal; Theorem 5.1, machine-checked). _Let Z be an admissible configuration with exact match, fixed before the oracle is sampled. In the random-oracle model with a second hash function H_<sup>_′_</sup> _fixed (Assumption 4.11), there is an extractor that reads weights M_<sup>_∗_</sup> _off the commitment and the committer’s hash queries, before any proof is made, such that for all committers and provers whose bounds on queries add up to at most q_ 



The property is index-binding knowledge soundness (Definition 2.7; Remark 2.8 (Names of the properties)). The weights _M_<sup>_∗_</sup> need not be the committer’s own, no witness is extracted, and the extractor is a function, not shown computable (Remark 5.3, Reading the theorem). For the committer compiled from the Lean provers (Definition 6.15), at files in use (Definition 6.11), within its budget of 2<sup>48</sup> hash queries (Proposition 6.33), the extractor gives back the committed weights except with probability about 2<sup>_−_155</sup> (Theorem 6.31). 

**Completeness** (informal; Theorem 6.29, machine-checked). _Let the files of Z be in use. At every_ served _request, that is, a well-formed request whose claimed output is the model’s result on it (Definition 6.12), VZ rejects the Lean prover’s proof with probability at most_ 2<sup>_−_40</sup> _, over the hash function and the committer’s and the prover’s coins._ 

**Zero knowledge** (informal; Theorem 6.35 and Theorem 6.37, machine-checked). _Let the files of Z be in use. Against any verifier with at most_ 2<sup>64</sup> _hash queries, none of them before the commitment, and up to 688,127 answered requests (replies) on one copy of the committed table: if every request is served, one simulator, fixed before the weights, that gets each request and the largest number of replies, and beside these nothing of the weights, makes a view within_ 2<sup>_−_154</sup> _of the real one; it may depend on the files of Z, and whether a request is served is not hidden. At all requests the same holds of a simulator that gets besides one bit for each request: served or not._ 

Both are in the model of the main theorem, with the files of _Z_ and the weights fixed before the hash function is sampled, and are stated for one request per proof, one commitment, one copy of the committed table and fresh random masks (Remark 6.42). The bit says whether the claimed output is the model’s result (Remark 6.44). The property is _zero knowledge with request leakage_ (Definition 6.34); neither it nor the form before compilation is stated to be index privacy (Remark 6.40). On served requests completeness and zero knowledge are stated through the texts of two definitions shaped with a person, each with one condition more, that the request can be served (Definition 3.11); with the condition that the output is the model’s, these are the served requests; the forms at every request are the Lean library’s own (Remark 6.47 and Limitation G.7). 

**Further machine-checked statements, and the check.** Table 1 gives the properties before compilation and on the commitment; Section 5.5 and Section 6 have results on other objects, each with its limits. 

3 

Table 1: What is proved, of which object: statement and bound, simplified. Every entry that does not say otherwise is _machine-checked_ : Lean’s kernel accepts a proof of the Lean statement that Appendix F names; where a statement is an instance of named machine-checked theorems by a substitution that no single Lean theorem states, its remark says so (Remark 6.32). Rows with hash queries are statements in the random-oracle model. A <u>property</u> that is not listed for an object is not stated of it. 

|_Object_; property|Statement and bound|
|---|---|
|_VZ, committer and prover compiled from _<br>index-binding knowledge soundness|_the Lean provers (6.15)_<br>5.1, about _q/_2<sup>124 </sup>+ 2<sup>_−_101</sup>; stated of Π_Z_ (4.15), which has the same verifier (6.16)|
|_files in use (6.11), one request per proof, _<br>soundness with extractability of the<br>commitment<br>correctness of extraction<br>completeness<br>zero knowledge with constant request<br>leakage, on served requests|_one commitment, fresh random masks_<br>6.31, about _q/_2<sup>124 </sup>+ 2<sup>_−_100 </sup>(_q_: the hash queries of whoever made the commitment<br>and of the prover or the opener, together); no two openings, twice that<br>6.31, about 2<sup>_−_155</sup>, at the honest committer’s budget<br>6.29, on served requests, _≤_2<sup>_−_40</sup>, on one copy of the committed table<br>6.35, _≤_2<sup>_−_154</sup>, at the counts of the next row; whether a request is served is not<br>hidden|
|zero knowledge with request leakage<br>(6.34), at every request, against any<br>verifier|6.37, _<_2<sup>_−_154</sup>; one copy of the committed table, 688,127 replies, 2<sup>64 </sup>hash queries,<br>none before the commitment; one bit per request is revealed|
|FlashProve_Z with the Lean provers, before _<br>_with_ Chk(_Z_) = 1 _(4.14)_|_compilation (Section 6.5); one request to a run; an admissible configuration (3.8)_|
|round-by-round soundness|6.20, 6.21: each round _≤_2<sup>_−_128</sup>, proof-of-work bits included; one whole run,<br>deterministic prover, _<_2<sup>_−_79 </sup>over the challenges that pass proof of work|
|completeness|6.23, on served requests, _≤_2<sup>_−_41 </sup>over the prover’s coins and the challenges that<br>pass proof of work|
|honest-verifier zero knowledge|6.27, _≤_2<sup>_−_135 </sup>over the life of one commitment; one bit per request is not hidden|



- _Batches and policies._ A proof made in a batch, which _VZ_ verifies singly, is covered at _ε_ ( _q_ ), and _N_ proofs sent at once at _Nε_ ( _q_ ) (Corollary 5.10). At every admissible _Z_ fixed before the oracle is sampled, with the error _ε_ ( _q_ ), an accepted output meets the policy of _Z_ under _M_<sup>_∗_</sup> (Theorem 5.11 and Remark 5.12). For a clipped-gap policy (Definition 3.4), under which a stock engine’s replies would be checked, this says only that an accepted output’s token costs meet the policy’s limits at the extracted weights; it does not prove that such an output is the model’s own. 

- _A verifier that runs._ An _executable form_ , a Lean term that can be evaluated, equals _VZ_ at every hash function with 32-byte values (Theorem 5.5), so the bound _ε_ ( _q_ ) holds for its own decision when its hash function is a random oracle (Corollary B.11). It runs as the command `lean-verify` ; its file parser, and Lean’s evaluator, compiler and runtime that execute it, are trusted, not proved (Remark 5.9 and Remark B.10). 

- _The main theorem’s proof._ Steps 4 to 9 of _VZ_ are the output of the Lean library’s compiler at three interactive oracle proofs (Remark B.7, Which compiler); the arithmetization of the whole integer model, the lemma on that compiler (Lemma A.15) and the coding theory below the Johnson radius (Lemma 5.13 and Lemma A.23; the first through files adapted from ArkLib and Mathlib, Appendix E.4) all lie inside the proof. As printed, the theorem joins two Lean theorems (Section 5, head). 

- _The check._ Lean’s kernel accepts the proofs, with the provisos of Section 1.4 and Remark B.17, and Lean prints no axiom beyond its three standard ones for the theorems that `lean/CHECK-RESULTS.md` names. An independent proof checker has accepted the soundness theorem on `verify` , `sound_of_verify_at` (the form of Theorem 5.1 that leaves the extractor open: Section 5, head), with every declaration beneath it, and nothing that does not lie beneath it, in a pass on that theorem’s export alone: the pass was made over a copy of the compiled files of the released Lean files’ build (that the compiled files it read were that build’s is known from a report alone) (Remark B.17, Checkers). 

**The programs.** The _released programs_ are `cp_commit` , `cp_prove` and `cp_verify` (the committer, the prover and the C++ verifier) as the release’s sources build them; the _timed runs_ are among the runs that are told as runs on them: the programs that made these runs were built from sources that differ from the released ones in comments, in the names of identifiers and files, in quoted texts that hold such a name and in the 

4 

wording of messages, and in nothing else (Appendix H.1, “Which programs ran”). A dagger (<sup>_†_</sup> ) marks a result not on the released programs, of any other programs, which is not shown to hold for the released programs (Section 7.1). Most of these other programs are built from _other sources_ : source text of this project’s programs that differs from the released text in more than the ways that Appendix H.1, “Which programs ran”, lists. In the timed runs (Appendix H.1) the C++ verifier (Section 7.1) accepted every proof (Section 7.2). In no reported test did the C++ verifier program accept an output that violates the run’s own policy (Table 7 and Table 4); most of these tests ran not on the released programs (Section 7.1). 

- _Honest proofs._ The executable form accepted (by `lean-verify` ; four by a separate build of the form alone) every honest proof of Section 7.3, at the size of _Z_ 0, on synthetic weights: 41 of the timed runs’ own as reported, at exact match; and, by other provers<sup>_†_</sup> , 363 and 18 at exact match and, where an acceptance says less, 1 and 4 under a clipped-gap policy: the first of each pair by one prover built from other sources, the second by other provers than that one (Section 7.3). One proof made at the default configuration’s size, on synthetic weights, by the programs that the release’s sources build, as reported, was accepted by `cp_verify` and, through `lean-verify` , by the executable form (Appendix H.1). 

- _False claims._ The released verifier rejected all 57 proof files that malicious provers made at the default configuration’s size, at exact match and under one commitment, on synthetic weights, and that were put to it, 24 of them claiming a false output (Section 7.6); in the timed runs, on the public weights, the programs rejected 240 of 240 altered copies of honest files and 9 of 9 altered calls. `lean-verify` and a separate build of the executable form together rejected all 26 proofs<sup>_†_</sup> of claims false by the run’s own policy, all from one test program, each rejected where the proof file is read (Section 7.3). 

- _Weight hiding_ , the programs’ goal (Definition H.1). We judge that weight hiding likely holds at the default configuration, in honest runs, if the provider keeps its operating conditions and the judgement’s own conditions (listed in `docs/limits.md` , not in this paper) hold. That judgement rests on the design, what is proved in Lean for the protocol, code review and tests; it is not a proof, and it does not cover a real model’s weights (Section 7.5). 

**Measurements.** The figures are of those timed runs (Appendix H.1) that used a third party’s public, mostly 4-bit weights: on one H100, at the default configuration as fitted to the weights (Section 3.3), at exact match, on windows of 1,024 tokens, with masking on (read off the runs’ own summaries, not tested: Appendix H.2); each proof checks one output token. Rates count prefill tokens; _±_ separates the mean and the sample standard deviation (Section 7.2). 

- In batches of 24 requests the steady rate is 111.23 ± 0.92 tokens/second (n = 3), from 110.20 to 111.99; counted from the program’s start, over five batches, it is 103.60 ± 0.84. The _±_ 0.92 is the standard deviation of three runs made side by side: three runs of the same kind on other programs, on another day on other machines, had a standard deviation of 3.9 (Limitation G.8). With one request in a batch the steady rate is 17.41 ± 0.15 tokens/second (n = 3). These rates are counted up to the moment at which the prover hands out its proofs; it runs no verifier, and a serving script’s check comes on top (Section 7.2). 

- A proof file has 18.74 to 18.81 MB; a client also downloads, once, the key file and the one copy’s file, 6.24 MB together; the C++ verifier checks a proof in 21.14 ± 0.03 seconds called with its option for one thread, on the prover’s machine (n = 3). 

- Per GPU, proving alone takes 93.0 times (92.2 to 94.1 over all pairs of runs) the GPU time of stock vLLM for prefill at the highest load tried ( _n_ = 3 prover runs, 8 engine runs; calculated; FP8 checkpoint, eight H100s). The multiple omits committing and serving and could be higher (Section 7.8). 

For its own integer model of Llama 4 Maverick, on other hardware and for a different statement, VerInf’s README [VerInf] reports 14.3 h for one proof on 1,000 tokens and a proof of 93.6 GB (Table 8); each proof of the timed runs on the public weights checks one output token. 

**Credit.** The definition of the primitive, the plan of the protocol (an offline phase, a phase for each batch, and link proofs between committed tables for the same weights), its zero-knowledge proximity tests [CFW26], and the step beyond unique decoding with out-of-domain samples, including the sample taken when a table is 

5 

committed, are due to Alessandro Chiesa, Giacomo Fenzi and Guy Weissenberg (personal communication). Table 11 lists further parts that draw on their work. 

### **1.4 Limits and open problems** 

The release lists the gaps that we know of, and no attack on the protocol is known to us at any gap while its conditions are kept, the opening budget among them (Limitation 8.4, Opening budget). Almost no attack was sought: that none is described is not a finding of safety. 

At least four things stand between the theorems and the measured runs. Exact match requires that the integer model itself be served: of the 96 honest replies scored from each of two stock engines none would pass it<sup>_†_</sup> (Section 7.4). The runs prove batches, with masks derived from a seed, which completeness and zero knowledge do not cover (Remark 6.42 and Remark G.15). Of the timed runs’ proofs only 41, all from the runs on synthetic weights, have been put to the Lean verifier, over the released Lean files; it accepted each (Section 7.3). That the programs run Lean’s protocol is tested, not proved, and differences are known (Limitation 8.2). 

**What the theorems do not say.** 

- _The programs._ The theorems concern Lean terms: _VZ_ , its extractor, and the Lean provers, which are definitions, not programs that run (Remark 6.43); the paper does not print the provers, and no statement compares two of the three provers with the honest provers that it describes (Definition 6.14 and Remark 6.16). Nothing is proved about the released programs, and no security level is claimed for them (Section 8.1). 

- _The model._ That a second hash function is fixed, and the oracle is taken as independent of it, is an assumption that is neither tested nor proved (Assumption 4.11 and Remark 5.3), nothing is claimed against a quantum adversary (Limitation G.6), and Fiat–Shamir attacks on protocols of this kind have not been assessed (Limitation G.5). 

- _The configuration._ Exact match is a hypothesis that _VZ_ does not test, and the provider’s scale file, with five numbers of the parameter file (Definition 3.7), fixes which integer model is meant, and the scale file can disable any chosen blocks of it: the client must check these, and that _y_ is as long as expected, since any prefix of the model’s output is a greedy output (Remark 5.3, Limitation 8.6, and Limitation G.4). The provider, not the verifier, chooses the configuration, and a commitment does not say whose weights it holds (Limitation 8.5 and Limitation 8.7). A choice of the configuration after querying the oracle, which is open to every provider, is not covered (Remark 5.3 and Limitation 8.5). For proofs sent one after another no bound is stated (Remark B.13). 

- _Trained weights._ No theorem says that a run on a trained model’s weights meets its hypotheses, and no proof for such weights has been put to the executable form. The measured runs, on the public weights, are at a configuration fitted to those weights (Section 3.3); the parameter checks were not evaluated at it (Remark D.5). Nor are such weights shown to meet the model’s range conditions, which limits what can be proved of such a model, not soundness (Remark 3.3 and Remark D.5). 

- _The integer model._ How far the integer model and the floating-point model differ is stated by no theorem: on the public weights its greedy token differs from that of the 16-bit weights, evaluated in float32, at 6.81 % of token positions on 216 texts and at 9.41 % on the 24 natural passages among them (a Python implementation<sup>_†_</sup> , no proofs), and its task quality has not been measured (Section 7.7 and Limitation G.9). All texts are built on WikiText-2 [MXBS17], on which the scales were fitted. 

- _Completeness._ It covers served requests alone, not batches, and that some request is served is shown at constructed weights. Where the model has no result, or another result than the claimed output, nothing is stated: the Lean prover returns the empty proof, which proves nothing and which a prover can also return at a request that it could serve (Remark 6.30; Limitation 8.9). A gap in completeness costs a rejection, with the provisos of Limitation 8.9. 

- _Zero knowledge._ Batches, a further commitment, after compilation a second copy of the committed table, clipped-gap policies, masks derived from a seed, a verifier’s hash queries before the commitment and 

6 

the C++ programs are not covered, and one bit per request, which depends on the weights, is revealed (Section 6.5). Over many requests the bits tell what the model answers and where it has no result, and no statement bounds what many bits reveal together (Remark 6.44). The simulators’ running time is not modelled, and nothing is said of timing, or of what the files of _Z_ tell of the weights (Remark 6.45), or of what the requests tell: each served request holds the model’s own output (Remark 6.36); the released tools compute the scale file and lines of the parameter file from the weights, and no statement bounds what they reveal (Remark 6.1). 

- _Clipped-gap policies._ A clipped-gap policy bounds a sum and a count, not each token (Limitation 8.8). As in every commit-and-prove argument, the committer chooses the weights. By itself, Theorem 5.11 therefore excludes nothing against a committer who commits to weights under which every output meets the policy (Remark 5.12). Some policies of admissible configurations exclude no token (Proposition B.1). _VZ_ does not reject a configuration with such a policy, and the C++ verifier rejects some such policies but not every one: no verifier sees the model’s logits (Remark B.2). 

- _Paper and Lean._ This paper gives _VZ_ in outline, and that its words match the Lean terms is checked by AI reading, not by Lean, except for the policy’s three numbers (Remark D.10 and Remark 5.3). 

- _How the proofs are checked._ Over a build of the released Lean files the proofs of completeness, zero knowledge and the theorem on the commitment have been checked by Lean’s kernel alone (Remark 6.48); a whole run of the library’s full check over all but one of the released Lean files (Remark B.17) passed at every step; its last line says that no theorem got a verdict on binders or on `Classical.choice` (the whole line, and what the check does not cover: `lean/CHECK-RESULTS.md` ; Remark B.17). The pinned version of Lean, 4.29.1, is older than the published fixes (from version 4.32.1 on) of soundness bugs in Lean’s kernel and runtime and in the big-number library (GMP) that its Linux builds link (Remark B.17). A re-checker, `leanchecker` , which is not the independent checker, stopped on 23 modules of a separate Lean library (counted over Lean files that differ from the released ones), which Lean’s kernel accepts at a thread stack larger than the default; by import lines at least one lies beneath each theorem of Section 6.5 on completeness, zero knowledge and the commitment and beneath Proposition 6.17, part 1, none beneath the soundness theorems of Section 5 (Remark 5.7; the finer finding, by a walk over the constants: Remark 6.48). 

#### **What is tested, not proved.** 

- _That the C++ programs run Lean’s protocol_ (Limitation 8.2). In tests the two verifiers, from whichever programs (Table 4 says which), returned the same decision on every proof file that both judged. Of four matched alterations of one honest proof<sup>_†_</sup> , each of which left its claim true, they agreed on one and disagreed on three (these four are not common files: the C++ verifier read the altered raw proof, the form the parsed file altered at the matching place), where the C++ verifier was the stricter. _VZ_ has sixteen tests that the C++ verifier, which clients run by default, lacks ( `lean/GAPS.md` , gap 25), and the one comparison of hash calls matched 881,280 of 1,166,983, a gap that the release rates as heavy. 

- _That VZ accepts the programs’ proofs._ None of the 363, 18, 1 and 4 is shown to be the released programs’ own, and the 363 are of little variety; the 41 are the timed runs’ own, all on synthetic weights. All but those put to it in the reference run and the 41 of the timed runs (Appendix H.3) were judged over Lean files that differ from the released ones or are not shown to be these ( `lean/GAPS.md` , gap 26). 

- _That the C++ verifier is sound._ The tests are few. Test programs that bypass the verifier program and call the library beneath it directly, against the library’s stated contract, did have false claims accepted, by at least two ways; that the verifier program itself never calls it so rests on AI readings of source text that differs in places from the released one, not on a test (Section 7.6). None of these ways went through the verifier program, so a client should check proofs with that program, not through the library. 

- _That the programs hide the weights._ Of the judgement’s five tests two have run on the released programs. The tests ran at the reduced configuration (Section 3.3), on synthetic weights; only one of the five has also run on a real model’s weights, once on the released programs and three times on other programs<sup>_†_</sup> . None searches for a leak: they test that the masks are there and that a loss is noticed (Table 6). 

7 

Of the injected losses<sup>_†_</sup> (Table 6), 8 of 8 (none a loss of a single pad) stopped the committer, 130 of 130 in GPU memory stopped, and 263 of 263 riskier losses stopped. Losses were unnoticed in three places: 30 of 165 injected losses of random values in CPU memory; 9 of 9 at samples that no self-check covers; and, with one self-check disabled, at a parameter file that counted no proof of work, a commitment without masks, which passed every other check. Neither the programs nor the judgement has had an independent review. 

**What is not covered.** Openings beyond the budget, or a lost mask, give away exact linear equations, unnoticed (Limitation 8.4, Limitation G.13, and Limitation G.14). Weight hiding rests on operating conditions of the provider, most of which no program can check, and the shipped vLLM integration breaks at least one (Limitation G.16). Section 8.1 and Appendix G state further limits, Section 8.2 the open problems. 

## **2 Preliminaries and definitions** 

### **2.1 The random-oracle model** 

**Definition 2.1 (Random oracle, query bound, trace).** A _hash function_ is a function _H_ : _{_ 0 _,_ 1 _}_<sup>_∗_</sup> _→{_ 0 _,_ 1 _}_<sup>_λ_</sup> , with _λ_ = 256. The _random oracle_ is the hash function with independent uniform values at distinct inputs. 

An _oracle algorithm_ is a finite rooted tree whose leaves are labelled by outputs. An inner node is a _query node_ , labelled by a string bs _∈{_ 0 _,_ 1 _}_<sup>_∗_</sup> , with one child for each element of _{_ 0 _,_ 1 _}_<sup>_λ_</sup> , or a _sampling node_ . A run under _H_ descends from the root, at a query node to the child _H_ (bs) and at a sampling node to a uniformly random child, and outputs the leaf’s label. Its _query–answer trace_ tr lists the pairs (bs _, H_ (bs)) of its query nodes, in order. 

A tree is _deterministic_ if it has no sampling node, and a _q-query algorithm_ if no path has more than _q_ query nodes. An algorithm with inputs is a family of trees, and has either property if every tree has. 

### **2.2 Codes and mutual correlated agreement** 

**Definition 2.2 (Mutual correlated agreement).** Let K be a finite field, _D_ a finite set of _positions_ , _C ⊆_ K<sup>_D_</sup> a linear code, _k ≥_ 2 an integer, and _G_ a distribution on K<sup>_k_</sup> , the _generator_ . A word _g ∈_ K<sup>_D_</sup> _agrees with C on_ a set _S ⊆ D_ if _g_ coincides on _S_ with some codeword. For _δ ∈_ [0 _,_ 1), the _mutual correlated agreement (MCA) error ε_ mca( _C, G, δ_ ) is the maximum, over all words _g_ 1 _, . . . , gk ∈_ K<sup>_D_</sup> , of 



The code _C_ has _mutual correlated agreement_ at the proximity parameter _δ_ with error _ε_ <u>,</u> under _G_ , if _ε_ mca( _C, G, δ_ ) _≤ ε_ . The _Johnson radius_ of a code of relative minimum distance ∆is 1 _− √_ 1 _−_ ∆. 

Mutual correlated agreement was introduced in [ACFY24b, Definition 4.9], for proximity generators; it strengthens the correlated agreement of [BCIKS20, Section 1.3]. Definition 2.2 follows [CFW26, version of February 25, 2026, Definition 3.14], which defines the MCA error as a maximum over the words. In [BCGM25, Definitions 3.10 and 3.14] a generator is a function of a random seed; here it is the distribution of the coefficients. The Johnson radius is as in [ABF26, Section 1]. 

### **2.3 Interactive oracle proofs, compilation, proof of work** 

**Definition 2.3 (Interactive oracle proof, batch).** An _interactive oracle proof (IOP) with external oracle strings_ has an input, the _IOP instance_ , and a fixed number of rounds. Its _external oracle strings_ are arbitrary, fixed before the first round, and not sent. In each round the prover sends an _oracle part_ (a string) and a _plain part_ ; the verifier answers with a uniform _challenge_ from a finite set. Finally the verifier accepts or rejects; it reads the IOP instance, the plain parts, the challenges, and the positions of the oracle strings that these determine. 

A _batch IOP_ for at most _R_ IOP instances marks each round as _shared_ or _own_ : a shared round has one challenge for all of them, an own round one for each. Each prover message has a shared component and one 

8 

own component per instance. The verifier decides on each from its _local transcript_ : the shared components, its own components, and its challenges. 

Interactive oracle proofs were introduced in [BCS16, Section 4], which states that [RRR16] independently introduces the same notion, as probabilistically checkable interactive proofs. 

**Definition 2.4 (Compilation (BCS transformation)).** The _compilation_ Compile( _Q, H_ ) of an IOP _Q_ (Definition 2.3) under a hash function _H_ is the following non-interactive argument; its queries to _H_ are of three kinds (leaf, inner node, Fiat–Shamir), coded injectively with disjoint ranges. 

1. Each oracle string is committed by the root of a Merkle tree whose leaves hash a symbol and a random salt. The external roots are an input of the verifier. 

2. The first Fiat–Shamir query holds the external roots, a salt and the IOP instance. The query of round _m_ holds a chaining value from the preceding answer and the round’s roots and plain part; its answer gives the round’s challenge and the next chaining value. 

3. The proof string consists of that salt, the other roots, the plain parts and, for each position read, symbol, salt and authentication path. 

4. The verifier recomputes the challenges, checks the paths, and decides as the verifier of _Q_ does. 

**Definition 2.5 (Proof of work (grinding)).** Round _m_ of an IOP (Definition 2.3) has _κm_ bits of _proof of work_ if its plain part contains a _nonce_ , a component that the verifier does not read, and the verifier rejects unless _κm_ designated bits of its challenge are 0. The compiled prover (Definition 2.4) tries values of the nonce, up to a fixed number, until these bits are 0. 

### **2.4 Commit-and-prove arguments and their properties** 

**Definition 2.6 (Indexed relation, commit-and-prove argument).** A _token_ is a natural number. An _instance_ is a pair ( _x, y_ ) of finite lists of tokens, the _prompt_ and the _output_ . An _indexed relation_ consists of a set _I_ of _indices_ , a set Rel of triples ( _M, x, y_ ) with _M ∈I_ (there is no witness), and a function pp on _I_ , the _public parameters of the index_ . The _language_ of _M_ is Lang( _M_ ) = _{_ ( _x, y_ ) : ( _M, x, y_ ) _∈_ Rel _}_ , and Lang( _⊥_ ) = _∅_ . 

A _commit-and-prove argument_ for this relation is a tuple Π = (Rnd _,_ Commit _, P, V_ ) of oracle algorithms (Definition 2.1): 

1. Rnd<sup>_H_</sup> outputs commitment randomness r; 

2. Commit<sup>_H_</sup> ( _M,_ r) is deterministic and outputs a _commitment c ∈{_ 0 _,_ 1 _}_<sup>_λ_</sup> ; 

3. _P_<sup>_H_</sup> ( _M,_ r _, x, y_ ) outputs a _proof π_ ; 

4. _V_<sup>_H_</sup> ( _c, x, y, π_ ) is deterministic and outputs 1 (accept) or 0 (reject). 

The _honest committer_ on _M_ samples r _←_ Rnd<sup>_H_</sup> , publishes _c_ = Commit<sup>_H_</sup> ( _M,_ r) and keeps ( _M,_ r) for the _honest prover P_ . In the _general form_ , instances and commitments range over arbitrary sets, and _P_ takes a list of instances. 

Indexed relations are as in [CHMMVW19, Section 3.1], where a triple holds an index, an instance and a witness. 

**Definition 2.7 (Index-binding knowledge soundness).** Let Π be a commit-and-prove argument with verifier _V_ for an indexed relation with index set _I_ and languages Lang( _·_ ) (Definition 2.6), and let _ε_ : N _→_ [0 _, ∞_ ]. An _extractor_ is a function Ext from pairs of a commitment and a query–answer trace to _I ∪{⊥}_ . For oracle algorithms _AC_ , the _adversarial committer_ , and _AP_ , the _adversarial prover_ , the experiment is: 

1. sample the random oracle _H_ ; 

2. run ( _c,_ st) _←A_<sup>_H_</sup> _C_<sup>,withquery–answertracetr</sup><sup>_C_;</sup> 

3. set _M_<sup>_∗_</sup> = Ext( _c,_ tr _C_ ); 

4. run ( _x, y, π_ ) _←A_<sup>_H_</sup> _P_<sup>(st);</sup> 

9 

#### 5. set acc = _V_<sup>_H_</sup> ( _c, x, y, π_ ). 

The argument Π is _index-binding knowledge sound_ for the relation, with error _ε_ , if there is an extractor Ext such that, for all _qC, qP ∈_ N, every _qC_ -query committer _AC_ and every _qP_ -query prover _AP_ , 



_Remark_ 2.8 (Names of the properties). The terms index-binding knowledge soundness, correctness of extraction and index privacy are due to [CFWpc], and Definition 2.6, Definition 2.7, Definition 2.12, Definition 2.13, and Definition 2.14 are adapted from that work; the name and the requirement of Definition 2.9 are taken from it too, and its experiment and error are this paper’s. Here the word _knowledge_ refers to the index: the relation has no witness, and the extractor returns an index or _⊥_ . The names of the roles, index committer, index verification key and index extractor, are from the same work. Table 12 lists uses of the same or like names in the literature, of other objects. From here on an index committer is called a committer, and an index extractor an extractor. The commitment is the 256-bit value that has the role of the index verification key. Such a key may be all that a committer publishes; the commitment is narrower: in the theorem it is these 256 bits alone. 

**Definition 2.9 (Correctness of extraction).** Let Π = (Rnd _,_ Commit _, P, V_ ) be a commit-and-prove argument with index set _I_ (Definition 2.6), Ext an extractor (Definition 2.7), _ε_ ext _∈_ [0 _,_ 1], and _H_ the random oracle (Definition 2.1). Π has _correctness of extraction_ for Ext with error _ε_ ext if, for every _M ∈I_ fixed before _H_ is sampled, 



where tr is the query–answer trace of this run of Rnd and Commit, and the probability is over _H_ and the random choices of Rnd. 

**Definition 2.10 (Extractability of the commitment, no two openings).** Let Π, Ext and _H_ be as in Definition 2.9 and _ε_ op : N _→_ [0 _, ∞_ ]. A pair ( _M,_ r) _opens_ a commitment _c_ if Commit<sup>_H_</sup> ( _M,_ r) = _c_ ; the queries of this test are not counted in _qC_ , _qO_ or _q_ . Probabilities are over _H_ and the algorithms’ random choices. 

1. Π has _extractability of the commitment_ (also called opening-form binding) for Ext with error _ε_ op if, for every _qC_ -query algorithm _AC_ that outputs a commitment _c_ and a state, with trace tr _C_ , and every _qO_ -query algorithm that outputs a pair ( _M,_ r) given this state, 



2. Π has _no two openings_ with error _ε_ op if, for every _q_ -query algorithm that outputs _c_ , ( _M_ 1 _,_ r1) and ( _M_ 2 _,_ r2), 



_Remark_ 2.11 (Standard names). No two openings is the standard binding property of a commitment scheme [CFQ19, Definition 2.1], for (Rnd _,_ Commit). Extractability of the commitment is the property that [BCS16, Section 3.1] calls extractability; correctness of extraction is its form for the honest committer’s own input. 

**Definition 2.12 (Completeness on a set of triples).** Let Π = (Rnd _,_ Commit _, P, V_ ) be a commit-and-prove argument (Definition 2.6) for the indexed relation Rel, _S_ a set of triples ( _M, x, y_ ), and _α ∈_ [0 _,_ 1]. The argument Π is _complete on S_ with error _α_ if, for every ( _M, x, y_ ) _∈ S ∩_ Rel, 



It is _complete_ with error _α_ if it is complete on the set of all triples. 

In what follows SD is the statistical distance. 

**Definition 2.13 (Real and ideal experiment).** Let Π be an argument in the general form (Definition 2.6), with arbitrary indices and one or several independent random oracles _H_ . A _batch_ is a list of instances, each marked _revealed_ or _hidden_ ; _P_ maps it to a proof or _⊥_ for each instance. For an oracle algorithm _V_<sup>_∗_</sup> , a function Env, the _environment_ , and _N ∈_ N, the _real experiment_ RealΠ( _M,_ Env _, V_<sup>_∗_</sup> _, N_ ) is: 

10 

1. sample _H_ ; the honest committer, on _M_ , outputs _c_ and keeps sk = ( _M,_ r) (Definition 2.6); 

2. run _V_<sup>_∗H_</sup> ( _c_ ), which asks for a batch at most _N_ times; 

3. each time, set bt = Env(view); if bt = _⊥_ , stop; 

4. run _P_<sup>_H_</sup> (sk _,_ bt) and give _V_<sup>_∗_</sup> the revealed instances with their proofs; 

5. output view: _c_ , what _V_<sup>_∗_</sup> was given, its query–answer trace and its coins. 

In the _ideal experiment_ IdealSim( _M,_ Env _, V_<sup>_∗_</sup> _, N_ ) a stateful Sim replaces _H_ , committer and prover (so the random oracle is programmable). Besides the queries of _V_<sup>_∗_</sup> , which it answers, it is given only the _leakage_ : pp( _M_ ), _N_ , and each batch’s revealed instances. 

**Definition 2.14 (Index privacy).** Let _S_ assign to each index _M_ of a set _J_ a set _SM_ of instances, let _R ≥_ 1 and _ζ_ : N<sup>2</sup> _→_ [0 _, ∞_ ]. An environment is _valid_ at _M_ if each of its batches has at most _R_ instances, all in _SM_ . An argument Π has _index privacy on S for batches of at most R, with error ζ_ , if a simulator Sim exists such that, for all _M ∈J_ , all environments Env valid at _M_ , all _q, N ∈_ N and every _q_ -query _V_<sup>_∗_</sup> , 



(Definition 2.13), and in the ideal experiment equal queries receive equal answers. 

What Section 6.5 proves of zero knowledge after compilation is stated for another notion, zero knowledge with request leakage (Definition 6.34), and there in two statements: on the served instances with a constant leakage (Theorem 6.35), and on all instances, those that are not served among them (Definition 6.12): at such an instance the prover returns the empty proof, and the simulator is told, for each instance, whether it is served, which depends on the weights (Theorem 6.37, with its bounds; Remark 6.40). Completeness says nothing at such an instance (Remark 6.30). 

## **3 The relation: model, policies, configuration** 

The complete definitions of the integer model, the placement rule, the arithmetization and the parameter checks are terms of the Lean development, which Appendix F names. This paper outlines them (Construction D.3 and Construction D.9, Definition 4.1 and Definition 4.14); Remark D.10 (Given in outline) lists what it does not print. Every admissible configuration is _Z_ 0 with its scale file, five of its model parameters and the three numbers of its policy replaced (Definition 3.7 and Definition 3.8); the layout file and all sizes of Table 2 are those of _Z_ 0. 

### **3.1 The integer model** 

The integer model is the function that one term of the Lean development computes. The release’s `docs/model.md` describes it step by step, and Construction D.3 outlines it, without defining it. 

**Definition 3.1 (Integer model).** The _integer model_ (Construction D.3) is a transformer in integer arithmetic with _window length L_ = 2<sup>10</sup> and _vocabulary V_ = _{_ 0 _, . . . ,_ 202047 _}_ . Its _weights M_ are integer matrices of fixed shapes whose entries lie in fixed intervals that contain 0; their set is _I_ . It rescales by its model parameters _µ_ (Definition 3.6). A _context_ is a list _x_ ¯ of at least one and at most _L_ tokens of _V_ . At valid _µ_ and _M ∈I_ , the model is _defined_ on _x_ ¯ if no intermediate value leaves its prescribed interval at any token position (the _range conditions_ ); it then maps _x_ ¯ to a vector of _logits_ logit _M_ (¯ _x_ ) _∈_ Z<sup>_V_</sup> . 

That Lean’s model function has a value exactly on such contexts is argued by hand; no Lean theorem states it. 

**Definition 3.2 (Greedy token, greedy output).** Let _µ_ be valid model parameters and _M ∈I_ (Definition 3.1). The _greedy token_ after a context _x_ ¯ on which the model is defined is the least token of largest logit, 



An output _y_ = ( _y_ 0 _, y_ 1 _, . . ._ ) is a _greedy output_ of _M_ on a prompt _x_ if, for every _i < |y|_ , the list _x ∥_ ( _y_ 0 _, . . . , yi−_ 1) is a context on which the model is defined, and _yi_ is the greedy token after it. The _relation of greedy outputs_ at 

11 

_µ_ is the indexed relation (Definition 2.6) that consists of these triples ( _M, x, y_ ), with index set _I_ and constant pp. 

_Remark_ 3.3 (Trained weights). The theorems’ weights are arbitrary elements of _I_ , all-zero matrices among them, and the theorems name no public model. No trained model’s weights are shown to meet the range conditions, which limits what can be proved of such a model, not soundness. Remark D.5 (No real weights shown) states what was counted and run. 

### **3.2 Acceptance policies** 

The clipped-gap policies are modelled on Token-DiFR [DiFR], at temperature 0 (Appendix E.2). 

**Definition 3.4 (Logit gap, acceptance policy, exact match, clipped gap).** Let _µ_ , _M_ , logit _M_ and greedy _M_ be as in Definition 3.2. The _logit gap_ of a token v _∈V_ after a context _x_ ¯ on which the model is defined is 



An _acceptance policy_ is a triple pol = ( _C, θ, cap_ ) of natural numbers: _clip_ , _threshold_ , _cap_ . 

Let _x_ and _y_ = ( _y_ 0 _, y_ 1 _, . . ._ ) be non-empty lists of tokens of _V_ with _|x|_ + _|y| ≤ L_ +1. Put _x_ ¯ _i_ = _x ∥_ ( _y_ 0 _, . . . , yi−_ 1) and _n_ ct = min( _|y|, L −|x|_ ); the _counted_ tokens are _y_ 0 _, . . . , yn_ ct _−_ 1. The output _y_ is _eligible_ on _x_ under _M_ if the model is defined on every _x_ ¯ _i_ and, in case _n_ ct _< |y|_ , _yi_ = greedy _M_ ( _x_ ¯ _i_ ) at _i_ = _|y| −_ 1. Put gap _i_ = gap _M_ ( _x_ ¯ _i, yi_ ); the _cost_ of a counted token _yi_ is cost _i_ = gap _i_ + 2<sup>8</sup> . The output _meets_ pol if it is eligible, 



No other pair ( _x, y_ ) meets a policy. _Exact match_ is the policy (2<sup>8</sup> _,_ 2<sup>8</sup> _,_ 0); every other policy is a _clipped-gap policy_ . 

Exact match forces every logit gap to 0, and the token after a full window has no cost. 

**Proposition 3.5 (Exact match and greedy outputs).** [machine-checked] _Let µ be valid model parameters, M ∈I, and x, y lists of tokens. The output y meets exact match on x under M (Definition 3.4) if and only if y is non-empty and a greedy output of M on x (Definition 3.2). In that case y meets every acceptance policy with θ ≥_ 2<sup>8</sup> _and C ≥_ 2<sup>8</sup> _._ 

Each of the two sentences is a single Lean theorem. 

_Outline._ A logit gap is at least 0, and it is 0 exactly at the greedy token. At exact match, _cap_ = 0 forces cost _i ≤_ 2<sup>8</sup> , hence gap _i_ = 0, at every counted token, and an eligible output’s token after a full window is greedy. Conversely, costs of 2<sup>8</sup> satisfy both bounds whenever _θ, C ≥_ 2<sup>8</sup> . 

### **3.3 Configurations** 

**Definition 3.6 (Model parameters).** The integer model has 48 layers, numbered from 0. Its _model parameters µ_ are natural numbers: in each layer eight shifts and the triples (M1 _,_ K1 _, ϵ_ 1) and (M2 _,_ K2 _, ϵ_ 2) of its two normalizations; in each odd layer three further shifts; and the triple (M _F ,_ K _F , ϵF_ ) of the final normalization. They are _valid_ if each lies in its range: 

|Number|Meaning|Range|
|---|---|---|
|_kq_, _kk_|shifts of queries, keys|5_, . . . ,_32; equal, or both at least 16|
|_kv_, _kg_, _ku_|shifts of values, gate, up|1_, . . . ,_24|
|_ko_|shift of attention’s output|1_, . . . ,_19|
|_kd_|shift of down projection|10_, . . . ,_19|
|_ks_|shift of softmax keys|4_, . . . ,_12; equal in the layers 4_j, . . . ,_4_j_+ 3|
|_kr_|shift of router (odd layers)|8_, . . . ,_24|
|_k_<sup>_′_</sup><br>_g_<sup>,</sup> <sup>_k′_</sup><br>_u_|routed expert’s gate, up|1_, . . . ,_24|
|M, K|exponents of a normalization|24_≤_M_≤_47, 17_≤_K_≤_32, K_≤_M_≤_K+ 14|
|_ϵ_|its regularizer|1_≤ϵ <_2<sup>48</sup>|



12 

**Definition 3.7 (Configuration, default configuration).** A _configuration_ is a triple _Z_ of byte strings: the _parameter file_ , the _layout file_ and the _scale file_ (their encoding: Definition D.1). The _default configuration Z_ 0 is one fixed configuration (Definition D.1). Its parameter file holds its policy ( _C, θ, cap_ ), which is exact match (Definition 3.4), and _ϵ_ 1, _ϵ_ 2, _ϵF_ , M _F_ , K _F_ of its model parameters _µ_ 0 (Definition 3.6); its scale file holds the rest of _µ_ 0; its layout file says where each cell (Definition 4.8) lies; a parameter check tests that each lies where a Lookup (Construction D.6) reads it (Definition 4.14, item 3). For model parameters _µ_ with the same _ϵ_ 1 and _ϵ_ 2 in every layer and an acceptance policy pol, _Z_ 0[ _µ,_ pol] denotes _Z_ 0 with these eight numbers and the scale file replaced by those of _µ_ and pol. Put _Z_ 1 = _Z_ 0[ _µ_ 0 _,_ (25856 _,_ 512 _,_ 1)]. 

The release’s pages call the three byte strings of a configuration its public files. 

**Definition 3.8 (Admissible configuration).** A configuration _Z_ is an _admissible configuration_ , written Adm( _Z_ ) = 1, if _Z_ = _Z_ 0[ _µ,_ pol] (Definition 3.7) for valid model parameters _µ_ (Definition 3.6) with one _ϵ_ 1 and one _ϵ_ 2 for all layers and an acceptance policy pol = ( _C, θ, cap_ ) with 



Then _µZ_ = _µ_ are the _model parameters of Z_ , and pol _Z_ = pol is the _policy of Z_ . 

The configuration _Z_ 1 is admissible, and Proposition B.3 describes its policy. No theorem ties the shipped files of _Z_ 1 to the Lean term (Remark D.2); five of the honest proofs of Section 7.3 are at these files. Section 7 uses two further kinds of configuration. 

- A configuration is _fitted_ to weights if its scale file and lines of its parameter file are computed from them (Remark D.5, No real weights shown). 

- A _reduced configuration_ is one for a synthetic model far smaller than the integer model, for instance with 8 layers and a window of 64 tokens; its parameter file states other sizes than that of _Z_ 0, so it is not an admissible configuration. 

### **3.4 Policies that exclude nothing** 

**Proposition 3.9 (All-zero weights).** [machine-checked] _Let µ be valid model parameters (Definition 3.6) and_ pol = ( _C, θ, cap_ ) _an acceptance policy (Definition 3.4) with θ, C ≥_ 257 _. Under the all-zero weights, every non-empty output y meets_ pol _on every non-empty prompt x, provided that all tokens lie in V, that |x|_ + _|y| ≤ L_ + 1 _, and that the last token of y is_ 0 _if |x|_ + _|y|_ = _L_ + 1 _._ 

_Outline._ The model is defined on every context. With an all-zero output matrix all logits are 0, so token 0 is the greedy token, every logit gap is 0 or 1, and every cost is at most 257. 

_Remark_ 3.10 (What a policy bounds). 

1. A clipped-gap policy bounds a sum and a count over the counted tokens of one output, which in a soundness statement the prover chooses; each token is scored after the output’s own earlier tokens. 

2. A logit gap is measured in the unit of the logits of the weights at hand, so what a policy excludes depends on the weights (Proposition 3.9). The parameter file states that unit, as 2<sup>_−_15</sup> nats at every admissible configuration; the model does not read this number, and nothing in the proof ties it to the weights. 

3. The bound on the sum has the form of the test of Token-DiFR [DiFR], whose checker runs a model that it knows, pools many outputs’ tokens and fits the clip to honest runs. Here each output is bounded alone, and the provider writes the clip, so an output that meets the policy has not thereby passed that test. 

### **3.5 Instances that can be served** 

**Definition 3.11 (Instances that can be served).** Let _µ_ be valid model parameters and _M ∈I_ (Definition 3.1). An instance ( _x, y_ ) with _y_ non-empty and all tokens in _V can be served_ at _M_ if the integer model at _µ_ is defined, for _M_ , on the first _L_ = 1 _,_ 024 tokens of _x ∥ y ∥_ 0 0 _· · ·_ . The set Serv _µ_ consists of all ( _M, x, y_ ) such that ( _x, y_ ) can be served at _M_ and _y_ is a greedy output of _M_ on _x_ (Definition 3.2). 

13 

Table 2: Parameters, with their values at _Z_ 0 <u>(Definition</u> 3.7). 

|Symbol|Parameter|Value|
|---|---|---|
|F, _p_|prime field, its order|_p_= 2<sup>64</sup> _−_2<sup>32 </sup>+ 1<br>|
|E|extension field, for _V_’s samples|F[_ϑ_], _ϑ_<sup>3 </sup>= 2|
|_L_; _G_|length of the window; segments of the model’s run<br>(Definition 4.8)|1_,_024; 25|
|_d_; _n_=_n_0+_v_|_A_ has _d_ matrix rows, 2<sup>_n_ </sup>matrix columns; a table 2<sup>_v_ </sup>rows|3_,_076_,_096; 18 = 7 + 11<br>|
|_D_|positions, _D ⊆_F|_|D|_= 2<sup>32</sup>|
|_ℓ_; _ℓ_0 =_ℓ−_3_s_|entries of a row’s message; plain entries<br>fi|402_,_653_,_184; 402_,_653_,_181|
|_t_|random coefficients of a row: opening budget|44_,_040_,_191|
|_s_|OOD answers for each row<br>|1|
|_ρ_|rate (_ℓ_+_t_)_/|D|_|0_._1041|
|_e_; _δ_ =_e/|D|_|decoder’s radius; proximity parameter|2_,_852_,_142_,_980; 0_._6640|
|_ν_off_, ν_first_, ν_copy_, ν_on|maximum number of opened columns (Construction 4.13)<br>i|62_,_62_,_62_,_64|
|_δ_seg|proximity parameter of a segment’s first proximity test<br>(Construction D.9); its positions|0_._2901; 219|
|_κ_req_, κ_bat_, κ_off|bits of proof of work: a request’s rounds; shared rounds;<br>offline and link proofs|20_,_28_,_32|
||Range test (Construction D.6): columns of its matrix Θ;<br>range of an entry of the product, sent as two digits|128; [_−_2<sup>17</sup>_,_2<sup>17</sup>)|
|_R_|largest batch|64|
|_K_|copies: at most, by _ν_off +_Kν_first _≤t_|710_,_324|
|_Nj_|batches on copy _j_: at most, by _ν_copy+_Njν_on _≤t_|688_,_127|



The parameter file states the sizes of the tables and the bits of proof of work; every other value is calculated from the parameter file and taken from no run; the rate _ρ_ is rounded up, proximity parameters down (Remark D.14, How the parameters are obtained). 

#### **Proposition 3.12 (Non-vacuity of the instances that can be served).** [machine-checked] 

- _(a) Valid model parameters exist, and for all valid µ the set_ Serv _µ (Definition 3.11) is not empty._ 

- _(b) There are valid model parameters µ and weights M ∈I at which no instance can be served._ 

_Remark_ 3.13 (Which instances can be served). Serv _µ_ leaves out the empty output, which is always a greedy output; whether each of the other conditions of Definition 3.11 is needed for the honest proof to be accepted is not known. At some valid model parameters and some weights no instance can be served (Proposition 3.12(b)). Instances are proved to lie in Serv _µ_ at constructed weights and, at the model parameters of _Z_ 0, at all valid weights whose layers’ matrices have every entry in _−_ 800 _.._ 800: there an instance can be served exactly if its tokens are in the vocabulary and its output is not empty. No statement says whether a real model’s weights lie within that bound, and no instance is proved to lie in Serv _µ_ at a real model’s weights. 

## **4 The protocol** 

Section 1.2 gives an overview of the protocol. This section gives its form, and Appendix D its rounds; what of _VZ_ the paper does not print is listed in Remark D.10. Sources are named where each idea is introduced; Table 11 lists the parts of the protocol and their sources. 

### **4.1 Parameters** 

Table 2 introduces the symbols for the sizes of the protocol. The parameter file states or determines them; every admissible configuration has the values of _Z_ 0. 

### **4.2 Encoding and commitment** 

The weights are encoded once, with a masked code, a zero-knowledge Reed–Solomon encoding ([CFWpc]; [CY24, Section 32.5]; [CFW26]). In the literature’s terms, a committed table (Definition 4.2) is a codeword of a 

14 

2<sup>_v_</sup> -fold interleaved Reed–Solomon code, with the columns as its symbols. The _masks_ are the pads and random coefficients of the tables and the mask rows of the provers’ oracle messages (Construction D.6); Section 7.1 defines the mode _masking on_ of a run. Any _t_ columns of a table sampled with fresh randomness, together with its OOD answers at an OOD seed chosen before the table is sampled, are distributed independently of the matrix (Theorem 6.3; Remark 6.4); _t_ + 1 columns are not (Proposition 6.5). This is the reason for the opening budget _t_ (Definition 4.2), a limit on the number of columns of one table that all proofs on it open in total (Definition C.6; Remark C.7). 

**Definition 4.1 (Digit matrix).** Each weight of _M_ (Definition 3.1) is written in signed digits to the base 2<sup>8</sup> : four digits for an entry of the embedding matrix, two for every other weight. A public placement rule (Remark D.10, Given in outline), fixed by the parameter file of _Z_ , assigns to each digit of each weight one entry of a matrix with _d matrix rows_ and 2<sup>_n_</sup> _matrix columns_ . An integer matrix _A_ of this shape is _in range_ if every entry lies in [ _−_ 2<sup>7</sup> _,_ 2<sup>7</sup> ) and 



for the entries _b_ 0 _, b_ 1 in each matrix column of each pair of matrix rows that the rule assigns to the low and the high digits of two-digit weights. For _A_ in range, wt( _A_ ) _∈I_ denotes the weights that the rule reads off _A_ . A _digit matrix_ of _M_ is a matrix _A_ in range with wt( _A_ ) = _M_ ; it need not be unique. 

**Definition 4.2 (Masked encoding, committed table, opening budget).** Let _D ⊆_ F with _|D| ≥ ℓ_ + _t_ . The _masked encoding_ of a message _f ∈_ F<sup>_ℓ_</sup> with random coefficients _τ ∈_ F<sup>_t_</sup> is the Reed–Solomon codeword 



Let _A_ be a matrix over F, or an integer matrix, of the shape of Definition 4.1. Let _n_ = _n_ 0 + _v_ , let _s_ be the number of OOD answers for each row (Definition 4.3), and let _d_ 2<sup>_n_0</sup> _≤ ℓ_ 0 = _ℓ −_ 3 _s_ . The _message of row r <_ 2<sup>_v_</sup> is _fr ∈_ F<sup>_ℓ_</sup> with 



with 0 at its other entries below _ℓ_ 0, and with uniform elements of F, the _pads_ , at its last 3 _s_ entries. The entries below _ℓ_ 0 are _plain_ . A _committed table_ for _A_ is **T** = (Enc( _fr_ ; _τr_ )) _r<_ 2 _v_ with independent uniform _τr_ . Its _column_ at the _position ϕ ∈ D_ is **T** [ _ϕ_ ] _∈_ F<sup>2</sup><sup>_v_</sup> . Its _opening budget_ is _t_ . 

**Definition 4.3 (Out-of-domain seed, answers, sample).** Let E = F[ _ϑ_ ] be of degree 3 over F, let eq( _z, j_ ) =<sup>�</sup> _k_ � _zkjk_ +(1 _− zk_ )(1 _− jk_ )�, where _jk_ is bit _k_ of _j_ , the lowest first, and let ( _fr_ ) _r<_ 2 _v_ be row messages of length _ℓ_ , over F or over E (Definition 4.2). An _out-of-domain (OOD) seed_ is a list _ς_ = ( _zi, βi_ ) _i<s_ of points _zi ∈_ E<sup>_⌈_log2</sup><sup>_ℓ⌉_</sup> and factors _βi ∈_ E _\ {_ 0 _}_ . The _OOD answers_ of ( _fr_ ) _r_ at _ς_ are **a** = ( _ar,i_ ) _r<_ 2 _v, i<s_ with 



The OOD answers of a committed table are those of its row messages. The pair _σ_ = ( _ς,_ **a** ) is the table’s _OOD sample_ at _ς_ . 

_Remark_ 4.4 (Standard terms). The masked encoding is the zero-knowledge encoding of [CY24, Section 32.5] and [CFW26]. The opening budget is the _query bound_ of bounded-query zero knowledge [BCRSVW18, Definition 4.4]; this paper keeps the word query for queries to _H_ . In [ACFY24b] and [CFW26] an out-ofdomain sample is the verifier’s point alone; here it is the OOD seed with the committer’s answers. Each answer is a masked sum of the weights. 

**Definition 4.5 (List, collision, decoder).** Let **T**<sup>_∗_</sup> _∈_ (F<sup>2</sup><sup>_v_</sup> )<sup>_D_</sup> be a word and _e_ the _radius_ , a number of positions. A _table of codewords_ is (Enc( _fr_ ; _τr_ )) _r<_ 2 _v_ for any _fr ∈_ E<sup>_ℓ_</sup> and _τr ∈_ E<sup>_t_</sup> . The _list_ Λ( **T**<sup>_∗_</sup> ) consists of the tables of codewords that differ from **T**<sup>_∗_</sup> in at most _e_ columns. The word has a _collision_ at an OOD seed _ς_ , written Coll( **T**<sup>_∗_</sup> _, ς_ ), if two members of Λ( **T**<sup>_∗_</sup> ) with different messages ( _fr_ ) _r_ have the same OOD answers at _ς_ (Definition 4.3). The _decoder_ maps **T**<sup>_∗_</sup> and _σ_ = ( _ς,_ **a** ) to a matrix or to _⊥_ . 

1. If Coll( **T**<sup>_∗_</sup> _, ς_ ), then Dec( **T**<sup>_∗_</sup> _, σ_ ) = _⊥_ . 

15 

2. Otherwise let **T**<sup>_′_</sup> be nearest to **T**<sup>_∗_</sup> among all tables of codewords with OOD answers **a** at _ς_ , ties broken by a fixed order. If **T**<sup>_′_</sup> _∈/_ Λ( **T**<sup>_∗_</sup> ), the value is _⊥_ . 

3. Otherwise the value is the matrix _A_ in range (Definition 4.1) whose residues modulo _p_ the messages of **T**<sup>_′_</sup> hold at the places of Definition 4.2, or _⊥_ if there is none. 

### **4.3 Copies and the link proof** 

Every proof opens columns, hence the provider makes copies of the table with fresh masks, each with a link proof ([CFWpc]; [CFW26]). 

Each committed table has an OOD sample ([CFWpc]; the technique: [BGKS20], as in [ACFY24b, Section 4.4]). The mask of an answer, a pad times a sampled nonzero factor (Definition 4.3), follows the private zero-evaders of [CFW26, version of February 25, 2026, Section 9.2]; that only list members with different messages count as a collision (Definition 4.5) follows its Section 9.1. 

The OOD samples allow a radius _e_ beyond half the code’s minimum distance, where the list of a word can have several members; the sample selects one, up to a small error (Lemma A.25). Definition 4.5 takes list members over E; those of a word over F are over F, since _e_ is below the minimum distance (proof of Lemma A.25). A word at relative distance _δ_ passes _ν_ sampled positions with probability (1 _− δ_ )<sup>_ν_</sup> , hence a larger _δ_ requires fewer opened columns, which gives shorter proofs and allows more proofs within one opening budget. **Definition 4.6 (First table, copy, link proof).** Let _A_ be in range. The _first table_ **T** 0 and the _copies_ **T** 1 _, . . . ,_ **T** _K_ are committed tables for _A_ (Definition 4.2) whose pads and random coefficients are mutually independent; _σj_ is an OOD sample of **T** _j_ . The _relation of the link proof_ is the set of all ( **T** 0<sup>_∗, σ_0</sup><sup>_,_</sup><sup>**T**</sup><sup>_∗_</sup> _j_<sup>_, σj_),two</sup> words with a sample each, such that 



with Dec and Coll as in Definition 4.5. A _link proof_ for the copy **T** _j_ is a proof string (Construction D.11) for IOP<sup>link</sup> _Z_ (Construction 4.13), whose IOP instance is ( _σ_ 0 _, σj_ ) and whose external oracle strings are **T** 0 and **T** _j_ . 

### **4.4 Requests and proofs** 

**Definition 4.7 (Encoded request, offline and online proof, files).** An _encoded request_ is _ξ_ = (win _, i_ 0 _, i_ 1 _, y_<sup>+</sup> ): a window win _∈V_<sup>_L_</sup> (Definition 3.1), token positions _i_ 0 _< i_ 1 _≤ L_ , and a token _y_<sup>+</sup> _∈V_ with _y_<sup>+</sup> = 0 if _i_ 1 _< L_ . For lists _x_ , _y_ , fill of tokens of _V_ , with _x_ and _y_ non-empty and _|x|_ + _|y| ≤ L_ + 1, build( _x, y,_ fill) is the encoded request with 



where _y_<sup>+</sup> is the last token of _y_ if _|x|_ + _|y|_ = _L_ + 1, and 0 otherwise; on all other inputs build is undefined. An _offline proof π_ key is a proof string (Construction D.11) for IOP<sup>key</sup> _Z_<sup>,an</sup><sup>_onlineproofπ_reqoneforIOPreq</sup> _Z_<sup>.With</sup> rt _j_ the Merkle root of table _j_ (Construction D.11) and _π_ link the link proof of copy _j_ (Definition 4.6), the _key file_ holds (rt0 _,_ **a** 0 _, π_ key), the _file of copy j_ holds (rt _j,_ **a** _j, π_ link), and a _proof file_ holds ( _j, π_ req). A proof _π_ of Π _Z_ (Construction 4.15) consists of fill, the contents of these three files and the two OOD seeds. 

In running text a _request_ is an instance ( _x, y_ ) that is put to the prover; build turns each into one encoded request. The request relation (Definition 4.8) and true requests (Definition 4.17) concern encoded requests. 

**Definition 4.8 (Segments, cells, request relation).** The arithmetization (Remark D.10, Given in outline) of the integer model at _µZ_ (Definition 3.1) with the policy of _Z_ (Definition 3.4) divides the model’s run on a window into _G segments_ , the last of which is the _head_ . Segment _g_ has 

- _cells Yg_ , integer matrices of intermediate values; 

- _identities_ , polynomials with integer coefficients in the entries of _A_ , the cells _Y_ 0 _, . . . , Yg_ and the components of _ξ_ ; 

16 

• _lookups_ , each a polynomial of the same kind, its _lookup key_ , with a public table. 

The _request relation_ Req _Z_ is the set of pairs ( _A, ξ_ ), with _A_ in range and _ξ_ an encoded request, for which there are cells of absolute value below the _relaxed bound_ 2<sup>18</sup> such that, modulo _p_ , every identity is 0 and every lookup key is in its table. 

**Lemma 4.9 (Meaning of the request relation).** [machine-checked] _Let µ be valid model parameters (Definition 3.6) and_ ( _C, θ, cap_ ) _a policy with C, θ <_ 2<sup>40</sup> _and cap <_ 2<sup>16</sup> _, and form the request relation of Definition 4.8 at these. Let A be in range, M_ = wt( _A_ ) _, and ξ_ = (win _, i_ 0 _, i_ 1 _, y_<sup>+</sup> ) _an encoded request. Put x_ ¯ _i_ = (win[0] _, . . . ,_ win[ _i_ ]) _and I_ = _{ i_ : _i_ 0 _≤ i <_ min( _i_ 1 _, L −_ 1) _}. Then_ ( _A, ξ_ ) _is in the relation if and only if_ 

_1. the model is defined on_ win _(Definition 3.1);_ 

_2. with_ gap _i_ = gap _M_ (¯ _xi,_ win[ _i_ + 1]) _(Definition 3.4),_ 



_3. if i_ 1 = _L, then_ gap _M_ (¯ _xL−_ 1 _, y_<sup>+</sup> ) = 0 _._ 

One Lean theorem proves the first direction for all cells within the relaxed bound, and for the second that the honest cells satisfy the constraints within the honest bound; a second Lean theorem passes from the honest bound to the relaxed one; the two are joined on paper, and items 1 to 3 put a Lean definition in words (Appendix F names all three). 

Beyond that join, no proof of the lemma is printed. Lemma A.33 is the form of its direction from the relation to conditions 1 to 3 that the proof of Theorem 5.1 uses: at an admissible configuration and at _ξ_ = build( _x, y,_ fill), membership in Req _Z_ gives that _y_ meets pol _Z_ on _x_ . 

### **4.5 The interactive protocol and the argument** 

A batch opens its copy once for all its requests, as in [GLSTW23] and [BFRW25] (see also [CFW26]); the requests of a batch are the leaves of a Merkle tree, the _batch tree_ (Construction D.11). The proximity tests are the zero-knowledge tests of [CFW26], with changes, of which this paper describes one (Table 11). A claim on the table (Construction 4.13, item 2), one linear claim with a linear form in committed masks, has the form of its committed sumcheck relation [CFW26, version of February 25, 2026, Section 2.4 and Definition 5.8]; the statements that are reduced to such a claim are this protocol’s, and the sources of the reductions are in Table 11. The rounds that sample positions have proof of work ([CFWpc], with published uses in [Sta21, Section 3.11.3], [ACFY24a, Section 6.2] and [ACFY24b, Section 6.2]). Compilation follows the BCS transformation [BCS16]; the committed tables are compiled as external oracle strings, as in [COS20, Section 10] and [CY24, Section 32.7], following [CFWpc]. 

**Assumption 4.10 (Random oracle).** The hash function _H_ that the algorithms of Π _Z_ (Construction 4.15) and all adversaries query is a random oracle (Definition 2.1) with outputs of _λ_ = 256 bits. It models the _format’s hash function O_<sup>_∗_</sup> , the function from byte strings to 32-byte strings, built on BLAKE2b, by which the release instantiates _H_ . Every probability in a statement that cites this assumption is taken over the choice of _H_ and over the random choices of the algorithms. 

That the Lean definition of BLAKE2b, beneath the Lean term for _O_<sup>_∗_</sup> , is BLAKE2b [RFC7693] rests on known answers alone. 

**Assumption 4.11 (Separate second hash).** The second hash function _H_<sup>_′_</sup> that the algorithms of Π _Z_ (Construction 4.15) evaluate is one particular function, the second hash function of the release’s format. It is not sampled, and its values do not depend on _H_ . It is assumed that _O_<sup>_∗_</sup> (Assumption 4.10) can be modelled by an oracle sampled independently of _H_<sup>_′_</sup> . 

_Remark_ 4.12 (On the second hash function). Like _O_<sup>_∗_</sup> , _H_<sup>_′_</sup> is built on the compression function of BLAKE2b. The statements that cite Assumption 4.11 use no further property of _H_<sup>_′_</sup> . They are proved for this one function, on which Chk( _Z_ ) depends, and are not stated for another function in its place. The programs, which check the 

17 

lookup tables against their four digests under _H_<sup>_′_</sup> in the parameter file, also assume that _H_<sup>_′_</sup> resists collisions. Remark D.13 (How the two hash functions differ) says how the two hash functions differ, and why a collision of _H_<sup>_′_</sup> merges no two of _VZ_ ’s queries. 

**Construction 4.13 (The interactive protocol** FlashProve _Z_ **).** FlashProve _Z_ consists of the encoder, which maps a matrix _A_ , pads and random coefficients to a committed table (Definition 4.2); the map from a table and an OOD seed to its OOD answers (Definition 4.3); Coll and Dec (Definition 4.5); the largest size _R_ of a batch; and three IOPs with external oracle strings (Definition 2.3). After a table is sent, _V_ samples its OOD seed and _P_ answers. The OOD sample belongs to no round of the three IOPs; it is part of their IOP instances. 

|IOP|IOP instance|Oracle strings|Set _L_|
|---|---|---|---|
|IOP<sup>key</sup><br>_Z_<br>|_σ_0|**T**<sup>_∗_</sup><br>0|Coll(**T**<sup>_∗_</sup><br>0<sup>_, ς_0) or Dec(</sup><sup>**T**</sup><sup>_∗_</sup><br>0<sup>_, σ_0)</sup><sup>_̸_ =</sup><sup>_⊥_</sup>|
|IOP<sup>link</sup><br>_Z_|(_σ_0_, σj_)|**T**<sup>_∗_</sup><br>0<sup>_,_</sup><sup>**T**</sup><sup>_∗_</sup><br>_j_|the relation of the link proof (Definition 4.6)|
|IOP<sup>req</sup><br>_Z_|(_ξ, σj_)|**T**<sup>_∗_</sup><br>_j_|Dec(**T**<sup>_∗_</sup><br>_j_<sup>_, σj_) =</sup><sup>_⊥_or (Dec(</sup><sup>**T**</sup><sup>_∗_</sup><br>_j_<sup>_, σj_)</sup><sup>_, ξ_)</sup><sup>_∈_Req</sup>_Z_|



_L_ is the set for which Proposition 5.14 states round-by-round soundness; Req _Z_ is the request relation (Definition 4.8). The rounds are those of Construction D.9, with this structure. 

1. IOP<sup>key</sup> _Z_ and IOP<sup>req</sup> _Z_ have a _first part_ , a _shared part_ and a _last part_ ; IOP<sup>link</sup> _Z_ has one part. In a batch of at most _R_ requests on one copy, the shared part is run once, the other two once for each request. 

2. The first part reduces its statement to one _claim on the table_ , _⟨h ⊗_ eq( _ψ, ·_ ) _, A⟩_ + _η_ = _b_ , with _h_ , _ψ_ , _b_ known to _V_ and _η_ a linear form in committed masks. In IOP<sup>key</sup> _Z_ the statement is that _A_ is in range; in IOP<sup>req</sup> _Z_ it is ( _A, ξ_ ) _∈_ Req _Z_ . 

3. The shared part folds the table’s rows at a sampled point, opens at most _ν_ off columns of **T** 0<sup>_∗_,oratmost</sup> _ν_ on of **T**<sup>_∗_</sup> _j_<sup>,andcomparesthefoldwiththemandwiththeOODanswers.Thelastpartrelatestheclaim</sup> to the fold. 

4. IOP<sup>link</sup> _Z_ folds both tables at one sampled point, opens at most _ν_ first columns of **T**<sup>_∗_</sup> 0<sup>and</sup><sup>_ν_copyof</sup><sup>**T**</sup><sup>_∗_</sup> _j_<sup>,and</sup> compares the folds on the plain entries at a second sampled point. 

5. Exactly the rounds _m_ in which _V_ samples positions have proof of work, of _κm_ bits (Definition 2.5). 

6. Each of the three verifiers rejects if Chk( _Z_ ) = 0 (Definition 4.14). 

**Definition 4.14 (Parameter checks).** Chk( _Z_ ) _∈{_ 0 _,_ 1 _}_ is a function of _Z_ and _H_<sup>_′_</sup> alone. Chk is a conjunction of tests, in four groups; oracle messages, Lookups and blocks are those of Construction D.6. 

1. _Errors._ A list of the IOPs’ rounds, each with an error bound, is derived from _Z_ ; every bound is at most the parameter file’s target. 

2. _Rounds._ The derived rounds, sizes and oracle messages are constants of _VZ_ . 

3. _Layout._ An oracle message is opened only after it is sent; the layout file places each cell where a Lookup reads it. 

4. _Sizes._ A challenge’s blocks, the bits of proof of work, the Merkle trees’ depths and the compiled verifiers’ queries are within fixed bounds. 

The closed list is the Lean term that Appendix F names for this definition (Remark D.10, Given in outline). 

In Construction 4.15 the committer and prover of Π _Z_ are those of Construction D.12. In a machine-checked statement Π _Z_ is the Lean term with the same verifier and extractor, whose committer and prover are constant functions (Remark 4.16). The statements of Section 5 on Π _Z_ depend on neither: Definition 2.7 names the verifier and an extractor alone. 

**Construction 4.15 (The argument** Π _Z_ **).** Fix _Z_ ; _H_<sup>_′_</sup> is the function of Assumption 4.11. _H_ is an oracle of all algorithms; in every experiment it is sampled after _Z_ is fixed. Π _Z_ = (Rnd _Z,_ Commit _Z, PZ, VZ_ ) is a commit-and-prove argument (Definition 2.6) for the relation of greedy outputs at the model parameters _µZ_ (Definition 3.2 and Definition 3.8). Construction D.11 defines Merkle roots rt, the map Seed, which derives an OOD seed from a root by two queries, and the compiled verifier of an IOP. code denotes fixed injective maps 

18 

to bit strings, one on configurations and one on pairs of a key and a digest; dg _Z_ = _H_ (code( _Z_ )) is the _digest_ of _Z_ . A proof _π_ is a list of tokens and five byte strings. _Verifier. VZ_<sup>_H_(</sup><sup>_c, x, y, π_)isdeterministic.Itevaluates</sup><sup>_H′_onpublicdataalone:onpartsof</sup><sup>_Z_,andon</sup><sup>_x_and</sup> 

_y_ . 

1. _Scope test._ Reject if Adm( _Z_ ) = 0 (Definition 3.8). 

2. _Unpacking._ Unpack _π_ , at dg _Z_ and by one query to _H_ (Construction D.11), into (fill _,_ k _, j,_ rt _j, σj, π_ key _, π_ link _, π_ req), where k = (rt0 _, σ_ 0) is the _key_ and each of the last three may be _⊥_ ; if this fails, continue with the fixed tuple of Construction D.11, at which step 4 rejects. 

3. _Public readings._ Compute the _public readings_ (dg _Z,_ par) (Construction D.11); they enter every string that steps 7 and 8 hash. 

4. Reject unless _ξ_ = build( _x, y,_ fill) is defined (Definition 4.7). 

5. Reject unless _H_ (code(k _,_ dg _Z_ )) = _c_ . 

6. Reject if one of _π_ key, _π_ link, _π_ req is _⊥_ . 

7. Reject unless _ς_ 0 = Seed(rt0) and _ςj_ = Seed(rt _j_ ): the OOD seeds that _V_ samples in Construction 4.13 are derived from the roots. 

8. Run the three compiled verifiers, each on a list of digests (roots, and OOD answers written as blocks of _λ_ bits), an IOP instance and a proof string: 



The third list holds the copy’s root twice. Each verifier includes the parameter checks Chk( _Z_ ) (Construction 4.13). Reject unless all three accept. 

9. Reject unless the plain parts in _π_ req hold the number that _π_ names for its copy. Otherwise accept. 

Steps 4 to 9 are the _compiled steps_ of _VZ_ . 

_Extractor._ Ext _Z_ ( _c,_ tr), for a query–answer trace tr, makes no query; its value is _⊥_ if a step fails. 

1. Find the first entry of tr whose answer is _c_ and whose query is code(k _,_ dg) for a key k = (rt0 _, σ_ 0) and a digest dg. 

2. Translate the queries of tr, at the digest dg, into queries for leaves and inner nodes of Merkle trees (Construction D.11). 

3. From these, rebuild the word **T**<sup>_∗_</sup> 0<sup>undertherootrt0.</sup> 

4. Return wt(Dec( **T**<sup>_∗_</sup> 0<sup>_, σ_0)).</sup> 

_Committer and prover_ are those of Construction D.12. _PZ_ serves up to _R_ instances as one batch; _VZ_ checks each proof by itself. 

_Remark_ 4.16 (Honest parties). The machine-checked statements on Π _Z_ concern _VZ_ , Ext _Z_ and the model, which are written in Lean, as above, up to what Remark D.10 (Given in outline) lists. Index-binding knowledge soundness (Definition 2.7) depends on nothing else: a commitment is whatever _λ_ = 256 bits the adversary presents. 

In the Lean term for Π _Z_ , Rnd _Z_ , Commit _Z_ and _PZ_ are constants, and the provers of the three IOPs send fixed messages. Hence completeness, index privacy (and with it anything on hiding the weights), correctness of extraction, extractability of the commitment and no two openings cannot be stated with any content of that term; a second Lean term, with the same verifier and with the committer and prover that the Lean function compile yields (Remark B.7 (Which compiler); its output: Definition 6.15) from the Lean provers, is the object of Section 6.5. Honest provers of the three IOPs, and the committer and prover compiled from them, are written in Lean as definitions, for one request to a proof; completeness and zero knowledge are machine-checked for these, with _VZ_ and with their limits (Section 6.5). Of the committer compiled from 

19 

them, with _VZ_ and its extractor, soundness with extractability of the commitment, no two openings and correctness of extraction are proved, at the files in use, with their limits, in Section 6.5; these read no prover. Of the released programs none of these properties is proved. The results of Theorem 6.3, Corollary 6.7, and Theorem 6.9 concern other objects: the table experiment, the whole-key argument and IOP<sup>link</sup> _Z_ . The extractor need not return the weights that a committer used. The results on the explicit committer (Definition 5.16) are statements on Π<sup>Γ</sup> _Z_<sup>,notonΠ</sup><sup>_Z_.</sup> 

The number that a proof names for its copy (step 9 of Construction 4.15) is tied to rt _j_ by no step, and an acceptance does not certify it. 

### **4.6 True requests** 

**Definition 4.17 (True request, constraints, honest bound).** Let _µ_ be valid model parameters (Definition 3.6), for a configuration _Z_ those that _Z_ states. Let _A_ be in range (Definition 4.1) and _ξ_ = (win _, i_ 0 _, i_ 1 _, y_<sup>+</sup> ) an encoded request (Definition 4.7), so that _y_<sup>+</sup> = 0 if _i_ 1 _< L_ . Under exact match, the encoded request is _true at A_ if, for the integer model at _µ_ and the weights wt( _A_ ) (Definition 3.1 and Definition 3.2), 

1. the model is defined on win, 

2. the token win[ _i_ + 1] is the greedy token after (win[0] _, . . . ,_ win[ _i_ ]) for all _i_ 0 _≤ i <_ min( _i_ 1 _, L −_ 1), and 

3. if _i_ 1 = _L_ , then _y_<sup>+</sup> is the greedy token after win. 

The _constraints_ of an arithmetization (Definition 4.8) are its identities and lookups. Cells _satisfy the constraints_ modulo _p_ , or over the integers, if every identity is 0 and every lookup key is an entry of its table, modulo _p_ or as integers. Cells are _within the honest bound_ if each has absolute value at most 2<sup>7</sup> . 

**Lemma 4.18 (True requests and greedy outputs).** [machine-checked] _Let µ be valid model parameters, A a digit matrix of M ∈I, and x, y,_ fill _lists of tokens._ 

- _(a) If_ build( _x, y,_ fill) _is defined and true at A (Definition 4.17), then y is a greedy output of M on x._ 

- _(b) If_ ( _M, x, y_ ) _∈_ Serv _µ (Definition 3.11), then_ build( _x, y,_ fill) _with empty_ fill _is defined and true at A._ 

## **5 Soundness** 

All theorems, propositions, lemmas and corollaries of this section are machine-checked, nine of them (Theorems 5.1 and 5.19 among them) by joining Lean theorems on paper, as the text beside each says. For each machine-checked statement, Appendix F names the Lean declarations that it renders. Lean proves Theorem 5.1 in two forms, at the same error: with the extractor left open, as in Definition 2.7, and with Ext _Z_ of Construction 4.15 named (Remark 5.2, What the theorem gives). The Lean form of Theorem 5.1 has one further hypothesis, that the three byte strings of _Z_ decode without error; admissibility implies it by a further Lean theorem; Theorem 5.1 as printed joins the two, and so does every statement of this section whose Lean form carries that hypothesis, Corollary 5.10 and Theorem 5.11 for this reason alone. 

### **5.1 The theorem** 

**Theorem 5.1 (Soundness).** [machine-checked] _Let Z be an admissible configuration (Definition 3.8) with the exact-match policy (Definition 3.4), fixed before H is sampled. Then the argument_ Π _Z (Construction 4.15) is index-binding knowledge sound (Definition 2.7) for the relation of greedy outputs at µZ (Definition 3.2), in the random-oracle model (Assumption 4.10 and Assumption 4.11), with error_ 



_where q is the sum of a bound on the committer’s queries to H and a bound on the prover’s._ 

In _T_ the 1 is the query of step 2 of _VZ_ (Construction 4.15 and Construction D.11), and the summand 2<sup>23</sup> covers the verifier’s queries in steps 3 and 6 to 9 (as the compiler counts them, not its calls to _H_ ) and one query that the proof adds; the query of step 5 is in neither, and the term ( _T_ + 2)<sup>2</sup> _/_ 2<sup>256</sup> accounts for it. Section 5.6 accounts for that term and for the factor 16. 

20 

_Remark_ 5.2 (What the theorem gives). Except with probability _ε_ ( _q_ ), an output that _VZ_ accepts is a greedy output of the weights that the straight-line extractor, Ext _Z_ of Construction 4.15, reads off the commitment and the committer’s queries; these weights are determined before any proof is made. The theorem holds for every proof string: a proof that is made in a batch, which _VZ_ verifies singly, with its path in the batch tree, is covered at _ε_ ( _q_ ), and _N_ proofs sent at once are covered at _N · ε_ ( _q_ ), _q_ being the committer’s bound and the whole list’s prover bound added up (Corollary 5.10). 

Here _ε_ ( _q_ ) _≈ q/_ 2<sup>124</sup> + 2<sup>_−_101</sup> : 124 bits per query (Remark B.5, Bits per query). The bounds on Reed– Solomon codes strictly below the Johnson radius that the proof uses (Lemma 5.13 and Lemma A.23) are machine-checked, resting in part on files adapted from ArkLib and Mathlib (Remark B.18, Typed statement). On how the Lean proof is checked, see Remark B.17. Where Chk( _Z_ ) = 0, _VZ_ rejects every input; Chk( _Z_ 0) = 1 (Corollary 5.8). 

_Remark_ 5.3 (Reading the theorem). 

1. The configuration is admissible and fixed before the oracle. A choice of it after querying _H_ , which is open to every provider, is not covered (Open problem 8.16; Limitation 8.5). 

2. A second hash function is fixed (Assumption 4.11), an assumption that is neither tested nor proved and on which no run bears. The proofs hold for this function, not for every function in its place. 

3. No witness is extracted: the relation has none, and no party is asserted to know the weights. The extractor need not be efficient: it is a function defined with the axiom of choice, not a program. 

4. The extracted weights need not be the committer’s: that they are is correctness of extraction (Section 5.5; Theorem 6.31). Ext _Z_ takes _Z_ as an argument: no statement says that Ext _Z_ and Ext _Z′_ agree, or ties a commitment to one configuration. 

5. The client checks three things. Exact match is a hypothesis that _VZ_ does not test: the client checks that pol _Z_ = (256 _,_ 256 _,_ 0); at other policies Theorem 5.11 applies. The conclusion holds whatever the other two are: the model parameters, which fix the integer model and lie in the scale file and, for _ϵ_ 1, _ϵ_ 2, _ϵF_ , M _F_ and K _F_ , in the parameter file (Definition 3.7; Limitation 8.6, Scales can disable blocks), and the length of _y_ (Limitation G.4, Prefixes count). The verifier is _VZ_ , not the C++ verifier (Limitation 8.1, Lean verifier only); on what a run trusts, see Remark B.10. 

6. That the parameter checks pass, which is necessary for acceptance and not sufficient, is proved only at _Z_ 0 and, for Theorem 5.11, at the configurations that differ from it only in the policy (Proposition 5.6). Besides the reduction of Remark 6.49, the one theorem on acceptance by _VZ_ is completeness with the Lean prover, on served requests and over a random oracle; that some request is served is stated by a lemma beside it, at constructed weights (Section 6.5). A verifier that rejects everything also meets Definition 2.7, and likewise round-by-round soundness (Proposition 5.14). 

At _Z_ 0, and at the configurations that differ from it only in the policy, the parameter checks are ruled out as the cause, and in runs at _Z_ 0 and _Z_ 1 the executable form, which equals _VZ_ by Theorem 5.5, accepted honest proofs on the parsed proof files (Section 7.3); so there _VZ_ is seen not to reject everything, at the format’s hash function and as far as a run is trusted (Remark B.10, Claim from files). These runs were made over Lean files that differ from the released ones, but for the calls of the reference run and of the second set of 41, which were made over the released Lean files (Appendix H.3). Elsewhere, and at any other hash function, the theorem may hold only because the verifier rejects everything (Limitation 8.9, Completeness on served requests only); over a random oracle, Section 6.5 excludes this where the files of _Z_ are in use (defined there), and some request is served. 

7. That Definition 3.8 and Construction 4.15 and the relation of greedy outputs are the Lean terms of Theorem 5.1 is checked by reading, not by Lean, except for the policy’s three numbers; so is Construction 4.13, its item 5 by a count by programs. 

8. A verifier that treated the programs’ hash function specially would also meet the theorem, whose bound is over a random oracle (Limitation G.5, Fiat–Shamir instantiation). 

Remark B.4 adds particulars to items 3 to 5. 

21 

### **5.2 The executable form and the default configuration** 

As a Lean term, _VZ_ cannot be evaluated, because parts of its definition use the axiom of choice. A second term, the executable form, can, and it computes the same decision. 

**Definition 5.4 (Executable form).** A function _O_ from byte strings to byte strings _induces_ the hash function _HO_ , where _HO_ (bs) consists of the first _λ_ = 256 bits of _O_ (byt(bs)); the injective coding byt of bit strings as byte strings and the order of bits are those of Construction B.8. The _executable form_ of _VZ_ at _O_ is the function _EZ_<sup>_O_thatthealgorithmofConstructionB.8computes.Itmapsacommitment</sup><sup>_c_,aprompt</sup><sup>_x_,an</sup> output _y_ and a proof _π_ (Construction 4.15) to 1 (accept), 0 (reject) or _⊥_ . 

**Theorem 5.5 (The executable form computes the verifier).** [machine-checked] _For all configurations Z, all functions O from byte strings to 32-byte strings, and all commitments c, prompts x, outputs y and proofs π (Definition 5.4),_ 



_In particular, EZ_<sup>_O_(</sup><sup>_c, x, y, π_)</sup><sup>_̸_=</sup><sup>_⊥._</sup> 

**Proposition 5.6 (The parameter checks pass at the default configuration).** [machine-checked] _The default configuration Z_ 0 _(Definition 3.7) is admissible (Definition 3.8), its policy is exact match, and_ Chk( _Z_ 0) = 1 _(Definition 4.14). Every configuration Z that differs from Z_ 0 _only in the values_ ( _C, θ, cap_ ) _of its policy, where_ 2<sup>8</sup> _≤ θ ≤ C <_ 2<sup>32</sup> _and cap ≤_ 1024 _, is admissible, and_ Chk( _Z_ ) = 1 _._ 

_Remark_ 5.7 (How the proposition is checked). Proposition 5.6, a conjunction made on paper of Lean theorems, concerns _Z_ 0 as written out in Lean, not the release’s files (Remark D.2, Written copy). The proofs of its statements on Chk lie in a separate Lean library, which nothing beneath Theorem 5.1 or Proposition 5.14 imports, so neither of these rests on it. One constant, on which that library rests, equates the layout file as one string with its pieces. At the default thread stack Lean’s kernel stops on it at too deep a recursion; at a larger one the kernel accepts it, both when the file is compiled and when the toolchain checks it again. The re-checker `leanchecker` stops on it in the same way at each stack limit that was tried, and no independent program has accepted it. It is not the only one. In passes over every other module of the main and the separate library, over Lean files that differ from the released ones, in code too, the re-checker stopped with the same message on 22 further modules of that library, and on no module of the main library. Replayed at the larger stack over the build of those files, by Lean’s own replay, Lean’s kernel accepted each of the 22. The check script replays all 1,438 modules of two libraries ( `lean/AtFullSide/` , 914 modules; `lean/ShapeLines/` , 524), not counting the file added later (Remark B.17), by Lean’s own replay at a larger stack and not by `leanchecker` ; 23 of them are shown to need it, all of them modules of `lean/AtFullSide/` ; on `lean/ShapeLines/` nobody has tried `leanchecker` . By import lines at least one of the 23 lies beneath the module of each theorem of Section 6.5 on completeness and zero knowledge, and beneath those of Theorem 6.31, of Proposition 6.17, part 1, and of Proposition 6.33, parts 1 and 4; and beneath none of the soundness theorems of Section 5 (Theorem 5.1; Theorem 5.17 and Theorem 5.19), nor beneath Theorem 5.5, Corollary B.11 or Theorem 6.20: no module of that library lies beneath any of them ( `lean/GAPS.md` , gap 73; `lean/LIMITS.md` , L69). 

**Corollary 5.8 (At the default configuration).** [machine-checked] _Let Z_ 0 _be the default configuration (Definition 3.7). In the random-oracle model (Assumption 4.10 and Assumption 4.11), the argument_ Π _Z_ 0 _is index-binding knowledge sound (Definition 2.7) for the relation of greedy outputs at µZ_ 0 _, with the extractor_ Ext _Z_ 0 _and the error ε_ ( _q_ ) _of Theorem 5.1. Moreover,_ Chk( _Z_ 0) = 1 _, and EZ_<sup>_O_</sup> 0<sup>=</sup><sup>_V_</sup> _Z_<sup>_H_</sup> 0<sup>_O_</sup> _for every function O from byte strings to 32-byte strings._ 

Lean has these three parts as separate theorems, joined here on paper; on the part Chk( _Z_ 0) = 1, see Remark 5.7. 

_Remark_ 5.9 (What a client gets). At _Z_ 0, comparing three digests (Definition D.1; Remark D.2) settles the policy and the model parameters, and of the three checks of Remark 5.3 (item 5) the length of _y_ remains. Corollary 5.8 is the conjunction of Theorem 5.1, Theorem 5.5, and Proposition 5.6 at _Z_ 0. The bound _ε_ ( _q_ ) also holds for the executable form’s own decision (Corollary B.11), and the executable form has accepted, in calls that ran to their end, every honest proof at _Z_ 0 counted in Section 7.3 (two of them by a separate build of the executable form alone, tied to _VZ_ by name and statement: Appendix H.3), in runs of Lean’s evaluator 

22 

on synthetic weights; it has also accepted parsed proofs altered at entries that it does not read, and at three of them a C++ verifier of other programs<sup>_†_</sup> rejected the matching altered raw proof. These calls were made over Lean files that differ from the released ones, but for the calls of the reference run and of the second set of 41, which were made over the released Lean files (Appendix H.3). 

The executable form is run inside Lean, by the command `lean-verify` , which evaluates _EZ_<sup>_O∗_</sup> (Assumption 4.10). The statements concern the values of terms, not whether or when an evaluation ends; Remark B.10 lists what a run trusts beyond them. 

**Corollary 5.10 (Many proofs).** [machine-checked] _Let Z and ε be as in Theorem 5.1. There is an extractor as in Definition 2.7, with output M_<sup>_∗_</sup> _, such that for all N , qC, qP , every qC-query committer, and every qP -query prover that receives the committer’s state and outputs at once a list of at most N triples_ ( _xi, yi, πi_ ) _: with probability at most_ 



_VZ accepts some_ ( _c, xi, yi, πi_ ) _with_ ( _xi, yi_ ) _∈/_ Lang( _M_<sup>_∗_</sup> ) _._ 

### **5.3 Other policies** 

**Theorem 5.11 (Soundness under any policy).** [machine-checked] _Let Z be an admissible configuration (Definition 3.8) with policy_ pol _, fixed before H is sampled. Then, in the random-oracle model (Assumption 4.10 and Assumption 4.11), the argument_ Π _Z (Construction 4.15) is index-binding knowledge sound (Definition 2.7), with the error ε_ ( _q_ ) _of Theorem 5.1, for the relation_ 



_of Definition 3.4 in the place of the relation of greedy outputs._ 

_Remark_ 5.12 (Committer-chosen weights). As in every commit-and-prove argument, the committer chooses the weights. By itself, Theorem 5.11 therefore excludes nothing against a committer who commits to weights under which every output meets the policy. Such weights exist for many policies: under all-zero weights, every non-empty output that fits one window together with a non-empty prompt, all tokens in _V_ , meets every policy with _θ, C ≥_ 257 (Proposition 3.9). The theorem constrains a provider only together with correctness of extraction at the same extractor (Definition 2.9; for the explicit committer, Theorem 5.17) and with knowledge of who made the commitment (Limitation 8.7, Unattributed commitment). At exact match it also excludes an accepted empty output (Proposition 3.5). 

### **5.4 Rounds and codes** 

**Lemma 5.13 (Mutual correlated agreement strictly below the Johnson radius).** [machine-checked] _Let_ K _be a finite field,_ K0 _a subfield and D ⊆_ K0 _a finite set. Let C ⊆_ K<sup>_D_</sup> _be the Reed–Solomon code of the polynomials over_ K _of degree at most a ≥_ 1 _, evaluated on D, and put ϱ_ = _a/|D|. Let e ≥_ 1 _be an integer with δ_ = _e/|D| <_ 1 _−_<sup>_√_</sup> _<u>ϱ.</u> Then C has mutual correlated agreement (Definition 2.2) at δ, for two words under the generator_ (1 _,_ z) _with_ z _uniform in_ K _, with error J/|_ K _|, where_ 



The lemma is the statement as formalized in Lean, after [BCHKS25, Theorem 4.6]; its Lean proof runs through files adapted from ArkLib and Mathlib [ArkLib, Mathlib] (Appendix E.4). Remark B.18 (Typed statement) relates it to the published theorem, which Theorem 5.1 does not need. 

**Proposition 5.14 (Round errors).** [machine-checked] _Let Z be an admissible configuration with_ Chk( _Z_ ) = 1 _(Definition 4.14). Then_ IOP<sup>key</sup> _Z and_ IOP<sup>link</sup> _Z are round-by-round sound, and_ IOP<sup>req</sup> _Z_<sup>_,asabatchIOPforatmost_</sup> _R encoded requests, is round-by-round knowledge sound (Definition A.1 and Definition A.2), each for its set L in Construction 4.13, with every round error, proof of work (Definition 2.5) included, at most_ 



23 

_For_ IOP<sup>req</sup> _Z the witness is empty: L_<sup>wit</sup> _consists of the triples_ (in _,_ ext _, ⋆_ ) _with_ (in _,_ ext) _∈L._ 

For the rounds of the Proximity tests, the source whose error terms the counts of bad challenges (Table 10) follow in form is named in Remark A.26. 

_Remark_ 5.15 (Single rounds). 

1. These are errors of single rounds; the probability of a forgery is bounded by Theorem 5.1, at 124 bits per query. The proposition joins on paper three Lean theorems, one for each IOP. Where _Z_ is admissible and Chk( _Z_ ) = 1 (Section 6.5), Lean proves these round errors for the three IOPs together, as one statement, and bounds the error of the whole run before compilation by 67 _,_ 391 _·_ 2<sup>_−_96</sup> _<_ 2<sup>_−_79</sup> ; after compilation the bound stays that of Theorem 5.1. 

2. In the rounds that sample positions, the bound holds only with the round’s 20, 28 or 32 bits of proof of work; without them the errors are about 2<sup>_−_96</sup> to 2<sup>_−_110</sup> (a program’s arithmetic, not machine-checked; where Lean joins the three, as in item 1, it is machine-checked that no round is worse than 2<sup>_−_96</sup> ). At _Z_ 0 the 28 bits for the 64 positions of a table are needed: with 27 the bound exceeds 2<sup>_−_128</sup> (Proposition D.8). 

3. The figures that the C++ verifier prints follow another accounting of round errors. Under stricter counts than that accounting (all columns of the matrix of cells that is tested; a factor for the list size), some of its rounds would miss, by a planning calculation that is not machine-checked, its target of 2<sup>_−_128</sup> (Remark G.2, Stricter counts). Proposition 5.14 does not depend on it: its bound is proved for the IOPs of Construction 4.13. 

### **5.5 The commitment** 

A client needs soundness for the weights that were committed, not for weights that an extractor returns. Theorem 5.19 gives it, for an explicit committer: except with a small probability, an accepted output meets the policy under the committed weights themselves. It rests on Theorem 5.11 and on Theorem 5.17: where the parameter checks pass, Ext _Z_ returns the committed weights, except with probability about 2<sup>_−_89</sup> . Theorem 5.17 states its errors in the form of the error of Theorem 5.1; Proposition 5.18 gives sharper ones for its parts 1 and 2, and Remark B.23 the form of part 1 of Theorem 5.19 with them. 

**Definition 5.16 (Explicit committer).** Let _Z_ be an admissible configuration (Definition 3.8). A _coin map_ Γ maps a secret seed in _{_ 0 _,_ 1 _}_<sup>_λ_</sup> and two values of _H_ to _coins_ : random coefficients, pads and leaf salts for the first table **T** 0 (Definition 4.6). The _explicit committer_ is the pair (Rnd _,_ Commit _Z,_ Γ): 

   1. Rnd outputs a uniform seed, without queries; 

   2. the randomness r of Commit<sup>_H_</sup> _Z,_ Γ<sup>(</sup><sup>_M,_r)iseithercoinsoraseedsd;inthesecondcasethecoinsare</sup> Γ(sd _,_ dg _Z,_ dg<sup>_′_</sup> ), where dg<sup>_′_</sup> is the value of _H_ at a code of dg _Z_ and the table’s sizes; 

3. from the coins and a digit matrix determined by _M_ (Definition 4.1), it computes **T** 0 and the key k = (rt0 _, σ_ 0) by the rules of Construction 4.15 and Construction D.11, and outputs only _c_ = _H_ (code(k _,_ dg _Z_ )). 

Π<sup>Γ</sup> _Z_<sup>isΠ</sup><sup>_Z_withthiscommitterandanyprover,</sup><sup>_VZ_andExt</sup><sup>_Z_unchanged.</sup> 

**Theorem 5.17 (Extraction and extractability for the explicit committer).** [machine-checked] _Let Z be an admissible configuration (Definition 3.8) at which the parameter checks pass,_ Chk( _Z_ ) = 1 _(Definition 4.14), and_ Γ _a coin map (Definition 5.16), both fixed before H is sampled. Let_ 



_In the random-oracle model (Assumption 4.10 and Assumption 4.11),_ Π<sup>Γ</sup> _Z_ 

_1. has correctness of extraction (Definition 2.9) for_ Ext _Z with error β_ (2<sup>35</sup> + 2) _;_ 

_2. has extractability of the commitment (Definition 2.10) for_ Ext _Z with error β;_ 

_3. has no two openings with error_ 2 _β._ 

24 

**Proposition 5.18 (Sharper errors).** [machine-checked] _Let_ 



_Under the hypotheses of Theorem 5.17, its parts 1 and 2 hold with γ in the place of β. Part 1 also holds, with β and with γ, for every_ r _, a secret seed or coins, that is fixed before H is sampled, in the place of_ r _←_ Rnd _._ 

**Theorem 5.19 (Soundness after a commitment by the explicit committer).** [machine-checked] _Let Z be an admissible configuration with policy_ pol _Z (Definition 3.8 and Definition 3.4),_ Γ _a coin map (Definition 5.16), M ∈I and AP a qP -query algorithm, all fixed before H is sampled. In the random-oracle model (Assumption 4.10 and Assumption 4.11), let_ r _←_ Rnd _, c_ = Commit<sup>_H_</sup> _Z,_ Γ<sup>(</sup><sup>_M,_r)</sup><sup>_,_(</sup><sup>_x, y, π_)</sup><sup>_←A_</sup> _P_<sup>_H_(</sup><sup>_c,_r)</sup><sup>_and_</sup> acc = _VZ_<sup>_H_(</sup><sup>_c, x, y, π_)</sup><sup>_,andlet_</sup> 



_with β as in Theorem 5.17._ 

_1._ Pr[ acc = 1 _and y does not meet_ pol _Z on x under M_ ] _≤ ε_ com( _qP_ ) _._ 

_2. If_ pol _Z is exact match,_ Pr[ acc = 1 _and y is empty or not a greedy output of M on x_ ] _≤ ε_ com( _qP_ ) _. Remark_ 5.20 (Sizes of the errors). Numerically, 



Here 2<sup>35</sup> + 2 bounds the explicit committer’s queries; for one test that a pair opens _c_ , _β_ uses the looser bound 2<sup>48</sup> , and _γ_ uses it twice. The function _β_ weakens _γ_ to the form of the error of Theorem 5.1. Each part of Theorem 5.17, and part 1 of Theorem 5.19 with _β_ or with _γ_ , is a single Lean theorem, joined as Theorem 5.1 is with the further theorem on admissibility, and the three parts of Theorem 5.17 are joined here on paper; each statement of Proposition 5.18, and part 2 of Theorem 5.19, is an application of named Lean theorems to one another, which no single Lean theorem states. Part 2 of Theorem 5.19 is Lean’s theorem at exact match taken, on paper, at the explicit committer’s 2<sup>35</sup> + 2 queries and the one query of step 2 of _VZ_ (Construction 4.15). 

_Remark_ 5.21 (Scope of the statements on the commitment). 

- _Object._ The explicit committer models one part of the release’s committing program. It was compared, in values only, with the committing program of other programs<sup>_†_</sup> , at the reduced configuration (Table 4), where none of these theorems is stated: what ran was a faster Lean definition of the committer, proved equal to it, with two further Lean definitions, for the table and for the coins, which no theorem on the committer mentions. Nothing is proved about the release’s committing program (Limitation G.1, Implementation unverified). 

- _Checks._ Chk( _Z_ ) = 1 is machine-checked only at the configurations of Proposition 5.6. There the hypothesis of Theorem 5.17 is met. Where Chk( _Z_ ) = 0, Theorem 5.19 is vacuous (Remark B.22, Proof outlines for the commitment). 

- _Acceptance._ The explicit committer outputs no key file and has no honest prover, so no proof is shown to be accepted under its commitment. Who made a given commitment is outside the statements (Limitation 8.7, Unattributed commitment). No test covers the three properties of Theorem 5.17. 

That the properties of Theorem 5.17 fail for a committer that replaces every digit matrix by the all-zero matrix, and that a committer whose commitment is a constant also meets the bound of Theorem 5.19, which constrains a committer only on the event that _VZ_ accepts, is Proposition B.24. 

25 

### **5.6 Proof outline** 

The proof of Theorem 5.1 has four parts (Appendix A). 

_Compilation._ Steps 4 to 9 of _VZ_ are the output of the Lean library’s compiler (Construction D.11) at the three IOPs (Remark B.7, Which compiler). The lemma on this compiler (Lemma A.15) rests on an argument over separate oracles (Lemma A.11), which fixes an event Bad on traces (Definition A.5): a new answer repeats a value of the trace, changes the state of an IOP from 0 to 1, or gives an OOD seed with a collision. A _Q_ -query algorithm meets Bad with probability at most _Q ε_ rd + 2 ( _Q_ + 1)<sup>2</sup> _/_ 2<sup>256</sup> (Lemma A.8). Outside Bad, an accepting run of a compiled verifier gives the relation of its IOP at the words under its roots (Lemma A.9). Hence the first table decodes, the copy decodes to the same matrix _A_ , and ( _A, ξ_ ) _∈_ Req _Z_ (Lemma A.10). 

_Rounds._ Proposition 5.14 gives _ε_ rd = 2<sup>_−_128</sup> where the parameter checks pass; elsewhere the three verifiers reject (Lemma A.22). 

_Meaning._ If ( _A,_ build( _x, y,_ fill)) _∈_ Req _Z_ , then _y_ meets pol _Z_ on _x_ under wt( _A_ ) (Lemma A.33). This proves Theorem 5.11; at exact match, _y_ is a greedy output (Proposition 3.5). 

_Error._ The compiled verifiers make _qV ≤_ 2<sup>20</sup> queries to the separate oracles of Construction D.11, which is not a bound on their queries to _H_ (Remark B.13, Lists sent at once). The public readings make eight queries to _H_ , and step 2 one, for dg _Z_ (Construction D.11; Lemma A.35); the proof adds one query that announces rt0, and to each query four queries for OOD seeds (Lemma A.11; Definition A.4). Hence _Q_ = 5 ( _q_ + _qV_ + 10) _≤_ 5 _T_ , since the 2<sup>23</sup> in _T_ covers _qV_ + 9. Then _Q ε_ rd _≤_ 5 _T/_ 2<sup>128</sup> , and where _ε_ ( _q_ ) _<_ 1 the quadratic term is at most 6 _T/_ 2<sup>128</sup> . The factor 16 is not tight: the arithmetic lemma is stated for round errors up to 2 _ε_ rd. The term ( _T_ + 2)<sup>2</sup> _/_ 2<sup>256</sup> accounts for replacing the key by its hash value _c_ (Lemma A.14 and Lemma A.36). 

## **6 Zero knowledge and completeness** 

This section is on zero knowledge, formally index privacy (Definition 2.14), and on completeness; what is proved is exact honest-verifier simulation up to private aborts before compilation (Definition 6.24; Theorem 6.27) and zero knowledge with request leakage after it (Definition 6.34; Theorem 6.35 and Theorem 6.37; Remark 6.40). For the Lean provers, honest provers of the three IOPs written in Lean as definitions, completeness and zero knowledge are machine-checked in the forms that Section 6.5 states, before and after compilation, with the limits stated there (Limitation 8.3 (Zero knowledge of definitions in Lean); Limitation 8.9 (Completeness on served requests only)); that the forms of zero knowledge are index privacy is not stated. On zero knowledge three further objects have machine-checked results. On completeness, further: Remark 6.49. 

- _The committed tables of_ FlashProve _Z ,_ before compilation. With fresh randomness and within the opening budget, their OOD answers and opened columns are independent of the weights (Theorem 6.3). 

- _The whole-key argument_ (Section 6.3). Its index privacy reduces to five conditions on its provers, five of twenty (Theorem 6.6). In Lean the compiled zero-knowledge statement of Section 6.5 is obtained through the theorem on all twenty conditions (Theorem C.25): inside its proof all twenty conditions are met by the Lean provers of Section 6.5, at one copy of the table. No Lean theorem states that by name; we have it from a reading of the proof’s calls. Every limit of the results before compilation holds for that statement as well (Remark 6.41). With constant-message provers, fresh random masks, one request to a proof and the whole key, not a 256-bit hash, as the commitment, and against any verifier, what it shows of the tables is simulated from the leakage alone, up to a stated error (Corollary 6.7). 

- _The IOP of the link proof,_ before compilation. At admissible _Z_ its honest prover has an exact honestverifier simulator, given the two tables’ OOD samples and opened columns, which is not a statement that the weights are hidden. At admissible _Z_ with Chk( _Z_ ) = 1, the link’s verifier accepts the same prover at every challenge that passes proof of work, for all weights, at one fixed public-files digest, at which _VZ_ is not proved ever to be called (Theorem 6.9). 

### **6.1 Leakage** 

The leakage (Definition 2.13) is what a simulator is given: pp( _M_ ), _N_ and each batch’s revealed instances. What a client of the released programs sees by design, including timing and failures, is listed in Remark G.10. 

26 

_Remark_ 6.1 (What the leakage contains). The simulator may depend on _Z_ , which is quantified before _M_ in every statement. In the statements on index privacy the simulator is given the leakage (Definition 2.13); what the simulators of Section 6.5 are given is listed there. In the machine-checked statements on index privacy pp is constant, and each concerns one commitment. Every revealed instance lies in _SM_ (Limitation G.11, Outcome leakage), and the released tools compute the scale file of _Z_ and lines of its parameter file from the weights (Section 3.3); these two files encode _µZ_ (Definition 3.7). No statement bounds what the revealed instances or these files reveal. The environment is quantified after _M_ and may depend on it. So its instances can carry information about the weights into both views, and no statement bounds that. 

### **6.2 Committed tables** 

The protocol has copies because a table is shown to hide the weights for at most _t_ opened columns. In the table experiment an unbounded _A_ names an OOD seed for each table before the table is sampled, and receives OOD answers, chosen columns, and runs of the three IOPs with constant-message provers. It is within the opening budget if at most _t_ columns of each table are opened, where each run is charged 62 or 64 columns. 

**Definition 6.2 (Constant-message prover).** A prover for an IOP (Definition 2.3) is a _constant-message prover_ if in every round it sends one fixed message, the same for all inputs, coins and challenges, and never aborts. Its external oracle strings are the committed tables of its input. 

**Theorem 6.3 (Committed tables before compilation).** [machine-checked] _Let Z be admissible (Definition 3.8), with opening budget t_ = 44 _,_ 040 _,_ 191 _. For all matrices A, A_<sup>_′_</sup> _over_ F _and every algorithm A within the opening budget (Definition C.6), the outputs of the table experiments with fresh randomness for each table and constant-message provers (Definition C.5),_ 



#### _are identically distributed._ 

_Remark_ 6.4 (No prover message is covered). By Theorem 6.3, what an unbounded, adaptive _A_ within the opening budget sees of the committed tables is perfectly independent of the weights. The theorem is not index privacy. 

- No message that an honest prover computes from _A_ is covered, alone or jointly with the tables. 

- Each table has one OOD sample, and its randomness is used at one configuration. Randomness derived by a pseudorandom function, as in the released programs, is not covered. An OOD seed is chosen before its table is sampled. A seed computed from the table, such as a hash of its root, as in the released programs, is not covered. 

For the OOD answers Theorem 6.3 rests on Lemma C.9, which follows [CFW26, version of February 25, 2026, Section 9.2]. 

**Proposition 6.5 (The opening budget is tight for one table).** [machine-checked] _Let t_ = 44 _,_ 040 _,_ 191 _and s_ = 1 _, at the table sizes of Z_ 0 _(Definition 3.7). Let A_ = 0 _and let A_<sup>_′_</sup> _have the single nonzero entry A_<sup>_′_</sup> [0][0] = 1 _, both in range (Definition 4.1); let ς_<sup>_◦_</sup> _be the OOD seed with point_ (1 _, . . . ,_ 1) _and factor_ 1 _. For every set Q ⊆ D of t_ + 1 _positions, the distribution of the pair of the OOD answers at ς_<sup>_◦_</sup> _and the columns_ ( **T** [ _ϕ_ ]) _ϕ∈Q of a committed table_ **T** _(Definition 4.2) differs between A and A_<sup>_′_</sup> _._ 

### **6.3 After compilation** 

The _whole-key argument_ Π<sup>wk</sup> _Z_<sup>[</sup><sup>**P**](DefinitionC.20)differsfromΠ</sup><sup>_Z_asfollows.</sup> 

- Matrices _A_ and encoded requests _ξ_ take the place of indices and instances. 

- The commitment is the whole key: the roots and OOD samples of **T** 0 _, . . . ,_ **T** _K_ (each sample at a seed that is a hash of its table’s root), an offline proof and _K_ link proofs. 

- A proof is for one request, on a uniformly chosen copy; _N_ c bounds the requests simulated on one copy. 

27 

- The proofs are compiled runs (Definition C.18) of provers **P** = ( _P_ key _, P_ link _, P_ req), which are parameters. 

- The verifier runs steps 6 to 9 of _VZ_ , at fixed values (dg _,_ par) of its public readings. 

Theorem 6.6 requires five conditions of each prover _P∗_ (in full: Definition C.21): 

1. its external oracle strings are the committed tables; 

2. a simulator Sim _∗_ , given the IOP instance, the challenges and the opened columns, reproduces what _P∗_ discloses, except with probability _ζ∗_<sup>sim</sup> ; 

3. _P∗_ aborts for a reason that depends on its input with probability at most _ζ∗_<sup>ab;</sup> 

4. the prefixes of its leaf salts are public; 

5. all coins are uniform. 

**Theorem 6.6 (Reduction of index privacy to five conditions).** [machine-checked] _Let Z be admissible, K ≤_ 710 _,_ 324 _and N_ c _≤_ 688 _,_ 127 _, and fix any values_ (dg _,_ par) _. Suppose that provers_ **P** _for the three IOPs of_ FlashProve _Z and simulators_ Sim _∗ satisfy the five conditions of Definition C.21 on a set J of matrices and a set Y of pairs_ ( _A, ξ_ ) _, with errors ζ∗_<sup>sim</sup> _, ζ∗_<sup>ab</sup><sup>_.LetSA⊆{ξ_: (</sup><sup>_A, ξ_)</sup><sup>_∈Y}forA ∈J .Thenthewhole-keyargument_</sup> Π<sup>wk</sup> _Z_<sup>[</sup><sup>**P**]</sup><sup>_(DefinitionC.20),withconstant_pp</sup><sup>_,hasindexprivacyonSforbatchesofatmost_1</sup><sup>_(Definition2.14),_</sup> _with error_ 



_where ζ_ hash _is the hashing error (Definition C.24)._ 

**Corollary 6.7 (Committed tables after compilation).** [machine-checked] _Let Z be admissible with_ Chk( _Z_ ) = 1 _, K ≤_ 710 _,_ 324 _and N ≤ N_ c _≤_ 688 _,_ 127 _, and fix any values_ (dg _,_ par) _. Let_ **P** _consist of constantmessage provers (Definition 6.2), with fresh uniform randomness and these parameters of the compiled runs (Definition C.18 and Definition C.19): search limit_ 2<sup>40</sup> _; ℓ_ s = 320 _uniform bits in each leaf salt; all-zero public prefixes; leaf_ 0 _of the batch tree. Let S assign to every matrix over_ F _any set of requests, and let_ 



_1._ Π<sup>wk</sup> _Z_<sup>[</sup><sup>**P**]</sup><sup>_hasindexprivacyonSforbatchesofatmost_1</sup><sup>_(Definition2.14),withanerrorζsuchthat_</sup> _ζ_ ( _q, N_ ) _≤ ζ_ max( _q, N_ ) _. For K_ = _N_ = 1 _and q ≤_ 2<sup>64</sup> _,_ 



_2. In the real experiment, for every matrix, every environment valid at it and every q-query V_<sup>_∗_</sup> _,_ 



The corollary’s search limit is 2<sup>40</sup> , that is, 2<sup>40</sup> + 1 attempts in each proof-of-work search. The corollary’s bound, with constant-message provers, fresh random masks, one request to a proof and the whole key as the commitment, and where the parameter checks pass, has these two values for _q ≤_ 2<sup>64</sup> . At one copy and one request, _ζ_ max _≤_ 2<sup>_−_135</sup> , as the corollary states. At the largest _K_ and _N_ that the corollary allows, _K_ = 710 _,_ 324 and _N_ = 688 _,_ 127 ( _N_ counts requests on all copies together), _ζ_ max _<_ 2<sup>_−_97</sup> , evaluated by hand. 

_Remark_ 6.8 (What holds after compilation). By Corollary 6.7, with constant-message provers, fresh random masks, one request to a proof and the whole key, not a 256-bit hash, as the commitment, what the commitment and the proofs show of the committed tables (roots, OOD samples, opened columns) is simulated up to _ζ_ max. This holds against any verifier within the corollary’s bounds. Besides the verifier’s queries the simulator is given the leakage alone, not the weights. 

28 

1. _Conditional._ Of 20 conditions (Theorem C.25) fifteen are proved (a count of conditions, not of work): those that mention no prover, coin or simulator; five of the fifteen hold by type or default. The other five concern the three honest provers, their simulators and coins; the Lean provers meet them inside the proof of the compiled zero-knowledge statement of Section 6.5, at one copy of the table; no Lean theorem states that by name, and we have it from a reading of the proof’s calls (Open problem 8.13). 

2. _Constant messages._ Fixed messages are trivially independent of the weights. Hence, for the provers’ own messages, a result on index privacy is meaningful only together with completeness of the same provers (for the Lean provers: Section 6.5). The corollary covers nothing that the three provers (in Corollary C.29: the other two) would compute from the weights in their rounds. 

3. _The whole key._ The commitment of Π _Z_ has 256 bits, not the whole key. No statement transfers either result to Π _Z_ . 

4. _Single requests._ Both results are for one request to a proof. Lean’s statements on completeness and zero knowledge do not cover batches, which the released programs use (Open problem 8.14). 

5. _Proofs are produced._ A constant-message prover never aborts, so its compiled run is _⊥_ only if a proof-ofwork search exhausts its limit (Definition C.18). Part 2 bounds the probability of that; without it, part 1 could be a statement on views that hold no proof. 

### **6.4 The link proof** 

**Theorem 6.9 (Link proof before compilation).** [machine-checked] _Let Z be admissible (Definition 3.8), and let P_ link _be the honest prover of_ IOP<sup>link</sup> _Z (Construction D.9), with uniform coins. For every matrix A over_ F _and all pads, random coefficients and OOD seeds of_ **T** 0 _and_ **T** _j:_ 

_1. P_ link _never aborts, and one honest-verifier simulator for all of them (Definition C.2), given the OOD samples_ ( _σ_ 0 _, σj_ ) _, the challenges_ ch _, and the at most_ 62 _columns that the verifier opens in each table, is exact at every_ ch _._ 

_2. Let_ Chk( _Z_ ) = 1 _, and let the proof-of-work bits of_ ch _be those that the compiled verifier (Construction D.11) marks at_ (dg _,_ par) = (0<sup>_λ_</sup> _, ∅_ ) _. Then the verifier of_ IOP _Z_<sup>link</sup> _accepts P_ link _at every_ ch _that passes the proofof-work checks._ 

_Remark_ 6.10 (Not a statement on the weights). Theorem 6.9 concerns one of the three provers, at the default configuration’s size, which every admissible _Z_ has. By part 1, to an honest verifier the link proof’s messages add nothing to what the two tables’ OOD answers and opened columns show. These are computed from the weights and the tables’ randomness (on their distribution: Theorem 6.3), so by itself part 1 does not state that the weights are hidden. Part 1 extends Corollary 6.7 to _P_ link, with the other two provers constant-message provers, at (dg _,_ par) = (0<sup>_λ_</sup> _, ∅_ ), values at which _VZ_ is not proved ever to be called (Corollary C.29). Nor is it proved at these values that the compiled verifier accepts _P_ link; on acceptance after compilation: Section 6.5. 

### **6.5 Completeness and zero knowledge of the Lean provers** 

This subsection concerns honest provers of IOP<sup>key</sup> _Z_<sup>,IOP</sup> _Z_<sup>link</sup> and IOP<sup>req</sup> _Z_ that are written in Lean as definitions, and the argument compiled from them, which proves one request per proof. The main results are the theorems on that argument, Theorem 6.29, Theorem 6.31, Theorem 6.35, and Theorem 6.37; of the definitions of this subsection their statements need only Definition 6.11, Definition 6.12, Definition 6.14, Definition 6.15, and Definition 6.34, so a reader may pass from Remark 6.18 to the paragraph “After compilation”. Zero knowledge is stated in two forms: before compilation as exact honest-verifier simulation up to private aborts (Definition 6.24 and Theorem 6.27), after compilation as zero knowledge with request leakage (Definition 6.34, Theorem 6.35, and Theorem 6.37). 

The theorems and propositions below are machine-checked, and no written proof is given: the formal statements are the Lean declarations that Appendix F lists under each number. Some printed statements say less than their Lean forms; where more than an instantiation is needed, the remark that follows the statement gives the step. Their Lean forms have one further hypothesis, that the three byte strings of _Z_ decode without 

29 

error; admissibility implies it by a further Lean theorem, and the statements below join the two. In the remarks a _request_ is an instance ( _x, y_ ), and _y_ is its _claimed output_ . 

#### **Hypotheses and objects.** 

**Definition 6.11 (Configuration in use).** Three conditions on a configuration _Z_ (Definition 3.7) are used as hypotheses. 

**(Admissible configuration)** Adm( _Z_ ) = 1 (Definition 3.8). 

**(Checks pass)** Chk( _Z_ ) = 1 (Definition 4.14). 

**(Exact match)** The policy pol _Z_ of an admissible _Z_ is exact match (Definition 3.4). 

The files of _Z_ are _in use_ if _Z_ is admissible, Chk( _Z_ ) = 1 and its policy is exact match; _Z_ itself is then also called _in use_ . At files in use the policy is exact match, which fixes the policy’s three numbers; the five numbers _ϵ_ 1, _ϵ_ 2, _ϵF_ , M _F_ , K _F_ stay free. 

**Definition 6.12 (Served instance, label, bit).** Let _µ_ be valid model parameters and _M ∈I_ (Definition 3.1). An instance ( _x, y_ ) is _well-formed_ if _x_ and _y_ are non-empty lists of tokens of _V_ with _|x|_ + _|y| ≤ L_ + 1, that is, if build( _x, y, ∅_ ) is defined (Definition 4.7). It is _served_ at _M_ if ( _M, x, y_ ) _∈_ Serv _µ_ (Definition 3.11), and _refused_ otherwise. The _bit_ bit _µ_ ( _M, x, y_ ) _∈{_ served _,_ refused _}_ says which. The _label_ of a well-formed instance is 



An instance that is not well-formed has the label other tokens. 

_Remark_ 6.13 (Served and can be served). 

1. _Two conditions._ “Can be served” (Definition 3.11) requires only that the model be defined on the _window_ of ( _x, y_ ), the first _L_ tokens of _x ∥ y ∥_ 0 0 _· · ·_ . “Served” requires in addition that _y_ be a greedy output of _M_ on _x_ . The bit is membership in Serv _µ_ , not the weaker condition. Every served instance is well-formed. 

2. _The Lean form._ In Lean the label is defined through the model’s _result_ on ( _x, y_ ): the tokens that its run on the window chooses for the positions of _y_ , or no result if the model is not defined on the window. A well-formed instance is served if the model has a result on it and _y_ is that result; its label is failed if there is no result, and other tokens if the result differs from _y_ . Lean theorems, at valid _µ_ , give the form of Definition 6.12 by propositional steps. 

3. _Zeros after the output._ The window includes the zeros that follow _y_ . Hence a greedy output _y_ of _M_ on _x_ has the label failed if the model is not defined on the whole window. 

4. _No token._ The label names no token: other tokens says only that the result differs from _y_ . 

**Definition 6.14 (Lean provers).** Let _Z_ be admissible. For _M ∈I_ let _A_ ( _M_ ) be a digit matrix determined by _M_ (Definition 4.1), and put 



The _Lean provers_ **P**<sup>lp</sup> _Z_<sup>=(</sup><sup>_P_lp</sup> key<sup>_, P_</sup> link<sup>lp</sup><sup>_, P_</sup> req<sup>lp)areproverswithabortsets(DefinitionC.2)forIOP</sup> _Z_<sup>key,IOP</sup> _Z_<sup>link</sup> and IOP<sup>req</sup> _Z_<sup>,thethirdforoneencodedrequest.TheyaretheLeantermsthatAppendixFnamesforthis</sup> definition; the paper does not print them. The input of each consists of a matrix _A_ , the randomness and the OOD seed of each committed table on which it runs and, for _P_ req<sup>lp,anaturalnumber,the</sup><sup>_copynumber_</sup> (Construction 4.15, step 9). 

**Definition 6.15 (Compiled Lean argument).** Let _Z_ be admissible. The _compiled Lean argument_ Π<sup>lp</sup> _Z_<sup>= (Rndlp</sup> _Z_<sup>_,_Commitlp</sup> _Z_<sup>_, P_</sup> _Z_<sup>lp</sup><sup>_, VZ_)isacommit-and-proveargument(Definition2.6)fortherelationofgreedy</sup> outputs at _µZ_ , with _VZ_ and Ext _Z_ of Construction 4.15. 

30 

   - Rnd<sup>lp</sup> _Z_<sup>samplesonenumber runiformlyfromafixedrange,withoutqueries:rdeterminesthecommitter’s</sup> coins. 

   - Commit<sup>lp</sup> _Z_<sup>(</sup><sup>_M,_r)forms</sup><sup>**T**0andonecopy</sup><sup>**T**1for</sup><sup>_A_(</sup><sup>_M_),makes</sup><sup>_π_keyand</sup><sup>_π_linkascompiledruns(Defini-</sup> tion C.18) of _P_ key<sup>lpand</sup><sup>_P_</sup> link<sup>lp,andoutputsthe256bits</sup><sup>_H_(code(k</sup><sup>_,_dg</sup> _Z_<sup>)),wherek = (rt0</sup><sup>_, σ_0).</sup> 

   - _PZ_<sup>lp(</sup><sup>_M,_r</sup><sup>_, x, y_)repeatsthisrun,queriesincluded,andmakes</sup><sup>_π_reqasacompiledrunof</sup><sup>_P_lp</sup> req<sup>on</sup><sup>**T**1and</sup> build( _x, y, ∅_ ); for ( _x, y_ ) not well-formed, on a fixed encoded request in no pair of Serv<sup>iop</sup> _Z_<sup>.Itoutputsthe</sup> _empty proof_ , the empty list of tokens with five empty byte strings, if one of the three proof strings is _⊥_ , else a proof with fill = _∅_ , k, _π_ key, the copy’s number 0 (the key’s list of copies starts at place 0 and does not hold _T_ 0, so this is _T_ 1), rt1, _σ_ 1, _π_ link and _π_ req. 

- _Remark_ 6.16 (Arguments with the same verifier). Π<sup>lp</sup> _Z_<sup>differsfromtheotherargumentsofthispaperasfollows.</sup> 1. Π _Z with the committer and prover of Construction D.12._ The verifier and the extractor are the same. That committer forms _K_ copies, and that prover serves batches of up to _R_ instances on a copy that it chooses within the opening budget. Π<sup>lp</sup> _Z_<sup>hasonecopy,provesoneinstanceperproof,andkeepsnocount</sup> and no state. _P_ link<sup>lpistheLeantermofwhichTheorem6.9speaks;nostatementcompares</sup><sup>_P_lp</sup> key<sup>or</sup><sup>_P_lp</sup> req with the honest provers of Construction D.9. 

   2. _The Lean term for_ Π _Z in the statements of Section 5_ (Remark 4.16, Honest parties). Its committer and prover are constant functions. Π<sup>lp</sup> _Z_<sup>isasecondLeanterm,withthesameverifier.Index-binding</sup> knowledge soundness depends on the verifier alone, so Theorem 5.1 holds for Π<sup>lp</sup> _Z_<sup>withthesameerror.</sup> For no other property is an equivalence of the two terms stated. 

   3. _The whole-key argument_ (Definition C.20). Its commitment is the whole key. The commitment of Π<sup>lp</sup> _Z_ has 256 bits; _π_ key, the copy’s root and OOD sample, and _π_ link are parts of each proof. 

**Proposition 6.17 (The hypotheses are met).** [machine-checked] 

_1. The default configuration Z_ 0 _is in use (Definition 6.11)._ 

_2. For every admissible Z there is a pair_ ( _A, ξ_ ) _∈_ Serv<sup>iop</sup> _Z_ 

   - <sup>iop</sup> _Z (Definition 6.14) with A in range._ 

_3. For every admissible Z with_ Chk( _Z_ ) = 1 _there are matrices A̸_ = _A_<sup>_′_</sup> _in range and an encoded request ξ such that_ ( _A, ξ_ ) _and_ ( _A_<sup>_′_</sup> _, ξ_ ) _lie in_ Serv<sup>iop</sup> _Z_<sup>_._</sup> 

_Remark_ 6.18 (Constructed weights). That some request is served (Proposition 6.17, parts 2 and 3) is shown at constructed weights, by lemmas and not by Theorem 6.23 and Theorem 6.29. In part 1, _Z_ 0 is the Lean value that Definition 3.7 names, the configuration as written out in Lean, not the release’s files (Remark D.2; Remark 5.7). 

**Before compilation.** A list of challenges of one of the three IOPs _passes_ if all its proof-of-work bits are 0 (Definition 2.5); the verifiers reject every other list. A star in a subscript, as in _P∗_ , Sim _∗_ , _α∗_ and _η∗_ , is one of key, link, req, and SD is the statistical distance. 

**Definition 6.19 (Whole run).** A _whole run_ of FlashProve _Z_ (Construction 4.13) with a prover _P_<sup>˜</sup> consists of the following rounds, in this order. 

1. _P_<sup>˜</sup> sends a word **T**<sup>_∗_</sup> 0<sup>asanoracle,</sup><sup>_V_samplesanOODseed</sup><sup>_ς_0(Definition4.3),and</sup><sup>_P_˜sendsanswers</sup><sup>**a**0;</sup> put _σ_ 0 = ( _ς_ 0 _,_ **a** 0). 

2. The rounds of IOP<sup>key</sup> _Z_ on _σ_ 0 and **T**<sup>_∗_</sup> 0<sup>.</sup> 

3. As in step 1, a word **T**<sup>_∗_</sup> 1<sup>with</sup><sup>_σ_1= (</sup><sup>_ς_1</sup><sup>_,_</sup><sup>**a**1).</sup> 

4. The rounds of IOP<sup>link</sup> _Z_ on ( _σ_ 0 _, σ_ 1) and **T**<sup>_∗_</sup> 0<sup>_,_</sup><sup>**T**</sup> 1<sup>_∗_.</sup> 

5. _P_<sup>˜</sup> names an encoded request _ξ_ . 

6. The rounds of IOP<sup>req</sup> _Z_ on ( _ξ, σ_ 1) and **T** 1<sup>_∗_.</sup> 

The run is _accepted_ if the verifiers of the three IOPs accept. 

31 

**Theorem 6.20 (Soundness of a whole run before compilation).** [machine-checked] _Let Z be admissible with_ Chk( _Z_ ) = 1 _. In a whole run (Definition 6.19) let_ G _be the event that_ 



_where_ Req _Z is the request relation (Definition 4.8). For every deterministic prover P_<sup>˜</sup> _, over uniform OOD seeds and uniform lists of challenges that pass,_ 



_Remark_ 6.21 (Reading the bound on a whole run). Let _Z_ be admissible with Chk( _Z_ ) = 1, as in Theorem 6.20. 

1. _Round by round._ It is machine-checked, as one Lean statement, that against every prover the protocol before compilation is round-by-round sound (Definition A.1) for Req _Z_ , the relation under the policy of _Z_ . Each round’s error, that of the round in which an OOD seed is sampled included, is at most 2<sup>_−_128</sup> of all that round’s challenges, proof-of-work bits among them (Proposition 5.14). 

2. _The factors._ Among the challenges that pass, a round’s error is therefore at most 2<sup>_−_96</sup> , since no round has more than 32 bits of proof of work. A whole run has 67 _,_ 391 rounds. Lean’s bound is for a deterministic prover; for a randomized one it follows by averaging over the prover’s coins, a step on paper. 

3. _Not a figure for the compiled argument._ 2<sup>_−_96</sup> is the bound for the rounds with the most proof of work, applied to every round. It is not a bound for the compiled argument, whose error is that of Theorem 5.1. 

4. _The relation._ Req _Z_ is taken at the policy of _Z_ , with the relaxed bound of Definition 4.8; at exact match it implies that each output token is the model’s. 

5. _The extractor._ The round-by-round statement of item 1 has an extractor for the weights. It is straightline: the decoding function Dec, applied to the first table and its OOD sample. It extracts the weights alone, and no running time is claimed for it. 

**Definition 6.22 (Completeness of provers for the three IOPs).** Let **P** = ( _P_ key _, P_ link _, P_ req) be provers with abort sets (Definition C.2) for the three IOPs of FlashProve _Z_ , _Y_ a set of pairs ( _A, ξ_ ), and _α ∈_ [0 _,_ 1]. A run of _P∗ fails_ at given coins and challenges ch if ch lies in the abort set, or if the verifier of the IOP rejects the transcript for some values of its nonces. **P** is _complete on Y with error α_ if there are _α_ key + _α_ link + _α_ req _≤ α_ such that, for every ( _A, ξ_ ) _∈Y_ , all randomness and all OOD seeds of committed tables **T** 0 _,_ **T** _j_ for _A_ , and every copy number: 

1. the run of _P_ key on **T** 0, of _P_ link on ( **T** 0 _,_ **T** _j_ ), and of _P_ req on **T** _j_ and _ξ_ , each with the tables’ OOD samples, fails with probability at most _α_ key, _α_ link, _α_ req, over the prover’s uniform coins and a uniform ch that passes; 

2. at all coins, challenges and nonces, the plain parts of _P_ req hold the copy number. 

**Theorem 6.23 (Completeness before compilation).** [machine-checked] _Let Z be admissible with_ Chk( _Z_ ) = 1 _. Then the Lean provers_ **P**<sup>lp</sup> _Z_<sup>_(Definition6.14)arecompleteon_Serviop</sup> _Z with error_ 2<sup>_−_41</sup> _(Definition 6.22)._ 

**Definition 6.24 (Exact simulation up to private aborts).** Let **P** and _Y_ be as in Definition 6.22, and _η_ key _, η_ link _, η_ req _∈_ [0 _,_ 1]. Honest-verifier simulators Simkey _,_ Simlink _,_ Simreq (Definition C.2), the third also given the copy number, are _exact for_ **P** _on Y up to private aborts_ ( _η_ key _, η_ link _, η_ req) if the following holds for every _A_ in range, all randomness and OOD seeds of the tables, with the tables’ OOD samples at these seeds, and every copy number; for _P_ req, items 2 and 3 are required at the _ξ_ with ( _A, ξ_ ) _∈Y_ . 

1. The external oracle strings of _P∗_ are the committed tables of its input. 

2. Sim _∗_ is exact at every ch that passes and lies outside the private part of the abort set of _P∗_ . 

3. A uniform ch that passes lies in that private part with probability at most _η∗_ . 

4. If ( _A, ξ_ ) _∈Y/_ , the abort set of _P_ req contains every ch. 

32 

**Definition 6.25 (Table experiment with served bits).** Let **P** and _Y_ be as in Definition 6.22. In the table experiment Tab<sup>**P**</sup> ( _A, A_ ) (Definition C.5) let _ς_ 0 be fixed in advance, let the honest verifier’s challenges be uniform among those that pass, let each run of the third IOP be on one encoded request _ξ_ , and let _A_ attach a bit to _ξ_ , which **P** does not read. The next two conditions are required for all answers that _A_ could be given, whether or not they occur. 

_A_ is _truthful_ at _A_ if the bit always says whether ( _A, ξ_ ) _∈Y_ . _A_ is _within the life_ ( _K, N_ ) _of one commitment_ if it asks for at most _K_ copies, at most one run of the first IOP, on **T** 0, at most one run of the second for each copy, at most _N_ runs of the third on each copy and none on **T** 0, and no single column. 

**Definition 6.26 (Simulated table experiment).** Let Simkey _,_ Simlink _,_ Simreq be honest-verifier simulators as in Definition 6.24. The _simulated experiment_ Tab<sup>Sim</sup> ( _A_<sup>_′_</sup> _, A_ ) is the experiment of Definition 6.25 with two changes. The tables are committed tables for _A_<sup>_′_</sup> . The output of a run is that of Sim _∗_ on the IOP instance, ch, the symbols that the verifier reads in these tables and, for a request, the copy number; at a request whose bit says that it is not in _Y_ , the output is _⊥_ and no column is read. 

**Theorem 6.27 (Honest-verifier zero knowledge before compilation).** [machine-checked] _Let Z be admissible with_ Chk( _Z_ ) = 1 _. There are honest-verifier simulators_ Simkey _,_ Simlink _,_ Simreq _such that:_ 

_1. they are exact for the Lean provers_ **P**<sup>lp</sup> _Z_<sup>_on_Serviop</sup> _Z up to private aborts_ (0 _,_ 0 _,_ 2<sup>_−_174</sup> ) _(Definition 6.24);_ 

_2. for all K ≤_ 710 _,_ 324 _and N ≤_ 688 _,_ 127 _, every assignment of copy numbers to the copies, every ς_ 0 _, all matrices A, A_<sup>_′_</sup> _in range, and every unbounded A that is truthful at A and within the life_ ( _K, N_ ) _of one commitment (Definition 6.25 and Definition 6.26),_ 



_Remark_ 6.28 (Reading the honest-verifier statement). 

1. _What the simulators are given._ The simulators are fixed first, and are given neither the weights nor a table’s randomness nor a prover’s coins. They may depend on the files of _Z_ . In part 1 each is given the verifier’s messages, the OOD samples, the columns of the tables that the verifier reads and, for a request, the request with its claimed output and the copy’s number (Definition 6.24). 

2. _The exception._ Part 1 excepts the challenges at which the prover aborts for a reason that depends on its input, the private part of its abort set. This has probability 0 offline, 0 for the link proof and at most 2<sup>_−_174</sup> for a served request. 

3. _Part 1 alone._ The OOD samples and the columns are computed from the weights, so run by run this is not by itself a statement that the weights are hidden (Remark 6.10, Not a statement on the weights). In part 2 they are those of tables for _A_<sup>_′_</sup> (in Lean, any matrix _A_<sup>_′_</sup> , in range or not). 

4. _The budget._ Part 2 is over the life of one commitment: up to 710,324 copies with 688,127 requests each, a budget that the statement fixes (one offline run, one link proof for each copy, requests on the copies alone, asked in any order). What an unbounded observer sees is within 2<sup>_−_135</sup> of what it sees of the simulators working from tables of any other fixed weights. There the simulators are also given, for each request, the bit whether it is served. These are the largest numbers that fit the opening budget _t_ = 44 _,_ 040 _,_ 191 of each table. Nothing ties a committer or a prover to this budget. 

**After compilation.** The committer and the prover are those of Π<sup>lp</sup> _Z_<sup>,compiledfromtheLeanprovers;the</sup> verifier is _VZ_ , and the hash function is a random oracle. 

**Theorem 6.29 (Completeness after compilation).** [machine-checked] _Let Z be in use (Definition 6.11), fixed before H is sampled. In the random-oracle model (Assumption 4.10 and Assumption 4.11), the compiled Lean argument_ Π<sup>lp</sup> _Z_<sup>_(Definition6.15)iscompleteon_Serv</sup><sup>_µ_</sup> _Z_<sup>_witherror_2</sup><sup>_−_40</sup><sup>_(Definition2.12andDefinition3.11)._</sup> 

_Remark_ 6.30 (Reading completeness). 

1. _In words._ Where the files of _Z_ are in use, at every well-formed request whose claimed output is the model’s own result, over the hash function and the committer’s and the prover’s coins, _VZ_ rejects the Lean prover’s proof with probability at most 2<sup>_−_40</sup> . 

33 

2. _Two bounds._ 2<sup>_−_41</sup> (Theorem 6.23) and 2<sup>_−_40</sup> are bounds over two distributions. The first is over the provers’ coins and the challenges that pass, at each encoding randomness, each OOD seed and each copy’s number, not on average. The second is over _H_ and all coins, and includes the proof-of-work searches that exhaust their limit. 

3. _Served instances only._ At a request that is not served, that is, one that is ill-formed or on which the model has no result or another result than the claimed output, Theorem 6.23 and Theorem 6.29 state nothing. In particular, no statement of completeness covers the well-formed requests on which the model has no result (Open problem 8.12). At a real failure the prover returns the empty proof, the same as for a claimed output that is not the model’s. That a request is refused is unchecked, as the label is: an empty proof proves nothing, and a prover can return it at a request that it could serve. The compiled verifier rejects it, under every hash function and at all public files (Proposition 6.33). A service that chooses to say “failed” does not prove it. 

**Theorem 6.31 (Soundness with extractability, openings and extraction).** [machine-checked] _Let Z be in use (Definition 6.11), fixed before H is sampled. With β as in Theorem 5.17, let_ 



_In the random-oracle model (Assumption 4.10 and Assumption 4.11), the compiled Lean argument_ Π<sup>lp</sup> _Z (Definition 6.15), with the extractor_ Ext _Z,_ 

_1. is index-binding knowledge sound for the relation of greedy outputs at µZ (Definition 2.7) and has extractability of the commitment (Definition 2.10), each with error ε_ b _;_ 

_2. has no two openings (Definition 2.10) with error_ 2 _ε_ b _;_ 

_3. has correctness of extraction (Definition 2.9) with error_ 3 (2<sup>48</sup> + 2<sup>49</sup> + 10)<sup>2</sup> _/_ 2<sup>256</sup> _<_ 2<sup>_−_155</sup> _._ 

_Remark_ 6.32 (Reading the theorem on the commitment). 

1. _Which parts are single Lean theorems._ Two statements are single Lean theorems to the letter: part 1 for some extractor, and part 3. Part 1 for Ext _Z_ by name, and part 2, are not. They follow a fortiori from one Lean theorem, which states, for Ext _Z_ , index-binding knowledge soundness with the error _ε_ of Theorem 5.1, extractability of the commitment with error _γ_<sup>lp</sup> ( _q_ ) = 3 ( _q_ + 2<sup>49</sup> + 10)<sup>2</sup> _/_ 2<sup>256</sup> , and no two openings with error 2 _γ_<sup>lp</sup> . Parts 1 and 2 as printed follow on paper: _ε ≤ ε_ b and min( _γ_<sup>lp</sup> _,_ 1) _≤ β ≤ ε_ b, each by a Lean lemma, and a probability is at most 1. 

2. _In words._ Nobody opens a commitment of the compiled committer to other weights than the extractor gives, except with probability about _q/_ 2<sup>124</sup> +2<sup>_−_100</sup> , where _q_ counts the hash queries of whoever made the commitment and of the opener together, and no commitment is opened to two different weights except with twice that. For the honest committer, within its budget of 2<sup>48</sup> hash queries (Proposition 6.33), the extractor gives back the committed weights except with probability about 2<sup>_−_155</sup> . 

3. _Openings._ To open (Definition 2.10) is to name the weights and one number, r. 

4. _What the commitment covers._ The 256-bit commitment covers the first table’s root, its OOD sample and the digest dg _Z_ . It does not cover _π_ key, the copy or _π_ link. 

5. _No prover._ Parts 1 and 2 depend on no prover. They hold for every argument that the same compilation yields from any three provers, provided that all leaf salts of its tables have one length, the same whatever the public readings return. Part 3 holds there with error 3( _N_ + 2<sup>49</sup> + 10)<sup>2</sup> _/_ 2<sup>256</sup> , for any bound _N_ on the committer’s own queries. 

6. _The extractor._ Ext _Z_ is a function, not shown computable. 

7. _The programs._ Nothing is stated about the committer of the released programs (Open problem 8.11). 

8. _Figures._ The error of Theorem 5.1 is about _q/_ 2<sup>124</sup> + 2<sup>_−_101</sup> , a different figure from _ε_ b. No figure here is 128 bits. 

34 

**Proposition 6.33 (Empty proofs, queries and opened columns).** [machine-checked] _The following holds of_ Π<sup>lp</sup> _Z_<sup>_(Definition6.15)._</sup> 

_1. Let Z be in use (Definition 6.11), M ∈I, and_ r _any commitment randomness. If an instance_ ( _x, y_ ) _is refused at M (Definition 6.12), then PZ_<sup>lp(</sup><sup>_M,_r</sup><sup>_, x, y_)</sup><sup>_outputstheemptyproof,foreveryHandallcoins._</sup> 

_2. Let Z be in use. One run of_ Rnd<sup>lp</sup> _Z_<sup>_,oneof_Commitlp</sup> _Z_<sup>_andoneofP_lp</sup> _Z_<sup>_togethermakeatmost_248</sup><sup>_queries_</sup> _to H._ 

_3. With ν_ off _, ν_ first _, ν_ copy _, ν_ on _and t_ = 44 _,_ 040 _,_ 191 _as in Definition C.6 and Theorem 6.3, ν_ off + _ν_ first _≤ t and ν_ copy + 688 _,_ 127 _· ν_ on _≤ t._ 

_4. For every configuration Z, VZ_<sup>_H_(</sup><sup>_c, x, y, π_) = 0</sup><sup>_fortheemptyproofπ,everyH,everycandeveryinstance_</sup> ( _x, y_ ) _._ 

**Definition 6.34 (Zero knowledge with request leakage).** Let Π be a commit-and-prove argument with index set _I_ (Definition 2.6), _W_ a set of triples ( _M, x, y_ ), _ℓ_ a function of ( _M, x, y_ ), and _ζ_ : N<sup>2</sup> _→_ [0 _, ∞_ ]. In the experiments of Definition 2.13 let every batch consist of one instance, which is revealed; an environment is _W-valid at M_ if ( _M, x, y_ ) _∈W_ for each of its instances ( _x, y_ ). A set of instances stands for the set of all triples whose instance lies in it. In Ideal<sup>_ℓ_</sup> Sim<sup>theleakageispp(</sup><sup>_M_),</sup><sup>_N_and,foreachinstance(</sup><sup>_x, y_),theinstance</sup> and the _request leakage ℓ_ ( _M, x, y_ ). The argument Π has _zero knowledge with request leakage ℓ on W, with error ζ_ , if a simulator Sim exists such that, for all _M ∈I_ , all environments Env that are _W_ -valid at _M_ , all _q, N ∈_ N and every _q_ -query _V_<sup>_∗_</sup> , 



and in the ideal experiment equal queries receive equal answers. Π[ _ℓ_ ] denotes Π with a prover that outputs ( _ℓ_ ( _M, x, y_ ) _, π_ ) in place of _π_ and a verifier that ignores the first component. 

**Theorem 6.35 (Zero knowledge on served instances).** [machine-checked] _Let Z be in use (Definition 6.11), fixed before H is sampled. In the random-oracle model (Assumption 4.10 and Assumption 4.11), the compiled Lean argument_ Π<sup>lp</sup> _Z_<sup>_(Definition 6.15) has zero knowledge with constant request leakage on_Serv</sup><sup>_µ_</sup> _Z_<sup>_(Definition 3.11_</sup> _and Definition 6.34), with an error ζ such that_ 



_Remark_ 6.36 (Reading zero knowledge on served instances). 

1. _In words._ At all files in use, for one commitment, up to 688,127 answered requests and 2<sup>64</sup> hash queries, and against any verifier within these two bounds: if every request is served, a simulator that gets each request and the largest number of answered requests, and beside these nothing of the weights, makes a view within 2<sup>_−_154</sup> of the real one. 

2. _The files are not counted._ The simulator is chosen after the files of _Z_ and may depend on them, so what these files reveal about the weights is not counted. The released tools compute the scale file of _Z_ and lines of its parameter file from the weights (Remark 6.1, What the leakage contains). Nor are the requests: each served request holds the model’s own output on its prompt, and the simulator gets the request, so what the requests reveal about the weights is not counted. “Nothing of the weights” concerns the experiment alone: there pp and the request leakage are constant. 

3. _Membership is not hidden._ Whether an instance is served depends on the weights, and the theorem does not hide it. The verifier does not name its requests: the environment does, as a function of the verifier’s view. The environment is chosen after the weights, may depend on them, and is the same in both experiments. That each of its instances is served at _M_ is a hypothesis on the environment; nothing in either experiment checks it. Every such environment is covered, so a verifier can pass a choice of request to the environment through its coins and queries. A verifier that gets a reply to a request that is not served is outside this theorem and inside Theorem 6.37. 

4. _One simulator._ One simulator serves all weights, all valid environments and all verifiers, and with probability 1 it gives no two answers to one query. Its running time is not modelled. 

35 

5. _The error._ The theorem states that an error function with this bound exists and names none; beyond the two bounds nothing is stated. _N_ counts the answered requests (Remark 6.39); at a served request the answer is a proof or, where a proof-of-work search exhausts its limit, the empty proof. 

6. _Scope._ The theorem is for fresh random masks and coins, one commitment, one copy of the table, one request per proof and exact match (Remark 6.42, Scope of the results). The weights are fixed before the hash function is sampled, and the verifier has no hash query before the commitment (Remark 6.45, The model). The prover and the committer are Lean definitions (Remark 6.43); the view is that of Remark 6.39, item 4. 

7. _Which requests._ At every valid _µ_ some request is served (Proposition 3.12(a): at constructed weights; more at _Z_ 0: Remark 3.13). At the model parameters of _Z_ 0 there are valid weights, constructed too, at which no instance can be served (Proposition 3.12(b)): at such weights the theorem concerns the commitment alone. No instance at a real model’s weights is proved to be one that can be served (Remark 3.13). 

**Theorem 6.37 (Zero knowledge at every instance).** [machine-checked] _Let Z be in use (Definition 6.11), fixed before H is sampled. In the random-oracle model (Assumption 4.10 and Assumption 4.11), the compiled Lean argument_ Π<sup>lp</sup> _Z_<sup>_(Definition6.15)haszeroknowledgewithrequestleakage_bit</sup><sup>_µ_</sup> _Z_<sup>_(Definition6.12)onthe_</sup> _set of all instances (Definition 6.34), with an error ζ such that_ 



**Corollary 6.38 (Zero knowledge with the label shown).** [machine-checked] _Under the hypotheses of Theorem 6.37, the argument_ Π<sup>lp</sup> _Z_<sup>[lab</sup><sup>_µ_</sup> _Z_<sup>]</sup><sup>_haszeroknowledgewithrequestleakage_lab</sup><sup>_µ_</sup> _Z_<sup>_(Definition6.12)onthe_</sup> _set of well-formed instances (Definition 6.34), with an error that satisfies the bound of Theorem 6.37._ 

_Remark_ 6.39 (Reading zero knowledge after compilation). 

1. _In words._ At all files in use and every request, well-formed or not, 

   - for one commitment, on one copy of the table, up to 688,127 answered requests and 2<sup>64</sup> hash queries, 

   - and against any verifier within these two bounds that has no hash query before the commitment (Remark 6.45), 

   - a simulator that gets each request, the largest number of answered requests and, of the weights, only one bit for each request (served or refused) 

   - makes a view within 2<sup>_−_157</sup> + 688 _,_ 127 _·_ 2<sup>_−_174</sup> _<_ 2<sup>_−_154</sup> of the real one; each answer is a proof or the empty proof. 

As a corollary (Corollary 6.38), for a service that chooses to say “failed”, the same holds at every well-formed request with one label (served, failed or other tokens) given to the simulator and shown with each proof. 

2. _The simulator._ The simulator is fixed before the weights, may depend on the files of _Z_ , and answers the hash queries in the hash function’s place. Its running time is not modelled. The public parameters of the index, pp, are constant for this relation (Definition 3.2), so they carry nothing about the weights. 

3. _The leakage._ The notion is only as strong as _ℓ_ is small: if Π has zero knowledge with a constant request leakage, then Π[ _ℓ_ ] with _ℓ_ ( _M, x, y_ ) = _M_ , whose prover outputs the weights, has zero knowledge with request leakage _ℓ_ . 

4. _The commitment and the view._ The commitment in both experiments is the 256-bit commitment of Definition 6.15, not the whole key. The view consists of this commitment, each answered request with what the prover outputs for it, the query–answer trace and the coins of _V_<sup>_∗_</sup> . 

5. _The two bounds. N_ counts the requests that the prover answers, by a proof or by the empty proof. Elsewhere in this paper an answered request is called a reply. Nothing in the prover counts the answered requests, and beyond the two bounds nothing is stated. 

36 

6. _The error._ 2<sup>_−_157</sup> bounds the error of the compilation at these numbers; 2<sup>_−_174</sup> is the bound on private aborts of Theorem 6.27, counted once for each answered request. A proof-of-work search that exhausts its limit is part of what is simulated: it gives the empty proof at a served request. 

7. _Derivation._ Theorem 6.37 follows from Theorem 6.35 and part 1 of Proposition 6.33 by a lemma on every commit-and-prove argument. Suppose that every served instance lies in the relation and that the prover outputs one fixed value at every refused instance. Then zero knowledge with constant request leakage on the served triples gives zero knowledge with request leakage bit on all instances, with the same error function: the simulator, given “refused”, outputs that value. With the _ζ_ of Theorem 6.35 this gives the bound 2<sup>_−_154</sup> . The sum in the display is the bound of a second Lean theorem, whose proof takes two steps of the same form at that number. Taken at that number, the proof term of Theorem 6.35 gives the sum as its bound too; no named Lean theorem states this, and the step has not been compiled. The converse, from Theorem 6.37 to Theorem 6.35, is not proved. 

_Remark_ 6.40 (Relation to index privacy). Zero knowledge with request leakage (Definition 6.34) differs from index privacy (Definition 2.14) in three respects: each batch is one revealed instance; the set _W_ need not lie in the relation, and in Theorem 6.37 it contains instances that are not served; and the simulator is given the request leakage. In Theorem 6.35 only the first respect is left: its environments are those valid at _M_ for _SM_ the set of instances served at _M_ , and a constant leakage carries nothing. It is not index privacy for batches of at most one either: an empty batch and a hidden instance, which Definition 2.14 admits, are not covered; for this form Lean has the other direction alone, and no Lean theorem derives index privacy from it. That zero knowledge in the forms of Theorem 6.27 and Theorem 6.37 is index privacy is not stated: no Lean theorem derives index privacy of Π<sup>lp</sup> _Z_<sup>fromTheorem6.37.</sup> 

_Remark_ 6.41 (From the IOPs to the argument). 

1. _Limits carry over._ Completeness and zero knowledge after compilation follow from the two results before compilation by a conditional theorem on the compilation, so every limit of these two holds for them as well. 

2. _Zero knowledge._ The Lean proof of Theorem 6.37 passes through Theorem C.25 on the whole-key argument: inside its proof all twenty conditions are met by the Lean provers, at one copy of the table. No Lean theorem states that by name; we have it from a reading of the proof’s calls (Open problem 8.13). 

3. _From the whole key to 256 bits._ The statement of Theorem 6.37 itself concerns the 256-bit commitment. The step from the whole key to that commitment is made inside the proof, in two parts. The whole-key argument is first carried from its separate oracles at fixed (dg _,_ par) to the one oracle _H_ , with the readings computed from it; this leaves the bound unchanged, and it is used as a bound for all values of (dg _,_ par). Then a verifier against Π<sup>lp</sup> _Z_<sup>isrunbyaverifieragainstthisargument,whichcomputes</sup><sup>_H_(code(k</sup><sup>_,_dg</sup> _Z_<sup>))</sup> from the key by one query of its own and forms each proof from the key and a proof of that argument. So the bound of Theorem C.25 is used at _q_ + 1 queries in place of _q_ , and no term is added. 

4. _Completeness._ Completeness does not pass through the reduction of Remark 6.49. 

#### **Limits common to these results.** 

_Remark_ 6.42 (Scope of the results). Theorem 6.23, Theorem 6.27, Theorem 6.29, Theorem 6.35, and Theorem 6.37 and part 3 of Theorem 6.31 are for fresh random masks and coins, one commitment, and one request per run and per proof, not for batches. The theorems after compilation hold at the exact-match policy, and Theorem 6.29, Theorem 6.35, and Theorem 6.37 for one copy of the table. Before compilation exact match is not a hypothesis; an output that the policy of _Z_ would accept and that is not the model’s own is not served, and nothing is stated about it. Every admissible configuration has the layout file of _Z_ 0 and its parameter file except for eight numbers, with _θ ≥_ 2<sup>8</sup> (Definition 3.8), so every result that asks for an admissible _Z_ is at the default configuration’s size. 

_Remark_ 6.43 (Definitions in Lean). The results concern the Lean provers exactly as written, default values included. By their definition in Lean, each prover’s coins are uniform and fresh in each run, and, if ( _A, ξ_ ) _∈/_ Serv<sup>iop</sup> _Z_<sup>,theabortsetof</sup><sup>_P_lp</sup> req<sup>at(</sup><sup>_A, ξ_)containseverylistofchallenges.Thattheproversarenot</sup> 

37 

trivial follows from the results together (Remark 6.46, The two together). These are definitions, not programs that run: for instance, the model’s run on a request is obtained by a non-constructive choice. Nothing is stated about the released programs (the C++ verifier and committer, the CUDA prover). 

_Remark_ 6.44 (Not hidden). 

1. _The bit before compilation._ Before compilation one bit per request is not hidden: whether the request is served, which depends on the weights. The simulator is given that bit, and where a request is not served the prover shows nothing. A bit says whether the claimed output is the model’s result on that request, so over many requests the bits tell what the model answers. 

2. _The bit after compilation._ After compilation one bit per request is revealed (served or refused). The bit depends on the weights, and it is declared, not hidden: for every request of its choice a verifier learns whether the claimed output is the model’s result. (It reads the bit off the reply: a refused request gets an empty proof; so, with small probability, does a served one, where a search for a nonce runs out.) 

3. _Many requests._ So over many requests, chosen adaptively, the verifier learns what the model outputs on inputs of its choice: whoever tries each possible token for a prompt learns the model’s next one from this bit alone, unless all are refused. 

4. _What the bit keeps back._ Against three labels the bit hides one thing: whether a refusal means that the model has no result or that its result is other tokens. The bit hides that only from a verifier that asks few requests. A request’s window (the 1,024 tokens that the model’s run reads: the prompt, the claimed output, then zeros) can itself be sent as a prompt, with one more token as the claimed output, which the run does not read: a verifier that tries every value of that token and is refused each time learns that the model has no result on that window, and one that is served once learns that it has one (argued from the definitions; no theorem states it). Each such search costs up to 202,048 of the 688,127 replies: one for each token of the vocabulary (Definition 3.1). 

5. _Further._ A true output is refused if the model has no value further on in the window. No statement bounds what many bits reveal together. In the corollary one label per request is revealed (served, failed or other tokens), and it depends on the weights. 

_Remark_ 6.45 (The model). 

1. _Order of choices._ In all results after compilation the files of _Z_ are fixed before the hash function is sampled; in Theorem 6.29, Theorem 6.35, and Theorem 6.37 and in part 3 of Theorem 6.31 so are the weights. In Theorem 6.35 and Theorem 6.37 the verifier has no hash query before the commitment. 

2. _Second hash function._ A second, fixed hash function inside the format, _H_<sup>_′_</sup> , is not treated as random (Assumption 4.11). 

3. _Running time._ The simulators’ running time is not modelled. 

4. _The configuration and timing._ Neither completeness nor zero knowledge says anything about what the files of _Z_ themselves reveal about the weights, or about timing. 

5. _Figures._ The figures on zero knowledge are statistical distances (2<sup>_−_135</sup> , 2<sup>_−_154</sup> ) or bounds on a probability (2<sup>_−_174</sup> ), not a security level. 

_Remark_ 6.46 (The two together). A prover that sends fixed messages and aborts at every request that is not served, or whose abort set has a public part that contains every list of challenges, also satisfies a statement of the form of Theorem 6.27. Completeness of the same provers excludes this, so the two results are to be read together. One Lean theorem, at every _Z_ in use, states round-by-round soundness (Remark 6.21, item 1), Theorem 6.23 and part 1 of Theorem 6.27 of one term for FlashProve _Z_ and one triple of provers. That term is the one from which _VZ_ , the verifier of Theorem 5.1, is compiled. Likewise Theorem 6.29, Theorem 6.31, Theorem 6.35, and Theorem 6.37 concern one committer, one prover and one verifier, those of Π<sup>lp</sup> _Z_<sup>;inCorollary6.38thelabelisaddedtotheprover’soutput,andtheverifierignoresit(Definition6.34).</sup> 

_Remark_ 6.47 (Definitions of the properties). The definitions of this subsection restate definitions of the Lean library. Before compilation, the definitions of completeness and of honest-verifier zero knowledge are this 

38 

library’s own and follow no published definition word for word; after compilation Theorem 6.29 and Theorem 6.35 are stated through `CompleteOn` and `ZeroKnowledgeOn` , the texts of `Complete` and `ZeroKnowledge` , each with one condition more, that the request can be served (Definition 3.11); with the condition that the output is the model’s, these are the served requests; the forms of Theorem 6.37 and Corollary 6.38 are the library’s own. The definitions before compilation, and the Lean statements of Theorem 6.23, Theorem 6.27, Theorem 6.37 and Corollary 6.38 other than the six top theorems and the final statement’s six fields, have not been independently reviewed (“Who wrote and checked this”). 

_Remark_ 6.48 (How they are checked). Of the library’s six top theorems the independent checker has accepted `sound` (Remark 6.16), `repliesFit` and `refusedHandsOutNoProof` (Proposition 6.33) by name, over a build of the released Lean files; over such a build the three on binding, completeness and zero knowledge have been checked by Lean’s kernel alone, and they rest on equations between closed terms that the kernel decides by computation. For those of the Lean theorems behind them that are named in `lean/CHECK-RESULTS.md` , Lean prints no axiom beyond its three standard ones. The reference run (the 109 calls of Appendix H.3) has been made over the released Lean files, and the independent checker has been run, on the soundness theorem’s export and on those of the theorems named at this remark’s head, as the exporter writes them over the released Lean files’ build; what it accepted: `lean/CHECK-RESULTS.md` ; of the theorems of this section only those named at this remark’s head are among what it accepted. A whole run of the library’s full check over all but one of the released Lean files (Remark B.17) passed at every step; its last line says that no theorem got a verdict on binders or on `Classical.choice` (the whole line, and what the check does not cover: `lean/CHECK-RESULTS.md` ; Remark B.17). What these checks are: `lean/CHECK-RESULTS.md` ; the limits of the independent checker’s pass, and the version of Lean: Remark B.17. The Lean modules of Theorem 6.23, Theorem 6.27, Theorem 6.29, Theorem 6.31, Theorem 6.35, Theorem 6.37, and Corollary 6.38, of Proposition 6.17, part 1, and of Proposition 6.33, parts 1 and 4, import the separate library of Remark 5.7; by import lines at least one of the modules on which the re-checker stopped lies beneath each of them. By a walk over the constants of a build of the released Lean files, proofs included (a program’s finding, no theorem; it covers twenty-three modules), the Lean theorems `finalSoundAndBinding_value` , `finalHandsOutNoProof_value` and `finalRefusedHandsOutNoProof_value` reach no constant of these modules; each of the theorems on completeness and zero knowledge listed above reaches at least one, but for Theorem 6.35, whose one Lean theorem was not among those walked; the modules of a third library, on which these theorems rest too, were not put to the re-checker; of the other Lean theorems that the listed statements cite it is not known. 

### **6.6 A reduction for completeness, and what is open** 

_Remark_ 6.49 (Completeness in Lean). Besides the results of Section 6.5, on completeness the following is machine-checked. 

1. _A reduction._ Let _µ_ be the model parameters (Definition 3.6) that the files of _Z_ fix. If thirteen conditions hold at ( _Z, α_ ), then _VZ_ , with the committer and prover built from a proving side given as a parameter, is complete on Serv _µ_ with error _α_ : an output that the model itself gives is rejected with probability at most _α_ . Three conditions are on the files: they parse, _Z_ is admissible, and Chk( _Z_ ) = 1. 

   - Ten are on the proving side. One, the largest, asks at every instance true under _Z_ ’s policy, whether it can be served or not, that each of the three IOP verifiers accept its prover’s run, at OOD seeds that are not excluded and at every challenge list that satisfies each proof-of-work check, outside an abort set and an exceptional set; the probabilities of the three sets enter _α_ . It also asks that each prover’s input oracles encode the index, that a challenge list lie in the union of these two sets whenever zeroing its proof-of-work bits gives a list in it, and that the request prover send the copy number handed to it. 

Three fix how the messages and coins of the key and request provers fit the byte format, and five constrain a salt length, a tree depth and the three search limits. The last asks that _α_ dominate these probabilities, the excluded seeds’ twice, and the failure probability of the searches, added up. The only acceptance assumed is that of the three IOP verifiers. 

2. _Conditions that hold._ The three on the files, at _Z_ 0 and at every _Z_ that differs from it only in the policy, in its three numbers, within the bounds of Proposition 5.6 (Remark 5.7, How the proposition is checked). For admissible _Z_ with Chk( _Z_ ) = 1: what the largest condition asks of the link proof, for _P_ link with 

39 

empty exceptional set and an abort set of probability 0 (Theorem 6.9); the three search limits, at 2<sup>_b_+6</sup> attempts per _b_ -bit proof of work, where the searches add at most 71 _·_ 2<sup>_−_92</sup> to _α_ ; and the salt length and the depth, which are met by choice. For the same _Z_ , in the whole-key argument at the fixed digest 0<sup>_λ_</sup> , wherever _π_ link _̸_ = _⊥_ , the verifier of IOP _Z_<sup>link</sup> , not the compiled one, also accepts _P_ link at the challenges derived by hashing (Remark C.30; Remark 5.7). 

3. _Searches succeed._ In the whole-key argument with constant-message provers and at least 2<sup>40</sup> attempts per search, which is more than in item 2, Corollary 6.7 bounds Pr[ _π_ = _⊥_ ] for each kind of proof _π_ ; on acceptance it is silent. 

Whether the Lean provers (Section 6.5) meet the ten conditions is not stated here. All thirteen are shown together only for idle provers, which always abort; then _α ≥_ 3 and the bound is vacuous. In the reduction the parts of the offline and the online prover are not shown, so that it gives no bound below 1. The proving side is a parameter, not that of Π _Z_ ; the reduction does not bound its queries, and no statement relates it to the released programs. Outside Serv _µ_ (Definition 3.11) nothing is stated. For no valid _µ_ is Serv _µ_ empty (Proposition 3.12(a)), but its members are exhibited at constructed weights (more at _Z_ 0: Remark 3.13). For one _µ_ there are valid weights _M_ , constructed too, for which Serv _µ_ contains no triple ( _M, x, y_ ) (Proposition 3.12(b)); by a further Lean theorem that _µ_ is the model parameters of _Z_ 0. 

Completeness and zero knowledge of the whole protocol, with batches, are open problems (Open problem 8.12 and Open problem 8.14). 

## **7 Evaluation** 

### **7.1 Set-up** 

The implementation has a committer, a prover and a verifier in C++ and CUDA (the _C++ verifier_ (Limitation 8.1, Lean verifier only)), an integration with vLLM [KLZ23], and the command `lean-verify` , which evaluates the executable form (Definition 5.4) on the prover’s proof files. Weights are _synthetic_ , generated by the implementation, or _public_ : a third party’s mostly 4-bit quantization of Llama 4 Maverick [Uns25], not Meta’s checkpoint, converted to the integer model (Construction D.3). 

Timed runs use one H100 (80 GB) of an eight-GPU host with 96 CPU cores: in the committing and serving runs the programs were bound to 24 of these cores, all of the GPU’s NUMA node, under a limit of 26 cores and 463.9 GB of memory for the container; as reported, when a machine was assigned the host’s other seven GPUs were in use by other tenants; whether the cores were reserved to a run, and what ran beside it later, the records do not tell; the prover’s process held at most 257.1 GB of host memory while serving at 24 requests a batch on the public weight files (269.7 GB in its container, file cache not counted); on synthetic weights both were higher (265.5 GB and 278.3 GB), and at 64 a batch on the public files the container reached 278.3 GB; the container held more with the middleware’s check beside the prover, and both held more in runs not on the released programs<sup>_†_</sup> (Appendix H.2); the committer’s process held at most 286.9 GB (299.6 GB). A largest size or peak is rounded up, the lower end of a range cut. 

Each request is proved on a window of 1,024 tokens and receives its own proof. Lean’s statements on completeness and zero knowledge do not cover batches. A _run_ is one commitment and one or two executions of the prover on one host; where there are two, its figure is their mean. _Masking on_ means that the committer samples the secret seed and no test option is set (Appendix H.2); fitted and reduced configurations are defined in Section 3.3. 

A result with no mark is of a run that Appendix H.1 names (the runs of `lean-verify` : the next paragraph): a run on the _released_ programs, which are the committer, the prover and the C++ verifier as the release’s sources build them (the programs that made these runs were built from sources that differ from the released ones in comments, in the names of identifiers and files, in quoted texts that hold such a name and in the wording of messages, and in nothing else: Appendix H.1, “Which programs ran”); the _timed runs_ are among these, and have a part of their own there. A result with no mark that Appendix H.1 does not name is to be taken as not on the released programs (Appendix H.1, “What this paper tells of other programs”). A dagger (<sup>_†_</sup> ) marks a result _not on the released programs_ , of any _other programs_ , which is not shown to hold for the released programs; a double dagger (<sup>_‡_</sup> ) marks the test programs of the second and the third test of hiding 

40 

(Table 6), built from the sources of other programs (two of the fault injection’s three from a copy of these in which four files differ: Section 7.5), which do not ship. 

The runs of `lean-verify` told here were made over Lean files that differ in code from the released ones, but for the calls of the reference run and of the second set of 41, which were made over the released Lean files (Appendix H.3). Those marked<sup>_†_</sup> used scripts that differ from the released ones, in code too (one lacked a bound on the length of a number that it reads), and a parser compiled from sources that differ from the released ones, in code too (its hashing class among what differs; the rest not compared file by file). The calls of a separate build of the executable form are another thing (Appendix H.3). Appendix H gives the conditions, the provenance and the limits of each table (Limitation G.7, No independent review). 

### **7.2 Cost** 

Table 3: Mean, sample standard deviation, least and most of _n_ runs. Rates in tokens per second, cut, never rounded up; times in seconds. The rate with whole outputs checked cannot be set against the steady rate at exact match; its difference from its control shows nothing. 

|Quantity|_n_|Mean|s.d.|Least to most|
|---|---|---|---|---|
|Rate at exact match, batches of 24|3|111.23|0.92|110.20 to 111.99|
|time per request (a batch’s time over 24)|3|9.206|0.077|9.143 to 9.292|
|rate with start-up counted|3|103.60|0.84|102.65 to 104.24|
|Rate at exact match, one request in a batch|3|17.41|0.15|17.24 to 17.51|
|Rate, whole outputs checked under a clipped gap, batches of 24|3|108.67|2.8|105.41 to 110.41|
|checked output tokens alone (calculated)|3|39.1|1.0|37.9 to 39.7|
|the same requests, last position alone checked|3|108.73|2.7|105.57 to 110.37|
|Committing, once: the first table and one copy (seconds)|3|2,323.77|18|2,307.45 to 2,343.76|
|Verifying one proof, C++ verifier, called with its option for one thread,<br>on the prover’s machine (seconds)|3|21.14|0.03|21.11 to 21.17|



Table 3 is for the public weights, with masking on (read off the runs’ own summaries, not tested: Appendix H.2); its unmarked rows are of the timed runs. At exact match a proof covers the model’s run over the whole prompt and checks one output token, the one that follows the prompt. Rates are steady-state rates: they count every prefill token of a window, checked or not, and exclude start-up and committing. A steady rate counts every interval between two releases. With start-up counted, in batches of 24 at exact match, each of the three runs passes 100 tokens per second (102.65 to 104.24); of the three sessions with whole outputs checked, and of their control, one each stays under it (99.33; 98.12). At the steady rate the slowest two executions gave 110.07 and 110.33. 

Three runs made side by side say little about other times and machines: three runs of the same kind on other programs<sup>_†_</sup> , on another day on other machines, gave 105.3 ± 3.9 tokens/second (100.85 to 108.06); there each service’s last interval is not counted; whether the programs, the machines, the day or that rule made the difference is not shown (Appendix H.2). A batch is taken from the queue just before the responses of the batch before it are released, so a request waits a little longer than the time between two releases: 220.7 to 225.9 seconds from a batch’s start to its release, from the second batch on, in each case 1.6 to 2.3 seconds more than the interval that ends at that release. That wait ends at a proof on which no verifier has run. Of what ships, the middleware of the vLLM integration hands on no response that has a proof before the C++ verifier has accepted every proof of its batch (Limitation G.16); that time is not in these figures. In one run of its own on the released programs, none of the timed runs, on stored proofs of one batch on synthetic weights, with no inference engine in front and no prover running beside it, on 24 CPUs of a 32-CPU machine, the middleware’s check of the 24 proofs, at its default there of six checks at once with one thread each, took 94.0 seconds from the hand-over of the proofs to the batch’s release; no spread is known. Beside a serving prover on the same 24 CPUs of another machine, in one more run on synthetic weights and on stored proofs, not that prover’s own, at the middleware’s default, six checks at once with one thread each, each of five checks took 88.74 to 89.15 seconds (alone on that machine the check took 92.1 seconds before the services and 90.3 after 

41 

them; what else ran there was not held the same), and the prover’s steady rate was 109.64 tokens per second, against 110.11 in that run’s second service with no check beside it; in the timed runs on synthetic weights two such services part by 0.1 to 0.7 % with nothing beside them, so no slowing can be told from one run. 

On other programs<sup>_†_</sup> , in the runs that answer to the three rows of Table 3 under a clipped gap (105.38, 37.9 and 105.26), the rates differed, row by row, by 8.0 to 9.0 % (the most over the least), and no cause is shown. The difference between the means there and in the timed runs is matched by the control’s and is smaller than the spread there; the timed runs had the lighter CPU load; no effect of the programs is shown. 

The configuration of these runs is the default one (the clipped-gap rows and their control apart from their policy) as fitted to the public weights; the three configuration files of one set of timed exact-match runs on these weights pass the scope test for an admissible configuration, which Theorem 5.1 requires, by one evaluation of a test that does not ship (no theorem); whether they are byte for byte the files behind the runs reported here is not shown, and the parameter checks were not evaluated at them (Remark D.5, No real weights shown). The jobs’ own calls of the C++ verifier, behind the timed stretches, accepted every proof; in these jobs the executable form was given none. Under the clipped-gap policy the accepting last line is `WITHIN LIMITS` , not `YES` . 

Batches of 64 are not to be relied on with such a GPU: on synthetic weights, in the timed runs, one call of the prover on 64 requests and one attempt at a service of five such batches, one run each, ended at a failed CUDA call, the service in its first batch, with 0 of 64 and 0 of 320 requests proved; every attempt at 64 on synthetic weights on record failed, eight in all, six of them on other programs<sup>_†_</sup> (Appendix H.2). The three runs at 64 on the public weights completed, with all 960 proofs accepted and at most 80,629 MiB of 81,559 MiB in use. At exact match, in batches of 24, a proof file has 18.74 to 18.81 MB. 

### **7.3 The executable form on the prover’s proofs** 

At the format’s hash function, at the default configuration’s size and on synthetic weights, the executable form accepted (by `lean-verify` ; four of them by a separate build of the executable form alone, Table 4), in calls that ran to their end, every proof from an honest run of the prover counted here. At exact match 363 were made by one prover built from other sources<sup>_†_</sup> , the part handed over of the 723 that it made (not the 723 that the three timed runs on synthetic weights made, 241 a run): 360 with one output token (96 requests of one shape from one list, each proved under the same three commitments, 24 of them twice under each) and three of one stored instance. 

Eighteen were made by other provers than that one: seven of that instance, nine of three others, and two beside the malicious proofs of the true claim of Appendix H.3, one of them of one of those three. 

Under the clipped gap of _Z_ 1 it accepted one by that prover<sup>_†_</sup> and four by others, as reported, all of one further instance. 

A second set of 41 honest proofs, from the timed runs’ own files on synthetic weights, counted apart from the 109 calls of Appendix H.3, has been put to the Lean verifier: each of three runs’ files holds proofs numbered 000 to 119 and one more; taken were that one and 000, 009 and so on, to 108 in two runs (13 each) and to 099 in the third (12): 3 + 13 + 13 + 12 = 41; it accepted each, at exact match, one call each. These are other files than the 41 of the 363 that are among the 109 calls. Like the 109 calls, the calls on them were made over the released Lean files; the command had accepted each of them before, over Lean files that differ from the released ones; 7 of their 14 requests are among those of the 47 honest proofs of the 109. Appendix H.3 gives their provenance. 

Under the clipped gap `lean-verify` has met honest proofs alone; a separate build of the executable form rejected the two proofs of false claims of Table 4. 

`lean-verify` and a separate build of the executable form together rejected all 26 proofs of claims false by the run’s own policy (Table 4) — 24 and two — all from one test program, claims fixed beforehand, each rejected where the proof file is read. 

With the form of the command marked<sup>_†_</sup> in Section 7.1, the library built in advance, three calls<sup>_†_</sup> on one stored proof, made one at a time for the timing, one after another on one shared CPU machine (an Intel Xeon Platinum 8375C, 64 CPUs), each accepted it, in 1,393 to 1,440 seconds, with at most 14.13 GB of memory in a call’s largest process, in these three calls. What an evaluation trusts, how narrow these proofs are, and what a reader can repeat: Appendix H.3. 

42 

Table 4: Evaluations of the executable form. These calls were made over Lean files that differ from the released ones, but for the calls of the reference run and of the second set of 41, which were made over the released Lean files (Appendix H.3). A dagger marks one of the other programs: the one that made the input or, where it stands on a verifier, that verifier. 

|Input|Decisions|
|---|---|
|Honest proofs at exact match<sup>_†_</sup>|18 of 18 accepted, 2 of them by a separate<br>build of the executable form<sup>_†_ </sup>alone|
|Honest proofs under the clipped gap of _Z_1<br>_†_|4 of 4 accepted, 2 of them by a separate build<br>of the executable form<sup>_†_ </sup>alone|
|Honest proofs by one prover built from other sources<sup>_†_</sup>, as reported,<br>at exact match|363 of 363 accepted (360 with one output token,<br>of 96 requests, and three of one stored instance,<br>each covering 1,001 positions; the C++ verifier<br>of those runs accepted each first, as reported)|
|The same under the clipped gap|1 of 1 accepted (an acceptance that says less;<br>not blind)|
|Proofs of false claims<sup>_†_</sup>: 24 at exact match, 2 under the clipped gap|26 of 26 rejected, the 2 under the clipped gap<br>by a separate build of the executable form<sup>_†_</sup><br>alone|
|Altered inputs<sup>_†_</sup>|by a separate build of the executable form<sup>_†_</sup><br>alone: 32 of 41 rejected; 9 accepted, all altered<br>proofs of claims true by the run’s policy,<br>altered at Merkle-path entries that the<br>verifier’s walk does not take, by a reading (two<br>of them had been expected to be rejected:<br>`lean/GAPS.md`, gap 14; Remark B.12)|
|`lean-verify`<sup>_†_</sup>: 16 honest proofs at exact match, 2 under the clipped<br>gap, the 24 at exact match, 11 malicious proofs of the true claim<br>Also to a C++ verifier<sup>_†_</sup>, besides a separate build of the executable<br>form<sup>_†_</sup>: 43 files|16 and 2 accepted, 34 rejected, one unparsed,<br>without a decision<br>the same decision on each (at 3, the C++<br>verifier’s as reported)|
|Also to a C++ verifier built from other sources<sup>_†_</sup>: the 42 (the 41<br>below and the one file the form could not read), the 22 malicious<br>proofs of the true claim ofAppendix H.3with their two honest proofs<br>and two controls, the 364 and three controls|the same decision wherever the form gave one<br>(432 of 435 sets)|
|Also to another C++ verifier<sup>_†_</sup>: the 41 (seven honest proofs, the 24,<br>ten malicious proofs of the true claim; no altered inputs) and the<br>nine proofs of three instances|the same decision on each|
|Four matched alterations of one honest proof<sup>_†_</sup>, to both verifiers|the same decision at 1; at 3 a C++ verifier<sup>_†_</sup><br>rejected the raw proof, a separate build of the<br>executable form<sup>_†_ </sup>accepted the parsed one|
|Explicit committer (Definition 5.16; what ran: Remark 5.21) against<br>the committing program<sup>_†_</sup>, reduced configuration|digests, root, OOD answers and commitment<br>equal in two runs; one public seed, synthetic<br>weights, nothing else compared|



### **7.4 What has run under each policy** 

Exact match requires that the integer model itself be served: of the 96 greedy outputs of each of two floating-point engines (16-bit, FP8), none matched the integer model’s greedy token at every token position (scored without proofs<sup>_†_</sup> ). A vLLM plug-in is designed to serve it, one request at a time: not on the released programs<sup>_†_</sup> , on the public weights and one H100, beside a helper process, it took 0.58 to 0.63 seconds per token after prompts of at most 5 tokens, and 0.58 to 1.46 over all 48 outputs. Clipped-gap parameters fitted on honest greedy outputs accept most outputs with one token replaced (Table 5) and most sampled outputs, 86 of 96 at temperature 0.7 and 85 of 96 with top-p 0.9 (scored likewise, 16-bit fit) (Limitation 8.8, Clipped gap is weak). Serving and proving have not run together as one service (Appendix H.4). 

43 

Table 5: Runs by policy, on the public weights. The runs with proofs (the first row’s on other programs<sup>_†_</sup> , the second row’s among the timed runs) use the honest <u>prover</u> and test neither soundness nor weight hiding. 

|What ran|Outcome|
|---|---|
|Exact match: vLLM serves the integer model and is stopped; the<br>prover then proves every served token; three runs<sup>_†_</sup>|48 of 48 outputs, to the same 16 synthetic<br>prompts in each run, received a proof that the<br>C++ verifier accepted|
|Clipped gap: stored outputs of a floating-point engine (Table 3)|all proved; the 48 outputs were selected as<br>meeting the policy: no acceptance rate|
|Scored, no proofs<sup>_†_</sup>: one token replaced by the second choice; by a<br>random token|28 of 32 and 24 of 32 meet the policy (16-bit fit,<br>FP8 fit); 27 of 32 and 21 of 32|



### **7.5 Tests of weight hiding** 

Table 6: The five tests behind the judgement of Section 7.5, one run of each: at the reduced configuration, on synthetic weights, under exact match, on selected cases, so no rates (four further runs of the first, on a real model’s weights: Appendix H.5). None of the five searches for a leak: they test that the masks are there and that a loss is noticed. Every “stopped” was observed in a test program or, in the fourth test, in a program that held test-only code <u>(Appendix</u> H.5). 

|Test|Outcome|
|---|---|
|OOD answers recomputed by a separate<br>program|all 1,536 answers as prescribed by the rules that the test carries; no pad<br>occurs twice or is 0. That the pads are random it does not show.|
|Injected losses of the OOD answers’<br>masks<sup>_‡_</sup>, at all-zero weights|8 of 8 (none a loss of a single pad) stopped the committer, all by one<br>self-check; with it disabled<sup>_†_</sup>, at a parameter file that counted no proof of<br>work, an unmasked commitment passed every other check|
|Fault injection<sup>_‡_</sup>: 295 selected losses of<br>random values|GPU memory: 130 of 130 stopped, 122 by one self-check alone. CPU<br>memory: 123 of 165 stopped, 12 changed no byte, 30 unnoticed|
|Riskier losses<sup>_†_</sup>, outside that selection<br>(its setting: as reported)|263 of 263 stopped; 9 of 9 at neighbouring samples, which no self-check<br>covers, unnoticed|
|Both shipped audits, with the provider’s<br>secrets|accepted honest files; in trial calls, which make the same checks, rejected<br>three false secrets. They compare 7.66 %<sup>_†_ </sup>(at the default configuration’s<br>size) to 33.00 % of the bytes by value, and no value that depends on the<br>weights (run by run: `docs/limits.md`).|



Section 6 concerns the protocol; for the programs, their goal, weight hiding (Definition H.1), is tested. We judge that weight hiding likely holds at the default configuration, in honest runs, if the provider keeps its operating conditions and the judgement’s own conditions (listed in `docs/limits.md` , not in this paper) hold. This judgement is given of the released programs: the first and the fifth of its five tests (Table 6) have run on them, under exact match, with the outcomes that Table 6 and Appendix H.5 tell, both at the reduced configuration on synthetic weights and the first at the default configuration too, once on synthetic weights and once, as fitted to them, on the public weights; the fifth ran once. The second and the third ran on test programs and the fourth on other programs, all three built from sources that differ from the released ones. The one reading of whole files for hiding was of a text from which the released one differs. It took in 384 files: all that a production build of the committer and the prover from that text reads, but for the C++ verifier’s own 36. The released sources differ in code in 68 of the 384, and in 136 more in comments, names and the wording of messages alone; two more of the 384 are not in the released sources; in the 68 the differences in code were read for hiding, by difference alone, apart from renamed names and reworded messages, and the files were not read whole again (Appendix H.5). That judgement rests on the design, what is proved in Lean for the protocol (Section 6; what that leaves out: Limitation 8.2 and Limitation 8.3), code review and tests; it is not a proof, it does not cover a real model’s weights, and neither the programs nor the judgement has had an independent review. Only one short search has looked for a leak: in two sets of honest published files, 

44 

three proofs among them, at the reduced configuration, not on the released programs<sup>_†_</sup> . It made several looks: at the OOD answers, for guessable seeds, and over the opened columns of the stored tables; of the proofs it studied nothing else by value. It found nothing, which is worth little: on the first of these sets the first look that it made cannot see any fault of a pad. Beyond that search no proof has been searched for a leak, and no file at the default configuration. 

None of that judgement’s five tests searches for a leak: they test that the masks are there and that a loss is noticed. The tests behind it ran at the reduced configuration, on synthetic weights; only one of the five has also run on a real model’s weights, once on the released programs and three times on other programs<sup>_†_</sup> , at the default configuration as fitted to those weights. That judgement reaches, on less evidence, a configuration that differs from the default one in a clipped-gap policy’s three numbers alone. Its whole scope, its conditions, and what is not shown or untested: Appendix H.5. 

A _self-check_ is a program’s recomputation of masked values; an _audit_ is a shipped tool that recomputes part of the published bytes from the provider’s secrets. Of the fault injection’s three test programs two were made from a copy, in which four files differ, of the sources of other programs. In Table 6 losses were unnoticed in three places: 30 of 165 injected losses of random values in CPU memory; 9 of 9 at samples that no self-check covers; and, with one self-check disabled, at a parameter file that counted no proof of work, a commitment without masks, which passed every other check. 

In one trial, not on the released programs, on a weight file with a tensor of 16 bits, which that judgement does not cover, a party that could write that file but not read it learned the magnitudes of two weights of their choice (Limitation G.18, One trial with such a file). 

### **7.6 Negative tests of the verifiers** 

Table 7: Negative tests of the C++ verifier, on synthetic weights at the small example’s setting unless stated. 

|Test|Outcome|
|---|---|
|A malicious prover<sup>_†_</sup>, at the reduced form of the default con-<br>figuration|113 of 113 falsifications rejected, each beside the true<br>output|
|A malicious committer<sup>_†_ </sup>alters a committed table at 1, 16 or<br>256 of 8,388,608 positions|24 whole calls accepted, in each of two tests, with the<br>honest request and output|
|One table<sup>_†_ </sup>altered at 257 to 524,287 positions|45 of 80 commitments accepted, each within the<br>unique-decoding radius of the unaltered table|
|Not the verifier program: test programs<sup>_†_ </sup>that call the shared<br>C++ library beneath it directly|false claims accepted, against the library’s stated<br>contract, by at least two ways (Section 7.6)|
|Altered copies of honest files, altered calls; public weights,<br>default configuration|240 of 240 and 9 of 9 rejected; where the altered bytes<br>lie is not reported: no evidence on the check of the<br>OOD sample|



In no reported test did the C++ verifier program accept an output that violates the run’s own policy (Table 7 and Table 4); most of these tests ran not on the released programs. On proof files the released verifier’s record is chiefly this (the whole of it: Appendix H.1): it rejected all 57 that malicious provers made at the default configuration’s size, at exact match and under one commitment, on synthetic weights, and that were put to it, 24 of them claiming a false output and 33 the true output with a falsification inside; and, at the small example’s setting, the 228 acceptances on 57 kept sets of files in which an edited committer had altered one stored table (at 1 to 131,072 of its 8,388,608 positions; the four full checks on each set; six of the sets a second making of six others) all carry a true claim, by a reading, not machine-checked (the release’s `docs/limits-all-rows.tsv` , rows LIM-63.1 and SO-15.7; Appendix H.1). Few falsifications test whether a false claim can pass: each was fixed beforehand, and a rejection shows that one is caught, not that none passes. 

The fourth row of Table 7 is not a test of that program: there, test programs call the library beneath it directly, against the library’s stated contract ( `docs/limits.md` ): a list of rules for C++ code that calls it, among them to name no private member of the library, to declare nothing at global scope before its headers, and to keep `operator new` . They run on small inputs of their own, which are not configurations. At the 

45 

128-bit target, on these small inputs, they had false claims accepted by at least two ways: in one, at all 9 tries that were carried through; in another, on synthetic data, at 8 of 48 tries against a control of 0 of 48. There may be others. 

That the verifier program cannot reach the library’s open ways rests on readings of that program’s source, not on a test; the source read differs in places from the released one (Appendix H.1). 

The altered tables that were accepted decode, by separate programs’ counts of the changed columns (no decoder was run; Appendix H.1), to the unaltered weights, which the design allows: an accepted proof is meant to show that the committed table decodes to the weights, not that its bytes are their honest encoding. No malicious program was built from the released sources, and none searches for a false proof (Appendix H.6). 

### **7.7 The integer model against floating point** 

On the public weights the integer model’s greedy token differs from that of the reference, public copies of the 16-bit weights (two, the same bit for bit) evaluated in float32, at 6.81 % of token positions on 216 texts and at 9.41 % on the 24 natural passages among them (Python code<sup>_†_</sup> , no proofs; the same two shares, to the digits shown, in each of three runs). The count behind the first share was not the same in each run, and the cause is not known. All texts are built on WikiText-2 [MXBS17], on which the scales were fitted, and no task benchmark was run ( `docs/limits.md` ) (Limitation G.9, Quality unmeasured). 

### **7.8 Comparison** 

Per GPU, proving alone takes 71.4 times (65.2 to 76.4 over all pairs of runs), 93.0 times (92.2 to 94.1 over all pairs of runs) and 20.1 times (19.7 to 20.5 over all pairs of runs) the GPU time of stock vLLM [KLZ23] for prefill of 24 windows at once, prefill at the highest load tried, and decoding at 512 streams ( _n_ = 3; the engine’s _n_ = 30, 8, 5). The engine served the FP8 checkpoint on eight H100s, on other machines; the multiples are calculated from the first rate of Table 3, omit committing and the serving that exact match requires (Section 7.4), and could be higher, since the engine’s rates are lower bounds. 

Table 8 sets our figures beside those of four earlier systems. By a simulation of 17 such machines [GHL26, Table 6], DeepProve’s rate for GPT-2 rises to 1,855 tokens per minute. 

Llama 4 Maverick is a mixture of experts: every second layer has 128 routed experts, of which a token uses one, together with a shared expert. Of the 400.7 billion committed weights, one token’s forward pass uses 16.15 billion (computed from the parameter file; the whole output head and the one embedding row that a token reads are counted). Each request’s proof also does work over all 400.7 billion committed weights, once per request and not once per token. The rate is therefore not comparable, parameter for parameter, with the rates for dense models. 

## **8 Limitations and open problems** 

### **8.1 Limitations** 

The release lists the gaps that we know of, and no attack on the protocol is known to us at any gap while its conditions are kept, the opening budget among them (Limitation 8.4, Opening budget). Almost no attack was sought: that none is described is not a finding of safety. Table 16 names the release’s pages. 

Theorems and proofs concern the protocol, or Lean’s IOPs and _VZ_ ; measurements and tests concern the implementation. Nothing is proved about the released programs, and no security level is claimed for them. Between the theorem and the command `lean-verify` the gap is smaller than between the theorem and the C++ programs (Limitation 8.1 (Lean verifier only); Limitation 8.2 (Protocol match unproved)): `lean-verify` evaluates the proved verifier’s executable form (Remark 5.9, What a client gets). 

_Limitation_ 8.1 (Lean verifier only). Theorem 5.1 concerns _VZ_ , not the C++ verifier, which clients run by default. Nothing is proved about that program, and no soundness is claimed for the programs as they run: an acceptance has the theorem’s meaning only if _VZ_ gives it, and neither 124 bits per query nor 2<sup>_−_128</sup> per round is stated of the C++ programs. No theorem states that the two decide alike on any file, or bounds a program’s probability of accepting a false claim. 

46 

Table 8: Proofs of language-model inference: each system’s results as its authors report them, without ratios, because the machines, models and <u>proved</u> statements differ. 

|System, model, input|Hardware|Proving|Proof|Verifying|Statement|
|---|---|---|---|---|---|
|This work (Table 3):<br>integer model of the size<br>of Llama 4 Maverick<br>(400.7B weights, mixture<br>of experts; 16.15B used<br>per token); 1,024 tokens,<br>batches of 24|one H100<br>(80 GB),<br>with 24<br>CPU cores<br>and up to<br>278.3 GB of<br>host memory<br>(Section 7.1)|111.23 ± 0.92<br>tokens/second<br>(n = 3),<br>counted in<br>prefill tokens;<br>9.206 ± 0.077<br>seconds per<br>request (n =<br>3), as<br>throughput;<br>released 220.7<br>to 225.9 s<br>after the<br>batch’s start;<br>committing<br>apart, about<br>39 minutes|18.74 to 18.81 MB|21.14 ± 0.03<br>seconds called<br>with its option<br>for one thread,<br>on the prover’s<br>machine (n = 3),<br>by the C++<br>verifier|non-interactive; the<br>greedy token that<br>follows the prompt|
|VerInf [VerInf], as its<br>README reports: its<br>integer model of Llama 4<br>Maverick; 1,000 tokens|one DGX<br>Spark|14.3 h; its<br>README<br>leaves open<br>whether that<br>includes<br>committing|93.6 GB at 40<br>opened columns,<br>below the 80 that its<br>README calls<br>deployment grade|17.7 h on 20<br>cores|interactive; a bound<br>on the “unexplained<br>information” of a<br>continuation|
|zkLLM [SLZ24]:<br>OPT-13B; LLaMA-2-13B;<br>2,048 tokens|one A100|713 s; 803 s;<br>committing<br>apart, about<br>1,270 s; 986 s|160 to 188 kB|3.71 s; 3.95 s|interactive; fixed<br>point, scaling factor<br>2<sup>16</sup>|
|DeepProve [GHL26,<br>Table 2]: GPT-2 (124M);<br>Gemma 3 (270M); 512<br>tokens; BaseFold<br>commitment|one 16-core<br>AMD<br>Ryzen 9, 128<br>GB|2.94 min; 5.9<br>min (174.32;<br>86.76 tokens<br>per minute),<br>without the<br>inference and<br>without<br>committing (a<br>setup of under<br>0.91; under<br>2.64 min)|24.33; 43.6 MiB|1.65 s; 2.13 s; its<br>abstract gives 1<br>to 3.7 s|all generated tokens;<br>12-bit quantization|
|Attestable [Attestable],<br>by the passages of its<br>blog post that we have<br>read, as first results:<br>Gemma 4 31B|one H100|53 tokens per<br>second on one<br>sequence of<br>16K tokens; 77<br>at batch four,<br>across four of<br>4K|4.35 to 7.92 MiB,<br>over all its successful<br>benchmark<br>configurations|157 to 648 ms<br>on its reference<br>CPU verifier|floating point, but<br>linear operations in<br>8-bit integers; 100<br>bits of security|



_Limitation_ 8.2 (Protocol match unproved). That Lean’s IOPs (Remark B.7, Which compiler) are the protocol of the C++ programs is not proved, nor that they are the technical report’s [TP]: no theorem relates them, and the only evidence is the runs. In the one comparison of hash calls, on the stored proof at _Z_ 0, a logging build of the C++ verifier<sup>_†_</sup> made 881,280 of the executable form’s 1,166,983 calls, with equal answers; the rest are explained only in part, a gap that the release rates as heavy. On that proof both decide alike; on three altered copies of an honest proof (one bit turned in a digest the form never reads) the executable form accepted the parsed file where a C++ verifier and a parser, both of other programs, rejected the raw proof; the claim was true ( `lean/GAPS.md` , gap 66). These calls were made over Lean files that differ from the released ones. Over the released Lean files the decision on the stored proof has been repeated (Appendix H.3); the comparison of hash calls and the calls on the altered copies have no run over the released Lean files. _VZ_ has sixteen tests that the C++ verifier lacks ( `lean/GAPS.md` , gap 25). 

47 

_Limitation_ 8.3 (Zero knowledge of definitions in Lean). The provers for which zero knowledge is machinechecked (Section 6.5) are the Lean provers: definitions in Lean, exactly as written, default values included. On the C++ programs nothing is stated. It is stated at the default configuration’s size, after compilation at the exact-match policy, with fresh random masks, for one commitment and one request per proof, not for batches; before compilation for an honest verifier; after compilation in the random-oracle model, on one copy of the committed table, up to 688,127 replies and 2<sup>64</sup> hash queries. All of it holds where _Z_ is admissible with Chk( _Z_ ) = 1, after compilation in use (Definition 6.11). Not hidden is one bit per request, before and after compilation: whether the request is served, which depends on the weights; in the corollary, one label per request (served, failed or other tokens). A bit says whether the claimed output is the model’s result on that request, so over many requests the bits tell what the model answers. Nothing is said of what the files of _Z_ themselves tell of the weights, or of timing. The simulators’ running time is not modelled ( `docs/limits.md` ). Nor is derived randomness covered (Remark G.15, Pseudorandom masks). At a refused request the Lean prover returns the empty proof. The programs’ service answers “failed” for every request that it does not prove, whatever the cause; a client can partly tell the causes apart ( `docs/limits-all-rows.tsv` , row HW-532). That answer is neither the bit nor the corollary’s label “failed”, and no theorem ties them. That the released programs hide the weights is the design’s aim, not an established property (Section 1.1); our judgement of it: Section 7.5. 

_Limitation_ 8.4 (Opening budget). A committed table can open only _t_ columns before its random coefficients are used up, with no margin (Proposition 6.5). Beyond that, its proofs give whoever holds them exact linear equations, permanently and unnoticed: _t_ + _k_ columns, 1 _≤ k ≤ d_ 2<sup>_n_0</sup> , give _k_ independent equations modulo _p_ in each row’s weight digits and pads and, for _k >_ 3 _s_ , at least _k −_ 3 _s_ in its digits alone (Proposition C.16). By a reading of other sources, a row that the prover keeps from one segment of a request’s proof to the next is encoded anew for each segment, and each encoding has 219 random coefficients, for the 219 positions that a proof opens in it, without slack: a number calculated from the default configuration and taken from no run; that it is enough is not proved, and it is enough only if each new encoding’s random coefficients are fresh, which by the code’s own list of conditions no function there tests. A commitment serves boundedly many batches. The released programs count batches on each copy, at most 688,127 at the default configuration (calculated), and no program counts proofs (Limitation 8.2, Protocol match unproved). Behind the programs’ counter no Lean theorem stands: nothing in Lean keeps count of the columns that a table has opened. The 688,127 of the compiled zero-knowledge result (Section 6.5) counts another thing: replies with one request to a proof, on one copy of the committed table. So a provider within the programs’ count is not thereby within that result. No program detects a count copied, restored or rolled back ( `docs/limits.md` ). 

_Limitation_ 8.5 (Provider-chosen configuration). The provider, not the verifier, chooses the configuration: all three files for the programs, and for Theorem 5.1 whatever may vary in an admissible configuration. Theorem 5.1 needs it fixed in advance, before the oracle is sampled and independently of the adversary (Open problem 8.16); one tuned to the concrete hash function is outside Theorem 5.1. The C++ verifier accepts forms of configuration that no statement covers, some of the release’s examples among them. 

_Limitation_ 8.6 (Scales can disable blocks). The scale file of an admissible configuration can disable any chosen subset of the model’s 96 attention and feed-forward blocks, up to all of them, whatever the weights, so the client must read it. This was worked out by hand and is not a theorem. 

_Limitation_ 8.7 (Unattributed commitment). As in every commit-and-prove argument, a commitment does not say whose weights, or what model’s, were committed: the provider chooses which weights, if any, it fixes, and a client has only the word of whoever published it. A client needs the commitment before it sends a prompt. 

_Limitation_ 8.8 (Clipped gap is weak). Under a clipped-gap policy an acceptance states only “within limits”, which is much less than exact match (unless _θ_ = 2<sup>8</sup> _< C_ , where an output meets the policy if and only if it meets exact match). Otherwise, meeting the policy bounds the sum of the counted tokens’ clipped costs and the number of them that cost more than the clip, not each token (Remark 3.10, What a policy bounds), and, by design, does not show whether any given counted token is the greedy token. With _C_ = _θ_ the bound on the sum says nothing; if _cap_ is also at least the number of counted tokens, as at (256 _,_ 256 _,_ 1024), the count says nothing either. _VZ_ does not reject such a configuration, and there only the token after a full window is certified ( `lean/THEOREM.md` ). 

48 

_Limitation_ 8.9 (Completeness on served requests only). Lean proves soundness of _VZ_ ; of its completeness it proves, besides the reduction of Remark 6.49, only that, on served requests, it rejects the Lean prover’s proof with probability at most 2<sup>_−_40</sup> , in the random-oracle model, where the files of _Z_ are in use ( _Z_ admissible, Chk( _Z_ ) = 1, exact-match policy), for one request to a proof and one commitment (Section 6.5). Where the model has no result, or another result than the claimed output, nothing is stated. No honest proof exists for a window on which the model is not defined. The Lean prover is a definition: that _VZ_ accepts the programs’ honest proofs is tested, on proofs made on other programs and on 41 of the timed runs’ own on synthetic weights, at the two stored configurations _Z_ 0 and _Z_ 1 alone (Remark B.6, Acceptance), not proved. These calls were made over Lean files that differ from the released ones, but for the calls of the reference run and of the second set of 41, which were made over the released Lean files (Appendix H.3). Nor does Theorem 5.11 show that _VZ_ accepts any non-greedy output. A gap in completeness could only cost a visible rejection: it cannot make _VZ_ accept a false claim beyond the bound of Theorem 5.1, under its hypotheses, provided the functions that form the claim are right (Remark B.10, Claim from files). 

_Limitation_ 8.10 (Released commitment). The released commitment has 32 bytes and hashes the three configuration files, the first table’s Merkle root, its out-of-domain answers and the out-of-domain seed. That it ties the provider to one set of weights is not proved for the released programs: what is proved of a commitment is proved of the committer compiled from the Lean provers (Section 6.5) and of an explicit committer that models part of Commit _Z_ (Open problem 8.11). Of a commitment a proof can certify at most the matrix to which the first table decodes, not that the table’s bytes are its honest encoding, and the commitment does not fix the offline proof in the key file: one set of weights can have several commitments, one commitment several key files, and a proof is tied to its copy. 

Appendix G states the further limitations: the implementation is unverified; only greedy outputs in a fixed window are covered, and prefixes count; the hash function is modelled as a random oracle, and adversaries are classical; there is no independent review, there are few runs, and the model’s quality is unmeasured; outcomes are visible by design; and eleven limitations concern how the released programs are operated. Among the eleven: a further commitment to the same weights, which renewal or a second GPU requires, is inside the judgement of Section 7.5 only on thin ground and on five conditions, and no test has searched two commitments together for leaks (Limitation G.12, Further commitments). 

### **8.2 Open problems** 

**Open problem 8.11 (Extraction and binding for the released committer).** [open] Prove correctness of extraction and extractability of the commitment (Definition 2.9 and Definition 2.10) for Commit _Z_ of the released programs. For the committer compiled from the Lean provers both are proved (Section 6.5). The hash function is drawn at random: these are statements in the random-oracle model. They hold at public files in use; the extractor is a function, not shown computable. The committer is a Lean term that no theorem ties to a program. For Commit _Z_ of the released programs nothing is proved: the explicit committer (Definition 5.16) models only part of it, and was compared, in values alone, with the committing program of other programs<sup>_†_</sup> (Table 4). 

Each property that names the honest parties requires its own proof (Remark 4.16, Honest parties). 

**Open problem 8.12 (Completeness beyond served requests).** [open] State and prove completeness of the compiled term, _VZ_ with the Lean committer and prover (Section 6.5; Remark 4.16 (Honest parties)), at well-formed requests on which the model has no result, and prove it for batches. 

**Open problem 8.13 (The twenty conditions as a theorem).** [open] State, as a theorem of its own, that the three Lean provers (Section 6.5) meet the twenty conditions of Theorem C.25. 

Zero knowledge of the Lean provers is machine-checked (Section 6.5). For the results on the whole-key argument the step to Π _Z_ is not proved; the compiled statement of Section 6.5, which is on Π<sup>lp</sup> _Z_<sup>(Definition 6.15),</sup> needs no such step for the commitment, which is the 256-bit one, or for the verifier; for the honest parties of Π _Z_ the step is not proved (Open problem 8.14). Batches are not covered. 

49 

**Open problem 8.14 (Zero knowledge of the whole protocol).** [open] Prove index privacy of Π _Z_ with its honest parties, for batches, on the instances that can be served. 

**Open problem 8.15 (Many proofs without the factor** _N_ **).** [open] Find _qH_ such that every run of _VZ_ makes at most _qH_ queries to _H_ . Bound the error for _N_ proofs sent one after another, each after the decision on the one before. 

Given such a _qH_ , which no theorem provides, the bound _ε_ ( _q_ + _NqH_ ), without the factor _N_ , is machinechecked for lists sent at once (Remark B.13, Lists sent at once). 

**Open problem 8.16 (A configuration chosen after the oracle).** [open] Prove index-binding knowledge soundness (Definition 2.7) of Π _Z_ against a committer that chooses the admissible configuration _Z_ after querying _H_ . 

## **9 Related work** 

**Proofs of language-model inference.** zkLLM [SLZ24] is interactive, uses a Hyrax commitment [WTSTW18] and hides the parameters from a semi-honest verifier. Our integer softmax and attention follow zkLLM, zkGPT [QSL25] and ZKML [CWS24]. zkAgent [WLL26] checks the greedy token by range-checking the logit differences over the whole vocabulary, as zkCNN [LXZ21, Section 4.3] does for max pooling; at exact match our protocol’s last segment does the same, by looking up the digits of the differences. DeepProve [GHL26] proves all generated tokens of an output, for GPT-2 and Gemma 3 models of 124M and 270M parameters. ZKTorch [CTK25] compiles a model into basic blocks and accumulates their proofs. ZK-DeepSeek [Wan25] turns DeepSeek-V3, with 671 billion parameters, into an integer model, composes proofs recursively into one of constant size, without a trusted setup, and reports times for several components. By its abstract, which is all that we have read of it, zkPyTorch [XLF25] compiles PyTorch models into circuits. By the passages of its blog post that we have read, Attestable [Attestable] reports early results for a hash-only zero-knowledge system over committed weights. 

**Proofs for the same model.** VerInf [VerInf], which precedes this work, reports in its README an interactive, hash-based zero-knowledge proof, on one DGX Spark and 1,000 tokens, for its own integer model of Llama 4 Maverick, the model whose shape we use. It proves a different statement from ours, a bound on the “unexplained information” of a continuation, and it hides both the tokens and the weights. Our choice of public 4-bit weight files [Uns25] comes from VerInf, and our rule for converting the stored weights to integers grew out of VerInf’s. 

**One committed model, many proofs.** In the literature’s terms the task of Π _Z_ is close to that of a function-hiding functional commitment [BNO21]: the committed function is the model, and Corollary B.14 has the form of evaluation binding. We state the likeness only; their definition has not been compared with ours clause by clause. [GGQZ25] builds such a commitment over KZG for convolutional networks, and zkCNN [LXZ21, Section 4.1] already has the prover commit to a network’s parameters before proving predictions. By the passages that we have read, Artemis [LVAKH24] is a commit-and-prove SNARK for any homomorphic polynomial commitment, and reports cutting the overhead of tying proofs of inference to the model’s commitment. 

**Attacks on committed models.** Nothing in FlashProve prevents the Hollow-LLM attack [GLL26], in which weights of the declared size yield valid proofs but compute no more than a much smaller model (Limitation 8.7, Unattributed commitment). [CPD24] recovers a model’s last layer, up to a linear map, from many vectors of its logits; whether vectors cut off at the clip suffice is not known, and as far as we know this has not been tried on these programs (Limitation G.21, Clipped gap reveals logits). 

**Alternatives to proofs.** A trusted execution environment has a reported overhead below 7% on Hopper GPUs for most typical queries [ZYDAZ24], but the client must trust the hardware and its vendor. TOPLOC [TOPLOC] hashes intermediate activations into 258 bytes per 32 new tokens, and its check is 

50 

reported to be much faster than the inference. Token-DiFR [DiFR] scores the tokens themselves, at no added cost to the provider. In both, the verifier runs the model again, so it must hold the weights. Our clipped-gap policy builds on Token-DiFR’s logit margin at temperature 0, but our verifier needs no weights (Remark 3.10, What a policy bounds). 

**Reported numbers.** Table 8 sets our figures beside those of four earlier systems, VerInf (as its README reports them), zkLLM, DeepProve and Attestable. Appendix E.1 and Appendix E.2 give the sources of our masks, segments, batches and policies. 

## **Who wrote and checked this** 

Under human direction, AI systems wrote this paper. The software and the Lean code are also almost entirely their work. The exceptions include code that they adapted from public sources, among them ArkLib and Mathlib (Appendix E.4) and the BLAKE2 reference code [BLAKE2]; the release’s `THIRD_PARTY.md` , `NOTICE` and `lean/CREDITS.md` name the sources and what was taken. AI systems also wrote nearly all of the technical report [TP], except where it marks words as taken from a source. 

For a machine-checked statement, Lean’s kernel accepts a proof of each Lean statement that Appendix F names for it, with two provisos: that kernel is the pinned version’s, of which Remark B.17 tells what is publicly known, and the re-checker stopped, for the depth of recursion, on constants in 23 modules of a separate library (counted over Lean files that differ from the released ones) which the build’s kernel accepts at a thread stack larger than the default (Remark 5.7, How the proposition is checked); Remark 6.48 names the statements of Section 6.5 whose modules import that library. The kernel does not check what a statement means. That a Lean statement says what this paper’s statement says is a reading; nothing machine-checks it. Six definitions of properties in the library ( `Sound` , `SoundAndBinding` , `SoundOpened` , `Complete` , `ZeroKnowledge` , `CompleteSoundZeroKnowledge` ; `lean/VerifiableInference/Statements.lean` ) were shaped with a person. The library’s six top theorems, which the release’s page `lean/STATEMENTS.html` shows first, are `sound` , `soundAndBinding` , `completeOnCanServe` , `zeroKnowledgeOnCanServe` , `refusedHandsOutNoProof` and `repliesFit` ; Appendix F names them under Remark 6.16, Theorem 6.31, Theorem 6.29, Theorem 6.35, and Proposition 6.33. The first two are stated through the first two of these definitions. The third and the fourth are stated through this library’s `CompleteOn` and `ZeroKnowledgeOn` , the texts of `Complete` and `ZeroKnowledge` , each with one condition more, that the request can be served (Definition 3.11); with the condition that the output is the model’s, these are the served requests. The last two are this library’s own, and so are the system, the bounds and the conditions on the files in all six. The library also has the _final statement_ , one Lean structure with six fields ( `lean/AUDIT.md` ), all proved by one Lean theorem; Appendix F names it under Theorem 6.29, Proposition 6.33, and Corollary 6.38. `completeOnCanServe` and `zeroKnowledgeOnCanServe` are no fields of it. 

AI systems made every test, run, comparison, calculation, reading and review that this paper reports: where it says “by a reading”, “by hand” or “as reported”, the reader, calculator or reporter is an AI system. Such checking as there was, of the match between the English statements and the Lean ones and of the written proofs, tests and judgements, was made by AI readers of this project; no written proof has been read whole by one reader, and, as far as we know, no person has checked any of these (Limitation G.7, No independent review). 

**Acknowledgments:** We thank Alessandro Chiesa, Giacomo Fenzi, Guy Weissenberg, Miranda Christ, Rahul Dalal, and Ben Mayer for their input and collaboration on FlashProve. We also thank James Petrie for VerInf [VerInf]. 

## **References** 

- [ABF26] G. Arnon, D. Boneh, G. Fenzi. Open Problems in List Decoding and Correlated Agreement. Cryptology ePrint Archive 2026/680 (print dated April 8, 2026). 

[ACFY24a] G. Arnon, A. Chiesa, G. Fenzi, E. Yogev. STIR: Reed-Solomon proximity testing with fewer queries. CRYPTO 2024; Cryptology ePrint Archive 2024/390 (sections as in the version of January 27, 2025). 

51 

- [ACFY24b] G. Arnon, A. Chiesa, G. Fenzi, E. Yogev. WHIR: Reed-Solomon proximity testing with super-fast verification. EUROCRYPT 2025; Cryptology ePrint Archive 2024/1586 (sections as in the version of November 21, 2024). 

- [ACY22] G. Arnon, A. Chiesa, E. Yogev. A PCP theorem for interactive proofs and applications. EUROCRYPT 2022; Cryptology ePrint Archive 2021/915. 

- [AER24] G. Angeris, A. Evans, G. Roh. A note on Ligero and logarithmic randomness. Cryptology ePrint Archive 2024/1399. 

- [AHIV17] S. Ames, C. Hazay, Y. Ishai, M. Venkitasubramaniam. Ligero: lightweight sublinear arguments without a trusted setup. ACM CCS 2017; Cryptology ePrint Archive 2022/1608. 

- [ArkLib] ArkLib Contributors. ArkLib. A public library for the proof assistant Lean (Apache-2.0), github.com/ Verified-zkEVM/ArkLib; commit 2817d8c7 (Appendix E.4). 

- [Attestable] Attestable. Proving LLMs at scale. Blog post, attestable.com/blog/proving-llms-scale (2026). [BCFGRS17] E. Ben-Sasson, A. Chiesa, M. A. Forbes, A. Gabizon, M. Riabzev, N. Spooner. Zero knowledge protocols from succinct constraint detection. TCC 2017, pp. 172–206. 

- [BCFW25] B. Bünz, A. Chiesa, G. Fenzi, W. Wang. Linear-time accumulation schemes. TCC 2025; Cryptology ePrint Archive 2025/753 (version of May 28, 2025). 

- [BCGJM18] J. Bootle, A. Cerulli, J. Groth, S. Jakobsen, M. Maller. Arya: nearly linear-time zero-knowledge proofs for correct program execution. ASIACRYPT 2018; full version: Cryptology ePrint Archive 2018/380. 

- [BCGM25] S. Bordage, A. Chiesa, Z. Guan, I. Manzur. All polynomial generators preserve distance with mutual correlated agreement. CCC 2026; Cryptology ePrint Archive 2025/2051, version of February 6, 2026. 

- [BCHKS25] E. Ben-Sasson, D. Carmon, U. Haböck, S. Kopparty, S. Saraf. On proximity gaps for Reed-Solomon codes. Cryptology ePrint Archive 2025/2055, version of November 6, 2025; the full version of [BCHKS26]. 

- [BCHKS26] E. Ben-Sasson, D. Carmon, U. Haböck, S. Kopparty, S. Saraf. On proximity gaps of Reed-Solomon codes. STOC 2026, pp. 1157–1167; doi 10.1145/3798129.3800827. 

- [BCIKS20] E. Ben-Sasson, D. Carmon, Y. Ishai, S. Kopparty, S. Saraf. Proximity gaps for Reed–Solomon codes. FOCS 2020; full version: Cryptology ePrint Archive 2020/654, version of July 3, 2021. 

- [BCRSVW18] E. Ben-Sasson, A. Chiesa, M. Riabzev, N. Spooner, M. Virza, N. P. Ward. Aurora: Transparent Succinct Arguments for R1CS. EUROCRYPT 2019; Cryptology ePrint Archive 2018/828 (print dated May 8, 2019). 

- [BCS16] E. Ben-Sasson, A. Chiesa, N. Spooner. Interactive oracle proofs. TCC 2016-B; Cryptology ePrint Archive 2016/116. 

- [BFRW25] B. Bünz, G. Fenzi, R. D. Rothblum, W. Wang. TensorSwitch: nearly optimal polynomial commitments from tensor codes. CRYPTO 2026; Cryptology ePrint Archive 2025/2065, version of November 15, 2025. 

- [BGKS20] E. Ben-Sasson, L. Goldberg, S. Kopparty, S. Saraf. DEEP-FRI: sampling outside the box improves soundness. ITCS 2020; Cryptology ePrint Archive 2019/336. 

- [BGKTTZ23] A. Block, A. Garreta, J. Katz, J. Thaler, P. Tiwari, M. Zając. Fiat-Shamir Security of FRI and Related SNARKs. ASIACRYPT 2023; Cryptology ePrint Archive 2023/1071. 

- [BGTZ23] A. Block, A. Garreta, P. Tiwari, M. Zając. On soundness notions for interactive oracle proofs. Cryptology ePrint Archive 2023/1256. 

- [BLAKE2] BLAKE2 reference source code package (reference C implementations): blake2b-ref.c, blake2.h, blake2-impl.h. Copyright 2012, Samuel Neves. github.com/BLAKE2/BLAKE2. Used under CC0 1.0. 

- [BNO21] D. Boneh, W. Nguyen, A. Ozdemir. Efficient functional commitments: how to commit to a private function. Cryptology ePrint Archive 2021/1342. 

- [CCHLRR18] R. Canetti, Y. Chen, J. Holmgren, A. Lombardi, G. Rothblum, R. Rothblum. Fiat-Shamir From Simpler Assumptions. Cryptology ePrint Archive 2018/1004 (print dated October 23, 2018). 

- [CF24] A. Chiesa, G. Fenzi. zkSNARKs in the ROM with Unconditional UC-Security. TCC 2024; Cryptology ePrint Archive 2024/724 (print dated September 4, 2024). 

- [CFQ19] M. Campanelli, D. Fiore, A. Querol. LegoSNARK: Modular Design and Composition of Succinct Zero-Knowledge Proofs. ACM CCS 2019; Cryptology ePrint Archive 2019/142. 

52 

|[CFS17]|A. Chiesa, M. A. Forbes, N. Spooner. A zero knowledge sumcheck and its applications. Cryptology ePrint<br>Archive 2017/305.|
|---|---|
|[CFW26]|A. Chiesa, G. Fenzi, G. Weissenberg. Zero-knowledge IOPPs for constrained interleaved codes. Cryptology<br>ePrint Archive 2026/391. Places are those of the version of February 25, 2026.|
|[CFWpc]|A. Chiesa, G. Fenzi, G. Weissenberg. Personal communication, 2026.|
|[CGSY23]|A. Chiesa, Z. Guan, S. Samocha, E. Yogev. Security Bounds for Proof-Carrying Data from Straightline<br>Extractors. TCC 2024; Cryptology ePrint Archive 2023/1646 (print dated September 11, 2024).|
|[CHMMVW|19] A. Chiesa, Y. Hu, M. Maller, P. Mishra, P. Vesely, N. Ward. Marlin: Preprocessing zkSNARKs with<br>Universal and Updatable SRS. EUROCRYPT 2020; Cryptology ePrint Archive 2019/1047 (print dated<br>January 1, 2020).|
|[CK11]|M. Chase, S. Kamara. Structured Encryption and Controlled Disclosure. ASIACRYPT 2010; full version:<br>Cryptology ePrint Archive 2011/010.|
|[CMS19]|A. Chiesa, P. Manohar, N. Spooner. Succinct arguments in the quantum random oracle model. TCC<br>2019; Cryptology ePrint Archive 2019/834 (definitions as in the version of January 14, 2020).|
|[CO25]|A. Chiesa, M. Orrù. A Fiat–Shamir Transformation From Duplex Sponges. Cryptology ePrint Archive<br>2025/536 (version of March 27, 2026).|
|[COS20]|A. Chiesa, D. Ojha, N. Spooner. Fractal: post-quantum and transparent recursive proofs from holography.<br>EUROCRYPT 2020; Cryptology ePrint Archive 2019/1076.|
|[CPD24]|N. Carlini, D. Paleka, K. Dvijotham, T. Steinke, J. Hayase, A. F. Cooper, K. Lee, M. Jagielski, M. Nasr,<br>A. Conmy, I. Yona, E. Wallace, D. Rolnick, F. Tramèr. Stealing part of a production language model.<br>ICML 2024 (PMLR 235); arXiv:2403.06634.|
|[CSZS25]|W. Cai, T. Shi, X. Zhao, D. Song. Are you getting what you pay for? Auditing model substitution in<br>LLM APIs. arXiv:2504.04715 (2025).|
|[CTK25]|B.-J. Chen, L. Tang, D. Kang. ZKTorch: compiling ML inference to zero-knowledge proofs via parallel<br>proof accumulation. arXiv:2507.07031 (2025).|
|[CWS24]|B.-J. Chen, S. Waiwitlikhit, I. Stoica, D. Kang. ZKML: an optimizing system for ML inference in<br>zero-knowledge proofs. EuroSys 2024.|
|[CY24]|A. Chiesa, E. Yogev. Building cryptographic proofs from hash functions. 2024. https://snargsbook.org.<br>Section numbers are those of Version 1.2 and may differ from those of the posted version.|
|[DG24]|B. E. Diamond, A. Gruen. Proximity gaps in interleaved codes. IACR Communications in Cryptology,<br>2025; Cryptology ePrint Archive 2024/1351.|
|[DiFR]|A. Karvonen, D. Reuter, R. Rinberg, L. Marks, A. Garriga-Alonso, K. Warr. DiFR: inference verification<br>despite nondeterminism. arXiv:2511.20621 (2025).|
|[DP23]|B. E. Diamond, J. Posen. Proximity testing with logarithmic randomness. IACR Communications in<br>Cryptology, 2024; Cryptology ePrint Archive 2023/630.|
|[Fen26]|G. Fenzi. How to prove more false statements: Fiat-Shamir limitations on (generated) R1CS. Cryptology<br>ePrint Archive 2026/1838.|
|[GGQZ25]|Y. Guo, Z. Guo, W. Qu, J. Zhang. Architecture-private zero-knowledge proof of neural networks.<br>Cryptology ePrint Archive 2025/2211.|
|[GHL22]|C. Gentry, S. Halevi, V. Lyubashevsky. Practical non-interactive publicly verifiable secret sharing with<br>thousands of parties. EUROCRYPT 2022; full version: Cryptology ePrint Archive 2021/1397.|
|[GHL26]|N. Gailly, I. Hishon-Rezaizadeh, T. Liu, N. Mainardi, D. Papadopoulos, C. Papamanthou, C. Pappas, S.<br>Srinivasan, Z. Youell, Y. Zhang. DeepProve: verifiable end-to-end large language model inference. ACM<br>CCS 2026; Cryptology ePrint Archive 2026/1112, version of May 30, 2026.|
|[GKR08]|S. Goldwasser, Y. T. Kalai, G. N. Rothblum. Delegating computation: interactive proofs for muggles.<br>STOC 2008.|
|[GLL26]|C. Gong, B. Liu, M. Li. Hollow-LLM attack: computationally trivial weights in zero-knowledge verification<br>of LLM inference. IEEE Symposium on Security and Privacy 2026; arXiv:2607.28884.|



53 

|[GLSTW23]|A. Golovnev, J. Lee, S. Setty, J. Thaler, R. S. Wahby. Brakedown: linear-time and field-agnostic SNARKs<br>for R1CS. CRYPTO 2023; full version: Cryptology ePrint Archive 2021/1043.|
|---|---|
|[Gru24]|A. Gruen. Some improvements for the PIOP for ZeroCheck. Cryptology ePrint Archive 2024/108.|
|[GW20]|A. Gabizon, Z. J. Williamson. plookup: a simplified polynomial protocol for lookup tables. Cryptology<br>ePrint Archive 2020/315.|
|[Hab25]|U. Haböck. A note on mutual correlated agreement for Reed-Solomon codes. Cryptology ePrint Archive<br>2025/2110.|
|[Hol19]|J. Holmgren. On Round-By-Round Soundness and State Restoration Attacks. Cryptology ePrint Archive<br>2019/1261.|
|[Jo26]|S. Jo. Interleaving stability for mutual correlated agreement and curve decodability. Cryptology ePrint<br>Archive 2026/891.|
|[KLZ23]|W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, I. Stoica.<br>Efficient memory management for large language model serving with PagedAttention. SOSP 2023;<br>arXiv:2309.06180.|
|[KRS25]|D. Khovratovich, R. D. Rothblum, L. Soukhanov. How to prove false statements: practical attacks on<br>Fiat-Shamir. CRYPTO 2025; Cryptology ePrint Archive 2025/118.|
|[LFKN92]|C. Lund, L. Fortnow, H. Karloff, N. Nisan. Algebraic methods for interactive proof systems. Journal of<br>the ACM 39(4), 1992.|
|[LNS21]|V. Lyubashevsky, N. K. Nguyen, G. Seiler. Shorter lattice-based zero-knowledge proofs via one-time<br>commitments. PKC 2021; full version: Cryptology ePrint Archive 2020/1448.|
|[LVAKH24]|H. Lycklama, A. Viand, N. Avramov, N. Küchler, A. Hithnawi. Artemis: efficient commit-and-prove<br>SNARKs for zkML. arXiv:2409.12055 (2024).|
|[LXZ21]|T. Liu, X. Xie, Y. Zhang. zkCNN: zero knowledge proofs for convolutional neural network predictions<br>and accuracy. ACM CCS 2021; Cryptology ePrint Archive 2021/673.|
|[LY26]|Y. Lim, S. Yong. Seal, then sample: sampled layerwise proofs for verifiable LLM inference from GPT-2<br>to 70B. arXiv:2609.27367 (2026).|
|[Mathlib]|Mathlib. A public library for the proof assistant Lean (Apache-2.0), github.com/leanprover-community/<br>mathlib4; its files name their own authors. The Lean library builds on one pinned revision of it (commit<br>5e932f97); 13 files with 18 declarations adapted from a later one (tag v4.34.0) come with this repository<br>(lean/ThirdParty/MathlibBackport/).|
|[MXBS17]|S. Merity,<br>C. Xiong,<br>J. Bradbury,<br>R. Socher. Pointer sentinel mixture models. ICLR 2017;<br>arXiv:1609.07843.|
|[QSL25]|W. Qu, Y. Sun, X. Liu, T. Lu, Y. Guo, K. Chen, J. Zhang. zkGPT: an efficient non-interactive<br>zero-knowledge proof framework for LLM inference. USENIX Security 2025; Cryptology ePrint Archive<br>2025/1184.|
|[RFC7693]|M.-J. Saarinen (ed.), J.-P. Aumasson. The BLAKE2 Cryptographic Hash and Message Authentication<br>Code (MAC). RFC 7693, RFC Editor, November 2015.|
|[RFC8439]|Y. Nir, A. Langley. ChaCha20 and Poly1305 for IETF protocols. RFC 8439, Internet Research Task<br>Force (IRTF), June 2018.|
|[RRR16]|O. Reingold, G. N. Rothblum, R. D. Rothblum. Constant-round interactive proofs for delegating<br>computation. STOC 2016, pp. 49–62.|
|[SGO26]|P. K. Sanjaya, C. Giannoula, V. Oktavian, M. Saeedi, G. Sines, G. Saileshwar, N. Vijaykumar. zkComposer:<br>decomposing proof construction to scale zkML. arXiv:2607.08095 (2026).|
|[SLZ24]|H. Sun, J. Li, H. Zhang. zkLLM: zero knowledge proofs for large language models. ACM CCS 2024;<br>arXiv:2404.16109.|
|[Sta21]|E. Ben-Sasson, D. Brownstein, D. Carmon, L. Goldberg, D. Levit, D. Medini, A. Shtaierman, E. Yogev.<br>ethSTARK documentation, version 1.2 (July 2023). Cryptology ePrint Archive 2021/582.|
|[Tha13]|J. Thaler. Time-optimal interactive proofs for circuit evaluation. CRYPTO 2013; full version: Cryptology<br>ePrint Archive 2013/351; arXiv:1304.3812.|



54 

|[TOPLOC]|J. M. Ong, M. Di Ferrante, A. Pazdera, R. Garner, S. Jaghouar, M. Basra, M. Ryabinin, J. Hagemann.<br>TOPLOC: a locality sensitive hashing scheme for trustless verifiable inference. ICML 2025 (PMLR 267);<br>arXiv:2501.16007.|
|---|---|
|[transformers|] The Hugging Face team. Hugging Face transformers, version 5.17.0 (Apache-2.0), github.com/<br>huggingface/transformers.|
|[TP]|Proving a Language Model’s Replies under Committed Weights: Protocol, Proofs and Evaluation.<br>Technical report, 2026. The technical report is not in this repository, and a reader of this release cannot<br>open it (Appendix E.1).|
|[Uns25]|Unsloth. Llama-4-Maverick-17B-128E-Instruct-GGUF (GGUF quantizations of Llama 4 Maverick; the<br>file set UD-Q4_K_XL). huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF.|
|[VCVio]|VCV-io. A public library for the proof assistant Lean (Apache-2.0), github.com/Verified-zkEVM/VCV-io;<br>its files name their own authors.|
|[VerInf]|J. Petrie. VerInf. Source code and README, github.com/JamesPetrie/VerInf, commit 89903c96 (August<br>2026).|
|[Wan25]|Y. Wang. Zero-knowledge proof based verifiable inference of models. arXiv:2511.19902 (2025).|
|[Wan26]|Z. Wang. NanoZK: privacy-preserving verifiable inference for large language models via layerwise<br>zero-knowledge proofs. ICICS 2026; arXiv:2603.18046.|
|[WLL26]|L. Wang, H. Lou, C. Li, Y. Yu, Y. Hu. zkAgent: verifiable LLM agent execution via one-shot transcript<br>proofs. Cryptology ePrint Archive 2026/199 (version of May 1, 2026; the version of February 2026 is<br>titled zkAgent: verifiable agent execution via one-shot complete LLM inference proof). Sections and<br>equations are numbered as in the version of February 2026.|
|[WTSTW18]|R. S. Wahby, I. Tzialla, a. shelat, J. Thaler, M. Walfish. Doubly-efficient zkSNARKs without trusted<br>setup. IEEE Symposium on Security and Privacy 2018; Cryptology ePrint Archive 2017/1132.|
|[XLF25]|T. Xie, T. Lu, Z. Fang, S. Wang, Z. Zhang, Y. Jia, D. Song, J. Zhang. zkPyTorch: a hierarchical<br>optimized compiler for zero-knowledge machine learning. Cryptology ePrint Archive 2025/535.|
|[XZZPS19]|T. Xie, J. Zhang, Y. Zhang, C. Papamanthou, D. Song. Libra: succinct zero-knowledge proofs with<br>optimal prover computation. CRYPTO 2019; Cryptology ePrint Archive 2019/317.|
|[YLFKM21]|T. Yurek, L. Luo, J. Fairoze, A. Kate, A. Miller. hbACSS: how to robustly share many secrets. NDSS<br>2022; Cryptology ePrint Archive 2021/159.|
|[ZXHSZ22]|J. Zhang, T. Xie, T. Hoang, E. Shi, Y. Zhang. Polynomial commitment with a one-to-many prover and<br>applications. USENIX Security 2022, pp. 2965–2982.|
|[ZYDAZ24]|J. Zhu, H. Yin, P. Deng, A. Almeida, S. Zhou. Confidential computing on NVIDIA Hopper GPUs: a<br>performance benchmark study. arXiv:2409.03992 (2024).|



55 

## **A Proof of the soundness theorem** 

Theorem 5.1 rests on the check of its Lean proof by Lean’s kernel; the printed proof is given to explain it. Each lemma restates a Lean lemma on which that proof rests, with its hypotheses and constants; part 1 of Lemma A.17 is true, but no Lean statement has its form. A proof marked _outline_ names Lean’s lemmas and does not argue them. For each machine-checked statement, Appendix F names the Lean declarations that it renders. 

### **A.1 Overview** 

Table 9: The chain of reductions, from the theorem downwards, with what each adds to the error or to _<u>q</u>_ . 

|Step|By|Cost|
|---|---|---|
|Greedy outputs to the policy|Proposition 3.5|none|
|Weights and tokens to (_A, ξ_)_∈_Req_Z_|Lemma A.33|none|
|Parser and public readings|Lemma A.15 and<br>Lemma A.35|1 + 8 queries|
|Commitment _c_ to the key k|Lemma A.14|(_q_+ 11)<sup>2</sup>_/_2<sup>256</sup>|
|Digests from _H_ to constants|Lemma A.13|none|
|_H_ to separate oracles|Lemma A.12|none|
|A false encoded request accepted to Bad|Lemma A.11|factor 5; _qV_ + 1 queries|
|Bad to single rounds|Lemma A.8|_ε_rd for each query; 2(_Q_+ 1)<sup>2</sup>_/_2<sup>256</sup>|
|Rounds to codes and polynomials|Proposition 5.14|_ε_rd _≤_2<sup>_−_128</sup>|



With _Q_ = 5 ( _q_ + 9 + _qV_ + 1) and _qV ≤_ 2<sup>20</sup> , the steps give the error 



In _T_ = _q_ + 1 + 2<sup>23</sup> the 1 is the parser’s query. The summand 2<sup>23</sup> covers _qV_ , the eight queries of the public readings and the announcing query, and _Q ≤_ 5 _T_ . The term 16 _T/_ 2<sup>128</sup> of _ε_ ( _q_ ) covers the first two terms where _ε_ ( _q_ ) _<_ 1, and ( _T_ + 2)<sup>2</sup> _/_ 2<sup>256</sup> the third (Lemma A.36). 

### **A.2 From the argument to three IOPs** 

Round-by-round soundness was introduced in [CCHLRR18, Definition 5.3]; [Hol19, Definition 2.5] restates it with a doomed set of partial transcripts. Related work on doomed sets: [BGKTTZ23, BGTZ23]. Round-by-round knowledge soundness is due to [CMS19, Definition 8.5], as round-by-round knowledge, in a form that differs from ours in detail; [CMS19, Definition 8.4] states round-by-round soundness for IOPs. Compare [CFW26, version of February 25, 2026, Definitions 3.6 and 3.7]; some of the Lean files that Definition A.1 and Definition A.2 follow are among those marked as written from that work (Table 13). That work has the notion from [BCFW25], extended to relaxed relations. 

**Definition A.1 (Round-by-round soundness).** Let _L_ be a set of pairs (in _,_ ext) of an IOP instance and external oracle strings (Definition 2.3). An IOP is _round-by-round sound_ for _L_ with _round errors_ ( _εm_ ) _m_ if there is a function State from such pairs with a partial transcript to _{_ 0 _,_ 1 _}_ such that: 

1. for (in _,_ ext) _∈L/_ , the empty transcript has state 0; 

2. no prover message changes the state from 0 to 1; 

3. the challenge of round _m_ does so with probability at most _εm_ ; 

4. the verifier rejects every complete transcript of state 0. 

In a batch IOP each IOP instance has the state of its local transcript, and in a shared round _εm_ bounds the probability that the state of some instance of the batch changes from 0 to 1. 

**Definition A.2 (Early part, knowledge soundness).** Let _Q_ be a batch IOP and early a map on IOP instances, the _early part_ . _Q_ is round-by-round sound for _L with early part_ early and round errors ( _εm_ ) _m_ if there is a function State that meets the conditions on prover messages and challenges (Definition A.1) and, for every (in _,_ ext) _∈L/_ : 

1. the empty transcript has state 0 at (early(in) _,_ ext); 

56 

2. the verifier rejects in on complete transcripts of state 0 at (early(in) _,_ ext). 

Let _L_<sup>wit</sup> be a set of such pairs with a witness, wit a _witness map_ , and _Q_<sup>_′_</sup> be _Q_ with a verifier that also rejects each complete transcript trs with (in _,_ ext _,_ wit(in _,_ ext _,_ trs)) _∈L_<sup>wit</sup> . _Q_ is _round-by-round knowledge sound_ for _L_<sup>wit</sup> under wit if _Q_<sup>_′_</sup> is sound in this sense for _L_ = _∅_ . 

Definition A.2 is used with the empty witness alone (Definition A.6). There, round-by-round knowledge soundness is round-by-round soundness with an early part. 

**Terms of Construction D.11.** The _separate oracles_ are Merkle oracles for leaves and for inner nodes, of trees over oracle strings and of the batch tree, and, for each of the three IOPs, a Fiat–Shamir oracle fs and an oracle g. An answer of fs gives a round’s challenge and a _chaining value_ , which the query of the next round holds. The first query of a compiled verifier holds a list cms of digests. The _early part_ of an encoded request _ξ_ is _ξ_ with _y_<sup>+</sup> replaced by 0. Seed(rt) is the OOD seed that two queries to fs derive from a root. The oracles ann and _spare_ occur in the analysis alone (Definition A.3). The function _H_<sup>_′_</sup> enters the proof as one fixed function (Assumption 4.11; Remark 4.12). 

**Definition A.3 (The argument over separate oracles).** Fix _Z_ and a value pv of the public readings (dg _Z ,_ par) of Construction 4.15. The argument Π<sup>_◦_</sup> _Z,_ pv<sup>queriestheseparateoraclesofConstructionD.11,whichareindependentand</sup> uniform. Further oracles exist, which its verifier never queries; among them are ann, with a single possible answer, and a _spare_ oracle on bit strings. 

- A matrix _A_ in range takes the place of the index, an encoded request _ξ_ that of an instance; the relation is Req _Z_ . 

- A commitment is a key k = (rt0 _, σ_ 0). 

- A proof is _π_<sup>_◦_</sup> = ( _j,_ rt _j, σj, π_ key _, π_ link _, π_ req). 

- The verifier _V_<sup>_◦_</sup> performs steps 6 to 9 of _VZ_ at the value pv. 

- The extractor is Ext<sup>_◦_</sup> (k _,_ tr) = Dec(Wd(tr _,_ rt0) _, σ_ 0), with Wd as in Definition A.4. 

The number _qV_ bounds the queries of _V_<sup>_◦_</sup> to the separate oracles, on every input. 

**Definition A.4 (Word under a root, rooted query, seed-first trace).** Let tr be a trace over the separate oracles. 

1. The _word under_ a digest rt, Wd(tr _,_ rt), has at each position the symbol of the leaf query that tr determines under rt, else a fixed symbol. 

2. rt is _announced_ in tr if tr contains the query ann(rt). 

3. A _start query_ is the first query to fs of step 1 of Construction D.11. A query to fs is _rooted_ in tr if it is a start query, or its chaining fields are chaining values of rooted queries of the preceding round; queries of Seed are not rooted. A rooted query determines a partial transcript: challenges from answers, plain parts from queries, oracle parts as the words under the queried roots. 

4. tr is _seed-first_ if each start query is preceded by the queries of Seed for the first two digests of its cms. Its _seed-first completion_ <u>tr</u> inserts these queries, at most four. 

**Definition A.5 (The event** Bad **).** Fix state functions of the three IOPs (Definition A.1 and Definition A.2). A trace lies in Bad if an entry that is not contained in the prefix pre before it causes one of four events. 

1. _Merkle event._ The answer of a Merkle oracle is a digest that occurs in pre. 

2. _Chaining event._ The chaining value of an answer of fs, or an answer of g, occurs in pre. 

3. _State event._ A rooted query to fs has an answer that changes the state of its partial transcript from 0 to 1; in a shared round, that of some encoded request under the queried root of the batch tree. External oracle strings are the words in pre under the first digests of cms; OOD seeds are read off pre, OOD answers off cms. 

4. _Seed event._ The seed of the second query of Seed(rt) is a collision of Wd(pre _,_ rt). 

By definition, Bad is closed under extension of traces. 

**Definition A.6 (Conditions on soundness).** Let _ε_ rd _≥_ 0. The sets are those of Construction 4.13. 

**(Sound-key).** IOP<sup>key</sup> _Z_ is round-by-round sound for its set, with round errors at most _ε_ rd (Definition A.1). **(Sound-link proof).** The same holds of IOP _Z_<sup>link</sup> . 

**(Sound-request).** IOP<sup>req</sup> _Z_<sup>,asabatchIOPforatmost</sup><sup>_R_encodedrequests,isround-by-roundknowledgesoundfor</sup> its set, with the empty witness, with the early part of _ξ_ , and with round errors at most _ε_ rd (Definition A.2). 

In the first two, a shared round’s error is counted once for each leaf of the tree of the call’s requests. 

57 

**Definition A.7 (Conditions of the compiler).** The fifteen _conditions of the compiler_ , at _ε_ rd _≥_ 0 and a value pv of the public readings, are the three of Definition A.6 and twelve _side conditions_ : 

**(No-decoding-at-a-collision).** Coll( **T**<sup>_∗_</sup> _, ς_ ) implies Dec( **T**<sup>_∗_</sup> _,_ ( _ς,_ **a** )) = _⊥_ . 

**(Local),** for each IOP. The verifier reads oracle strings only at declared positions. 

**(Coherent),** for each IOP. A plain part is the pair of its two components’ plain parts; each oracle position lies in one component. 

**(Rare-collisions).** For every word, the OOD seed read off uniform hash blocks is a collision with probability at most _ε_ rd. 

**(Layout).** Queries determine rounds and openings; uniform answers give uniform challenges. 

**(Answers-code),** two conditions. The code of OOD answers is injective and of one length. 

**(Hashed-key).** The implication of Lemma A.14, for Π<sup>_◦_</sup> _Z,_ pv<sup>.</sup> 

The conditions on soundness are the subject of Appendix A.3. (Layout) is stated informally: it summarizes the laws of four Lean structures, which this paper does not print. (Hashed-key) holds by Lemma A.14. 

**Lemma A.8 (** Bad **is rare).** [machine-checked] _Assume the conditions of Definition A.7 other than (Hashed-key), and form_ Bad _(Definition A.5) with the state functions that (Sound-key), (Sound-link proof) and (Sound-request) of Definition A.6 provide. Then the trace_ tr _of every q-query algorithm over the separate oracles satisfies_ 



_Proof._ We bound the probability that the answer to a new query causes an event, given pre, and take a union bound. 

_Step 1: Merkle and chaining events._ The answer, or its chaining value, is uniform in _{_ 0 _,_ 1 _}_<sup>_λ_</sup> and independent of pre, in which each entry holds at most four values that count as occurring. The _i_ -th query causes such an event with probability at most 4 _i/_ 2<sup>_λ_</sup> ; the sum over _i < q_ is at most 2 _q_<sup>2</sup> _/_ 2<sup>_λ_</sup> . 

_Step 2: state events._ If a query to the oracle fs of one IOP is rooted, pre determines its partial transcript, the IOP instance and the external oracle strings; among several preceding queries with one chaining value the first is taken. By (Layout) the challenge is uniform, and by the condition on that IOP it changes a state from 0 to 1 with probability at most _ε_ rd. In a shared round pre determines at most one encoded request for each leaf of the batch tree, and the round’s error bounds the event for all of them jointly. 

_Step 3: seed events._ A second query of Seed causes no state event. The word Wd(pre _,_ rt) is determined before its answer, and (Rare-collisions) applies. 

**Lemma A.9 (Outside** Bad **).** [machine-checked] _Assume the conditions of Definition A.7 other than (Hashed-key). Let_ tr _∈/_ Bad _._ 

_1._ Stable words. _If_ rt _is announced in a prefix_ tr0 _of_ tr _, then_ Wd(tr0 _,_ rt) = Wd(tr _,_ rt) _._ 

_2._ Seeds. _If_ tr _contains the queries of_ Seed(rt) _, with value ς, then_ Coll(Wd(tr _,_ rt) _, ς_ ) _does not hold._ 

_3._ Accepting runs. _Suppose that_ tr _is seed-first and has equal answers at equal queries, and that it contains an accepting run of the compiled verifier of one of the three IOPs and, for each root of an external oracle string, the queries of_ Seed _, with the OOD seed of the IOP instance as value. Then the relation of that IOP (Construction 4.13) holds of the IOP instance and the words under these roots in_ tr _._ 

_Proof. Part 1._ Without a Merkle event, no query after tr0 has as its answer a digest that occurs in tr0; the tree under rt can neither change nor grow. 

_Part 2._ Without a chaining event, the second query of Seed(rt) follows the first, in which rt occurs; from then on the word under rt is stable. A collision at _ς_ would be a seed event. 

_Part 3._ Without chaining events, the queries to fs of the accepting run occur in tr in the order of the rounds, and each is rooted. Because tr is seed-first, the OOD seeds are read off the prefix before the start query; by (Answers-code) the OOD answers are read off their injective code. The partial transcripts therefore belong to one IOP instance and to the stable words under the roots. Suppose for contradiction that the relation fails. The state is 0 at the start and stays 0, since there is no state event, and the IOP’s verifier rejects the complete transcript. By (Local), (Coherent) and the verified openings, the compiled verifier decides as the IOP’s verifier does. For IOP<sup>req</sup> _Z_ the start query holds only the early part of _ξ_ ; this is why (Sound-request) is stated with that part. 

58 

**Lemma A.10 (The three relations in sequence).** [machine-checked] _Assume the conditions of Definition A.7 other than (Hashed-key). Let_ k = (rt0 _, σ_ 0) _, and let_ tr = tr _C ∥_ ann(rt0) _∥_ tr<sup>_′_</sup> _be a trace whose answers are those of fixed oracle functions, where_ tr<sup>_′_</sup> _contains an accepting run of V_<sup>_◦_</sup> (k _, ξ, π_<sup>_◦_</sup> ) _. If_ <u>tr</u> _∈/_ Bad _, then_ 



_Proof._ The inserted entries of <u>tr</u> are queries to fs, and words under roots are the same in tr and <u>tr.</u> Let **T**<sup>_∗_</sup> 0<sup>and</sup><sup>**T**</sup><sup>_∗_</sup> _j_<sup>be</sup> the words under rt0 and rt _j_ . We apply Lemma A.9. 





3. By part 3 for IOP<sup>link</sup> _Z_ , Dec( **T**<sup>_∗_</sup> _j_<sup>_, σj_) = Dec(</sup><sup>**T**</sup> 0<sup>_∗, σ_0) =</sup><sup>_A_.</sup> 4. By part 3 for IOP<sup>req</sup> _Z_<sup>,(</sup><sup>_A, ξ_)</sup><sup>_∈_Req</sup> _Z_<sup>.</sup> 

5. By part 1, **T**<sup>_∗_</sup> 0<sup>= Wd(tr</sup><sup>_C,_rt0),sincert0isannounceddirectlyaftertr</sup><sup>_C_.</sup> 

**Lemma A.11 (Index-binding knowledge soundness over separate oracles).** [machine-checked] _Assume the conditions of Definition A.7 other than (Hashed-key). Then_ Π<sup>_◦_</sup> _Z,_ pv<sup>_isindex-bindingknowledgesoundfor_Req</sup> _Z_<sup>_,inthe_</sup> _general form of Definition 2.6 and Definition 2.7, over the separate oracles in the place of H. The extractor is_ Ext<sup>_◦_</sup> _, and the error is_ 



_Proof._ Join the committer, one query ann(rt0), the prover and _V_<sup>_◦_</sup> into one algorithm, with at most _q_ + 1 + _qV_ queries. Modify it so that it makes the queries of the seed-first completion. This changes no output and yields the trace <u>tr,</u> with at most four further queries for every query: at most _Q_ in all. If _V_<sup>_◦_</sup> accepts while Ext<sup>_◦_</sup> (k _,_ tr _C_ ) is _⊥_ or is not related to _ξ_ , then <u>tr</u> _∈_ Bad by Lemma A.10. Lemma A.8, applied to the modified algorithm, bounds this probability. 

**Lemma A.12 (One hash function).** [machine-checked] _Let a bijection between bit strings and triples (oracle, query, number of an answer block) be given, under which an answer of a Merkle oracle over oracle strings is one block. If an argument over the separate oracles is index-binding knowledge sound with the extractor_ Ext<sup>_◦_</sup> _, then so is the argument that queries H through the bijection, with the same error and the extractor that translates the trace and applies_ Ext<sup>_◦_</sup> _._ 

_Proof._ A uniform _H_ induces independent uniform oracles, and a query to _H_ is one block of one query to a separate oracle. The extractor reads only entries at oracles with one block. 

**Lemma A.13 (Public readings).** [machine-checked] _Let_ Pub _be a deterministic oracle algorithm that repeats no query, and let S contain every string that_ Pub _queries under some hash function. Let_ Pub _and S be fixed before H is sampled. Suppose that, for every_ pv _in the codomain of_ Pub _:_ 

_1. the conditions of Definition A.7 hold at_ pv _;_ 

_2._ (Spare) _under the bijection at_ pv _, every string of S codes a query to the spare oracle;_ 

_3._ (Same-word) _the extractor of Lemma A.12 at_ pv _depends on_ pv _only through_ dg _Z ._ 

_Consider the argument over H with commitments_ (dg _,_ k) _whose verifier computes_ pv = Pub<sup>_H_</sup> _, rejects unless_ dg _is the digest in_ pv _, and then runs V_<sup>_◦_</sup> _at_ pv _through the bijection at_ pv _. It is index-binding knowledge sound for_ Req _Z with error ε_ sep _. Its extractor, at_ (dg _,_ k) _, is that of Lemma A.12 at any_ pv _with the digest_ dg _, and ⊥ if there is none._ 

_Proof._ By induction on the queries of Pub. Its first query str _∈ S_ is fixed before _H_ is sampled, and _H_ can be sampled by first sampling val = _H_ (str). Fix val, and give (str _,_ val) to the committer and the prover as a constant. By (Spare), _V_<sup>_◦_</sup> does not query str, and deleting the entries at str from a trace changes no word under a root. When no query is left, pv is a constant. The hypotheses hold at every pv, hence at this one, and Lemma A.11 and Lemma A.12 apply with the error _ε_ sep. 

This step requires that _Z_ is fixed before _H_ is sampled: the induction resamples _H_ at strings that no adversary chooses. 

**Lemma A.14 (The commitment as the hash of the key).** [machine-checked] _Let_ Π _be an argument over H whose commitments are keys and whose verifier is deterministic. Let_ Π _be index-binding knowledge sound with extractor_ Ext _and error ε_<sup>_′_</sup> _, and let_ code _be an injective partial map from keys to bit strings. Consider the argument in which a commitment is c_ = _H_ (code(k)) _, a proof contains_ k _, and the verifier rejects unless H_ (code(k)) = _c and then decides as the verifier of_ Π _does. It is index-binding knowledge sound with error_ 



59 

_Its extractor applies_ Ext _to the first key whose code the trace maps to c._ 

_Proof._ Let k<sup>_∗_</sup> be the first key whose code the committer’s trace maps to _c_ , and let the verifier accept a proof with the key k. If k = k<sup>_∗_</sup> , the outcome is that of the experiment for Π with the committer that outputs k<sup>_∗_</sup> ; this gives _ε_<sup>_′_</sup> ( _q_ ). Otherwise consider, in this order, the committer’s queries, one query at _c_ read as a bit string, the prover’s queries and the verifier’s query at code(k). The query at _c_ is made in the analysis alone: it makes _c_ a string that has occurred, for the case that the trace holds no k<sup>_∗_</sup> . Since code is injective, at some new entry of this list the answer equals a preceding answer or, read as a bit string, a preceding query or the entry’s own query. After _i_ entries this has probability at most (2 _i_ + 1) _/_ 2<sup>_λ_</sup> , and<sup>�</sup> _i<q_ +2<sup>(2</sup><sup>_i_+ 1) = (</sup><sup>_q_+ 2)2.</sup> 

**Lemma A.15 (The compilation lemma).** [machine-checked] _Let Z be fixed before H is sampled. Assume the hypotheses of Lemma A.13, with_ Pub _the public readings of VZ , and let_ code _be injective on pairs_ (dg _,_ k) _. Let the public readings and the parser together make at most q_ pre _queries. Let_ Rel _be an indexed relation such that, for all A, x, y and_ fill _,_ 

( _A,_ build( _x, y,_ fill)) _∈_ Req _Z_ = _⇒_ wt( _A_ ) _is defined and_ (wt( _A_ ) _, x, y_ ) _∈_ Rel _._ 

_Then every argument whose verifier accepts only where the public readings, the parser and the compiled steps of VZ accept is index-binding knowledge sound for_ Rel _, with error_ 



_Its extractor is that of Lemma A.14, formed from that of Lemma A.13, followed by_ wt _._ 

_Proof._ Lemma A.14 replaces the commitment (dg _,_ k) of Lemma A.13 by _c_ . An instance ( _x, y_ ), with fill from the proof, is mapped to _ξ_ = build( _x, y,_ fill); by the hypothesis, an instance outside Rel at wt( _A_ ) gives an encoded request outside Req _Z_ at _A_ . The queries of the public readings and the parser are charged to the prover: _q_ becomes _q_ + _q_ pre. 

_Remark_ A.16 (Which form of the lemma). Two forms of the compilation lemma are machine-checked. One takes the public readings as constants and has sixteen hypotheses: the fifteen of Definition A.7 and the implication for greedy outputs. It does not apply to _VZ_ , and the proof of Theorem 5.1 does not go through it: the strings that _VZ_ hashes hold digests that it obtains from _H_ . The other is Lemma A.15. It requires the fifteen at every value of the public readings, the implication for Rel in the place of the sixteenth, and further conditions; for FlashProve _Z_ each of these is a theorem, so that Theorem 5.1 has none of them as a hypothesis. Which form Lean’s proof uses is known by a reading of the Lean code and by a program that follows the theorem’s dependencies; no theorem states it. Limits of both forms: `lean/LIMITS.md` . 

**Lemma A.17 (The side conditions hold).** [machine-checked] _The tag is for parts 2 and 3 alone: no Lean statement has the form of part 1 (the outline below). Let Z be a configuration and ε_ rd = 2<sup>_−_128</sup> _._ 

_1. If (Sound-key), (Sound-link proof) and (Sound-request) hold for_ FlashProve _Z , then all fifteen conditions of Definition A.7 hold at every value_ pv _of the public readings._ 

_2. The public readings of VZ are deterministic, repeat no query and make at most eight queries. They satisfy (Spare) and (Same-word) of Lemma A.13._ 

_3. The map_ code _is injective on pairs_ (dg _,_ k) _._ 

_Outline._ No single Lean statement has the form of part 1: besides the three conditions, the Lean lemma behind it takes nine numeric premises, the numeric tests of (Layout) below. Further Lean lemmas prove these at every _Z_ ; they are joined with it only inside the proof of another Lean statement. 

(No-decoding-at-a-collision) is step 1 of Definition 4.5. (Rare-collisions) is Lemma A.25: for every configuration, Lean’s construction forms committed tables at the sizes of _Z_ 0. (Hashed-key) is Lemma A.14. (Local), (Coherent), (Layout), (Answers-code) and parts 2 and 3 are properties of the verifiers and codings of Construction D.9, Construction D.11, and Construction 4.15, which Lean proves by unfolding them; (Layout) rests also on numeric tests, which Lean proves at every _Z_ . Only four of the fifteen depend on pv. Within (Layout), the coding of a start query of IOP<sup>req</sup> _Z_ is injective by the encoded request’s window and token positions alone, whatever the values of _H_<sup>_′_</sup> . (Spare) is domain separation between the strings of the public readings and the coded queries. 

### **A.3 Round-by-round soundness of the three IOPs** 

For sub-protocols, Lean states round-by-round soundness without a state function. The proofs of Lemma A.28 and Lemma A.30 are outlines: for the arithmetization and for two of the three IOPs the printed text is a guide to Lean’s proof of Proposition 5.14, not an argument. 

60 

**Definition A.18 (Doomed, reduction, leads).** Consider _k_ rounds, a set _B_ of complete transcripts, and _counts_ cnt = (cnt1 _, . . . ,_ cnt _k_ ) _∈_ N<sup>_k_</sup> . The prover is _doomed towards B_ with these counts if 

- _k_ = 0 and the empty transcript lies in _B_ ; or 

- _k >_ 0 and, for every first message msg, at most cnt1 challenges ch1 are _bad_ : after (msg _,_ ch1) the prover is not doomed, with counts (cnt2 _, . . . ,_ cnt _k_ ), towards _{_ trs : (msg _,_ ch1 _,_ trs) _∈B}_ . 

The _error_ of round _m_ is cnt _m_ divided by the number of its challenges. 

A _reduction_ consists of rounds and a partial map out from an input statement and a complete transcript to an output statement. It _leads_ , with counts cnt, from a condition _C_ on inputs to a relation _R_ between inputs and outputs if, from every input stm with _C_ (stm), the prover is doomed, with these counts, towards the set of all trs such that out(stm _,_ trs) is undefined or _R_ (stm _,_ out(stm _,_ trs)) holds. 

**Lemma A.19 (State functions; composition).** [machine-checked] 

_1. If, outside L, the prover of an IOP is doomed towards rejection with counts_ cnt _, then the IOP is round-by-round sound for L (Definition A.1) with the errors of these counts._ 

_2. Let_ Red1 _lead from C_ 1 _to R_ 1 _with counts_ cnt _, and_ Red2 _from C_ 2 _to R_ 2 _with counts_ cnt<sup>_′_</sup> _. If C_ 1(stm) _and R_ 1(stm _,_ stm<sup>_′_</sup> ) _imply C_ 2(stm<sup>_′_</sup> ) _, then_ Red1 _followed by_ Red2 _leads from C_ 1 _to the composition of R_ 1 _and R_ 2 _, with counts_ (cnt _,_ cnt<sup>_′_</sup> ) _; likewise for a finite list._ 

_Proof._ For part 1, let the state of a partial transcript be 0 if and only if the prover is doomed after it. Part 2 is by induction on the rounds of Red1. 

**Lemma A.20 (Proof of work; challenges from hash blocks).** [machine-checked] 

_1. Let_ Ch _be a finite set and_ Ch<sup>_′_</sup> _⊆_ Ch _. The number of pairs_ ( _a, z_ ) _∈_ Ch _× {_ 0 _,_ 1 _}_<sup>_κ_</sup> _with z_ = 0 _and a ∈_ Ch<sup>_′_</sup> _is |_ Ch<sup>_′_</sup> _|._ 

_2. Let a sub-protocol be stated with challenges in sets_ Ch _m, and let the prover be doomed towards B with at most a share_ shr _m of_ Ch _m bad in round m. In_ FlashProve _Z the challenge of round m is a tuple of_ blk _m hash blocks, from which an element of_ Ch _m and κm bits of proof of work are computed. Over these rounds the prover is doomed towards the transcripts whose image lies in B or in which a proof-of-work test fails, with round errors at most_ 



_Proof._ For part 1, _a �→_ ( _a,_ 0) is a bijection onto the pairs in question. For part 2, at most fib _m_ tuples of hash blocks are mapped to one element of Ch _m_ , where fib _m |_ Ch _m|_ is at most (1 + 2<sup>_−_128</sup> )<sup>blk</sup><sup>_m_</sup> times the number of tuples. By part 1, proof of work leaves the count unchanged, while the number of challenges grows by the factor 2<sup>_κm_</sup> . 

The conditions of Definition A.6 do not mention proof of work: they bound the bad challenges among all challenges of a round. A further nonce is a further query, which counts in _q_ . In the next lemma the implications between claims run backwards: if the claim after a part holds, so does the claim before it. Informally, a false claim therefore remains false, except at bad challenges, until the last part refutes it. 

**Lemma A.21 (From three parts to a batch IOP).** [machine-checked] _Let a batch IOP for at most R IOP instances consist of reductions_ Red1 _,_ Red2 _,_ Red3 _: its first, shared and last part. Let C_ first _, C_ shared _, C_ last _be conditions, P_ first _a relation between inputs and outputs of_ Red1 _, and P_ shared _, P_ last _properties of the outputs of_ Red1 _,_ Red2 _. Suppose:_ 

_1._ Red1 _leads from C_ first _to C_ shared _∧_ ( _P_ shared _⇒P_ first) _;_ 

_2._ Red2 _leads from C_ shared _to C_ last _∧_ ( _P_ last _⇒P_ shared) _, with counts_ cnt<sup>key</sup> + _R_ cnt<sup>own</sup> _, where the challenges counted in_ cnt<sup>key</sup> _depend only on the external oracle string and its OOD sample;_ 

_3._ Red3 _leads from C_ last _to P_ last _;_ 

_4. every accepting transcript is a run of the three reductions on the early part of the IOP instance;_ 

_5. on an accepting transcript, C_ first _⇒P_ first _implies membership in the IOP’s set;_ 

_6. every round error is at most ε_ rd _._ 

_Then the batch IOP is round-by-round knowledge sound in the form of (Sound-request) (Definition A.6)._ 

_Proof._ By Lemma A.19, from every input with _C_ first the prover is doomed towards _P_ first: the last part gives _P_ last, hence _P_ shared, hence _P_ first. By hypotheses 4 and 5 it is doomed towards the event that the verifier rejects or the IOP instance lies in the set. The three reductions run on the early part; hence the state function of Lemma A.19 is one at the early part, as Definition A.2 requires. In a batch, the challenges counted in cnt<sup>key</sup> form one set for all encoded requests, which share the copy and its OOD sample. 

61 

**Lemma A.22 (Where a parameter check fails).** [machine-checked] _Let Z be a configuration with_ Chk( _Z_ ) = 0 _. Then (Sound-key), (Sound-link proof) and (Sound-request) of Definition A.6 hold for_ FlashProve _Z with ε_ rd = 2<sup>_−_128</sup> _._ 

_Proof._ Each of the three verifiers rejects every transcript (Construction 4.13). 

**Codes.** Two bounds strictly below the Johnson radius are used: Lemma 5.13 and a bound on lists. Related work for Lemma A.24: [DP23]; further sources are in Appendix E. 

**Lemma A.23 (List size strictly below the Johnson radius).** [machine-checked] _Consider a linear code over a field, with set of positions D and minimum distance at least d_ min _positions. Let e ≤|D| be a natural number with |D|_ ( _|D| − d_ min) _<_ ( _|D| − e_ )<sup>2</sup> _. Let_ **T**<sup>_∗_</sup> _be a table of words with any set of rows. Then at most_ 



_tables of codewords differ from_ **T**<sup>_∗_</sup> _in at most e columns._ 

_Outline._ This is the Johnson bound for the code of tables, with columns as symbols. 

**Lemma A.24 (Folding a table).** [machine-checked] _Let |D| be a power of two, at most_ 2<sup>32</sup> _, let ℓ_ + _t ≥_ 2 _, and put d_ min = _|D| −_ ( _ℓ_ + _t_ ) + 1 _. Let_ 1 _≤ e_<sup>_′_</sup> _≤ e < d_ min _with |D|_ ( _|D| − d_ min) _<_ ( _|D| − e_<sup>_′_</sup> )<sup>2</sup> _. Let_ **T**<sup>_∗_</sup> _be a word (Definition 4.5), and σ_ = ( _ς,_ **a** ) _a sample at which_ Coll( **T**<sup>_∗_</sup> _, ς_ ) _does not hold. Call γ ∈_ E<sup>_v_</sup> bad _if some u ∈_ E<sup>_ℓ_+</sup><sup>_t_</sup> _satisfies:_ 



_3. step 2 of Definition 4.5 gives no table, or a table whose messages, folded at γ, differ from the message of u._ 

_Let J be the bound of Lemma 5.13 at degree ℓ_ + _t −_ 1 _and radius e_<sup>_′_</sup> _. Then the number of bad γ is at most_ 



_Proof._ Call z _∈_ E _exceptional_ for _g_ 0 _, g_ 1 if it realizes the event of Definition 2.2 for _g_ 0 + z _g_ 1. 

_Step 1: tables._ Let _g_ 0 _, g_ 1 be tables with a set Rw of rows, and read agreement as agreement of all rows on a common set of columns. For each exceptional z fix such a set _S_ z for _g_ 0 + z _g_ 1. The vectors _a ∈_ E<sup>Rw</sup> for which<sup>�</sup> _r_<sup>_ar g_0</sup><sup>_,r_</sup> and<sup>�</sup> _r_<sup>_ar g_1</sup><sup>_,r_bothagreewiththecodeon</sup><sup>_S_zformasubspace.Itisproper,sincesomerowof</sup><sup>_g_0or</sup><sup>_g_1doesnotagree.</sup> There are at most _|_ E _|_ such subspaces. Each has at most _|_ E _|_<sup>_|_Rw</sup><sup>_|−_1</sup> elements, and all contain 0; hence some _a_ lies in none. Every z that is exceptional for the tables is exceptional for the two words formed with this _a_ , and the bound _J_ holds for tables. 

_Step 2: folds._ Folding 2<sup>_v_</sup> rows with eq( _γ, ·_ ) is a sequence of _v_ steps. Step _k_ maps two half tables _g_ 0 _, g_ 1 to (1 _− γk_ ) _g_ 0 + _γk g_ 1 = _g_ 0 + _γk_ ( _g_ 1 _− g_ 0), the line of Step 1 for the tables _g_ 0 and _g_ 1 _− g_ 0; these agree with the code on a set of columns if and only if _g_ 0 and _g_ 1 do. At most _v ⌊J⌋|_ E _|_<sup>_v−_1</sup> points _γ_ are exceptional in some step. 

_Step 3: other points._ If Enc( _u_ ) is within _e_<sup>_′_</sup> positions of the folded word, then **T**<sup>_∗_</sup> agrees with a table of codewords on the same columns, and _u_ is the fold of that table. At most ls( _|D|, d_ min _, e_<sup>_′_</sup> ) tables are within _e_<sup>_′_</sup> columns of **T**<sup>_∗_</sup> (Lemma A.23); since _e_<sup>_′_</sup> _≤ e_ , each lies in Λ( **T**<sup>_∗_</sup> ). 

_Step 4: the list._ Since there is no collision, such a table with the OOD answers **a** is the decoded table, which condition 3 excludes. For one with other answers, condition 2 is a nonzero multilinear equation in _γ_ , with at most _v |_ E _|_<sup>_v−_1</sup> solutions. 

Steps 2 to 4 run as the second of the error bounds in the proof of [CFW26, version of February 25, 2026, Lemma 6.5], and for each variable the bound has the form of that lemma’s second error. That work, in its Section 2.5, likens its analysis to those of [ACFY24b] and [BCFW25] and takes mutual correlated agreement from prior work; its own care, shared here, is to count the lists on the interleaved code. 

**Lemma A.25 (Collisions are rare).** [machine-checked] _Let the sizes of a committed table, the radius e and the number s of OOD answers be those of Z_ 0 _. Then, for every word_ **T**<sup>_∗_</sup> _, the OOD seed that is read off uniform hash blocks is a collision of_ **T**<sup>_∗_</sup> _(Definition 4.5) with probability at most_ 2<sup>_−_128</sup> _._ 

_Proof._ A collision requires two of the at most ls members of Λ( **T**<sup>_∗_</sup> ) with different messages and equal answers. The word **T**<sup>_∗_</sup> is over F, and _e_ is below the minimum distance. A codeword over E within _e_ positions of a word over F agrees with it at _ℓ_ + _t_ or more positions of _D ⊆_ F, and is therefore over F. The difference of two members thus has its pad entries in F, where the map from three entries to a pad is injective. Such a nonzero message has all OOD answers 0 at a share of at most shr of the OOD seeds. Lean’s bound is (1 + 2<sup>_−_128</sup> )<sup>blk�ls</sup> 2� shr. At the sizes of _Z_ 0, _s_ = 1, a seed’s point has 29 coordinates, ls = 26 and shr = 29 _/|_ E _|_ . Since<sup>�26</sup> 2 � _·_ 29 _·_ 2<sup>178</sup> _< p_<sup>3</sup> = _|_ E _|_ , the bound is about 2<sup>_−_178</sup> ; Lean compares it with 2<sup>_−_178</sup> and then with 2<sup>_−_128</sup> . 

62 

Lemma A.25 is in substance [CFW26, version of February 25, 2026, Lemma 9.1], at this protocol’s code and OOD seeds; the share shr in the proof above corresponds to the evading error in its Lemma 9.3. 

**Sub-protocols.** In Table 10, ls bounds the list of the oracle message or group at its radius _e_ , mca is a count for mutual correlated agreement at that radius, _ν_ is a number of positions, and vano, vanb count the seeds at which a nonzero message has all answers 0. 

|Table 10|: Bad challenge|s in a round of a sub-protocol (Rema|rk A.26).|
|---|---|---|---|
|Round|Challenges|Bad, at most|Source|
|OOD seeds (Send, Code<br>switch)|seeds|�ls<br>2<br>�<br>vano, summed over the messages|two members of a list answer<br>alike|
|Batching|E|the largest exponent|roots of a polynomial|
|Masked sumcheck, κ|E|1|a linear equation|
|Masked sumcheck, variable|E; a further<br>one, E_\ {_0_}_|the degree of the round polynomial|roots of a polynomial|
|Sampled element on a line|E|1|a linear equation|
|Sampled point, _k_<br>coordinates|E<sup>_k_</sup>|_k |_E_|_<sup>_k−_1</sup>|zeros of a multilinear<br>polynomial|
|Fold, κ|E|ls|one value for each member<br>of the list|
|Fold, variable _i_|E|mca_i_+1+ deg_·_ls_i_, and mca of the<br>two mask tables while these are<br>folded|Lemma 5.13; the list after _i_<br>variables|
|Positions (Code switch,|_D_<sup>_ν_</sup>|(_|D| −e −_1)<sup>_ν_</sup>|distance above _e_|
|Base case)||||
|Code switch in a Proximity<br>test|_D_<sup>_ν _</sup>_× S_b|(_|D| −e −_1)<sup>_ν_</sup>_|S_b_|_+_|D|_<sup>_ν_ </sup>ls vanb|also the list, for the batching<br>element|
|Base case, κ|E|mca+ls, the list for two rows|Lemma 5.13; the list|
|Lookup, _ϖ_|E|2<sup>_k_+1</sup> _−_1 for trees of depth _k_|roots of a polynomial|
|Lookup, joined tables|E|the entries of the joined table|roots of a polynomial|
|Range test, Θ|matrices|the share 2<sup>_−_128 </sup>of all matrices|128 independent columns|



_Remark_ A.26 (Round counts). Table 10 is read off several machine-checked lemmas on reductions of the Lean library. It is not one Lean statement, and the proof of Theorem 5.1 uses only the common count of Lemma A.28. For Masked sumcheck, Fold, the Code switch of a Proximity test, Base case and Range test, Lean proves that the reduction leads (Definition A.18) from the reduction’s condition to the relation that its witness map sends each witness of the output’s claims to one of the input’s. For OOD seeds and positions it proves the count alone; for Batching and Lookup the entries are the counts that Lean compares with 2<sup>_−_128</sup> . 

Six rows follow in form, as counts for this protocol’s codes and not term for term, error terms of [CFW26, version of February 25, 2026]: OOD seeds, the first error of its Lemma 9.9; the two rows on Fold, the errors of its Lemma 6.5; Positions, the spot checks in its Lemmas 7.4 and 9.9; Code switch in a Proximity test, the other two errors of its Lemma 9.9 (in its overview, the second error of Lemma 2.17); Base case, the first error of its Lemma 7.4. 

Of the two rows on Masked sumcheck, the first, on the challenge that joins mask and claim, has the form of the first error of [CFW26, version of February 25, 2026, Lemma 6.5], and the count for the further variable of the second, sampled off one point (there, off two), that of the third error of its Theorem 11.3, each without the list sizes; the count for the other variables is that of a sumcheck round [LFKN92]. 

#### **The request.** 

**Definition A.27 (Conditions and claims of** IOP<sup>req</sup> _Z_<sup>**).**Fix</sup><sup>_Z_,aword</sup><sup>**T**</sup><sup>_∗_</sup> _j_<sup>andasample</sup><sup>_σj_= (</sup><sup>_ςj,_</sup><sup>**a**</sup><sup>_j_).Atapoint</sup> between two sub-protocols, the _condition_ states that neither **T**<sup>_∗_</sup> _j_<sup>noranuntestedoraclemessagehasacollisionatits</sup> OOD seeds, and that the verifier’s state has the form fixed for that point. _C_ first, _C_ shared, _C_ last are the conditions at the three parts’ starts. _P_ first **.** Some cells and the token that the verifier holds at the first part’s end satisfy each segment’s identities, lookups and relaxed bound (Definition 4.8), at the early part of _ξ_ and at Dec( **T**<sup>_∗_</sup> _j_<sup>_, σj_).</sup> 

- _P_ shared **.** The claim ( _h, ψ, η, b_ ) on the table (Construction D.7) holds of Dec( **T**<sup>_∗_</sup> _j_<sup>_, σj_).</sup> 

63 

_P_ last **.** _⟨_ eq( _ψ, ·_ ) _, w⟩_ + _η_ = _b_ and _⟨_ eq( _ψ_<sup>_∗_</sup> _, ·_ ) _, w⟩_ = _⟨h, ω⟩_ . 

In the last two, _w_ , _ω_ and mask entries are what their oracle messages decode to, and the state records no failed test. The closed lists of claims and of states’ forms are the Lean terms that Appendix F names for this definition (Remark D.10, Given in outline). 

**Lemma A.28 (The first part of** IOP<sup>req</sup> _Z_<sup>**).**[machine-checked]</sup><sup>_LetZbeanadmissibleconfigurationwith_Chk(</sup><sup>_Z_) = 1</sup><sup>_._</sup> _The first part of_ IOP<sup>req</sup> _Z_<sup>_,ontheearlypartofξ,leads(DefinitionA.18)fromC_first</sup><sup>_to_</sup> 



_(Definition A.27). The count of each round m, with challenge set Hm, is ⌊|Hm|/_ 2<sup>128</sup> _⌋._ 

_Outline._ Lean fixes, for each point between two sub-protocols, the set of claims that are open there. It divides the sub-protocols into thirteen kinds: Send; Code switch; Lookup; the own rounds of layers, of attention and of the head; the round that sends _y_<sup>+</sup> ; the matrix of the Range test; Batching; Batching with Fold; the merging of the head’s claims on the table; the Proximity test; and the reduction to one claim on the table. 

For each kind it proves that a sub-protocol leads, from every state that the first part can reach and that meets the condition before it, to the condition after it and to the implication that, if the claims open after it hold, so do those open before it; in Lean an explicit map sends values that satisfy the former to values that satisfy the latter. By Lemma A.19 these compose, since what is open after one sub-protocol is open before the next. 

The counts have the forms of Table 10, at the sizes that _Z_ fixes; in the round that sends _y_<sup>+</sup> no challenge is bad. They are transferred to challenges from hash blocks by Lemma A.20 and compared with _|Hm|/_ 2<sup>128</sup> , as natural numbers; for the attention segments this comparison is itself a parameter check. Since a larger count gives a weaker statement, every round is bounded by the common count. What the lemmas require of _Z_ is tested by Adm and Chk. 

**Lemma A.29 (The shared part and the last part).** [machine-checked] _Let Z be an admissible configuration with_ Chk( _Z_ ) = 1 _._ 

_1. Hypotheses 2 and 3 of Lemma A.21 hold for_ IOP<sup>req</sup> _Z_<sup>_,attheconditionsandclaimsofDefinitionA.27,withevery_</sup> _round error at most_ 2<sup>_−_128</sup> _._ 

_2. In step 1 of Construction D.7, ψ_<sup>_∗_</sup> = ( _ψ_ 0<sup>_∗, γ_)</sup><sup>_iscountedin_cntkey</sup><sup>_ifγisbad(LemmaA.24),andin_cntown</sup><sup>_if_</sup> _w̸_ = _h_<sup>_⊤_</sup> _A and ⟨_ eq( _ψ_<sup>_∗_</sup> _, ·_ ) _, w − h_<sup>_⊤_</sup> _A⟩_ = 0 _. For N_ req _≤ R requests the count is_ 



_with J and_ ls _as in Lemma A.24._ 

_3. In the other shared rounds the counts hold for any number of requests._ 

_Proof._ By _C_ shared, **T**<sup>_∗_</sup> _j_<sup>hasnocollisionat</sup><sup>_ςj_,andLemmaA.24applies.Let</sup><sup>_γ_notbebad.OncetheProximityteston</sup> the group of _u_ has established them, the constraints of steps 3 and 4 of Construction D.7 state conditions 2 and 1 of Lemma A.24 for _u_ . Hence _u_ is the fold of the decoded table. The constraint of step 7 then gives _ω_ = _A_ eq( _ψ_<sup>_∗_</sup> _, ·_ ), outside at most _k |_ E _|_<sup>_k−_1</sup> points _χ_ , where _k_ = _⌈_ log2 _d⌉_ . 

If _w̸_ = _h_<sup>_⊤_</sup> _A_ , the constraint _⟨_ eq( _ψ_<sup>_∗_</sup> _, ·_ ) _, w⟩_ = _⟨h, ω⟩_ is a nonzero multilinear equation in _ψ_<sup>_∗_</sup> , with at most _n |_ E _|_<sup>_n−_1</sup> solutions. This set depends on the encoded request, and it is counted _N_ req times. With _w_ = _h_<sup>_⊤_</sup> _A_ , the first constraint of _P_ last is _P_ shared. The other rounds have counts as in Table 10; Proposition D.8 evaluates steps 1 and 4 at _Z_ 0. 

**Lemma A.30 (The two other IOPs).** [machine-checked] 

_1. Let Z be an admissible configuration with_ Chk( _Z_ ) = 1 _. Then (Sound-key) of Definition A.6 holds with ε_ rd = 2<sup>_−_128</sup> _._ 

_2. Let Z be any configuration. Then (Sound-link proof) holds with ε_ rd = 2<sup>_−_128</sup> _._ 

_Outline. Part 1._ Lean proves a lemma for each of the 21 sub-protocols of IOP<sup>key</sup> _Z_ and composes them as for IOP<sup>req</sup> _Z_<sup>,</sup> with a batch of one. _Part 2._ IOP<sup>link</sup> _Z_ is treated as a whole, with radii _e_ 0 and _e_ 1 for the two tables. The count for _γ_ is the sum of the bounds of Lemma A.24 for the two tables and _v |_ E _|_<sup>_v−_1</sup> . For the positions it is the larger of ( _|D| − e_ 0 _−_ 1)<sup>_ν_first</sup> _|D|_<sup>_ν_copy</sup> and _|D|_<sup>_ν_first</sup> ( _|D| − e_ 1 _−_ 1)<sup>_ν_copy</sup> , and for _χ_ it is _k |_ E _|_<sup>_k−_1</sup> with _k_ = _⌈_ log2 _ℓ⌉_ . 

_Proof of Proposition 5.14._ For IOP<sup>key</sup> _Z_ and IOP<sup>link</sup> _Z_ this is Lemma A.30. For IOP<sup>req</sup> _Z_ we verify the hypotheses of Lemma A.21: the first holds by Lemma A.28, the second and third by Lemma A.29, the fourth by the definition of the verifier, the fifth by Lemma A.31, the sixth by the counts in these lemmas. 

64 

### **A.4 From the relation to the model** 

**Lemma A.31 (From the open claims to the request relation).** [machine-checked] _Let Z be an admissible configuration with_ Chk( _Z_ ) = 1 _. On every accepting transcript of_ IOP<sup>req</sup> _Z_<sup>_,ifC_first</sup><sup>_impliesP_first</sup><sup>_(DefinitionA.27),then_</sup> 



_Proof._ Let Dec( **T**<sup>_∗_</sup> _j_<sup>_, σj_)</sup><sup>_̸_=</sup><sup>_⊥_.Then</sup><sup>_C_firstholds,by(No-decoding-at-a-collision),and</sup><sup>_P_firstprovidescells.Bystep4for</sup> IOP<sup>req</sup> _Z_ in Construction D.9, the stored token is unchanged after each sub-protocol other than the one that receives it; hence the head’s claims concern the token received. By step 3, that token is _y_<sup>+</sup> . 

_Remark_ A.32 (Stored token). The test of step 4 concerns the verifier’s own state, not a prover message, and the proof above relies on it. That it changes no decision is known in part by a Lean theorem, by which the decision is the same for the written IOP and a second text of it, and in part by a reading, that the second text is the IOP without the test; at worst it costs completeness, not soundness. 

**Lemma A.33 (From the request relation to the policy).** [machine-checked] _Let Z be an admissible configuration. Let A be in range, and let x, y,_ fill _be lists of tokens such that ξ_ = build( _x, y,_ fill) _is defined and_ ( _A, ξ_ ) _∈_ Req _Z . Then_ wt( _A_ ) _is defined, and y meets_ pol _Z on x under the weights_ wt( _A_ ) _(Definition 3.4), in the integer model at µZ ._ 

_Proof._ The relation is stated modulo _p_ , for cells below 2<sup>18</sup> in absolute value whose lookup keys are entries of public tables. The model parameters _µZ_ are valid, since _Z_ is admissible, and _A_ is in range. Under these bounds the two sides of each identity are integers that differ by less than 2<sup>61</sup> _−_ 1 _< p_ in absolute value; a multiple of _p_ of that size is 0, and the identity holds over the integers. By induction on the groups of layers, the cells are the values of the integer model on the window, at the weights wt( _A_ ) and for every fill, and the model’s range conditions hold. The head’s identities and lookups then state the conditions of Definition 3.4. The bounds for the single identities are Lean’s and are not printed. 

**Lemma A.34 (The extractor).** [machine-checked] _Let Z be an admissible configuration. Then_ Ext _Z (Construction 4.15) is the extractor of Lemma A.15._ 

_Proof._ By unfolding the two definitions. 

### **A.5 Adding up the error** 

**Lemma A.35 (Numbers of queries).** [machine-checked] _Let Z be any configuration._ 

_1. At every value of the public readings, V_<sup>_◦_</sup> _makes at most qV ≤_ 2<sup>20</sup> _queries to the separate oracles._ 

_2. The public readings of VZ make at most_ 8 _queries to H._ 

_3. The parser, step 2 of VZ , makes at most_ 1 _query to H (Construction D.11)._ 

A query to a separate oracle costs one query to _H_ for each block of its answer; hence _qV_ does not bound the queries of _VZ_ to _H_ (Remark B.13). That _ε_ rd denotes 2<sup>_−_128</sup> is a definition; that the round errors and the probability of a collision are at most this number is Proposition 5.14 and Lemma A.25. 

**Lemma A.36 (The arithmetic of the bound).** [machine-checked] _Let q, q_ par _, qV ∈_ N _and ε_ rd _≥_ 0 _with ε_ rd _≤_ 2 _·_ 2<sup>_−_128</sup> _and qV_ + 9 _≤_ 2<sup>23</sup> _. Put T_<sup>¯</sup> = _q_ + _q_ par + 2<sup>23</sup> _and Q_ = 5 ( _q_ + 8 + _q_ par + _qV_ + 1) _, and suppose that_ 



_Proof._ The hypotheses give 16 _T_<sup>¯</sup> _<_ 2<sup>128</sup> , _Q ≤_ 5 _T_<sup>¯</sup> and _q_ + 8 + _q_ par + 2 _≤ T_<sup>¯</sup> + 2. Multiplied by 2<sup>256</sup> , the three terms on the left are at most 10 _T_<sup>¯</sup> 2<sup>128</sup> , 2 (5 _T_<sup>¯</sup> + 1)<sup>2</sup> _≤_ 6 _T_<sup>¯</sup> 2<sup>128</sup> and ( _T_<sup>¯</sup> + 2)<sup>2</sup> . 

_Proof of Theorem 5.1._ Let _Z_ be an admissible configuration. 

_Step 1._ The three conditions on soundness hold with _ε_ rd = 2<sup>_−_128</sup> . If Chk( _Z_ ) = 1, this is Lemma A.30 and, as in the proof of Proposition 5.14, Lemma A.21; otherwise it is Lemma A.22. Lemma A.17 gives the other hypotheses of Lemma A.15. 

65 

_Step 2._ Let Rel consist of the triples ( _M, x, y_ ) such that _y_ meets pol _Z_ on _x_ under _M_ . Lemma A.33 is the implication that Lemma A.15 requires, and Lemma A.34 identifies the extractor. By Lemma A.35, Π _Z_ is index-binding knowledge sound for Rel with the error (1). 

_Step 3._ Where _ε_ ( _q_ ) _<_ 1, Lemma A.36 with _q_ par = 1, hence _T_<sup>¯</sup> = _T_ , bounds (1) by _ε_ ( _q_ ). This proves Theorem 5.11. _Step 4._ At exact match, an output that meets the policy is a greedy output (Proposition 3.5). 

## **B Further statements and proofs for Section 5** 

Here, in Appendix C and Appendix D, and at Proposition 3.12, Lemma 4.9, Lemma 4.18, Lemma 5.13, and Lemma A.35, some machine-checked statements have neither a printed proof nor an outline. Their ground is the kernel’s check of the Lean statements that Appendix F names for them. 

**Proposition B.1 (Policies that every eligible output meets).** [machine-checked] _Let µ be valid model parameters (Definition 3.6) and_ pol = ( _C, θ, cap_ ) _an acceptance policy (Definition 3.4)._ 

_1. If_ M _F ≤_ K _F_ + 11 _and θ, C ≥_ 2436593409 _, then, for all weights M , every output that is eligible under M meets_ pol _under M . The policy_ (2<sup>32</sup> _−_ 1 _,_ 2436593409 _,_ 0) _is an instance._ 

_2. Let µ_ = _µZ_ 0 _(Definition 3.8), θ ≤ C <_ 2<sup>32</sup> _and_ 1 _≤ n_ ct _≤ L −_ 1 _. Then θ_ = _C and cap ≥ n_ ct _if and only if, for all weights M , every output with n_ ct _counted tokens that is eligible under M meets_ pol _under M ._ 

_Remark_ B.2 (All weights). What a clipped-gap policy excludes depends on the committed weights and on numbers that the provider writes into _Z_ . 

1. Up to _cap_ counted tokens may be any tokens, where the others leave room in the sum (Remark 3.10, What a policy bounds). 

2. Some policies of admissible configurations exclude no token (Proposition B.1). _VZ_ does not reject a configuration with such a policy, and the C++ verifier rejects some such policies but not every one: no verifier sees the model’s logits. 

3. At M _F ≤_ K _F_ + 11 some threshold of a configuration with these exponents, if it is admissible (which has been evaluated at three parameter sets, not proved), excludes no token, whatever the weights; the C++ verifier never prints these two exponents. 

4. The weights by which the proof of Proposition B.1 violates a policy are synthetic. Nothing follows for a trained model, under whose weights every eligible output may still meet such a policy. 

**Proposition B.3 (The example clipped-gap policy).** [machine-checked] _Let_ pol = (25856 _,_ 512 _,_ 1) _, the policy of Z_ 1 _(Definition 3.7), let µ be valid model parameters and M ∈I, and let y be an output that is eligible on a prompt x under M , with n_ ct _counted tokens and logit gaps_ gap _i (Definition 3.4)._ 

_1. If y meets_ pol _and n_ ct _≤_ 99 _, then_ gap _i <_ 25600 _for all i < n_ ct _, and_<sup>�</sup> _i<n_ ct<sup>gap</sup><sup>_i≤_256</sup><sup>_n_ct</sup><sup>_._</sup> 

_2. If y meets_ pol _, then for every integer g ≤_ 25600 _,_ 

_g ·_ # _{ i < n_ ct : gap _i ≥ g } ≤_ 256 _n_ ct _._ 

_3. If n_ ct _≥_ 100 _and all counted tokens but at most one are greedy tokens, then y meets_ pol _._ 

_Remark_ B.4 (Reading the theorem: particulars). To the items of Remark 5.3: 

3. No form of the theorem with a witness is stated, and Ext _Z_ reads nothing of the prover. It is straight-line, and no running time is stated. 

4. The weights that Ext _Z_ reads off a commitment and the committer’s queries need not be the same at all configurations. 

5. Configurations with other policies are admissible. 

_Remark_ B.5 (Bits per query). The bound _ε_ ( _q_ ) of Theorem 5.1 is at most about 16 ( _q_ + 2<sup>23</sup> ) _/_ 2<sup>128</sup> : 124 bits per query. It is never below about 2<sup>_−_101</sup> , it is about 2<sup>_−_60</sup> at _q_ = 2<sup>64</sup> , and it says nothing from about _q_ = 2<sup>124</sup> on. A query returns 32 bytes. A call of BLAKE2b that returns 64 bytes, as each call for a Fiat–Shamir challenge does, an attempt at a proof of work among them, counts as two queries, and every other call as one. So 124 bits per query are at least 123 bits per call; this conversion is a reading, not a theorem, and it counts an attempt at a proof of work as a whole call, whereas in the released programs one attempt can cost less than one hash call, so that _κ_ counted bits mean fewer than 2<sup>_κ_</sup> calls; by how much is not known. 

66 

_Remark_ B.6 (Acceptance). What is known beyond Limitation 8.9 (Completeness on served requests only): the runs in which the executable form accepted honest proofs are at _Z_ 0 and _Z_ 1, at the format’s hash function (Section 7.3); no run concerns another hash function, and, besides the reduction of Remark 6.49, the one theorem on acceptance is over a random oracle and on the Lean prover’s proofs (Section 6.5). Beyond Proposition 5.6, the parameter checks have been evaluated, and pass, at seven admissible configurations, over Lean files that differ from the released ones, by a report alone ( `lean/GAPS.md` , gap 52; `lean/THEOREM.md` ); at all others nothing is known. _VZ_ can reject on _Z_ alone at further places, hundreds by a rough count, not traced one by one; lemmas settle two of them at _Z_ 0, and no statement says that the list is complete. Where _Z_ is admissible, Chk( _Z_ ) = 1 and the policy is exact match, as at _Z_ 0, Section 6.5 gives more: some request is served there, at constructed weights, and _VZ_ rejects the Lean prover’s proof for it with probability at most 2<sup>_−_40</sup> over the random oracle and the honest coins. So no test whose outcome is a function of _Z_ and _H_<sup>_′_</sup> alone makes _VZ_ reject every input, and a test that also reads _H_ at strings formed from _Z_ does so with probability at most 2<sup>_−_40</sup> over _H_ ; at the format’s hash function nothing follows for such a test, and no lemma settles a test that only other requests meet. This is an inference from these statements; no Lean theorem states it. At other _Z_ the question is open. 

_Remark_ B.7 (Which compiler). Two compilers occur: the Lean library’s own (Construction D.11) and the generic compilation of Definition 2.4. Steps 4 to 9 of _VZ_ (Construction 4.15) are, by definition, the output of the first at a hand-written input (the three IOPs, the assignment of their rounds to hash queries, and the codings) that depends on _Z_ and the digests of step 3. Lemma A.15 is machine-checked for this compiler, and Theorem 5.1 rests on it. That the first is an instance of the second is not machine-checked; Theorem 5.1 does not need it. In the Lean term for Π _Z_ the model, the committer and the prover are fields, the last two constants (Remark 4.16, Honest parties). A further Lean function, compile (Appendix F at 6.15; its output is the argument of Definition 6.15), which the release’s page `lean/STATEMENTS.html` shows, is a layer around the first: it returns _VZ_ (by a proved equation); soundness, completeness and zero knowledge are all stated of its output; the soundness theorem is proved not through it but through the compilation of Construction D.11 (Lemma A.15). The library also has a second theorem with the statement of Theorem 5.1 to the letter ( `sound_of_verify_at_through_the_IOP` ), which draws the three protocols’ soundness from the one round-by-round statement of Remark 6.21, item 1, and hands it to the same compilation lemma: it is not an independent proof. 

### **B.1 The executable form** 

**Construction B.8 (Algorithm of the executable form).** _Coding._ A bit string is _plain_ if its length is a multiple of 8 and its first eight bits do not spell the byte 4. For a plain bs, byt(bs) is the byte string with the bits of bs. Otherwise byt(bs) consists of the bytes 4 and _|_ bs _|_ mod 8 and of bs, extended by zeros to whole bytes. Bytes are read from the lowest bit, and a byte string of fewer than 32 bytes is extended by zeros. 

_Algorithm._ On input _Z_ , _O_ , _c_ , _x_ , _y_ and _π_ = (fill _, π_ 1 _, . . . , π_ 5): 

1. If Adm( _Z_ ) = 0, return 0. 

2. Let _o_ 0 = _O_ (byt(code( _Z_ ))), and let _o_ 1 _, . . . , ok_ , _k ≤_ 7, be the values of _O_ at the codings of the further public strings of step 3 of Construction 4.15, formed with _o_ 0 as the digest of _Z_ . 

3. If some _oi_ does not have 32 bytes, return _⊥_ . If Chk( _Z_ ) = 0, return 0. 

4. Unpack ( _π_ 1 _, . . . , π_ 5), with _o_ 0 and without a call to _O_ , into k, _j_ , rt _j_ , _σj_ and the data of three proof strings. If this fails, return 0. 

5. Evaluate steps 4 to 9 of Construction 4.15 as tests, with _O_ in the place of the hash function: build( _x, y,_ fill) is defined; k has a code; the hash value of that code is _c_ ; each of _π_ key, _π_ link, _π_ req can be formed from its data; the plain part of _π_ req holds _j_ ; _ς_ 0 and _ςj_ are the seeds of rt0 and rt _j_ ; the three compiled verifiers accept. Return 1 if all hold, and 0 otherwise. 

**Corollary B.9 (Program verdict).** [machine-checked] 

_1. For all Z, all functions O from byte strings to byte strings and all c, x, y, π: if EZ_<sup>_O_(</sup><sup>_c, x, y, π_)</sup><sup>_̸_=</sup><sup>_⊥,then_</sup> _EZ_<sup>_O_(</sup><sup>_c, x, y, π_) =</sup><sup>_V_</sup> _Z_<sup>_HO_</sup> ( _c, x, y, π_ ) _._ 

_2. Let_ g _be a tuple of eleven byte strings: the three files of a configuration Z_ g _, three files of the client, and five of a proof. Let_ claim _and_ verdict _be the two Lean functions that the command_ _`lean-verify` evaluates on_ g _, where_ claim(g) _reads the client’s files and is ⊥ or a tuple_ ( _c, x, y,_ fill) _, and let π_ g _consist of_ fill _and the five. Then_ verdict(g) = _⊥ if_ claim(g) = _⊥, and otherwise, with O_<sup>_∗_</sup> _of Assumption 4.10,_ 

verdict(g) = _VZH_ g _O∗_ ( _c, x, y, π_ g) _._ 

_Proof of Theorem 5.5 and Corollary B.9._ Part 1 follows the cases of Construction B.8. If Adm( _Z_ ) = 0 or Chk( _Z_ ) = 0, then _VZ_ rejects at every hash function. If _π_ cannot be unpacked, _VZ_ rejects (at step 4, on the fixed tuple of 

67 

Construction D.11). Otherwise one lemma, the main step, equates the conjunction of these tests with the value of steps 2 to 9 at _HO_ . If every answer of _O_ has 32 bytes, the algorithm does not return _⊥_ ; this holds of _O_<sup>_∗_</sup> , and verdict is claim followed by _E_<sup>_O∗_</sup> . 

_Remark_ B.10 (Claim from files). _Z_ , _c_ , _x_ and _y_ must come from the client. That claim reads the client’s files as the format’s specification intends is not proved, and an error there would make _VZ_ decide another claim than the client’s. The proof needs no trust, nor does the program outside Lean that makes _π_ from the release’s proof files: Theorem 5.1 holds for every proof string. 

A run also trusts the part of the command that reads files and prints, and Lean’s evaluator, compiler and runtime. That none of the library’s definitions beneath the executable form is replaced by other code when compiled rests on a walk in Lean over what the command runs, repeated whenever the library is compiled, which lists every constant there that carries replaced or external code (none is this library’s). The command is part of the release’s source tree, not a stand-alone program. In Corollary B.11 the functions _OH_ are defined with the axiom of choice, have no code and are not _O_<sup>_∗_</sup> . 

**Corollary B.11 (Soundness of the executable form).** [machine-checked] _Let Z be as in Theorem 5.1. For each hash function H let OH be a function from byte strings to 32-byte strings that induces H (Definition 5.4); such a family exists. Then the bound of Theorem 5.1 holds, with the same error ε_ ( _q_ ) _, for the experiment of Definition 2.7 in which the decision on_ ( _c, x, y, π_ ) _is EZ_<sup>_OH_</sup> ( _c, x, y, π_ ) _._ 

_Proof._ By Theorem 5.5 the two experiments have the same event at every _H_ . A family exists because byt is injective: let _OH_ (byt(bs)) be _H_ (bs), written as 32 bytes, and let _OH_ take any 32 bytes elsewhere. 

_Remark_ B.12 (Malleable proofs). That an accepted proof cannot be altered into another accepted proof of the same instance is proved nowhere, and altered proofs were accepted in runs (Section 7.3). A false instance gains nothing by this: Theorem 5.1 bounds its acceptance whatever a proof holds. 

### **B.2 Many proofs** 

_Proof of Corollary 5.10._ With the extractor of Theorem 5.1, for _i < N_ let _AP,i_ run the given prover and output entry _i_ of its list, or a fixed triple; it makes the same queries. At every _H_ and all random choices, the event of the corollary implies that of Definition 2.7 for some _AP,i_ , which has probability at most _ε_ ( _qC_ + _qP_ ). 

_Remark_ B.13 (Lists sent at once). In Corollary 5.10 the prover sees no decision of the verifier and no honest proof before it sends its list. For proofs that are sent one after another, each after the decision on the one before, no bound is stated. For lists sent at once, the bound _ε_ ( _q_ + _NqH_ ), without the factor _N_ , is machine-checked if every run of _VZ_ makes at most _qH_ queries to _H_ . Such a _qH_ is not known (Open problem 8.15): the bound _qV_ of Lemma A.35 counts queries to the separate oracles, each of which costs one query to _H_ for each digest of its answer. 

**Corollary B.14 (Evaluation binding).** [machine-checked] _Let Z and ε be as in Theorem 5.1, and let N , qC , qP , the committer and the prover be as in Corollary 5.10. With probability at most N · ε_ ( _qC_ + _qP_ ) _, VZ accepts two entries i, j of the prover’s list with xi_ = _xj, |yi|_ = _|yj| and yi̸_ = _yj._ 

_Proof._ For fixed weights, prompt and length determine a greedy output. Of two accepted entries with equal prompts and lengths and different outputs, at most one lies in Lang( _M_<sup>_∗_</sup> ), so the event of Corollary 5.10 occurs. 

_Remark_ B.15 (Evaluation binding). Corollary B.14 has the form of evaluation binding for functional commitments [BNO21], for the function that maps a prompt and a length to an output and for two proofs in one list. It is neither extractability of the commitment nor the absence of two openings (Definition 2.10). 

### **B.3 Other policies, parsers and the parameter checks** 

Theorem 5.11 is proved in step 3 of the proof of Theorem 5.1 (Appendix A.5), and Proposition 5.14 in Appendix A.3. 

**Corollary B.16 (Soundness with any parser).** [machine-checked] _Let Z be as in Theorem 5.1. In VZ (Construction 4.15), replace the parsing of step 2 by any deterministic oracle algorithm that makes at most q_ par _queries to H and maps x, y and a proof, from any set of proofs, to a tuple of the form that step 2 produces. Then the resulting argument is index-binding knowledge sound (Definition 2.7) for the relation of greedy outputs at µZ , under the same assumptions, with error_ 



_At every admissible Z, the same holds for the relation of Theorem 5.11, with the extractor_ Ext _Z ._ 

68 

_Proof._ The proof of Theorem 5.1 uses of the parser only its number of queries (Lemma A.15 and Lemma A.36). 

_Proof of Proposition 5.6._ Admissibility and the policy of _Z_ 0 follow from tests on its byte strings that Lean’s kernel evaluates. Chk( _Z_ 0) = 1 is unfolded into closed facts, many of them proved by evaluation in the kernel (Remark 5.7, How the proposition is checked). Among admissible configurations Chk does not depend on the policy. 

_Remark_ B.17 (Checkers). Lean’s kernel accepts the proofs of the Lean theorems that the machine-checked statements of Section 5 and this appendix render or join, in a build of the released Lean files with no error and no `sorry` ; for those named in `lean/CHECK-RESULTS.md` Lean prints no axiom beyond its three standard ones. No lemma about a step of the protocol and no conjecture on codes is assumed, and the random oracle and _H_<sup>_′_</sup> belong to the statements. An independent proof checker has accepted the soundness theorem on `verify` , `sound_of_verify_at` (the form of Theorem 5.1 that leaves the extractor open: Section 5, head), with every declaration beneath it, and nothing that does not lie beneath it, in a pass on that theorem’s export alone: the pass was made over a copy of the compiled files of the released Lean files’ build. Its passes by name on `sound` , `repliesFit` and `refusedHandsOutNoProof` were made over that build too; that the compiled files that these passes read were that build’s is known from a report alone; Lean’s own files were not compared with a list of checksums. The checker’s other passes were made over Lean files that differ from the released ones. It has accepted neither Proposition 5.6 (Remark 5.7, How the proposition is checked) nor Theorem 5.17 nor Theorem 5.19. The Lean verifier’s reference run (the 109 calls of Appendix H.3) has been made over the released Lean files; what these are, and what has run: `lean/CHECK-RESULTS.md` . One whole run of the library’s full check ( `lean/scripts/check.py` ) over all but one of the released Lean files has the last line “All steps passed: tree, words, imports, build (kernel replay, audit). . . . ”. That line also says that no theorem got a verdict on binders or on `Classical.choice` : every line of `lean/scripts/top-theorems.txt` says “general” (the whole line, and what the check does not cover: `lean/CHECK-RESULTS.md` ). It is one run, on one machine. In its kernel replay Mathlib and the other packages are taken as they were compiled beforehand, and are not replayed ( `lean/GAPS.md` , 81 (g)); which modules go by another call than `leanchecker` : Remark 5.7. One Lean file, with one definition and four theorems, which nothing else imports, was added after the runs reported here and checked on its own. Three of the check script’s pinned values were changed and three of its stored lists were extended for it; nothing else in the script or the lists changed. The full check’s run builds the three libraries from nothing in one call of Lake; the other results told here stand on a second build of the same files, from nothing, in several calls ( `lean/CHECK-RESULTS.md` ). The checker judges a text that this project’s own exporter writes from the compiled library; the exporter is not in the repository, so a reader cannot repeat the pass; and no run shows the checker rejecting a wrong proof. The pinned version of Lean, 4.29.1, is older than the published fixes (from version 4.32.1 on) of soundness bugs in Lean’s kernel and runtime and in the big-number library (GMP) that its Linux builds link. No proof in this library is known to use such a fault; the proofs have not been searched for one, and the check script’s word list and audit would not detect one. This paper tells no check of the library with a later version of Lean ( `lean/LIMITS.md` , L45). 

_Remark_ B.18 (Typed statement). Lemma 5.13 is the statement as typed in the Lean library. That it follows from the two-word case of the published theorem [BCHKS25, Theorem 4.6] rests on a reading; no theorem says it, and the lemma does not make that theorem, in its own form, formally verified. The Lean proof derives it from two theorems in files adapted from ArkLib [ArkLib] and Mathlib [Mathlib]. 

### **B.4 The commitment** 

**Definition B.19 (Soundness after an honest commitment).** Let Π = (Rnd _,_ Commit _, P, V_ ) and _I_ be as in Definition 2.9, let _Q_ be a predicate on triples ( _M, x, y_ ) of an index, a prompt and an output, and let _ε_ : N _→_ [0 _, ∞_ ]. For an index _M_ and an oracle algorithm _AP_ , the experiment is: sample the random oracle _H_ ; then 



The argument Π is _sound after an honest commitment_ for _Q_ with error _ε_ if, for every _M ∈I_ , all _qP ∈_ N and every _qP_ -query algorithm _AP_ , 



_Remark_ B.20 (Composition). Suppose that the bound of Definition 2.7 holds for Π, a relation with languages Lang( _·_ ), a given extractor Ext and an error _ε_ ; that Π has correctness of extraction for the same Ext with error _ε_ ext; and that Rnd followed by Commit makes at most _q_ 0 queries. Then Π is sound after an honest commitment (Definition B.19) for the predicate ( _x, y_ ) _∈_ Lang( _M_ ), with error 

_ε_ ( _q_ 0 + _qP_ ) + _ε_ ext _._ 

The implication is machine-checked for every argument. Theorem 5.19 states this property of Π<sup>Γ</sup> _Z_<sup>.</sup> 

_Remark_ B.21 (Implications). For every argument and every extractor, two implications are machine-checked. 

69 

- Extractability of the commitment with error _ε_ op implies no two openings with error 2 _ε_ op. 

- If Rnd makes no query and outputs a function of one uniform number, and Commit makes at most _q_ 0 queries on every input, it also implies correctness of extraction for the same extractor with error _ε_ op( _q_ 0). 

No two openings does not exclude two openings to one index with different randomness. 

_Remark_ B.22 (Proof outlines for the commitment). The outlines follow the Lean proofs. 

- _Part 2 of Theorem 5.17 with γ (Proposition 5.18)._ Where Chk( _Z_ ) = 1, at every hash function the explicit committer outputs the commitment of a second committer, which is stated over the separate oracles of Construction D.11. This rests on five equations on single queries and two on the encoder and the OOD answers; the equation on leaves needs that both committers’ leaf salts have one length, which Chk( _Z_ ) = 1 gives. For the second committer, take the trace of _AC_ and then the traces of the opener and of the test that ( _M,_ r) opens _c_ . With probability at most _γ_ ( _qC_ + _qO_ ), a hash event occurs on these traces, or the word under rt0 changes when the other two traces are added. Otherwise _c_ fixes the key, and Ext _Z_ reads off the trace of _AC_ the table that the test forms. The decoder returns the digit matrix of every honestly formed table (Definition 4.5). 

- _Theorem 5.17._ Part 2 follows because _γ ≤ β_ wherever _β <_ 1. Parts 1 and 3 follow from part 2 by Remark B.21, with _q_ 0 = 2<sup>35</sup> + 2. 

- _Theorem 5.19._ Where Chk( _Z_ ) = 0, _VZ_ rejects every input. Where Chk( _Z_ ) = 1, part 1 is the implication of Remark B.20, applied to Theorem 5.11 with the extractor Ext _Z_ . Its other two inputs are part 1 of Theorem 5.17 and _q_ 0 = 2<sup>35</sup> + 2. Part 2 follows by Proposition 3.5. 

_Remark_ B.23 (Constants of the errors). 

   - The bound 2<sup>35</sup> + 2 on the explicit committer’s queries is machine-checked, on every input. It is the argument of _β_ in part 1 of Theorem 5.17 and a summand of _q_ in Theorem 5.19: the honest committer’s queries are counted among the adversary’s. 

   - In _β_ , the summand 2<sup>23</sup> is the allowance for the verifier’s queries in Theorem 5.1, although no verifier runs here. 

   - With _γ_ (2<sup>35</sup> + 2) (Proposition 5.18) in the place of _β_ (2<sup>35</sup> + 2), part 1 of Theorem 5.19 is likewise a single Lean theorem, joined with the theorem on admissibility as in Remark 5.20. 

- **Proposition B.24 (Two degenerate committers).** [machine-checked] _Let Z and_ Γ _be as in Theorem 5.19._ 

   _1. If the explicit committer replaces every digit matrix by the all-zero matrix, none of the three properties of Theorem 5.17 holds, for_ Ext _Z and the errors stated there._ 

   _2. The_ constant committer _always outputs_ 0<sup>_λ_</sup> _, without randomness or queries. With it in the place of the explicit committer, and any predicate on_ ( _M, x, y_ ) _in the place of “y meets_ pol _Z on x under M ”, part 1 of Theorem 5.19 holds with q_ = _qP and without the term β_ (2<sup>35</sup> + 2) _._ 

_Remark_ B.25 (What the two committers show). All of the following is machine-checked. 

- Part 2 of Proposition B.24: soundness after an honest commitment (Definition B.19) does not separate the explicit committer from the constant one. For the predicate that never holds, it states that _VZ_ accepts after the constant commitment with probability at most the error of Theorem 5.1. 

- Part 1 holds in a stronger form: with the all-zero matrix, correctness of extraction fails for every extractor at every error below 1 _/_ 2, and no two openings for every error function whose value at 2<sup>35</sup> + 2 is below 1. 

- With completeness, the property excludes that committer. If Π<sup>Γ</sup> _Z_<sup>withtheall-zeromatrixanda</sup><sup>_q_-queryprover</sup> is complete on the instances that can be served (Definition 2.12 and Definition 3.11) with error _α_ , and sound after an honest commitment for greedy outputs with error _ε_ , then _α_ + _ε_ ( _q_ ) _≥_ 1. Completeness is proved for no committer of Section 5.5. 

## **C Proofs on zero knowledge** 

Appendix C.2 proves Theorem 6.3 and Proposition 6.5: Lemma C.8, Lemma C.9, and Lemma C.10 treat one table, and Lemma C.12 many tables together with runs of the IOPs. Appendix C.3 proves Theorem 6.6 and Corollary 6.7, and Appendix C.4 concerns the link proof. 

70 

### **C.1 The definition of index privacy** 

_Remark_ C.1 (Environment and simulator). Definition 2.14 is adapted from [CFWpc]. Its reach and its limits are these. 

- The environment sees the whole view of _V_<sup>_∗_</sup> and may decline, but it is a function of that view and has no coins of its own: no machine-checked statement extends a bound to environments with coins. 

- Batches, with their revealed and hidden instances, are this paper’s addition to the definition. The addition has had no independent check. 

The simulator’s running time is not bounded. 

### **C.2 Committed tables** 

**Definition C.2 (Disclosed view, honest-verifier simulator).** Let _P_ be a prover for an IOP (Definition 2.3) in which each oracle part is marked _masked_ or not. Let ch denote the challenges of a run. The _abort set_ of _P_ at an input is a set of values of ch, the union of an input-independent _public part_ and a _private part_ . 

The _disclosed view_ at ch consists of the plain parts of _P_ , its oracle parts that are not masked, and the symbols that the verifier reads in the masked ones and in the external oracle strings. The _output_ of _P_ at ch is _⊥_ if ch lies in the abort set, else the disclosed view. 

An _honest-verifier simulator_ is a randomized algorithm given the IOP instance, ch, and the symbols that the verifier reads in the external oracle strings. It is _exact_ at an input and ch if its output is distributed as that of _P_ . 

The core of this notion, a view of the verifier and a simulator with limited access to the oracle strings, goes back to [BCFGRS17], as [CFW26, version of February 25, 2026, Definition 2.1] attributes it; here a prover may abort, and a simulator is exact at single values of ch. 

**Definition C.3 (Table system).** A _table system_ consists of a set of indices; an encoder from an index and randomness, uniform in a finite set, to a _table_ , a string of columns; a finite set of OOD seeds with a map from index, randomness and seed to OOD answers; and three IOPs (Definition 2.3). Their external oracle strings are one table, two tables, and one table; their IOP instances are the table’s OOD sample, both samples, and, in a batch IOP, pairs of a request and the sample. FlashProve _Z_ (Construction 4.13) is a table system whose indices are the matrices over F shaped as in Definition 4.1, encoded as in Definition 4.2 and Definition 4.3. 

**Definition C.4 (Four conditions on the tables).** Let a table system (Definition C.3), a set _U_ of OOD seeds and _εU ∈_ [0 _, ∞_ ] be given. 

**(Bounded reads)** The first verifier reads at most _ν_ off columns of its table, the second at most _ν_ first of its first table and _ν_ copy of its second, the third at most _ν_ on for a batch. For every seed in _U_ and every set _Q_ of at most _t_ positions, the OOD answers with the columns at _Q_ have the same distribution for all indices. 

**(First-table budget)** _ν_ off + _Kν_ first _≤ t_ . 

**(Copy budget)** _ν_ copy + _N_ c _ν_ on _≤ t_ . 

**(Good seeds)** A seed derived from uniform digests lies outside _U_ with probability at most _εU_ . 

**Definition C.5 (Table experiment).** Let a table system (Definition C.3), provers **P** for its IOPs and an index _A_ be given. In Tab<sup>**P**</sup> ( _A, A_ ) an unbounded randomized algorithm _A_ names an OOD seed _ς_ 0; then **T** 0 is sampled, and _A_ receives its OOD answers. Next _A_ adaptively asks for: 

- _copy_ : _A_ names _ςj_ ; then **T** _j_ is sampled with fresh randomness, and _A_ receives its answers; 

- _column_ : _A_ receives **T** _j_ [ _ϕ_ ] for _j_ and _ϕ_ of its choice; 

- _run_ : the first IOP on **T** _j_ , the second on ( **T** 0 _,_ **T** _j_ ) with _j ≥_ 1, or the third on **T** _j_ and a batch of its choice is run with the honest verifier; _A_ receives ch and the output of _P∗_ (Definition C.2), in the third IOP for the revealed requests. 

The output is that of _A_ . Tab _Z_ is the case of FlashProve _Z_ with constant-message provers. 

**Definition C.6 (Within the opening budget).** Let _ν_ off _, ν_ first _, ν_ copy _, ν_ on be as in (Bounded reads) of Definition C.4; for FlashProve _Z_ they are 62 _,_ 62 _,_ 62 _,_ 64. An algorithm _A_ in the table experiment (Definition C.5) is _within the opening budget t_ if there are natural numbers _K_ and _Nj_<sup>key</sup> _, Nj_<sup>link</sup> _, Nj_<sup>req</sup> _, Nj_<sup>col</sup> , for 0 _≤ j ≤ K_ , such that, for every _j_ , 



and in every execution _A_ asks for at most _K_ copies and, for each _j_ , at most _Nj_<sup>key</sup> runs of the first IOP on **T** _j_ , _Nj_<sup>link</sup> of the second on ( **T** 0 _,_ **T** _j_ ), _Nj_<sup>req</sup> of the third on **T** _j_ , and _Nj_<sup>col</sup> columns of **T** _j_ . 

71 

_Remark_ C.7 (The budget covers a table’s whole use). The inequality of Definition C.6 counts every column of **T** _j_ that any run ever opens. The numbers 62 and 64 are upper bounds; where Chk( _Z_ ) = 0 the verifiers of IOP<sup>key</sup> _Z_ and IOP<sup>req</sup> _Z_ read no column. With one offline proof, _K_ link proofs and _N_ c online proofs on each copy it reads 62 + 62 _K ≤ t_ and 62 + 64 _N_ c _≤ t_ , that is, _K ≤_ 710 _,_ 324 and _N_ c _≤_ 688 _,_ 127. 

**Lemma C.8 (Columns are uniform).** [machine-checked] _Let fr ∈_ F<sup>_ℓ_</sup> _for r <_ 2<sup>_v_</sup> _be arbitrary messages, and let Q ⊆ D with |Q| ≤ t. Over independent uniform τr ∈_ F<sup>_t_</sup> _, the values_ Enc( _fr_ ; _τr_ )( _ϕ_ ) _, for r <_ 2<sup>_v_</sup> _and ϕ ∈ Q, are independent and uniform in_ F _._ 

_Proof._ By Definition 4.2, Enc( _f_ ; _τ_ )( _ϕ_ ) = _τ_ ( _ϕ_ ) + _ϕ_<sup>_t_</sup> _f_ ( _ϕ_ ), where _τ_ ( _X_ ) =<sup>�</sup> _j<t_<sup>_τjXj_.Thelinearmap</sup><sup>_τ�→_(</sup><sup>_τ_(</sup><sup>_ϕ_))</sup><sup>_ϕ∈Q_</sup> from F<sup>_t_</sup> to F<sup>_Q_</sup> is given by a Vandermonde matrix on _|Q| ≤ t_ distinct points, so it is surjective. Hence ( _τ_ ( _ϕ_ )) _ϕ∈Q_ is uniform, also after the fixed vector ( _ϕ_<sup>_t_</sup> _f_ ( _ϕ_ )) _ϕ∈Q_ is added. Rows have independent coefficients. 

**Lemma C.9 (Pads mask the answers).** [machine-checked] _Let s_ = 1 _. For every OOD seed_ ( _z_ 0 _, β_ 0) _and all plain entries of a row, the map from the row’s three pads to its OOD answer (Definition 4.3) is a bijection from_ F<sup>3</sup> _onto_ E _. Hence, over uniform pads, the OOD answers of a table are independent and uniform in_ E _, for every matrix._ 

_Proof._ By Definition 4.3, _ar,_ 0 is the sum of _β_ 0 pad _r,_ 0 and a term that depends on the plain entries and on _z_ 0 alone. The three pads are the coordinates of pad _r,_ 0 in the basis 1 _, ϑ, ϑ_<sup>2</sup> of E over F, and _β_ 0 _̸_ = 0. Hence pads map to _ar,_ 0 bijectively. Rows have independent pads. 

Masking an OOD answer (Definition 4.3) with fresh random entries of the message follows the private zero-evaders of [CFW26, version of February 25, 2026, Section 9.2]: Lemma C.9 is in substance the privacy part of its Lemma 9.3, with its Definition 9.2, at one answer in E with three pads in F. 

**Lemma C.10 (One table).** [machine-checked] _Let t_ = 44 _,_ 040 _,_ 191 _and s_ = 1 _, at the table sizes of the default configuration. For every OOD seed and every Q ⊆ D with |Q| ≤ t, the distribution of the pair of the OOD answers and the columns_ ( **T** [ _ϕ_ ]) _ϕ∈Q of a committed table_ **T** _for A is the same for all matrices A over_ F _._ 

_Proof._ Fix the pads, and with them all messages _fr_ and all answers. By Lemma C.8 the columns at _Q_ are uniform, whatever the pads are. Hence the columns are independent of the pads, and so of the answers. By Lemma C.9 the answers are uniform; so is the pair, for every _A_ . 

**Lemma C.11 (Column counts).** [machine-checked] _Let Z be admissible. The verifier of_ IOP<sup>key</sup> _Z reads at most_ 62 _columns of its table, that of_ IOP<sup>link</sup> _Z at most_ 62 _of each of its two tables, and that of_ IOP<sup>req</sup> _Z at most_ 64 _for a batch. With Lemma C.10,_ FlashProve _Z satisfies (Bounded reads) of Definition C.4 at all OOD seeds._ 

**Lemma C.12 (Tables and runs before compilation).** [machine-checked] _Let a table system satisfy (Bounded reads) at all OOD seeds (Definition C.4), and let Y be a set of pairs of an index and a request. Let_ **P** _be provers whose external oracle strings are the tables of their input. An input is_ valid _if its IOP instance holds the OOD samples of its tables and its requests form pairs of Y with its index. Let each_ Sim _∗ be exact, at every valid input, except with probability ζ∗_<sup>sim</sup> _over uniform_ ch _(Definition C.2). Fix an index A_ 0 _, and let_ Tab<sup>Sim</sup> _be the table experiment (Definition C.5) with_ Sim _∗, on the samples and opened columns, in place of each run of P∗. For every index A and every A within the opening budget (Definition C.6) whose requests form pairs of Y with A, at most R_ rev _≥_ 1 _of a batch being revealed,_ 



_Proof. Step 1: simulated runs on the true tables._ Replace each run of _P∗_ by Sim _∗_ , given the true OOD samples and the true columns that the verifier reads. One replacement changes the distribution by at most _ζ∗_<sup>sim</sup> , or _R_ rev _ζ_ req<sup>simfora</sup> batch; the sum is the stated bound. 

_Step 2: one reader. A_ with the simulators is one randomized algorithm that depends on the tables only through their OOD answers and columns that it reads adaptively, at most _t_ of each table by (Bounded reads) and Definition C.6. 

_Step 3: all tables at the start._ Sample one independent table for every pair ( _j, ς_ ) at the start; when _A_ names _ςj_ , use the table of ( _j, ςj_ ). Since _A_ names one seed for each _j_ , the distribution is unchanged. 

_Step 4: from fixed to adaptive positions._ The probability of a transcript of the reader is the product of the probabilities of its own coins and, for each table, of the event that the answers and the columns at the transcript’s positions take the transcript’s values. These positions form a fixed set of size at most _t_ , so by (Bounded reads) the probability is the same for _A_ and _A_ 0. 

Steps 1 and 2 of the proof above, which replace the runs by simulators and take _A_ with these as one reader, follow the composition of honest-verifier zero knowledge in [CFW26, version of February 25, 2026, Definitions 4.4 and 4.7 and Theorem 4.5]. 

72 

**Definition C.13 (Table game).** Let _U_ be a set of OOD seeds, and let _K_ , _Nj_<sup>key</sup> , _Nj_<sup>link</sup> , _Nj_<sup>req</sup> , _Nj_<sup>col</sup> be numbers that satisfy the inequality of Definition C.6. The _table game_ is the table experiment (Definition C.5) that answers _⊥_ beyond these numbers and at seeds outside _U_ , with one request in a batch. In each run an _auxiliary value_ is sampled from a fixed distribution, which may depend on the IOP, the table’s number and the IOP instance; ch, or _⊥_ , is a function of it. _A_ receives the auxiliary value and, unless ch = _⊥_ , the output at ch. 

**Lemma C.14 (Tables and runs in the table game).** [machine-checked] _In the setting of Lemma C.12, let (Bounded reads) hold for U in place of all OOD seeds. Fix an index A, and let each_ Sim _∗ be exact except with probability ζ∗_<sup>sim</sup> _under the distribution of_ ch _in the table game (Definition C.13), at the valid inputs with index A and seeds in U . For every index A_ 0 _and every A that asks for at most N runs of the third IOP, at requests that form pairs of Y with A, the outputs of the game at A with_ **P** _and of the game at A_ 0 _with_ Sim _∗ are within statistical distance_ 



_Proof._ As for Lemma C.12; Step 1 uses the hypothesis on Sim _∗_ at _A_ alone. 

_Proof of Theorem 6.3._ By Lemma C.11, (Bounded reads) holds at all seeds. The simulator that outputs the fixed messages and the symbols that it is given is exact at every ch. Let _Y_ hold all pairs and _A_ 0 = _A_ . By Lemma C.12, Tab _Z_ ( _A, A_ ) and Tab _Z_ ( _A_<sup>_′_</sup> _, A_ ) both have the distribution of Tab<sup>Sim</sup> ( _A_ 0 _, A_ ). 

_Proof of Proposition 6.5._ Suppose that the distributions agree for some _Q_ . The value that _A_<sup>_′_</sup> produces with all pads and random coefficients 0 then has positive probability for _A_ = 0, at some pads and ( _τr_ ) _r_ . At _ς_<sup>_◦_</sup> the entry _f_ 0[0] has coefficient eq((1 _, . . . ,_ 1) _,_ 0) = 0, and the other plain entries of _f_ 0 vanish for both matrices. Hence both answers _a_ 0 _,_ 0 are images of the pads alone, and by Lemma C.9 the pads of _f_ 0 are 0 for _A_ too. The codeword of _f_ 0 is then the evaluation of _X_<sup>_t_</sup> for _A_<sup>_′_</sup> and of _τ_ 0( _X_ ) for _A_ . They agree on _Q_ , so the nonzero polynomial _X_<sup>_t_</sup> _− τ_ 0( _X_ ) of degree _t_ has _t_ + 1 roots, a contradiction. 

_Remark_ C.15 (What tightness shows). Proposition 6.5 negates Lemma C.10 at _t_ + 1 columns. It follows, though not machine-checked, that the conclusion of Theorem 6.3 fails for the algorithm that names _ς_<sup>_◦_</sup> , asks for the columns at _Q_ and outputs what it receives. It bounds no distance and concerns one seed, at which eq((1 _, . . . ,_ 1) _, j_ ) = 0 for every _j < ℓ_ 0. It is not shown that wt( _A_ ) _̸_ = wt( _A_<sup>_′_</sup> ). 

**Proposition C.16 (Linear equations beyond the opening budget).** [argued on paper] _Let_ **T** _be a committed table with row messages fr (Definition 4.2), and let Q ⊆ D be a set of t_ + _k positions with_ 1 _≤ k ≤ d_ 2<sup>_n_0</sup> _. There are a matrix_ mat _Q ∈_ F<sup>_k×ℓ_</sup> _and a linear map_ lin _Q_ : F<sup>_Q_</sup> _→_ F<sup>_k_</sup> _, both depending on Q alone, such that:_ 

_1._ mat _Q fr_ = lin _Q_ �( **T** [ _ϕ_ ] _r_ ) _ϕ∈Q_ � _for every row r and all random coefficients;_ 

_2. the first k columns of_ mat _Q, which belong to digits of A, are linearly independent;_ 

_3. the row space of_ mat _Q contains a subspace of dimension at least k −_ 3 _s whose vectors vanish at the_ 3 _s pads._ 

_Hence the columns at Q determine k independent linear equations in the digits and pads of each row, and at least k −_ 3 _s in its digits alone._ 

_Proof of Proposition C.16._ Row _r_ of **T** is the evaluation on _D_ of _gr_ ( _X_ ) = _τr_ ( _X_ ) + _X_<sup>_t_</sup> _fr_ ( _X_ ), where deg _τr < t_ and _fr_ ( _X_ ) =<sup>�</sup> _j<ℓ_<sup>_fr_[</sup><sup>_j_]</sup><sup>_Xj_.</sup> 

_Step 1: the random coefficients cancel._ Let 



The _t ×_ ( _t_ + _k_ ) Vandermonde matrix on _Q_ has rank _t_ , so dim Υ = _k_ . For _υ ∈_ Υ, 



_ϕ_<sup>_υ_</sup> _ϕ_<sup>(</sup><sup>_i_)</sup><sup>_aϕ_</sup> � _i≤k_<sup>.This</sup> 

For a basis _υ_<sup>(1)</sup> _, . . . , υ_<sup>(</sup><sup>_k_)</sup> of Υ, let mat _Q_ have the rows (c _j_ ( _υ_<sup>(</sup><sup>_i_)</sup> )) _j<ℓ_ , and let lin _Q_ map ( _aϕ_ ) _ϕ∈Q_ to<sup>��</sup> proves part 1. 

_Step 2: independence._ If _υ ∈_ Υ and c _j_ ( _υ_ ) = 0 for all _j < k_ , then<sup>�</sup> _ϕ_<sup>_υϕ ϕj_= 0forall</sup><sup>_j< t_+</sup><sup>_k_,and</sup><sup>_υ_= 0since</sup> the square Vandermonde matrix on _Q_ is invertible. Since _k ≤ d_ 2<sup>_n_0</sup> , the entries _fr_ [ _j_ ] with _j < k_ are digits. This proves part 2. 

_Step 3: eliminating the pads._ The map _υ �→_ (c _j_ ( _υ_ )) _ℓ_ 0 _≤j<ℓ_ from Υ to F<sup>3</sup><sup>_s_</sup> has a kernel Υ<sup>_′_</sup> of dimension at least _k −_ 3 _s_ . This proves part 3. The entries of _fr_ between _d_ 2<sup>_n_0</sup> and _ℓ_ 0 are 0, so the equation of _υ ∈_ Υ<sup>_′_</sup> involves digits alone; by Step 2 these equations are independent. 

73 

_Remark_ C.17 (Few further columns). Proposition C.16 rests on the printed proof alone, which is not machine-checked and has had no independent check. For 1 _≤ k ≤_ 3 _s_ each of the _k_ equations may involve the pads. At _ς_<sup>_◦_</sup> (Proposition 6.5) the OOD answers are the pads, so the _k_ equations are independent equations in the digits alone, for every _k ≥_ 1. It is not shown that this holds at every OOD seed. 

### **C.3 After compilation** 

Theorem C.25 reduces index privacy of a whole-key argument (Definition C.20) to twenty conditions: four on the tables (Definition C.4), eight on nonces and positions (Definition C.22), three on formats (Definition C.23) and five on the provers (Definition C.21). Proposition C.26 gives the first fifteen for FlashProve _Z_ at admissible _Z_ and bounded _K_ and _N_ c, Lemma C.28 the five for constant-message provers, and Lemma C.27 bounds the error where _ζ∗_<sup>sim</sup> = _ζ∗_<sup>ab</sup> = 0. The idea of this step, from honest-verifier zero knowledge of a public-coin IOP to zero knowledge of the argument compiled from it, is public: compare [CF24, Lemma 2.20 (informal) and Lemma 9.9] and [CO25, Section 2.6, Corollary 1 (informal), with Theorem 7.1 for interactive proofs]. 

**Definition C.18 (Compiled run of a prover).** Let _P∗_ be a prover for an IOP of a table system, with a _search limit_ for each round with proof of work. A _compiled run_ of _P∗_ computes a proof string, over the oracles of Construction D.11. 

- The chain salt is sampled; challenges are derived as the compiled verifier recomputes them. 

- In a round with proof of work the nonces 0 _,_ 1 _,_ 2 _, . . ._ up to the search limit are tried, one query each, until the challenge passes. 

- Oracle parts are committed with the leaf salts of Definition C.19. 

The result is _⊥_ if a search exhausts its limit or the challenges lie in the abort set. 

**Definition C.19 (Coins of a compiled run).** Let _ℓ_ s _≥_ 0. In a compiled run of _P∗_ (Definition C.18): 

- a leaf salt over an oracle part is a _public prefix_ , a function of a value of the coins of _P∗_ , the round, the component and the chain salt, followed by _ℓ_ s uniform bits; 

- a leaf without a position has a fixed public body; 

- the leaf number in the batch tree is fixed and public, and the path’s digests are uniform. 

A leaf salt of a committed table is a fixed public prefix followed by _ℓ_ s uniform bits. 

**Definition C.20 (Whole-key argument).** Let _Z_ be a configuration, and fix values (dg _,_ par) in place of the two public readings of _VZ_ (Construction 4.15). Let **P** = ( _P_ key _, P_ link _, P_ req) be provers for the three IOPs of FlashProve _Z_ , with search limits and _ℓ_ s (Definition C.18 and Definition C.19), and _K ≥_ 1. The _whole-key argument_ Π<sup>wk</sup> _Z_<sup>[</sup><sup>**P**]hasthe</sup> matrices over F in the place of indices (Definition C.3), encoded requests _ξ_ in the place of instances, and the separate oracles of Construction D.11, independent and random. 

- The committer samples **T** 0 _, . . . ,_ **T** _K_ with independent randomness, hashes each into a salted Merkle tree with root rt _j_ , computes _ςj_ = Seed(rt _j_ ) and _σj_ , and makes one compiled run of _P_ key and _K_ of _P_ link. The commitment is<sup>�</sup> rt0 _, σ_ 0 _, π_ key _,_ (rt _j, σj, π_ link<sup>(</sup><sup>_j_))1</sup><sup>_≤j≤K_</sup> �. 

- The stateless prover samples _j_ uniformly, makes a compiled run of _P_ req on **T** _j_ and one encoded request, and outputs ( _j, π_ req). On larger batches it outputs _⊥_ . 

- The verifier consists of steps 6 to 9 of _VZ_ at (dg _,_ par). 

**Definition C.21 (Five conditions on the provers).** For **P** as in Definition C.20, let Sim _∗_ be honest-verifier simulators (Definition C.2), _J_ a set of matrices, _Y_ a set of pairs ( _A, ξ_ ), and _ζ∗_<sup>sim</sup> _, ζ∗_<sup>ab</sup> _∈_ [0 _, ∞_ ]. Probabilities are over uniform ch with proof-of-work bits 0. The conditions are required for all table randomness, OOD seeds and copy numbers. 

**(Encoded tables)** For all matrices: the external oracle strings of _P∗_ are the committed tables of its input. 

- **(Exact simulation)** For _A ∈J_ , ( _A, ξ_ ) _∈Y_ : except with probability _ζ∗_<sup>sim</sup> , ch lies in the private part of the abort set or Sim _∗_ is exact. 

- **(Rare private aborts)** For the same _A_ , _ξ_ : ch lies in the private part with probability at most _ζ∗_<sup>ab</sup> 

_∗_<sup>.</sup> 

- **(Readable prefixes)** For _A ∈J_ and all _ξ_ : unless _P∗_ aborts, its plain parts determine the value of its coins in the public prefixes (Definition C.19). 

- **(Uniform coins)** All coins are uniform and independent, as in Definition C.19; salts have the compiled verifier’s lengths. 

74 

**Definition C.22 (Eight conditions on nonces and positions).** Let a table system and data for the compiled verifiers of its three IOPs (Construction D.11) be given. A condition stated for each IOP counts three times. 

**(Nonce-free positions)** For each IOP: the positions read are independent of the nonces. **(Nonces readable)** For each IOP: a plain part determines its nonce, up to the search limit. **(Plain seeds)** The two queries of Seed have the nonce 0. 

**(Local verifiers)** Each decision depends on the oracle strings only at the positions read. 

**Definition C.23 (Three conditions on formats).** In the setting of Definition C.22: 

- **(Round formats)** The rounds are own rounds, then shared rounds, then own rounds, beginning with an own round. In an own round the shared component is empty; in a shared round other than the first, the own components are. 

- **(Coherent rounds)** The plain part of a message is the pair of its components’ plain parts, and each oracle position is answered from one component alone. 

- **(Consistent layout)** A uniform oracle answer gives a uniform challenge, of which the proof-of-work bits are a separate coordinate. Each position of an oracle part is one leaf of one tree of its component. A table fits its Merkle tree, position _ϕ_ at leaf _ϕ_ , and a batch its tree. Symbols in leaves can be read back. Openings and tree depths are within declared bounds. The code of the OOD answers into digests is injective and of fixed length. 

**Definition C.24 (Hashing error).** In the setting of Definition C.20, the queries of one run of the committer and _N_ of the prover are counted at _names o_ : the oracles of Construction D.11 and, for each IOP and _i ≥_ 2, a further name for attempt number _i_ of all proof-of-work searches of that IOP. 

- m _o_ is the number of queries at _o_ that contain a secret string: all leaves of Merkle trees over oracle strings; attempt number 1 of each round’s query to fs; the further attempts, up to the search limits; each compiled run’s query to g, leaf and inner nodes in the batch tree. 

- e _o_ is 2( _K_ + 1) at the oracle of Seed and 0 elsewhere. 

- b _o_ is _ℓ_ s at such a leaf, the minimum of _λ_ and the length of the chain salt at a query to fs, and _λ_ elsewhere. 

The _hashing error_ is 



over names with m _o̸_ = 0. 

**Theorem C.25 (Reduction of index privacy to twenty conditions).** [machine-checked] _Let a table system (Definition C.3), data for its compiled verifiers, provers_ **P** _, simulators_ Sim _∗, and J , Y, U , K, N_ c _, ζ∗_<sup>sim</sup> _, ζ∗_<sup>ab</sup><sup>_,ε_</sup> _U_<sup>_,ℓ_</sup> s<sup>_be_</sup> _given. Suppose that the fifteen conditions of Definition C.4, Definition C.22, and Definition C.23 hold, and the five of Definition C.21, of which (Exact simulation) and (Rare private aborts) are required at the OOD seeds of U alone. Let S assign to each index of J a set of instances with which it forms a pair of Y. Then the whole-key argument formed from these data as in Definition C.20, with constant_ pp _, has index privacy on S for batches of at most_ 1 _(Definition 2.14), with error_ 



_where ζ is the error of Theorem 6.6._ 

_Proof._ Fix _A ∈J_ , a valid environment and _V_<sup>_∗_</sup> . 

_Step 1: lazy sampling._ Committer and prover answer their queries that contain a secret string with fresh uniform values, without consulting the table of oracle answers. Such an answer is entered into that table only where a proof string discloses the query: at opened leaves and at queries that are not leaves, never at an unopened leaf or after _⊥_ . The distribution changes only if such a query coincides with another query. A query at the name _o_ contains b _o_ uniform secret bits, and two attempts of one search differ in the nonce. A union bound over the _q_ queries of _V_<sup>_∗_</sup> gives the first term of _ζ_ hash( _q, N_ ) (Definition C.24), one over the pairs of queries of committer and prover at one name the second. Only (Uniform coins), (Consistent layout), (Plain seeds) and (Nonces readable) are used. 

_Step 2: a reduction to the table game._ After Step 1 a table’s leaf digests are fresh and independent of it. Root and OOD seed are therefore computed from sampled digests before the table is asked for, which is the order of the table experiment. Hence the experiment is obtained by a reduction from the table game (Definition C.13) with provers **P** , one offline proof, one link proof for each copy and _N_ c online proofs on each copy. They differ if some copy is sampled more than _N_ c times, with probability at most _K ·_ Pr[Bin( _N,_ 1 _/K_ ) _> N_ c], or a seed lies outside _U_ , with probability at most ( _K_ + 1) _εU_ by (Good seeds). 

_Step 3: simulating the game._ By the two budget conditions the game’s numbers satisfy the inequality of Definition C.6. Challenges are derived from fresh answers; by (Consistent layout), the event that no search of a 

75 

compiled run fails and ch lies in a given set has at most that set’s probability under uniform ch with proof-of-work bits 0. By (Rare private aborts) and (Exact simulation), a run differs from its simulation with probability at most _ζ∗_<sup>ab+</sup><sup>_ζ_</sup> _∗_<sup>sim</sup> . By Lemma C.14, the game at _A_ with **P** is replaced by the game at a fixed _A_ 0 with Sim _∗_ , at the sum of these terms over its 1 + _K_ + _N_ runs. 

_Step 4: the ideal experiment._ The reduction around the simulated game depends on _A_ 0 and the revealed instances alone; it is the simulator, and never answers one query in two ways. 

**Proposition C.26 (The conditions that mention no prover hold).** [machine-checked] _Let Z be admissible (Definition 3.8), let_ (dg _,_ par) _be arbitrary, K ≤_ 710 _,_ 324 _and N_ c _≤_ 688 _,_ 127 _, and let the search limits be arbitrary. Then_ FlashProve _Z and the data of the compiled verifiers of VZ at_ (dg _,_ par) _satisfy the conditions of Definition C.4, Definition C.22, and Definition C.23 with_ 



_and with U the set of all OOD seeds._ 

_Proof of Theorem 6.6._ By Proposition C.26 its conditions hold with _εU_ = 0 and _U_ the set of all seeds. The claim follows from Theorem C.25. 

**Lemma C.27 (Bound on the error after compilation).** [machine-checked] _Let Z be admissible and_ (dg _,_ par) _arbitrary. In Theorem 6.6 let ζ∗_<sup>sim</sup> = _ζ∗_<sup>ab</sup> = 0 _, N ≤ N_ c _and ℓ_ s _≥_ 256 _, and let the search limits of the three IOPs be at most_ limkey _,_ limlink _,_ limreq _. Then_ 



_where q_ hon = ( _K_ + 1) 2<sup>32</sup> + 2<sup>37</sup> + _K_ 2<sup>25</sup> + _N_ 2<sup>39</sup> _and n_ nm = 9 + limkey + limlink + limreq _._ 

_Proof._ For _N ≤ N_ c the binomial term vanishes. At admissible _Z_ every m _o_ is at most _q_ hon, e _o ≤_ 2( _K_ + 1), b _o ≥_ 256, and at most _n_ nm names have m _o̸_ = 0; substitute in Definition C.24. 

**Lemma C.28 (Constant-message provers satisfy the five conditions).** [machine-checked] _For any Z and_ (dg _,_ par) _, let_ **P** _consist of constant-message provers (Definition 6.2) with arbitrary search limits. Let their coins be as in Definition C.19, with leaf number_ 0 _, with ℓ_ s _at most the length that the compiled verifier fixes for a leaf salt, and with all-zero public prefixes that complete this length. Let each_ Sim _∗ output the fixed messages together with the symbols that it is given. Then the five conditions of Definition C.21 hold with ζ∗_<sup>sim</sup> = _ζ∗_<sup>ab</sup> = 0 _, for J the set of all matrices and every Y._ 

_Proof of Corollary 6.7. Part 1._ Where Chk( _Z_ ) = 1 a leaf salt has 576 bits, so Lemma C.28 admits _ℓ_ s = 320. Theorem 6.6 applies with _ζ∗_<sup>sim</sup> = _ζ∗_<sup>ab</sup> = 0, and Lemma C.27 gives _ζ ≤ ζ_ max. For _K_ = _N_ = 1 and _q ≤_ 2<sup>64</sup> the numerator is at most 2<sup>121</sup> . 

_Part 2._ A search fails only if 2<sup>40</sup> + 1 fresh answers miss a test of at most 32 bits, with probability at most (1 _−_ 2<sup>_−_32</sup> )<sup>240</sup> _≤_ 2<sup>_−_256</sup> . The three IOPs have at most 19, 6 and 46 rounds with proof of work. The committer runs before _V_<sup>_∗_</sup> , and its 19 + 6 _K_ searches fail with probability below the stated bound. The event that some _π_ req = _⊥_ is a set of views, so its probability exceeds that in the ideal experiment by at most _ζ_ max( _q, N_ ); there the Fiat–Shamir queries of a simulated run receive fresh answers. 

On the scope of Corollary 6.7: 

- Beyond one copy and one request the bound is a formula, which no theorem evaluates at the default configuration (Section 6.3 has a value evaluated by hand). 

- That the simulated view holds the roots, OOD answers and opened columns is read off the definitions, not proved. 

- That each proof opens a column of a committed table is proved for the run of committer and prover alone and for one verifier that makes no query, not for every verifier. 

- For two matrices of _J_ and an environment valid at both, the real views are within 2 _ζ_ ( _q, N_ ), which is below 1 only if _ζ_ ( _q, N_ ) _<_ 1 _/_ 2. 

76 

### **C.4 The link proof** 

_Proof of Theorem 6.9, outline. Part 1._ The abort set of _P_ link is empty by definition. Fix the input and ch. Prover and simulator have the same finite set of coins, vectors co of field elements; let dis _P_ (co) and disSim(co) be what each discloses at the coins co. There are maps sh, sh<sup>_′_</sup> on coins such that, for all co, 



For intuition, consider a masked value: a secret plus a uniform coin is distributed as the coin alone; here sh plays the role of the secret. By the second equation, two coins with the same image under co _�→_ co + sh(co) have the same sh, so they are equal. The map is injective, so a bijection of the finite set of coins, and by the first equation the two outputs are identically distributed under uniform coins. The equations use that no oracle part is opened at more positions than its rows have random coefficients, and that the claim left at the last sub-protocol is true in the honest run. 

_Part 2._ The verifier accepts _P_ link whenever the OOD answers are those of the row polynomials, the tables are their evaluations, both encode the same plain entries, and the symbols have the format’s form, as for two committed tables of one matrix. The other conditions of acceptance hold at the challenges that pass the proof-of-work checks. 

For its own constructions, [CFW26, version of February 25, 2026, Theorem 10.2] proves honest-verifier zero knowledge of the proximity test that IOP<sup>link</sup> _Z_ runs with changes (Section 4.5), resting in substance on its Lemmas 6.4, 7.3 and 9.8, through its Theorems 9.10 and 7.1, by its composition theorem (its Theorem 4.5); Part 1 is proved for the whole of IOP<sup>link</sup> _Z_ at once, by a bijection on coins. 

**Corollary C.29 (Committed tables and link proofs after compilation).** [machine-checked] _Let_ (dg _,_ par) = (0<sup>_λ_</sup> _, ∅_ ) _, and let the public prefixes of the leaf salts of P_ link _be zeros. Part 1 of Corollary 6.7, with its bound_ 2<sup>_−_135</sup> _, and for K_ = 1 _part 2, hold with the honest prover P_ link _of Theorem 6.9 in place of the constant-message prover for_ IOP<sup>link</sup> _Z ._ 

_Proof._ In Lemma C.28 replace the prover for IOP<sup>link</sup> _Z_ and its simulator by those of Theorem 6.9, whose part 1 gives _ζ_ link<sup>sim=</sup><sup>_ζ_</sup> link<sup>ab= 0.Theotherthreeconditionsholdforasuitablechoiceofthecompiledrun’scoins.LemmaC.27gives</sup> the same bound, and part 2 uses only that no prover aborts. _Remark_ C.30 (Reach of the corollary). Corollary C.29 holds at the one fixed value dg = 0<sup>_λ_</sup> in place of the digest of the configuration, at which _VZ_ is not proved ever to be called. The public prefixes are zeros by choice. The other two provers send constant messages: nothing that these two would compute from the weights is covered. At these values and every admissible _Z_ with Chk( _Z_ ) = 1, the interactive verifier of IOP _Z_<sup>link</sup> , not the compiled one, accepts _P_ link at the challenges that the oracle answers give, wherever _π_ link _̸_ = _⊥_ (machine-checked). 

## **D Sub-protocols, parameters and round errors** 

**Definition D.1 (The default configuration, encoding).** The default configuration _Z_ 0 is the triple of byte strings that the Lean development writes out as a term. Its strings have the lengths 2629, 224975 and 2784 and the SHA-256 digests 

```
ce9e126bb3ba7179319a31981a3fc36ca505d738348971f379dfc795f94a3fae
a99b964bbd3648d6e3b4007fc9370b5c70c7e9668848daa039820bc9e03e8f80
1c3191164c5113fd04d5108a469d7b3d17f4ccb3256e8a676d15bdc94435b919
```

The configuration _Z_ 0[ _µ,_ pol] (Definition 3.7), with pol = ( _C, θ, cap_ ), consists of: 

1. the parameter file of _Z_ 0 with the values of _ϵ_ 1, _ϵ_ 2, _ϵF_ , M _F_ , K _F_ , _C_ , _θ_ , _cap_ replaced, in decimal without leading zeros, each _ϵ_ in two lines as _ϵ_ mod 2<sup>32</sup> and _⌊ϵ/_ 2<sup>32</sup> _⌋_ ; 

2. the layout file of _Z_ 0; 

3. the scale file of 13 little-endian 32-bit words for each layer _l_ , 

_kq, kk, kv, ko, kg, ku, kd,_ M1 _,_ K1 _,_ M2 _,_ K2 _, ks,_ w _l,_ 

where w _l_ = 0 if _l ≡_ 3 (mod 4) and w _l_ = 1 otherwise, and, after the last layer’s words, _kr, kg_<sup>_′, k_</sup> _u_<sup>_′_foreachodd</sup> layer, in increasing order of the layers. 

In _µ_ 0 the first twelve words are 29 _,_ 29 _,_ 17 _,_ 17 _,_ 16 _,_ 20 _,_ 19 _,_ 44 _,_ 30 _,_ 44 _,_ 30 _,_ 8 in every layer, ( _kr, kg_<sup>_′, k_</sup> _u_<sup>_′_)=(17</sup><sup>_,_16</sup><sup>_,_20),</sup> _ϵ_ 1 = _ϵ_ 2 = _ϵF_ and (M _F ,_ K _F_ ) = (44 _,_ 30). 

77 

_Remark_ D.2 (Written copy). The release ships _Z_ 0 and _Z_ 1 with its two stored proofs for the integer model at synthetic weights. No theorem relates the shipped files to the Lean terms; the release contains a program that compares them. The digests in Definition D.1 are the terms’, computed outside Lean. The value of _ϵ_ 1 = _ϵ_ 2 = _ϵF_ in _µ_ 0 stands in the shipped `examples/full-size/config/policy.txt` . 

**Construction D.3 (The integer model).** Put 

( _d_ model _, nh, ng, dh, f, fe, NE_ ) = (5120 _,_ 40 _,_ 8 _,_ 128 _,_ 16384 _,_ 8192 _,_ 128) _._ 

The weights _M_ (Definition 3.1) are the following integer matrices: an embedding matrix ( _|V| × d_ model), an output matrix ( _d_ model _× |V|_ ) and, in each layer, 



with _d_ hid = _f_ in an even layer; an odd layer has _d_ hid = _fe_ , a router matrix ( _d_ model _× NE_ ) and _NE_ further triples of the shapes of _Wg, Wu, Wd_ , the routed experts. Entries of the embedding matrix lie in [ _−_ 2155905152 _,_ 2139062143], all others in [ _−_ 2<sup>12</sup> _,_ 2<sup>12</sup> ). 

Let _µ_ be valid (Definition 3.6) and _x_ ¯ a context (Definition 3.1). At each token position of _x_ ¯ the model starts from the token’s row of the embedding matrix; each layer adds the output of a causal attention block and then of a gated feed-forward block, each applied to a normalized input; an odd layer evaluates _Wg, Wu, Wd_ and the routed expert that the router ranks first. The logits logit _M_ ( _x_ ¯) are the normalized vector at the last token position times the output matrix. All steps are exact integer operations. Eight kinds of intermediate value have prescribed intervals; the model is defined on _x_ ¯ if these hold at every token position. 

_Remark_ D.4 (Model specification). Construction D.3 is an outline of the Lean term that defines the integer model (Section 3.1). The release’s `docs/model.md` gives every step, the eight intervals and the five constant lookup tables; cosine and sine share one of the four digests of tables in the parameter file. In Lean the model holds its five tables as constants: no theorem describes their content, and none ties them to these digests. On synthetic weights at small sizes the Lean function agrees with the release’s reference program in every value that was compared, over Lean files and programs that the integer-model package’s pages name ( `integrations/vllm/integer_model/tests/lean_vectors/README.md` ); no theorem relates the two. 

_Remark_ D.5 (No real weights shown). On trained weights (Remark 3.3, Trained weights): 

1. _Counted outside Lean._ A reference program like the release’s, not the released one, under a parameter file that the release does not contain, found no range condition failing on 544 of 544 windows, cut from 96 prompts (what the count does not show: `lean/LIMITS.md` , “At real weights: a count, no theorem”), at a public model’s 4-bit weights. 

2. _Run._ The C++ programs have proved requests on these weights; the timed runs of Section 7.2 do so, at a configuration fitted to the weights (Section 7.1). The release ships the fitting tools and no fitted configuration. A fitted configuration of timed exact-match runs on these weights passes the scope test Adm (Definition 3.8) by one evaluation (whether it is these runs’ own files, byte for byte, is not shown), with Lean sources as in Section 7.1 (no theorem); every security line of its parameter file is the default’s. The parameter checks (Definition 4.14) were not evaluated at it; where they fail, _VZ_ rejects every input. The configuration of the clipped-gap rows was not put to the scope test. 

3. _Not tried._ The executable form of _VZ_ has been run on proofs for synthetic weights (Section 7.3), not on proofs for a trained model’s weights. 

The sources of the sub-protocols are in Table 11; the Proximity test’s Fold, Code switch and Base case are, with changes (Section 4.5), Constructions 6.3, 9.7 and 7.2 of [CFW26, version of February 25, 2026]. Send has the masking of the answers from its Section 9.2, here with the pads in the message itself (in its Construction 9.7, in a mask oracle); the main course of Masked sumcheck, with committed masks whose values are not revealed but join the output claim, has the form of steps 1 to 4 of its Construction 6.3 and, for a polynomial summand, of steps 4 to 6 of its Construction 11.4. Steps 2 and 4 of Construction D.7 have the form of steps 1 to 4 of its Construction 9.7, which carries over to zero-knowledge encodings a known template for code switching, here with the table’s random coefficients in the new message. 

**Construction D.6 (Sub-protocols).** This construction gives the course of each sub-protocol (Remark D.10, Given in outline). An _oracle message_ is a list of rows, each sent as a masked encoding (Definition 4.2) under a code that _Z_ fixes; its _mask rows_ are uniform. A _group_ is a set of these under one code. A _linear constraint_ is an equation _⟨_ co _, f ⟩_ = tg on the entries _f_ of a group, with co, tg computed by _V_ . A challenge is a uniform tuple of blocks of _λ_ bits, from which fixed maps compute what _V samples_ (Lemma A.20). 

78 

**Send.** _P_ sends an oracle message _f_ ; _V_ samples OOD seeds ( _zi, βi_ ) _i<s_ ; _P_ sends answers _ai_ . The constraints _⟨_ eq( _zi, ·_ ) _, f ⟩_ + _βi_ pad _i_ = _ai_ join the group of _f_ , with pad _i_ as in Definition 4.3. **Batching.** _V_ samples κ _∈_ E. Claims or constraints cl0 _,_ cl1 _, . . ._ are replaced by<sup>�</sup> _j_<sup>κ</sup><sup>_j_cl</sup><sup>_j_.</sup> 

- **Masked sumcheck,** for a claim that _F_ has the sum _S_ over the Boolean cube. _P_ sends _S_<sup>_′_</sup> , the sum of a mask polynomial _F_<sup>_′_</sup> with committed coefficients; _V_ samples κ; the running claim is _S_<sup>_′_</sup> + κ _S_ . For each variable, _P_ sends the round polynomial of _F_<sup>_′_</sup> + κ _F_ without the coefficient that the running claim determines, and _V_ samples an element. The output is a claim on _F_<sup>_′_</sup> + κ _F_ at the sampled point. Those of steps (d), (f) of Construction D.9, and all but one of a Lookup, end with two further rounds: one more variable, with a nonzero sample; then masked end values. 

- **Fold,** for one constraint on a group. A masked sumcheck with sampled point _γ_ replaces it by one constraint on � _r_<sup>eq(</sup><sup>_γ, r_) row</sup><sup>_r_andonmaskrows.</sup> 

- **Code switch.** Send _f_<sup>_′_</sup> under another code, said to hold the folded row, with a nonce beside the answers; _V_ samples positions and opens the old group there. Each opened column, folded at _γ_ , gives a constraint on _f_<sup>_′_</sup> ; these, the fold’s and the answers’ constraints form a new group. 

- **Base case,** for one constraint _⟨_ co _, f ⟩_ = tg on rows _f_ sent with masking rows _f∗_ . _P_ sends tg _∗_ ; _V_ samples κ; _P_ sends _f∗_ + κ _f_ with its random coefficients, and a nonce; _V_ samples positions, and checks _⟨_ co _, f∗_ + κ _f ⟩_ = tg _∗_ + κ tg and the encoding against the opened symbols. 

- **Proximity test,** on a group. Batching; for each stage but the last, Fold and Code switch, whose second challenge also batches the new group’s constraints; Base case. 

- **Lookup,** for lookup keys ky _i_ and a table (tb _j_ ) _j_ . _P_ has sent, in bits, the multiplicity mu _j_ of each tb _j_ . _V_ samples _ϖ_ . The claim<sup>�</sup> _i_<sup>(</sup><sup>_ϖ −_ky</sup><sup>_i_) = �</sup> _j_<sup>(</sup><sup>_ϖ −_tb</sup><sup>_j_)mu</sup><sup>_j_,withapublicfactorforpadding,isreducedalongthetwotreesof</sup> products, by masked sumchecks, to claims on the lookup keys and the bits; one more shows that the bits are bits. 

- **Range test,** for cells _Y_ already sent. Send a vector _U_ , said to hold the sums of cells that the segment’s claims name. _V_ samples Θ (Table 2). Send the digits of _Y_ Θ with their multiplicities; a Lookup places them in their ranges. Claims relate the digits to _Y_ and Θ, and _U_ to _Y_ . 

**Construction D.7 (Test on a committed table).** _Input._ An external oracle string **T**<sup>_∗_</sup> with a sample _σ_ = ( _ς,_ **a** ); numbers _ν_ and _κ_ ; and, for each request of a batch, a claim on the table (Construction 4.13) with _h ∈_ E<sup>_d_</sup> , _ψ ∈_ E<sup>_n_</sup> , _b ∈_ E, and an oracle message _w ∈_ E<sup>2</sup><sup>_n_</sup> , said to be _h_<sup>_⊤_</sup> _A_ , with the constraint _⟨_ eq( _ψ, ·_ ) _, w⟩_ + _η_ = _b_ . Brackets hold what the honest prover sends. 

_Shared part,_ run once for the batch. 

1. _V_ samples _ψ_<sup>_∗_</sup> = ( _ψ_ 0<sup>_∗, γ_)</sup><sup>_∈_E</sup><sup>_n_0</sup><sup>_×_E</sup><sup>_v_.</sup> 

2. Send _u_<sup>�</sup> _u_ =<sup>�</sup> _r_<sup>eq(</sup><sup>_γ, r_) (</sup><sup>_fr, τr_),with</sup><sup>_fr, τr_ofDefinition4.2 �</sup> , with a mask message. 

3. For _i < s_ the following constraint is added to the group of _u_ : 



4. _V_ samples _ν_ positions, with _κ_ bits of proof of work. **T**<sup>_∗_</sup> is opened there. For each position _ϕ_ the constraint that the encoding of _u_ has the symbol<sup>�</sup> _r_<sup>eq(</sup><sup>_γ, r_)</sup><sup>**T**</sup><sup>_∗_[</sup><sup>_ϕ_]</sup><sup>_r_at</sup><sup>_ϕ_isadded.</sup> 

5. Send _ω_<sup>�</sup> _ω_ = _A_ eq( _ψ_<sup>_∗_</sup> _, ·_ ) _∈_ E<sup>_d_�</sup> , with a mask message. 

6. _V_ samples _χ ∈_ E<sup>_⌈_log2</sup><sup>_d⌉_</sup> . 

7. _P_ sends one masked value. With it the equation 



is split into one constraint on the group of _u_ and one on the group of _ω_ . 

8. Proximity test on the group of _u_ , then on the group of _ω_ (Construction D.6). 

_Last part,_ run for each request. 

9. _ω_ joins the group of _w_ , and the constraint _⟨_ eq( _ψ_<sup>_∗_</sup> _, ·_ ) _, w⟩_ = _⟨h, ω⟩_ is added. 

10. Proximity test on that group. 

79 

**Proposition D.8 (Two rounds at the default configuration).** [machine-checked] _Let the sizes of a committed table, e, ν_ on _, κ_ bat _and R be those of Z_ 0 _(Table 2); in particular v_ = 11 _, n_ = 18 _, ν_ on = _R_ = 64 _and κ_ bat = 28 _. Let d_ min _, J and_ ls _be as in Lemma A.24, with e_<sup>_′_</sup> = _e._ 

_1. |D|_ ( _|D| − d_ min) _<_ ( _|D| − e_ )<sup>2</sup> _and e < d_ min _._ 

_2. ⌊J⌋·_ 2<sup>132</sup> _≤|_ E _|._ 

_3. At a word without a collision, at most_ (11 _⌊J⌋_ + 286) _|_ E _|_<sup>10</sup> _points γ are bad._ 

_4. The bound of Lemma A.20 is at most_ 2<sup>_−_128</sup> _for step 1 of Construction D.7 with at most_ 64 _requests._ 

_5. It is at most_ 2<sup>_−_128</sup> _for step 4, with the count_ ( _|D| − e −_ 1)<sup>64</sup> _and_ 28 _bits of proof of work. With_ 27 _bits it exceeds_ 2<sup>_−_128</sup> _._ 

**Construction D.9 (The rounds of the three IOPs).** The sub-protocols are those of Construction D.6 and Construction D.7. Brackets hold what the honest prover sends. 

_Opening rounds._ In IOP<sup>key</sup> _Z_ and IOP<sup>req</sup> _Z_ each part opens with a round whose message is a plain part alone. The first part opens with OOD answers and a number, the shared part with OOD answers and a free value, the last part with a free value. _V_ rejects unless both lists of answers are the IOP instance’s and, in IOP<sup>key</sup> _Z_<sup>,thenumberis0;inIOPreq</sup> _Z_<sup>,</sup> step 9 of Construction 4.15 reads it. The shared part closes with an empty round. 

IOP<sup>key</sup> _Z_<sup>_._</sup> 

1. _First part._ Send a mask message; Send the multiplicities. Two Lookups: for the entries of _A_ , and for the sums _b_ 0 + 2<sup>8</sup> _b_ 1 of Definition 4.1. Proximity test on these messages’ group. Batching and a masked sumcheck over the _n_ variables of the matrix columns reduce the Lookups’ claims to one claim ( _h, ψ, η, b_ ). Send _w_ . 

2. _Shared and last part._ Construction D.7 with _ν_ = _ν_ off, _κ_ = _κ_ off and a batch of one. 

IOP<sup>link</sup> _Z ._ 

1. _V_ samples _γ ∈_ E<sup>_v_</sup> . 

2. Send one oracle message with _u_ 0, _uj_ �each table’s fold at _γ_ , as _u_ in Construction D.7<sup>�</sup> and a mask. 

3. For each table and each _i < s_ , the constraint of step 3 of Construction D.7. 

4. _V_ samples _ν_ first + _ν_ copy positions, with _κ_ off bits of proof of work. **T** 0<sup>_∗_isopenedatthefirst</sup><sup>_ν_first,</sup><sup>**T**</sup><sup>_∗_</sup> _j_<sup>atthe</sup> others, with the constraints of step 4 of Construction D.7. 

5. _V_ samples _χ ∈_ E<sup>_⌈_log2</sup><sup>_ℓ⌉_</sup> . The constraint<sup>�</sup> _k<ℓ_ 0<sup>eq(</sup><sup>_χ, k_) (</sup><sup>_u_0[</sup><sup>_k_]</sup><sup>_−uj_[</sup><sup>_k_]) = 0isadded.</sup> 

6. Proximity test on the one group. 

- IOP<sup>req</sup> _Z_<sup>_._</sup> 

1. _First part._ For each segment _g < G_ (Definition 4.8) in turn: 

   - (a) in the head alone, _P_ first sends a token in a plain part, and _V_ stores it; 

   - (b) for _g_ = 0, Send the segment’s cells and a mask message for the claims on the table; for _g >_ 0, Code switch, with _κ_ req bits of proof of work, from the group of segment _g −_ 1 to a message that also holds the cells of segment _g_ and the lower segments’ cells that higher segments read; 

   - (c) Lookup for the segment’s lookups; 

   - (d) the segment’s own rounds: sampled points, Batching and masked sumchecks reduce its identities to claims on cells and on the table; 

   - (e) Range test on the cells that the segment reads; 

   - (f) in the head alone, Batching and a masked sumcheck on its claim on the table; 

   - (g) Batching of the segment’s claims into one constraint on its group; Batching of the group’s constraints; Fold. 

Then Code switch and Proximity test on the last group; Batching and a masked sumcheck, as in IOP<sup>key</sup> _Z_<sup>,leave</sup> one claim ( _h, ψ, η, b_ ); Send _w_ . 

2. _Shared and last part._ Construction D.7 with _ν_ = _ν_ on, with _κ_ = _κ_ bat in the shared part and _κ_ req in the last. 

3. _V_ rejects unless the token of step (a) is the _y_<sup>+</sup> of _ξ_ . 

4. After every sub-protocol of the first part other than step (a), _V_ rejects if its stored value of that token has changed. 

Each verifier also rejects if a group is left untested. 

_Remark_ D.10 (Given in outline). This paper gives _VZ_ in outline. Its Lean code defines, and the paper does not print: 

80 

- the placement rule and wt (Definition 4.1); 

- the arithmetization (Definition 4.8), with the rounds of step (d) of Construction D.9; Lemma 4.9 states its meaning; that the lemma applies at an admissible configuration joins Lean theorems by hand, a step that Lean’s proof of Theorem 5.1 does not use, since that the three byte strings of _Z_ decode without error, which admissibility implies, is one of that theorem’s hypotheses in Lean; 

- the closed list of the parameter checks (Definition 4.14), which are computed with a constant in the place of dg _Z_ ; among them are a second derivation of the rounds, from the sizes alone, with one decoding radius for each shape of oracle message, and, in attention segments, a count of each round’s bad challenges; 

- the sub-protocols in detail (Construction D.6); 

- the maps code, the strings of the queries and the bijection (Construction D.11). 

The release’s format specification states what is hashed and sent. 

**Construction D.11 (Oracles, seeds, and the compiled verifier of one IOP).** _Oracles._ The algorithms of Π _Z_ query separate oracles: for leaves and inner nodes of Merkle trees over oracle strings; for those of the _batch tree_ , with one leaf for each request of a batch; and, for each IOP, a Fiat–Shamir oracle fs and an oracle g. A bijection between bit strings and triples (oracle, query, number of an answer block) implements them by _H_ . The public readings determine it. 

_Public readings._ The pair (dg _Z ,_ par), where par consists of the values of _H_ at up to seven strings, formed from _Z_ , dg _Z_ and _H_<sup>_′_</sup> , that state the parameters of the three IOPs. Ext _Z_ chooses public readings whose digest is dg, and translates every query of tr by their bijection. 

_Unpacking._ Step 2 of Construction 4.15 unpacks the byte strings of _π_ at dg _Z_ , by a query of its own; step 3 queries code( _Z_ ) again. If _π_ cannot be unpacked, _VZ_ continues with a fixed tuple whose fill is not a list of tokens of _V_ ; step 4 then rejects. A byte string that is unpacked but gives no proof string of its IOP’s shape gives _⊥_ , and step 6 rejects. 

_Commitment to an oracle string._ A Merkle root; the leaf at a position hashes the symbol there and a _leaf salt_ . _Seed of a root._ Seed(rt) is read off the second of two answers of one Fiat–Shamir oracle: the first query holds rt, dg _Z_ and the table’s sizes, the second the first answer’s chaining value. 

_Proof string._ A _chain salt_ ; the roots of the prover’s oracle parts; its plain parts (Definition 2.3); openings; the batch tree’s root, a leaf number, a path and a leaf salt. 

- _Compiled verifier,_ given a list cms of digests, an IOP instance and a proof string. For IOP<sup>req</sup> _Z_ the list holds the 

- copy’s root twice. 

   1. Recompute the challenges. Every round’s query to fs holds the round’s number, roots and plain part, nonce included. The first also holds cms, the chain salt and, for IOP<sup>req</sup> _Z_<sup>,the</sup><sup>_earlypart_of</sup><sup>_ξ_,whichis</sup><sup>_ξ_with</sup><sup>_y_+replaced</sup> by 0; each further one the chaining value of the preceding answer. An answer gives the challenge, bits of proof of work included, and a chaining value. 

   2. At the first shared round, g maps the request’s chaining value and own component to its _batch value_ , and the query to fs holds cms, the batch tree’s root and the shared component. The last part continues from the shared part’s chaining value and the batch value. 

   3. Reject unless the salts have their fixed lengths and the openings are exactly those that the IOP’s verifier reads. 

   4. Verify every opening against its root and, where the IOP has shared rounds, the batch value, with its leaf salt, against the batch tree’s root. 

   5. Decide as the IOP’s verifier does on the challenges, the plain parts and the opened symbols. 

**Construction D.12 (Committer and prover of** Π _Z_ **).** Rnd _Z_ outputs the masks and salts. Commit _Z_<sup>_H_(</sup><sup>_M,_r)forms</sup> **T** 0 _, . . . ,_ **T** _K_ (Definition 4.6) for a digit matrix determined by _M_ , computes rt _j_ , _ςj_ = Seed(rt _j_ ) and **a** _j_ , makes _π_ key and each _π_ link as the honest provers of Construction D.9 do at the recomputed challenges, and returns _H_ (code(k _,_ dg _Z_ )). _PZ_<sup>_H_(</sup><sup>_M,_r</sup><sup>_, x, y_)makes</sup><sup>_π_reqlikewise,withemptyfill;forabatchitchoosesacopywithintheopeningbudget(Defini-</sup> tion C.6) and places the requests at distinct random leaves of the batch tree. 

_Remark_ D.13 (How the two hash functions differ). _O_<sup>_∗_</sup> (Assumption 4.10) and _H_<sup>_′_</sup> (Assumption 4.11) both iterate the compression function of BLAKE2b, which takes a counter of 128 bits. _O_<sup>_∗_</sup> evaluates BLAKE2b as specified, on labelled strings of fewer than 2<sup>32</sup> bytes; the upper 64 bits of the counter are 0 in each of its calls. _H_<sup>_′_</sup> starts from the state of BLAKE2b with 64-byte output and sets these 64 bits to num + 2<sup>32</sup> _k_ in every call, where _k_ is the number of 32-bit words of its input and num is one of the four numbers in use, none of them 0. The two functions hence differ in that half of the counter in every call. That this keeps them apart is part of Assumption 4.11: no formal proof states it, and the published analysis of BLAKE2b does not cover _H_<sup>_′_</sup> . 

_VZ_ evaluates _H_<sup>_′_</sup> on public tables and on parts of the statement, _x_ and _y_ among them. Its digest of the encoded request is part of the first string that an online proof hashes (Construction D.11). The digest is redundant there: the 

81 

same string holds the window, with _x_ and _y_ , and the two token positions; the digest is compared with nothing; and the coding of that string is injective by the encoded request alone, whatever the values of _H_<sup>_′_</sup> (proof of Lemma A.17). Hence a collision of _H_<sup>_′_</sup> merges no two queries. 

_Remark_ D.14 (How the parameters are obtained). In Table 2 the parameter file states the sizes of the tables and the bits of proof of work. Every other value is calculated from that file and taken from no run. A verifier program calculates these values without checking a proof; _δ_ and the bound on _K_ are calculated from its values, and no program outputs them. Rates are rounded up, proximity parameters down. The reduced configurations of Section 7 are synthetic models far smaller than the default one. 

- _Range test. V_ samples a matrix Θ with 128 columns whose entries are 0 with probability 1 _/_ 2 and _±_ 1 with probability 1 _/_ 4 each. _P_ sends each entry of the product as two digits; an entry that two digits can express lies in [ _−_ 2<sup>17</sup> _,_ 2<sup>17</sup> ). 

- _Copies and batches._ A copy serves at most 688 _,_ 127 batches of up to 64 requests, and a commitment has at most 710 _,_ 324 copies; both numbers are calculated. The committer fixes _K_ when it commits. Every started batch counts towards _Nj_ , whatever its size. These 688 _,_ 127 batches are about 4.81 years (1,759 days) of batches of 24 requests back to back on one GPU that never pauses, calculated from the measured time of such a batch (Section 7), and less with smaller ones. The programs choose a batch’s copy at random (from code review). 

## **E Sources and credit** 

### **E.1 The sources of each part** 

Table 11 lists each part of the protocol with the section or numbered item that describes it, its sources and, for some parts, a pointer to the technical report, which proves it, cites it or takes it as given (its written proofs are not verified, those for soundness and for zero knowledge are not settled, and nearly every reading of them has found errors: Limitation G.7; `docs/limits.md` ). Where a place in [CFW26] is named, it is that of the version of February 25, 2026; citations that name none refer to that work as its authors gave it to this project [CFWpc]. 

The report is written for other programs (Section 7.1). Among other differences, those programs depart from the report’s protocol at three points, which lie in the compilation, and none of the report’s written proofs covers them. How the report’s theorems relate to the released programs: `docs/limits.md` . 

Table 11: The parts of the protocol and their sources. 

|Part|Section; statement|Source|
|---|---|---|
|The definitions of the primitive:<br>a commitment to the weights,<br>then proofs of requests;<br>completeness, index-binding<br>knowledge soundness, index<br>privacy|Section 2.4; [TP]|adapted from [CFWpc]; the report’s experiment<br>hands the extractor the OOD seed, unlike<br>Definition 2.7, whose extractor is given the<br>commitment and the committer’s trace alone|
|The plan: an offline phase, a<br>phase for each batch, copies with<br>their link proofs|Definition 4.6 and Definition 4.7|adapted from [CFWpc]|
|The link proof|Definition 4.6|after [CFWpc]; [CFW26]|
|The weights as a matrix of digits;<br>one test of the committed table,<br>then tests of its folded rows|Definition 4.1 and<br>Construction D.7|[CFWpc]; [GLSTW23, Section 4.1], whose scheme<br>with a Reed–Solomon code is, by its own statement,<br>implicit in Ligero [AHIV17]|
|Committed tables under a code<br>that hides|Definition 4.2|[CFWpc]; [CY24, Section 32.5]; [CFW26]; [ACY22,<br>Section 1.2], for data encoded apart from the<br>instance and committed once for many statements,<br>without hiding|
|The OOD sample of each<br>committed table, taken when the<br>table is committed, with<br>distances beyond unique<br>decoding in the tests of a<br>committed table|Definition 4.3 and Definition 4.5|[CFWpc]; the mask of an answer: [CFW26, version<br>of February 25, 2026, Section 9.2]; that only different<br>messages collide: its Section 9.1|



82 

|Part|Section; statement|Source|
|---|---|---|
|The step from two words to a<br>fold of several rows|Lemma A.24; a lemma and a<br>corollary of [TP]|by the report’s list of statements: the lemma after<br>[AER24] and [DG24, Theorem 3.6], in part the<br>report’s own;<br>the corollary the report’s own, with [BCGM25,<br>Corollary 8.6 and Theorem 9.2] and [CFW26] as its<br>sources;<br>see also [BCGM25, Lemma 4.4]: a tensor product of<br>two generators with MCA has MCA, with the sum<br>of their errors|
|The repeated mask codes|a lemma and a corollary of [TP]|repeating a mask’s code is in [CFW26], where the<br>verifier enforces the repetition;<br>the report departs from that [TP]: a mask row is<br>sent copied, and the errors are those of the repeated<br>code.<br>By the report’s list of statements: the lemma and<br>the corollary are stated and proved in the report;<br>they are not verified here, and no machine-checked<br>statement of this paper takes them as a premise;<br>the list names no source for the lemma, and for the<br>corollary it names [CFW26], for the repetition of a<br>mask’s code alone;<br>the corollary holds given MCA strictly below the<br>Johnson radius (Lemma 5.13, machine-checked),<br>by way of the lemma on interleaved rows [TP]; on<br>that lemma see the row “No factor for the number of<br>interleaved rows” below|
|One reading of the committed<br>table per batch|Construction D.7|after [GLSTW23, Section 4.1] and [BFRW25,<br>Section 2.1], the second of which, by its own<br>statement, follows Ligero [AHIV17]; [CFW26];<br>[CFWpc]|
|The count of revealed positions:<br>query bound and opening budget|Definition 4.2 and Definition C.6|[CFWpc]|
|The zero-knowledge proximity<br>tests: the folding sumcheck, code<br>switching, the final test|Construction D.6|[CFW26]<br>with three changes [TP], of which this paper<br>describes one|
|The zero-knowledge sumcheck for<br>a polynomial summand|Construction D.6|adapted from a theorem of [CFW26];<br>the sumcheck [LFKN92]; its round polynomials<br>[Gru24]; its hiding polynomial [CFW26];<br>that paper’s version of February 25, 2026 (its<br>Section 2.5) likens its own small masks to the small<br>mask for each round in [XZZPS19],<br>and finds a full mask, a random polynomial as in<br>[CFS17], too costly<br>the two lemmas on the hiding polynomial [TP]: both<br>are adapted from [CFW26]|
|The lookup’s two products<br>(grand products)|Construction D.6|[GKR08]; [Tha13, Section 5.3.1]|
|Lookups into several tables<br>combined as one|Construction D.6|adapted from [BCGJM18, Appendix B.5]; several<br>tables in one: [GW20, Sections 4 and 4.1];<br>[BCGJM18, Appendix B.6]; the report states the<br>lookup’s interface as a sub-protocol of its own; the<br>form in which it states sub-protocols is after<br>[CFWpc]|
|Range checks|Construction D.6; [TP]|adapted from [LNS21, Lemma 2.5]; [GHL22, Section<br>3.4]|
|Congruences that hold over the<br>integers|[TP]|[CFWpc]; [GHL22, Section 3]|
|The sumcheck for a matrix<br>product, on committed weights|Construction D.9 and<br>Construction D.7|[Tha13]; [LXZ21, Section 4.3]; [SLZ24]|
|Proving the model segment by<br>segment|Definition 4.8 and<br>Construction D.9|no source named; earlier systems that split a model<br>into parts: Appendix E.2|



83 

|Part|Section; statement|Source|
|---|---|---|
|Proximity below the Johnson<br>radius|Lemma 5.13|in the report cited and taken as given: [BCHKS25,<br>Theorem 4.6]; see also [Hab25]; in Lean<br>machine-checked in the typed form of Lemma 5.13<br>(Appendix E.4)|
|Out-of-domain samples|Construction D.6; a definition, a<br>lemma and a theorem of [TP]|the technique, from [BGKS20], as in [ACFY24b,<br>Section 4.4], whose results and proofs are, by its own<br>statement, adapted from [ACFY24a];<br>by the report’s list of statements: the lemma is<br>adapted from [CFW26];<br>the definition is after a suggestion [CFWpc], with its<br>statements and proofs the report’s own and parts<br>from [CFW26] and [CFWpc];<br>the theorem is after [CFW26], in part the report’s<br>own<br>following a suggestion [CFWpc], the verifier takes<br>such a sample after each message that carries data,<br>as in [BGKS20], [ACFY24a, Section 4.3],<br>[ACFY24b, Section 4.4] and [CFW26]|
|No factor for the number of<br>interleaved rows in the MCA<br>error of a line|Lemma A.24|a case of a statement in the abstract of [Jo26],<br>which has no such factor for generators with no<br>more seeds than the field has elements;<br>of that preprint only the abstract has been read; the<br>case used here is machine-checked within<br>Lemma A.24; see also [CFW26, Lemma 3.15]: for<br>additive codes, interleaving costs at most a factor of<br>the number of rows|
|Reed–Solomon encodings that<br>hide: the lemma|Remark 4.4 and Lemma C.8; a<br>lemma of [TP]|adapted from lemmas of [CFW26]|
|The compilation: Merkle trees,<br>the Fiat–Shamir transformation,<br>proof of work|Definition 2.4|the BCS transformation [BCS16] with proof of work,<br>adapted from [CY24, Section 25.1] and [Sta21,<br>Section 6.3];<br>the compiled argument, adapted from [CFWpc] and<br>the COS transformation [COS20, Section 10], [CY24,<br>Section 32.7]; [CFW26];<br>the lemma on proof of work, adapted from [Sta21,<br>Theorem 6];<br>the reduction, [CY24, Sections 26.1 and 26.2];<br>[BCFW25, Theorem B.4];<br>proof of work on the rounds that sample positions,<br>[CFWpc], with published uses in [Sta21, Section<br>3.11.3], [ACFY24a, Section 6.2] and [ACFY24b,<br>Section 6.2];<br>the outlines of the report’s proofs of index-binding<br>knowledge soundness and of index privacy, and the<br>first term of its bound on index privacy, after<br>[CFWpc]<br>committed tables, each committed once, are<br>compiled as external oracles with the COS<br>transformation, following [CFWpc]<br>how proof of work counts: two lemmas of [TP], one<br>of them adapted from [Sta21, Theorem 6]|
|A single bound on the error of<br>every round|Proposition 5.14 and<br>Definition 4.14|follows a suggestion [CFWpc]; as in [ACFY24b,<br>Section 6] and [ACFY24a]|
|The field|Table 2|the choice of the field takes up a suggestion [CFWpc]|
|Logit gap, cost, acceptance<br>policy; tokens that the provider<br>can place|Definition 3.4|as in the technical report’s appendix on acceptance<br>policies [TP]; on Token-DiFR: Appendix E.2|



### **E.2 The forward pass, other systems, batches and policies** 

**The forward pass.** Softmax, attention, and rounding by a lookup of the remainder follow [SLZ24, QSL25, CWS24]. 

84 

zkComposer [SGO26] cuts the computation into parts, by layers or by input positions, and proves each part on its own. What passes from one part to the next is committed to, and that ties the parts’ proofs together. They can be made in parallel or, to save memory, one at a time. DeepProve [GHL26, Section 4.5] likewise partitions the model into chunks of consecutive layers, for several machines to prove. NanoZK [Wan26] proves each layer apart and links the proofs by a chain of hash commitments. 

Our segments are not proofs of their own: one proof runs through all of them in order, and a closed segment’s folded row and the rows that a later segment still needs are sent on, encoded anew, with the claims on them (Construction D.9). 

**Batches.** Our batches rest on the observation of SLP [LY26] that the weights, not the tokens, dominate the cost. In SLP, one proof covers many requests; here, each request gets a separate proof. A Merkle tree over several verifiers’ transcripts, whose root serves as the challenge for all of them, appears in [YLFKM21]; [GGQZ25, Section 4.2] uses one for proofs of neural networks and credits [ZXHSZ22]. 

**Many proofs.** The hiding theorems of [SLZ24, GGQZ25] cover one prompt or one query. For many proofs under one commitment, [GGQZ25, Section 4.3], whose own aim is to hide the architecture too, notes that a one-time mask fails, and masks the committed polynomial afresh for each proof through the commitment’s homomorphism. 

As we understand it, a design like ours, in which each proof reveals positions of a table committed once, needs an opening budget; a homomorphic commitment can be masked afresh for each proof and needs none. A hash-based commitment needs no number-theoretic assumption, where zkLLM’s Hyrax commitment [WTSTW18] rests on the discrete logarithm problem; neither needs a trusted setup. 

**Policies.** At temperature 0, the logit margin of Token-DiFR [DiFR, Section 4.2] equals our logit gap up to one unit of the integer logits, which our rule for ties can add (Definition 3.4). The equality is one of formulas on the same logits: Token-DiFR’s are a reference model’s floating-point logits, and ours are the committed integer model’s integer logits. 

**Reported numbers.** Table 8 gives four earlier systems’ results as their authors report them. 

### **E.3 Standard terms** 

Table 12 sets terms of this paper beside those of the literature. It concerns the words alone: the sources of the definitions are those of Table 11. 

Table 12: Terms of this paper and of the literature. 

|This paper|The cited work|Source|
|---|---|---|
|commit-and-prove argument|commit-and-prove SNARK|[CFQ19, Definition 3.1]|
|a commitment to the weights, then<br>proofs of outputs|function-hiding functional commitment, a<br>close primitive|[BNO21, Definition 5]|
|offline phase, online phase|the same, of a preprocessing argument|[COS20, Sections 1.1 and 9]|
|index privacy|the same, of an algebraic holographic proof|[BNO21, Definition 6]|
|trace|query-answer trace|[CGSY23, Definition 3.3]|
|round-by-round soundness, doomed|round-by-round soundness, state function,<br>doomed set|[CCHLRR18, Definition 5.3];<br>[Hol19, Definition 2.5]; [CFW26,<br>version of February 25, 2026,<br>Definitions 3.6 and 3.7]|
|leakage function|the same, of structured encryption|[CK11, Section 4.2]|



### **E.4 The Lean library** 

**Codes.** The proof of Theorem 5.1 calls theorems on Reed–Solomon codes, above all mutual correlated agreement below the Johnson radius (Lemma 5.13, the statement that Lean’s theorems use in the place of [BCHKS25, Theorem 4.6]). As typed in the library they are proved there, not assumed, from files adapted from ArkLib and Mathlib [ArkLib, Mathlib] ( `lean/ThirdParty/` ). 

A public Lean library, ArkLib [ArkLib] (Apache-2.0), has a theorem of this kind: mutual correlated agreement of Reed–Solomon codes on affine lines, for every radius below the Johnson radius (the probability form, which we have not checked); the source’s own form of the bound is not proved there. The proof runs through adapted proofs of two theorems of ArkLib’s: a counting form of such a statement and a bound on its count. Files of our own ( `lean/Cb/Bridge/` ) prove that, under the statement’s premises, the estimate in the second equals the bound in the statement, and derive the 

85 

statement from the two. Two of these files ( `CodeListRSFromLines.lean` , `CodeListRSProved.lean` ) write out ArkLib’s estimate and restate what its two theorems give together. Their heads say that the two cannot count as independent of one more file of ArkLib’s, `Johnson/Probability.lean` , the file of the probability form, which the next paragraph names. 

The commit of [ArkLib] is cited as named in the copy that we used; we have not looked that commit up on ArkLib’s public repository. ArkLib’s declarations in question are ReedSolomon.exists_johnson_line_exactCorrelatedPair and ReedSolomon.HiddenDerivative.johnsonExceptionCount_le_comparisonEstimate, and ReedSolomon.mcaError_ affineLine_johnson_le (in ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/Johnson/ Probability.lean), the probability form, which we have not checked; two files of our own do not count as written independently of that file (the paragraph above). Files adapted from it come with this repository’s Lean library (lean/ThirdParty/). Comments in six of these files credit further sources, among them the formalization kz99/rs-ld-mca (“adapted, with permission”, in ArkLib’s words; that work, its licence and the permission were not seen) and the public project proximity-prize; lean/ThirdParty/README.md lists them. 

**Numbered items.** Table 13 lists the numbered definitions and statements that follow Lean files which `lean/CREDITS.md` lists under four of its headings, and, in four further rows, those over files marked as written from [CFW26], files that follow or build on further work of the three researchers whom the acknowledgments name first [CFWpc], or files not counted as written independently of that work or of ArkLib. That page also points to the published works that the files cite and names, for adapted code, Mathlib [Mathlib], Lean 4’s core library and the BLAKE2 reference source code package [BLAKE2]. Under two of the four headings that page lists files in which a comment names a public work: fifteen files name the Lean library VCV-io [VCVio], and one, `lean/IntegerModel/Model.lean` , names the file `modeling_llama4.py` of Hugging Face transformers, version 5.17.0 [transformers]. 

Among these works: a file that Definition A.1, Definition A.18, and Lemma A.19 follow takes the idea of the doomed set from [Hol19] and [CCHLRR18] and names [BGKTTZ23] and [BGTZ23] as related work; a file that Lemma A.24 follows cites [DP23]. In Table 13, as in Table 11, [CFW26] without a place is that work as its authors gave it to this project (Appendix E.1). 

Table 13: Numbered items that follow credited Lean files. 

|The Lean files|Numbered items|
|---|---|
|Listed under “ArkLib” [ArkLib]|Lemmas 5.13, A.24|
|Listed under “VCV-io”|Definitions 2.6, 2.7; Theorem 5.1|
|Listed under “The model” (values that<br>depend on Meta’s models)|Definitions 3.1, 3.2, 3.6, 3.7, 3.8, 5.16, D.1; Constructions 4.15, D.3;<br>Assumption 4.11; Theorems 5.1, 5.17, 5.19;<br>Propositions 3.9, 5.6, 5.18, 6.17, B.1, B.3, B.24; Corollaries 5.8, B.9, B.16|
|Listed under “Hugging Face<br>transformers”|Definitions 3.1, 3.2, 3.6; Construction D.3|
|Marked as written from [CFW26]|Definitions 2.2, 4.3, 4.5, 4.6, 5.4, A.1, A.2, A.6, A.7, C.3, C.4, C.5, C.6, C.13;<br>Constructions 4.13, B.8; Theorem 6.3;<br>Proposition 5.14; Lemmas 5.13, A.21, A.24, A.25, A.30, A.33, C.9, C.10, C.12|
|Follow or build on further work of the<br>three researchers whom the<br>acknowledgments name first [CFWpc]|Definitions 2.3, 2.6, 2.7, 2.13, 2.14, A.1, A.2, A.6, A.7, A.18, C.22, C.23;<br>Construction 4.13; Theorem 5.1; Proposition C.26;<br>Lemmas A.11, A.30, C.11; Corollaries 5.10, B.14|
|Do not count as written independently<br>of that work|Definitions 2.3, A.1, A.7, A.18, C.2, C.4, C.5, C.6, C.18, C.20, C.22;<br>Construction 4.13; Theorems 6.3, 6.6, 6.9; Proposition C.26;<br>Lemmas C.11, C.12, C.27; Corollaries 6.7, B.11, C.29|
|Do not count as written independently<br>of ArkLib [ArkLib]|Construction 4.13; Theorem 5.1 (through its statement); Lemmas 5.13, A.24, A.29|



The last four rows do not come from `lean/CREDITS.md` or from its table of files. The row on [CFW26] rests on a mark that stood at the head of Lean files while the library was written. The mark named no title; we read it as meaning that work. It was set on groups of files, in some directories on every file, and for a time on every new file, and it passed to files that a program made from marked ones; so it may name more files than were written from that work, and it does not show, file by file, what a file’s writer had at hand. 

The row on [CFWpc] is over files that comments in the Lean files tie to that work, and over files that state, prove or use the definitions of the Lean files that follow or render it. The row after it is over files that do not count as written independently of that work. The last row is over the files adapted from ArkLib, which that page lists too, and 

86 

over files whose writers had seen its source or had it available while they wrote: the heads of some say so, and for the others the ground is the record of who wrote them, which is not in this repository. 

Some places in further Lean files are like places in public Lean libraries (ArkLib, VCV-io, Mathlib and others); this table does not list items for that likeness. Among the items that follow such files are Definitions 2.1, 2.13 and 2.14. The lists behind the rows on [CFW26] and [CFWpc] are known to lack files; no other list of files behind this table, on that page or here, is claimed to be exhaustive. The Lean statement of Theorem 5.1 is over a description (Construction 4.13) that uses definitions from twelve files whose writers had seen the source of ArkLib or had it available while they wrote: three files under `lean/Johnson/Bridge/` , `lean/Johnson/Desc/Lens.lean` , `lean/Johnson/Desc/Ok/FormRules.lean` , `lean/Johnson/Desc/First/Parts.lean` and six files under `lean/Johnson/Desc/First/TailClaims/` . The heads of the three files under `lean/Johnson/Bridge/` and of `lean/Johnson/Desc/Lens.lean` say so; for the other eight the ground is the record of who wrote them, which is not in this repository. The twelve may not be all. This paper does not decide whether these files call for further credit. 

## **F Notation and Lean names** 

Table 14 lists the notation. The list below gives, for the numbered statements and the section that it names, the Lean names that each renders. Elsewhere in this paper a Lean name stands only where a sentence speaks of that declaration itself. 

<u>Table 14: Notation.</u> 

|Symbol|Meaning|
|---|---|
|_H_, _H_<sup>_′_</sup>, _λ_|hash function (random oracle); second hash function (fixed); output length<br>of _H_|
|_Z_; Adm(_Z_); Chk(_Z_)|configuration; scope test; parameter checks|
|_M_; _A_; wt|index (weights); digit matrix, _d ×_2<sup>_n_</sup>, _n_ = _n_0 +_v_; map from matrices in<br>range to weights|
|_x_, _y_; _V_; _L_; _ξ_|prompt, output; vocabulary; length of the window; encoded request|
|Serv_µ_|instances that can be served, with their outputs<br>|
|_c_, _π_; k; dg_Z_, par<br>|commitment, proof; key (rt0_, σ_0); digest of _Z_, second public reading|
|_π_key, IOP<sup>key</sup><br>_Z_ <sup>,</sup> <sup>_ν_off,</sup> <sup>_κ_off</sup>|one object, the_offline proof_, on the first table**T**0: proof string, IOP, columns<br>opened, bits of proof of work; stored in the key file|
|_π_link, IOP<sup>link</sup><br>_Z_<br>, _ν_first, _ν_copy|the _link proof_, on **T**0 and a copy **T**_j_; stored in the file of copy _j_|
|_π_req, IOP<sup>req</sup><br>_Z_ <sup>,</sup> <sup>_ν_on,</sup> <sup>_κ_req,</sup> <sup>_κ_bat</sup><br>code; Seed|the _online proof_, of an encoded request _ξ_ on a copy; stored in a proof file<br>codes as bit strings; OOD seed of a root|
|build, fill|map from (_x, y,_fill) to an encoded request; filling tokens|
|F, _p_; E=F[_ϑ_]|prime field and its order; extension field of the verifier’s samples|
|eq(_z, j_)|�<br>_k_<sup>(</sup><sup>_zkjk_ + (1</sup><sup>_−zk_)(1</sup><sup>_−jk_)), with</sup> <sup>_jk_ bit</sup> <sup>_k_ of</sup> <sup>_j_, the lowest first</sup>|
|_D_; _ℓ_, _ℓ_0; _t_; _s_|positions; entries of a row’s message, plain entries; opening budget; answers<br>a row|
|Enc(_f_;_τ_); **T**, **T**[_ϕ_]|masked encoding; committed table with 2<sup>_v_ </sup>rows, its column at _ϕ ∈D_|
|_ς_, **a**, _σ_|OOD seed, OOD answers, OOD sample (_ς,_**a**)|
|_e_, _δ_, _ρ_|radius in positions; _δ_ =_e/|D|_; rate (_ℓ_+_t_)_/|D|_|
|Λ, Coll, Dec|list of a word; collision at a seed; decoder|
|**T**0; **T**_j_; rt_j_|first table; copy _j_, 1_≤j ≤K_; Merkle root of **T**_j_|
|_ν_off_, ν_first_, ν_copy_, ν_on|maximum number of opened columns: offline proof; link proof in **T**0, in its<br>copy; batch|
|_K_; _Nj_; _R_; _G_|number of copies; number of batches on copy _j_; largest batch; segments|
|IOP<sup>key</sup><br>_Z _<sup>_,_ IOPlink</sup><br>_Z_<br>_,_IOP<sup>req</sup><br>_Z_|the three IOPs of FlashProve_Z_|
|Req_Z_|request relation<br>|
|(_h, ψ, η, b_); _w_|claim on the table; _w_ =_h_<sup>_⊤_</sup>_A_|
|_ψ_<sup>_∗_</sup>= (_ψ_<sup>_∗_</sup><br>0<sup>_, γ_);</sup> <sup>_u_;</sup> <sup>_ω_;</sup> <sup>_χ_</sup>|shared point; rows folded at _γ_; _ω_ =_A_eq(_ψ_<sup>_∗_</sup>_, ·_); second sampled point|
|_m_, _κm_, _εm_; _ε_rd|round, its bits of proof of work, its error; bound on every round’s error|
|_κ_req_, κ_bat_, κ_off|values of _κm_: a request’s own rounds; shared rounds; offline and link proofs|
|Π_Z_, _VZ_, Ext_Z_|the argument, its verifier, its extractor|
|_q_; _ε_(_q_)|hash queries of committer and prover together; soundness error|



87 

##### **Section 2** 

- **2.1** Random oracle, query bound, trace: `VerifiableInference.Hash` , `VerifiableInference. uniformHash` , `VerifiableInference.RandomizedAlgorithm` , `VerifiableInference.DeterministicAlgorithm` , `VerifiableInference.RandomizedAlgorithm.QueryBound` , `VerifiableInference.QueryLog` , `VerifiableInference. RandomizedAlgorithm.withQueryLog` . 

- **2.3** Interactive oracle proof, batch: `Iop. InteractiveOracleProof` , `Iop.InteractiveOracleProof. IsLocal` , `Iop.BatchInteractiveOracleProof` . 

- **2.5** Proof of work (grinding): `Iop.InteractiveOracleProof. grind` , `Iop.GrW` , `Iop.GrW.ok` . 

- **2.6** Indexed relation, commit-and-prove argument: `VerifiableInference.ArgumentSystem` , `VerifiableInference.ArgumentSystem.uploadRelation` , `VerifiableInference.ArgumentSystem.uploadArgument` , `ZKCP.IndexedRelation` , `ZKCP.IndexedRelation.L` , `ZKCP. Arg` , `ZKCP.Gen.Arg` . 

- **2.7** Index-binding knowledge soundness: `VerifiableInference.Sound` , `ZKCP.HasIndexBinding` , `ZKCP.bindingExp` , `ZKCP.bindingBad` , 

   - `VerifiableInference.sound_iff` . 

- **2.9** Correctness of extraction: `VerifiableInference. ExtractsCommittedWeights` . 

- **2.10** Extractability of the commitment, no two openings: `VerifiableInference.BindsWith` , `VerifiableInference. NoTwoOpenings` . 

- **2.12** Completeness on a set of triples: `VerifiableInference. CompleteOn` , `VerifiableInference.Complete` . 

- **2.13** Real and ideal experiment: `ZKCP.Gen.Arg` , `ZKCP.Gen. realExp` , `ZKCP.Gen.idealExp` , `ZKCP.Gen.ViewEnv` , `ZKCP. Gen.Simulator` , `ZKCP.Gen.View` , `VerifiableInference. realExperiment` , `VerifiableInference. simulatedExperiment` . 

- **2.14** Index privacy: `ZKCP.Gen.PrivacyOn` , `ZKCP.Gen. PrivacyFor` , `ZKCP.Gen.HasIndexPrivacyWith` . 

##### **Section 3** 

- **3.1** Integer model: `VerifiableInference.ModelWeights` , `VerifiableInference.settingsAreValid` , `VerifiableInference.modelNextToken` , `Policy. modelLogits` . 

- **3.2** Greedy token, greedy output: `VerifiableInference. modelNextToken` , `VerifiableInference.ArgumentSystem. IsModelOutput` , `Policy.modelNextToken_eq_firstArgmax` . 

- **3.4** Logit gap, acceptance policy, exact match, clipped gap: `Policy.logitGap` , `Policy.tokenCost` , `Policy.outputCost` , `Policy.countedTokens` , `Policy.HasScores` , `Policy. MeetsPolicy` , `Inst.Model.AcceptancePolicy` , `Inst.Model. AcceptancePolicy.exact` . 

- **3.5** Exact match and greedy outputs: `Policy.meetsPolicy_ exact_iff` , `Policy.meetsPolicy_of_isModelOutput` . 

- **3.6** Model parameters: `IntegerTransformer.Settings` , `VerifiableInference.settingsInRange` , `VerifiableInference.settingsAreValid` , `Inst.Model. Lawful` . 

- **3.7** Configuration, default configuration: `VerifierAndGoals. PublicFiles` , `VerifierAndGoals.storedPublicFiles` , `VerifierAndGoals.zGap` , `VerifierAndGoals.storedLines` , `VerifierAndGoals.policyFile` . 

- **3.8** Admissible configuration: `VerifierAndGoals.InScope` , `VerifierAndGoals.readFiles` , `VerifierAndGoals. settingsOfRead` , `VerifierAndGoals.FreeLines` , `VerifierAndGoals.policyFile` . 

- **3.9** All-zero weights: `Policy.meetsPolicy_zeroWeights` . 

- **3.11** Instances that can be served: `VerifiableInference. CanServe` , `IntegerModel.zeroFilledWindow` . 

- **3.12** Non-vacuity of the instances that can be served: `VerifiableInference.validSettings_exist` , `VerifiableInference.canServe_exists` , `VerifiableInference.canServe_nowhere_ atHaltingWeights` , `VerifiableInference. referenceSettings_valid` . 

- **3.13** Which instances can be served: `VerifiableInference. Compilation.complete_at_the_stored_files_at_ weights_within_800` , `VerifiableInference.canServe_ iff_of_passesSomeWorstCaseCheck` , `VerifiableInference. passesSomeWorstCaseCheck_atReferenceSettings` , `VerifierAndGoals.AtFiles.settingsOfRead_named` . 

##### **Section 4** 

- **4.1** Digit matrix: `Cm.Idx` , `Inst.Model.Layout.WFW` , `Cm.Idx. weights` , `VerifiableInference.OnPublicFiles.Flat. InRange` . 

- **4.2** Masked encoding, committed table, opening budget: `OnBytes.Flat.codingCells` , `OnBytes.Flat.cellEntryAt` , `OnBytes.Flat.codingHonest` . 

- **4.3** Out-of-domain seed, answers, sample: `Iop. HolographicProtocol.Rec` , `Hiop. ConfigurationWithOutOfDomainSample.smp` , `OnBytes. Slice.recZ` . 

- **4.5** List, collision, decoder: `OnBytes.Slice.DecR.dcd` , `OnBytes.Slice.DecR.coll` , `OnBytes.Slice.DecR. configurationOfStoredTable` , `Hiop.` 

   - `ConfigurationWithOutOfDomainSample.D` , `OnBytes.Flat. cellsOf` . 

- **4.6** First table, copy, link proof: `Iop.HolographicProtocol. link` , `VerifiableInference.OnPublicFiles. CompilerSoundnessConditionStatements.entrySound` . 

- **4.7** Encoded request, offline and online proof, files: `IntegerModel.requestOf` , `VerifiableInference. OnPublicFiles.Flat.buildReq` , `VerifiableInference. OnPublicFiles.Flat.Req` . 

- **4.8** Segments, cells, request relation: `VerifiableInference. OnPublicFiles.Flat.RelLAt` , `Policy.LedgerRel.Rno` , `Policy.LedgerRel.Cond` , `OnBytes.Flat.claimsReqAt` . 

- **4.9** Meaning of the request relation: `Policy.ledger_ relation_at_protocol_prime` , `Policy.IsTrueUnder` , `Policy.ledger_sub` . 

- **4.10** Random oracle: `VerifiableInference.Hash` , `VerifiableInference.uniformHash` , `VerifiableInference. Prob` , `Fmt.oracle` , `Fmt.size_oracle` . 

- **4.11** Separate second hash: `VerifierAndGoals. atSecondHashFunction` , `OnBytes.secondHashFunction` . 

- **4.13** The interactive protocol FlashProve _Z_ : `VerifierAndGoals.theDescription` , `Iop. HolographicProtocol` , `VerifiableInference. OnPublicFiles.CompilerSoundnessConditionStatements. keySound` , `VerifiableInference.OnPublicFiles. CompilerSoundnessConditionStatements.entrySound` , `VerifiableInference.OnPublicFiles.` 

   - `CompilerSoundnessConditionStatements.proofSound` . 

- **4.14** Parameter checks: `FlashProve.` 

   - `protocolDataWellFormed` , `FlashProve. protocolDataWellFormedAt` , `FlashProve. dataWellFormedAtFullSizeDepth` , `VerifiableInference. OnPublicFiles.sizeNumbersFitAtFullSize` , `LPFD.lawsAsk` , `VerifiableInference.Desc.FileDescription.planOk` . 

- **4.15** The argument Π _Z_ : `VerifierAndGoals.systemOfRecord` , `VerifierAndGoals.atSecondHashFunction` , `VerifierAndGoals.theDescription` , `VerifierAndGoals. verify` , `VerifierAndGoals.verifyAt` , 

88 

###### `VerifiableInference.OnPublicFiles.` 

`innerCompiledVerifier` , `TypedOracleCompiler.Compile. Batch.Cms.verifyAtFullKey` , `ZKCP.verifyDigestKey` , `ZKCP.Arg.buildInstance` , `FlashProve.theExtractor` . 

- **4.17** True request, constraints, honest bound: `VerifiableInference.IsTrueRequest` . 

- **4.18** True requests and greedy outputs: `IntegerModel. isModelOutput_of_isTrueRequest` , `IntegerModel. isTrueRequest_of_canServe` , `VerifiableInference. canServe_weightsOfIndex_iff` , `VerifiableInference. canServe_iff` , `VerifiableInference.modelNextToken_eq_ nextToken` , `VerifiableInference.lawful_scalesOf` , `VerifiableInference.settingsOf_scalesOf` . 

##### **Section 5** 

- `VerifierAndGoals.AtTheName.read_of_scope` : each statement below whose Lean form asks that the three byte strings decode is joined with it. 

- **5.1** Soundness: `VerifierAndGoals.AtTheName.sound_of_ verify_at` , `VerifierAndGoals.AtTheName.nearestSound_ of_verify` , `VerifierAndGoals.AtTheName.indexBinding_ of_verify_at` , `VerifierAndGoals.AtTheName. indexBindingWith_of_verify_at` . 

- **5.2** What the theorem gives: `VerifierAndGoals.AtTheName. indexBindingWith_of_verify_at` . 

- **5.4** Executable form: `PV.Ride.theForm` , `FlashProve. FormBytes` , `VerifiableInference.OnPublicFiles.hashOf` . 

- **5.5** The executable form computes the verifier: `PV.Ride. theForm_eq_verify` , `PV.Ride.theForm_total` , `PV.Ride. theForm_at_hash` , `PV.Ride.verdictThe_eq` , `PV.Ride. verdictThe_some` . 

- **5.6** The parameter checks pass at the default configuration: `VerifierAndGoals.filesRead` , `VerifierAndGoals. inScope_zAt` , `AtFullSide.Top.theLive_zFull` , `AtFullSide.Top.theLive_zGap` , `AtFullSide.Top.theLive_ zAt` . 

- **5.8** At the default configuration: `VerifierAndGoals. AtTheName.sound_of_verify_at` , `VerifierAndGoals. AtTheName.indexBindingWith_of_verify_at` , `VerifierAndGoals.filesRead` , `AtFullSide.Top.theLive_ zFull` , `PV.Ride.theForm_eq_verify` , `AtFull.Top.sound_ zFull` . 

- **5.9** What a client gets: `PV.Ride.theForm_accepts_only` , `PV. Ride.sound_of_the_form_events` . 

- **5.10** Many proofs: `VerifierAndGoals.AtTheName. soundManyProofs_union_of_verify_at` , `VerifierAndGoals. AtTheName.sessionAugmented_union_of_verify_at` . 

- **5.11** Soundness under any policy: `VerifierAndGoals. AtTheName.soundUnderPolicy_of_verify_at` , `VerifierAndGoals.AtTheName.indexBindingUnderPolicy_ of_verify_at` . 

- **5.13** Mutual correlated agreement strictly below the Johnson radius: `Cb.BCHKS46At` , `Cb.Bridge.bchks46At` , `Cb.Gl. bchks46At` . 

- **5.14** Round errors: `AtPublicFiles.AtFiles. soundAsIOPAtTheProtocol` , `VerifierAndGoals.AtTheName. keySound_at_live_holds` , `FlashProve.entrySound` , `VerifierAndGoals.AtTheName.proofSoundAt_of_verify` . 

- **5.16** Explicit committer: `Bind.Cpp.theCpp` , `Bind.Cpp. theTwin` , `Bind.Cpp.cppSide` , `Bind.Cpp.cppCommit` , `Bind. Twin.twinID` , `Bind.Twin.commitW` . 

- **5.17** Extraction and extractability for the explicit committer: `Bind.Cpp.extractsCommitted` , `Bind.Cpp.bindsAt` , `Bind. Cpp.noTwoOpenings_live` , `VerifierAndGoals. bindingErrorInQueries` , `Bind.Cpp.twinBudget` . 

- **5.18** Sharper errors: `Bind.Cpp.extractsCommitted_square_ salt` , `Bind.Cpp.bindsWith_of_tieAt` , `Bind.Cpp.tieAt` , 

`Bind.Cpp.saltAt_of_live` , `H2K.Sizes.inScope_readFiles` , `Bind.Cpp.extracts_salt` , `Bind.Cpp.extracts_square_ salt` , `VerifierAndGoals.Fit.epsBind` . 

- **5.19** Soundness after a commitment by the explicit committer: `Bind.Cpp.soundAtCommitted_own` , `Bind.Cpp. soundAtCommitted_exact_nonempty` . 

- **5.20** Sizes of the errors: `Bind.Cpp.honest_count` , `Bind.Cpp. bindsAtZ_cpp` . 

- **5.21** Scope of the statements on the commitment: `AtFullSide.Tie.extractsCommitted_zFull` , `AtFullSide. Tie.bindsAt_zFull` , `AtFullSide.Tie.noTwoOpenings_ zFull` , `AtFullSide.Tie.extractsCommitted_zAt` , `AtFullSide.Tie.bindsAt_zAt` , `AtFullSide.Tie. noTwoOpenings_zAt` , `Bind.Cpp.rejects_of_dead` . 

##### **Section 6** 

- **6.2** Constant-message prover: `OnBytes.WholeKey.silent` , `OnBytes.Flat.Files.fixedMessageProvers` , `AtPublicFiles.Premises.silentAtC` , `OnBytes.WholeKey. extEnc_silent` . 

- **6.3** Committed tables before compilation: `AtPublicFiles. AtFiles.tables_own` , `AtPublicFiles.AtFiles.tables_ own_named` , `AtPublicFiles.Reads.hidE_filesAt` . 

- **6.5** The opening budget is tight for one table: `AtPublicFiles. Table.every_set_objAt` , `AtPublicFiles.Table.every_ set_filesAt` , `AtPublicFiles.Table.not_hidE_filesAt` , `AtPublicFiles.AtFiles.not_hidE_own` . 

- **6.6** Reduction of index privacy to five conditions: `AtPublicFiles.Premises.privacyFromFive_description` , `OnBytes.WholeKey.wholeKeyPrivacy` , `OnBytes.WholeKey. zetaW` . 

- **6.7** Committed tables after compilation: `AtPublicFiles. ZetaAt.privacyWithBound_description` , `AtPublicFiles. Shown.keyInGame_description` , `AtPublicFiles.Shown. sessionsInGame_description` , `AtPublicFiles.ZetaAt. zeta_number_caps` , `AtPublicFiles.Shown.allAtOnce_ description` , `AtPublicFiles.ZetaAt.privacyWithNumber_ description` , `AtPublicFiles.ZetaAt.sMT_description` . 

- **6.9** Link proof before compilation: `OnBytes.LinkSim.AtHome. homeExactAt` , `OnBytes.LinkSim.AtHome.home_withheld` , `OnBytes.LinkSim.AtHome.home_beside` , `OnBytes.LinkSim. AtHome.homeSide_accepted` , `FlashProve.linkFits_at` , `Johnson.Prover.Link.link_yes` . 

- **6.11** Configuration in use: `VerifiableInference. PublicSetup` , `VerifiableInference.Compilation. FilesInUse` , `VerifiableInference.Compilation. FilesTaken` , `AtPublicFiles.AtFiles.read_of_inScope` . 

- **6.12** Served instance, label, bit: `VerifiableInference. AtEveryRequest.WellFormedRequest` , `VerifiableInference.RequestForm` , `VerifiableInference. AtEveryRequest.requestLabel` , `VerifiableInference. AtEveryRequest.Label` , `VerifiableInference. AtEveryRequest.requestOutcome` , `VerifiableInference. AtEveryRequest.Outcome` , `VerifiableInference. AtEveryRequest.modelResult` , `VerifiableInference. AtEveryRequest.served_iff_canServe` , `VerifiableInference.AtEveryRequest.wellFormed_and_ result_iff` , `VerifiableInference.AtEveryRequest. requestLabel_eq_served_iff` , `VerifiableInference. AtEveryRequest.requestLabel_eq_failed_iff` , `VerifiableInference.AtEveryRequest.requestLabel_eq_ otherTokens_iff` , `VerifiableInference.AtEveryRequest. bitOf_requestLabel` , `VerifiableInference.canServe_ iff_requestResult` , `VerifiableInference.canServe_and_ modelOutput_iff` , `VerifiableInference.requestForm_ iff_requestOfTokens` , `VerifierAndGoals.valid_of_read` . 

- **6.14** Lean provers: `VerifiableInference.Compilation.` 

89 

`theProvers` , `AtPublicFiles.AtFiles.HonestTerm. honestProvers` , `AtPublicFiles.AtFiles.HonestTerm. thePartsAt` , `AtPublicFiles.AtFiles.ServedRequest` , `VerifiableInference.Compilation.proversAtFiles` , `VerifiableInference.Compilation.` 

- `completeOnServedRequests_refuse_iff` , `VerifiableInference.Compilation.theProvers_eq` , `FlashProve.backThe` , `OnBytes.FrontRoad.inRange_ backThe` , `FlashProve.sec_the` . 

- **6.15** Compiled Lean argument: `VerifiableInference. flashProve` , `VerifiableInference.Compilation. honestProversAndSettings` , `VerifiableInference. Compilation.finalSystem` , `VerifiableInference. Compilation.flashProveAtFiles` , `VerifiableInference. Compilation.compile` , `VerifiableInference.Compilation. compiledSide` , `VerifiableInference.Compilation. compile_flashProve_verify_eq` , `VerifiableInference. Compilation.finalExtractor` , `VerifiableInference. Compilation.noProofForm` , `VerifiableInference. Compilation.compiled_provers_are_the_triple` , `VerifiableInference.Compilation.compiled_protocol_ is_the_bound` , `OnBytes.FrontRoad.familyLaws_the` . 

- **6.16** Arguments with the same verifier: `VerifiableInference. sound` , `VerifiableInference.Compilation.finalSound` , `VerifiableInference.Compilation.compile_flashProve_ sound_iff` . 

- **6.17** The hypotheses are met: `VerifiableInference. storedSetup` , `VerifiableInference.Compilation. filesInUse_stored` , `VerifierAndGoals. storedPublicFiles` , `VerifiableInference.Compilation. someRequestIsServed_holds` , `VerifiableInference. Compilation.zkOnServedRequests_inScope` . 

- **6.19** Whole run: `Iop.HolographicProtocol.wholeRunShape` , `Iop.HolographicProtocol.WholeRun` , `Iop. HolographicProtocol.WholeRun.Accepted` , `Iop. HolographicProtocol.WholeRun.Concludes` . 

- **6.20** Soundness of a whole run before compilation: `AtPublicFiles.AtFiles.probabilityInScope` , `AtPublicFiles.AtFiles.probabilityInScope79` , `AtPublicFiles.AtFiles.bound_le_79` , `AtPublicFiles. AtFiles.soundAsIOPAtTheProtocol` , `AtPublicFiles. AtFiles.roundsInScope` , `AtPublicFiles.AtFiles. relationAtFiles` , `AtPublicFiles.AtFiles. ConclusionFails` , `AtPublicFiles.AtFiles. clearingIsTheWires` . 

- **6.21** Reading the bound on a whole run: `AtPublicFiles. AtFiles.soundAsIOPAtTheProtocol` , `AtPublicFiles. AtFiles.wholeRunAtFiles` , `AtPublicFiles.AtFiles. mostWorkInScope` , `AtPublicFiles.AtFiles. roundsInScope` . 

- **6.22** Completeness of provers for the three IOPs: `VerifiableInference.Compilation.CompleteIOP` , `VerifiableInference.Compilation.CompleteInParts` , `Iop. FailsRarely` , `Iop.FailsAt` , `Iop.chanceAtPassing` , `Iop. PartsWork.Faithful` , `AtPublicFiles.AtFiles. workFaithful_atFiles` . 

- **6.23** Completeness before compilation: `AtFullSide.Honest. complete_in_scope` , `AtPublicFiles.AtFiles.HonestTerm. CompleteAtFiles` , `AtPublicFiles.AtFiles. CompleteOnServedRequests` , `AtPublicFiles.AtFiles. completenessTarget` . 

- **6.24** Exact simulation up to private aborts: `VerifiableInference.Compilation.RunByRun` , `VerifiableInference.Compilation.ExactOutsideStops` , `VerifiableInference.Compilation.PrivateStopsAtMost` , `VerifiableInference.Compilation.RunSimulators` , `VerifiableInference.Compilation.` 

###### `HonestVerifierZeroKnowledge` . 

- **6.25** Table experiment with served bits: `VerifiableInference.Compilation.WholeLife` , `VerifiableInference.Compilation.commitmentLife` , `VerifiableInference.Compilation.everyRequest` , `VerifiableInference.Compilation.realSide` , `Hiop. Seeded.EnvF.Adm` , `Hiop.Seeded.HostF.view` . 

- **6.26** Simulated table experiment: `VerifiableInference. Compilation.simulatedRuns` , `VerifiableInference. Compilation.simulatorsByTable` , `Hiop.Seeded.simWholeF` , `Hiop.Seeded.ViewNearF` . 

- **6.27** Honest-verifier zero knowledge before compilation: `VerifiableInference.Compilation.zkOnServedRequests_ inScope` , `VerifiableInference.Compilation. ZeroKnowledgeOnServedRequests` , `VerifiableInference. Compilation.life_number_at_most` , `VerifiableInference. Compilation.life_err_at_zero_zero` , 

- `VerifiableInference.Compilation.budgetAtFiles` . 

- **6.29** Completeness after compilation: `VerifiableInference. completeOnCanServe` , `VerifiableInference. completeAtWellFormedRequests` , `VerifiableInference. Compilation.theFinalStatementHolds` , `VerifiableInference.Compilation.FinalComplete` , `VerifiableInference.AtEveryRequest. ProofAcceptedWhere` , `VerifiableInference.Compilation. finalComplete_iff_completeOn` , `VerifiableInference. Compilation.finalAtT1_filled_withRoom` . 

- **6.30** Reading completeness: `VerifiableInference. Compilation.rejectsNoProof_value` . 

- **6.31** Soundness with extractability, openings and extraction: `VerifiableInference.soundAndBinding` , `VerifiableInference.Compilation. finalSoundAndBinding_value` , `VerifiableInference. Compilation.bindingOf_value` , `VerifiableInference. Compilation.finalExtracts_value` , `VerifiableInference. Compilation.BindingOf` , `VerifiableInference. Compilation.finalBindingError` , `VerifiableInference. Compilation.finalOpeningError` , `VerifierAndGoals.Fit. epsBind_min_le_bitsBinding` , `VerifiableInference. Compilation.finalSoundnessError_le_ finalBindingError` , `VerifiableInference.Compilation. bitsBinding_le_finalBindingError` , `VerifierAndGoals. Fit.BindsWith.min_one` , `VerifiableInference.BindsWith. mono` . 

- **6.33** Empty proofs, queries and opened columns: `VerifiableInference.refusedHandsOutNoProof` , `VerifiableInference.repliesFit` , `VerifiableInference. rejectsEmptyProof` , `VerifiableInference.Compilation. finalHandsOutNoProof_value` , `VerifiableInference. Compilation.handsOutNoProof_value` , 

- `VerifiableInference.Compilation.` 

- `FinalRefusedHandsOutNoProof` , `VerifiableInference. Compilation.finalRefusedHandsOutNoProof_value` , `VerifiableInference.Compilation.rejectsNoProof_ value` , `VerifiableInference.Compilation. FinalHandsOutNoProof` , `VerifiableInference. Compilation.RepliesFit` , `VerifiableInference. Compilation.RepliesFitAtFiles` , `VerifiableInference. Compilation.TableBudgetFits` , `VerifiableInference. AtEveryRequest.WithinBudget` , `VerifiableInference. Compilation.theFinalStatementHolds` . 

- **6.34** Zero knowledge with request leakage: `VerifiableInference.AtEveryRequest. ZeroKnowledgeWhere` , `VerifiableInference. AtEveryRequest.ZeroKnowledgeBelowWhere` , `VerifiableInference.AtEveryRequest. ZeroKnowledgeShownBelowWhere` , `VerifiableInference.` 

90 

- `AtEveryRequest.Simulator` , `VerifiableInference. AtEveryRequest.simulatedExperiment` , `VerifiableInference.realExperiment` , `VerifiableInference.labelled` , `VerifiableInference. ZeroKnowledgeOn` , `VerifiableInference.Simulator` , `VerifiableInference.simulatedExperiment` . 

- **6.35** Zero knowledge on served instances: `VerifiableInference.zeroKnowledgeOnCanServe` , `VerifiableInference.ZeroKnowledgeOn` , `VerifiableInference.PublicSetup` , `VerifiableInference. flashProve` , `VerifiableInference.Compilation.compile` , `VerifiableInference.CanServe` , `VerifiableInference. Compilation.finalZeroKnowledgeError` , `VerifiableInference.Compilation.` 

- `finalVerifierQueries` , `VerifiableInference. Compilation.finalReplyCount` . 

- **6.37** Zero knowledge at every instance: `VerifiableInference. zeroKnowledgeAtEveryRequest` , `VerifiableInference. zeroKnowledgeAtEveryRequest_of_lines` , `VerifiableInference.zeroKnowledgeBelow_of_ onCanServe` , `VerifiableInference.AtEveryRequest. zeroKnowledgeFollowsFromOn` , `VerifiableInference. AtEveryRequest.ZeroKnowledgeFollowsFromOn` , `VerifiableInference.AtEveryRequest.served_iff_` 

- `canServe` , `VerifiableInference.refusedHandsOutNoProof` , `VerifiableInference.Compilation.` 

- `finalServedIsModelOutput` , `VerifiableInference. Compilation.zeroKnowledgeOn_of_domain` , `VerifiableInference.Compilation.` 

- `theFinalStatementOneBitHolds_at` , `VerifiableInference. Compilation.finalZeroKnowledgeOneBit_at` , `VerifiableInference.Compilation.stepErrorAtTheStops` , `VerifiableInference.Compilation.` 

- `theFinalStatementOneBitHolds` , `VerifiableInference. Compilation.theFinalStatementBothHold` , `VerifiableInference.Compilation.` 

- `FinalZeroKnowledgeOneBit` . 

- **6.38** Zero knowledge with the label shown: `VerifiableInference.zeroKnowledgeWithLabel` , `VerifiableInference.Compilation.` 

- `theFinalStatementHolds_at` , `VerifiableInference. Compilation.theFinalStatementOneBitHolds_at` , `VerifiableInference.Compilation.finalZeroKnowledge_ of_oneBit` , `VerifiableInference.Compilation. theFinalStatementHolds` , `VerifiableInference. Compilation.finalStatement_of_oneBit` , 

- `VerifiableInference.Compilation.FinalZeroKnowledge` . 

- **6.39** Reading zero knowledge after compilation: `VerifiableInference.AtEveryRequest. toldWeightsSaysNothing` , `VerifiableInference. Compilation.served_at_T1` , `VerifiableInference. Compilation.termLeft_at` . 

- **6.46** The two together: `VerifiableInference.Compilation. threeAtFiles_in_use` , `AtPublicFiles.OneProtocol. ThreeAtOneProtocol` , `AtPublicFiles.OneProtocol. ThreeAtFiles` . 

- **6.47** Definitions of the properties: `VerifiableInference. CompleteOn` , `VerifiableInference.ZeroKnowledgeOn` , `VerifiableInference.Complete` , `VerifiableInference. ZeroKnowledge` . 

- **Section 6.5** Completeness and zero knowledge of the Lean provers: `VerifiableInference.Compilation.theProvers` , `AtPublicFiles.AtFiles.HonestTerm.honestProvers` , `AtPublicFiles.AtFiles.HonestTerm.thePartsAt` , `AtPublicFiles.AtFiles.soundAsIOPAtTheProtocol` , `AtPublicFiles.AtFiles.probabilityInScope` , `AtFullSide. Honest.complete_in_scope` , `VerifiableInference.` 

###### `Compilation.someRequestIsServed_holds` , 

`VerifiableInference.Compilation.zkOnServedRequests_ inScope` , `VerifiableInference.Compilation.finalAtT1_ filled` , `VerifiableInference.Compilation. theFinalStatementHolds` , `VerifiableInference. Compilation.theFinalStatementOneBitHolds` , 

- `VerifiableInference.Compilation.finalStatement_of_` 

`oneBit` , `VerifiableInference.Compilation. theFinalStatementBothHold` , `VerifiableInference. Compilation.rejectsNoProof_value` , 

`VerifiableInference.Compilation.finalExtracts_value` , `VerifiableInference.Compilation.life_number_at_most` , `AtPublicFiles.AtFiles.probabilityInScope79` , `VerifiableInference.Compilation.threeAtFiles_in_use` , `VerifiableInference.Compilation.filesInUse_stored` , `VerifiableInference.Compilation.` 

   - `someRequestIsServedInUse` . 

- **6.49** Completeness in Lean: `FlashProve.completeOn_last` , `FlashProve.LastArgs` , `FlashProve.LastFits` , `FlashProve. verify_record` , `FlashProve.caps_home` , `FlashProve. number_home` , `FlashProve.bytes_forty` , `FlashProve. lastFits_idle` , `FlashProve.three_le_idleAlpha` , `AtFullSide.Home.lastFits_idle_named` , `Meaning.served_ met` , `VerifierAndGoals.settingsOfRead_full` , `VerifiableInference.canServe_nowhere_ atHaltingWeights` . 

##### **Section 8** 

- **8.15** Many proofs without the factor _N_ : `VerifierAndGoals. AtTheName.soundManyProofs_of_verify_at` . 

##### **Appendix A** 

- **A.1** Round-by-round soundness: `Iop.` 

   - `InteractiveOracleProof.Sound` , `Iop.Doomed` , `Iop. BatchInteractiveOracleProof.SndB` , `Iop.JointLaw` . 

- **A.2** Early part, knowledge soundness: `Iop. BatchInteractiveOracleProof.SndBL` , `Iop. BatchInteractiveOracleProof.guardP` , `Iop. BatchInteractiveOracleProof.SndBP` , `Iop. BatchInteractiveOracleProof.SndBKT` . 

- **A.3** The argument over separate oracles: `VerifiableInference.OnPublicFiles. schemeOverTypedOracles` , `TypedOracleCompiler.Compile. Batch.Cms.verifyAtKeyRoot` , `TypedOracleCompiler. Compile.Batch.Cms.verifyAtFullKey` , `TypedOracleCompiler.Compile.Batch.Cms. indexExtractorAtRootAndOutOfDomainSample` , `TypedOracleCompiler.Compile.Batch.Cms.tVC` , `TypedOracleCompiler.Compile.Batch.Cms.detBound_VRA` . 

- **A.4** Word under a root, rooted query, seed-first trace: `TypedOracleCompiler.Compile.Batch.extEnc` , `TypedOracleCompiler.Compile.Batch.Announced` , `TypedOracleCompiler.Compile.Batch.rooted` , `TypedOracleCompiler.Compile.Batch.Cms.SeedsFirstAll` , `TypedOracleCompiler.Compile.Batch.Cms.preAll` , `OnBytes.Slice.between` . 

- **A.5** The event Bad: `TypedOracleCompiler.Compile.Batch. Cms.BadA` , `TypedOracleCompiler.Compile.Batch.EM` , `TypedOracleCompiler.Compile.Batch.EF` , `TypedOracleCompiler.Compile.Batch.stepS` , `TypedOracleCompiler.Compile.Batch.Lane.step` , `TypedOracleCompiler.Compile.Batch.Cms.stepSeed` . 

- **A.6** Conditions on soundness: `VerifiableInference. OnPublicFiles.CompilerSoundnessConditionStatements. keySound` , `VerifiableInference.OnPublicFiles. CompilerSoundnessConditionStatements.entrySound` , `VerifiableInference.OnPublicFiles. CompilerSoundnessConditionStatements.proofSound` , `Iop.` 

91 

   - `BatchInteractiveOracleProof.SndBL` , `Iop. BatchInteractiveOracleProof.SndBKT` . 

- **A.7** Conditions of the compiler: `VerifiableInference. OnPublicFiles.CompilerSoundnessConditions` , `VerifiableInference.OnPublicFiles. CompilerSoundnessConditionStatements.seedLaw` , `VerifiableInference.OnPublicFiles. CompilerSoundnessConditionStatements.wires` , `VerifiableInference.OnPublicFiles.` 

   - `CompilerSoundnessConditionStatements.commitment` . 

- **A.8** Bad is rare: `TypedOracleCompiler.Compile.Batch.Cms. cmsA_bcs` , `TypedOracleCompiler.Compile.Batch. hashEvents` , `TypedOracleCompiler.Compile.Batch.Cms. pr_step4_le` , `TypedOracleCompiler.Compile.Batch.pr_ stepS_le` . 

- **A.9** Outside Bad: `TypedOracleCompiler.Compile.Batch.Cms. BcsAtA` , `TypedOracleCompiler.Compile.Batch.Cms.cmsA_ bcs` . 

- **A.10** The three relations in sequence: `TypedOracleCompiler. Compile.Batch.Cms.coreA` . 

- **A.11** Index-binding knowledge soundness over separate oracles: `TypedOracleCompiler.Compile.Batch.Cms. bindingGAt` , `TypedOracleCompiler.Compile.Batch.Cms. cmsA_binding_at` , `TypedOracleCompiler.Compile.Batch. Cms.cmsRA_binding_at` , `VerifiableInference. OnPublicFiles.CompilerSoundnessConditions.binds` , `TypedOracleCompiler.Compile.Batch.Cms.` 

   - `soundnessBoundOverTypedOracles` . 

- **A.12** One hash function: `TypedOracleCompiler.Compile. Batch.Cms.bindingR_one` , `Rom.binding_overOne` , `Rom. OracleCoding` . 

- **A.13** Public readings: `TypedOracleCompiler.Compile.Batch. Cms.pub_lift` , `VerifiableInference.OnPublicFiles. told_binds_core` , `VerifiableInference.OnPublicFiles. LayoutsByPublicReading` , `VerifiableInference. OnPublicFiles.LayoutsByPublicReading.ext` . 

- **A.14** The commitment as the hash of the key: `ZKCP. HasIndexBindingWith.digestKey` , `VerifiableInference. Rec.soundForWith_systemE` , `VerifiableInference. OnPublicFiles.epsCode` . 

- **A.15** The compilation lemma: `VerifiableInference. OnPublicFiles. verifierAcceptingNoMoreThanBehindReadingAndParser_ soundFor` , `VerifierAndGoals.Fit.soundForWith_system_ of_toldO` , `VerifiableInference.OnPublicFiles. innerCompiledVerifier_soundFor_byPublicReading` , `VerifiableInference.soundForWith_via` , `VerifiableInference.SoundForWith.bindWeights` . 

- **A.16** Which form of the lemma: `VerifiableInference. OnPublicFiles.innerCompiledVerifier_sound` , `VerifiableInference.OnPublicFiles. verifierAcceptingNoMoreThanBehindReadingAndParser_ soundFor` . 

- **A.17** The side conditions hold: `FlashProve. compilerSoundnessConditions_layoutsByPublicReading` , `FlashProve.publicReadingFitAtDescriptionOfFields` , `VerifierAndGoals.AtTheName.soundUnderPolicyAtZ_ numbers` , `FlashProve.layoutsByPublicReading_spare` , `FlashProve.layoutsByPublicReading_word` , `FlashProve. layoutsByPublicReading_code` , `FlashProve. readingsQueryBound_descriptionOfFields` , `FlashProve. layoutsByPublicReading_noCoin` , `FlashProve. layoutsByPublicReading_once` . 

- **A.18** Doomed, reduction, leads: `Iop.Doomed` , `Iop.Err` , `Iop. Err.Below` , `Iop.Reduction.Leads` . 

- **A.19** State functions; composition: `TypedOracleCompiler. Compile.Batch.stOf` , `TypedOracleCompiler.Compile.` 

   - `Batch.stOf_start` , `TypedOracleCompiler.Compile.Batch. stOf_prover` , `TypedOracleCompiler.Compile.Batch.stOf_ ver` , `TypedOracleCompiler.Compile.Batch.stOf_full` , `Iop.Doomed.append` , `Iop.Reduction.Leads.seq` , 

   - `VerifiableInference.Desc.Plain.fold_chain` . 

- **A.20** Proof of work; challenges from hash blocks: `Iop.GChal. cnt_pass` , `OnBytes.Flat.CarryS.doomed` , `OnBytes.Flat. Maps.below_err` , `Iop.kapSl` . 

- **A.21** From three parts to a batch IOP: `TypedOracleCompiler. Joined.proofSound_plainJB` , `VerifierAndGoals. AtTheName.proofSound_of_rows` , `VerifierAndGoals. AtTheName.RowsOfProofFile` , `Iop.Reduction.LeadsB` , `Iop. Reduction.LeadsBN` , `Iop.BatchInteractiveOracleProof. SndBKT` . 

- **A.22** Where a parameter check fails: `FlashProve.keySound_ theLay` , `FlashProve.entrySound_theLay` , `FlashProve. proofSound_theLay` . 

- **A.23** List size strictly below the Johnson radius: `Cb.johnson_ tables` , `Cb.johnson_tables_cnt` , `Cb.listMax` , `Cb. InJohnson` . 

- **A.24** Folding a table: `Cb.Bridge.foldErr_ix` , `Cb.LdIx.Bad` , `Cb.rows_cost_nothingAt` , `Cb.rows_free_abs` , `Cb.rows_ avoid` , `Code.cnt_bad_eqk_mul_sum` , `OnBytes.Slice.list_ johnson` . 

- **A.25** Collisions are rare: `OnBytes.Flat.entryFull_seedLawK` , `OnBytes.Slice.recBound` , `OnBytes.Slice.fr_bound` , `OnBytes.Slice.coll_count_of_list` , `Cb.ldFc_overF` , `OnBytes.Slice.recZ_okOn` . 

- **A.26** Round counts: `Cb.ldSendFam_cnt` , `Cb.ldSendFamErr` , `VerifiableInference.Desc.K9L.num_k9` , `Cb.psc_know` , `Cb. ld_sum_know` , `Cb.Pv.SendLoc.errs` , `Cb.cnt_ldSwBad1` , `Cb. cnt_ldSwBad2` , `Cb.ldSwitch_know` , `Cb.ldBase_good` , `TwRound.lb_entN` , `Cb.ldRng_err` , `Cb.ldRng_goodT` . 

- **A.27** Conditions and claims of IOP<sup>req</sup> _Z_<sup>:</sup><sup>`VerifiableInference.`</sup> `OnPublicFiles.meansSome` , `Meaning.firstMeans` , `VerifiableInference.Desc.First.start` , `VerifiableInference.Desc.First.openSets` , `VerifiableInference.Desc.Shared.startS` , `VerifiableInference.Desc.Shared.endS` , `VerifiableInference.Desc.Shared.promsAt` . 

- **A.28** The first part of IOP<sup>req</sup> _Z_<sup>:</sup><sup>`VerifiableInference.`</sup> `OnPublicFiles.rows_at_descB` , `VerifierAndGoals. AtTheName.kindsLaws_at_head` , `VerifiableInference. Desc.Sound.Piece.GoodIn` , `Iop.Err.cap` , `VerifiableInference.Desc.Sound.chainOfPromisesR` , `P5J. good_of_kinds` , `VerifiableInference.Desc.First. K2Caller.k2_err_cap` , `OnBytes.Road.K4.Plan4.herr_cap` , `VerifiableInference.Desc.First.Att.P.K05_law_one_ guard` , `VerifiableInference.Desc.Kinds.Head.Law.Bears. k06BearsCap` , `VerifiableInference.Desc.Kinds.Head.Law. Came.k7_own` , `VerifiableInference.Desc.First.MergeLaw. merge_goodIn_cap` , `P5J.headA_at_name` , `P5J.conds_of_ laid` , `P5J.owed_of_live` , `H2K12.lastTypedAt_at_head` . 

- **A.29** The shared part and the last part: `P5J.five_at_free2_ laid2` , `Desc.K14.Law.k14_at_free_head` , `Desc.K14.Law. point_leadsB_at_k14` , `Desc.K14.Law.ownPt_below` , `VerifiableInference.Desc.Sound.eachLawsBNOwn_plain` , `P5J.AtMgA.leads3_of_leftAt` , `P5J.AtMgA.lastLeft_at_ head_alone` , `P5J.headA_at_name` , `P5J.conds_of_laid` , `P5J. owed_of_live` . 

- **A.30** The two other IOPs: `VerifierAndGoals.AtTheName. keySound_at_live_holds` , `LPFD.The21` , `LPFD.keyDebtCap_ of_21` , `LPFD.keySound_of_debt` , `FlashProve.hole_ entrySound` , `Desc.K15.H_fullV` , `VerifiableInference. OnPublicFiles.entrySoundOfFullD_of_joh` . 

- **A.31** From the open claims to the request relation: `VerifiableInference.OnPublicFiles.tie_of_some` , 

92 

`VerifiableInference.OnPublicFiles.TieOwesSome` , `P5J. meansOfStart` , `Meaning.layersSay` , `VerifiableInference. Desc.Kinds.Head.Law.head_cuts_of_guard` , `P5J.reply_ leaves_at_mg` , `P5J.owed_of_live` . 

- **A.33** From the request relation to the policy: `OnBytes.Road. L5.sound_ledger` , `VerifierAndGoals.AtTheName.meaning` , `VerifierAndGoals.AtTheName.carries` , `Policy.LedgerRel. Rno` , `Policy.ledger_noWrap` , `Cm.TI.eq_of_dvd` , `Cm.TP.Lay. stream_eqK` . 

- **A.34** The extractor: `VerifierAndGoals.AtTheName.hext` , `FlashProve.theExtractor` . 

- **A.35** Numbers of queries: `VerifierAndGoals.AtTheName. count_the` , `FlashProve. publicReadingFitAtDescriptionOfFields` , `VerifierAndGoals.theDescription_within` , `FlashProve. readingsQueryBound_descriptionOfFields` . 

- **A.36** The arithmetic of the bound: `VerifierAndGoals.frame_ le_bits` , `VerifierAndGoals.soundnessErrorInQueries` , `VerifierAndGoals.AtTheName.errorAtBits` . 

##### **Appendix B** 

- **B.1** Policies that every eligible output meets: `Policy. meetsPolicy_of_close_scales` , `Policy.exists_policy_ of_close_scales` , `VerifierAndGoals.AtFiles.every_ output_at_meets_iff` . 

- **B.3** The example clipped-gap policy: `Policy.gaps_of_ meetsPolicy_qExample_short` , `Policy.card_gap_ge_mul_ le_qExample` , `Policy.meetsPolicy_qExample_of_one_ free` . 

- **B.7** Which compiler: `VerifierAndGoals.AtTheName.sound_ of_verify_at_through_the_IOP` , `VerifiableInference. OnPublicFiles.innerCompiledVerifier` , 

- `TypedOracleCompiler.Compile.Batch.verifyOneProtocol` , `VerifiableInference.Compilation.compile` . 

- **B.8** Algorithm of the executable form: `Run.toBytes` , `Run. bytesToDig` , `Fmt.bitAt` , `PV.Ride.verdictThe` , `PV.Ride. verdictFrom` , `PV.Ride.rideOf` , `PV.Ride.letsThe` , `FlashProve.tiePremB` , `FlashProve.decodeD` , `PV.Head. cThru` , `PV.Head.sevenThru` , `PV.RunPure.Full. stepsFullAllAt` , `VerifiableInference.OnPublicFiles. SlotsBehindCutD.fastSteps` . 

- **B.9** Program verdict: `PV.Ride.verdictThe_is_ theDescription` , `PV.Program.claimOf` , `PV.Program. verdictOf` , `PV.Program.verdictOf_is_verify` , `PV. Program.verdictOf_eq` , `PV.Program.verdictOf_none` , `PV. Program.verdictOf_map` . 

- **B.11** Soundness of the executable form: `PV.Ride.sound_of_ the_form_events` , `PV.Ride.sound_of_the_form_events_ oOf` , `PV.Ride.soundOfDecision_form_iff` , `PV.Ride.sound_ of_the_form_at` , `Run.size_oOf` . 

- **B.13** Lists sent at once: `VerifierAndGoals.AtTheName. soundManyProofs_of_verify_at` , `VerifierAndGoals. AtTheName.sessionAugmented_of_verify_at` . 

- **B.14** Evaluation binding: `VerifierAndGoals.AtTheName. noTwoAnswers_of_verify_at` . 

- **B.16** Soundness with any parser: `VerifierAndGoals. AtTheName.sound_of_verify_parsers_at` , `VerifierAndGoals.AtTheName.soundUnderPolicy_of_ verify_at` . 

- **B.19** Soundness after an honest commitment: `VerifiableInference.SoundAfterHonestCommitmentFor` . 

- **B.20** Composition: `VerifiableInference.SoundForWith. soundAfterHonestCommitmentFor` , `Bind.Cpp. soundAtCommitted` . 

- **B.21** Implications: `VerifiableInference.BindsWith. noTwoOpenings` , `VerifiableInference.BindsWith. extractsDrawn` . 

- **B.22** Proof outlines for the commitment: `Bind.Cpp.tieAt` , `Bind.Cpp.bindsWith_of_tieAt` , `FlashProve.bindsWith_ plain_law` , `VerifiableInference.OnPublicFiles.told_ binds_written` , `VerifiableInference.OnPublicFiles. told_bindsWith_on` , `Bind.Cpp.bindsAtZ_cpp` , `Bind.Cpp. soundAtCommitted` . 

- **B.23** Constants of the errors: `Bind.Cpp.honest_count` , `Bind. Cpp.commit_count` , `Bind.Cpp.soundAtCommitted_square` . 

- **B.24** Two degenerate committers: `Bind.Cpp.blindTwin_ fails` , `Bind.Cpp.blindTwin` , `Bind.Cpp.hollow_meets_ join` , `Bind.Cpp.commit_D` 0, `Bind.Cpp.sample_D` 0. 

- **B.25** What the two committers show: `Bind.Cpp.hollow_ meets_join_never` , `Bind.Cpp.blind_extracts` , `Bind.Cpp. blind_twoOpenings` , `Bind.Cpp.blindTwin_exact_join_ and_complete` . 

##### **Appendix C** 

- **C.2** Disclosed view, honest-verifier simulator: `Iop.Honest` , `Iop.DView` , `Iop.InteractiveOracleProof.dview` , `Iop. Honest.output` , `Iop.Honest.withheld` , `Iop.Honest.HVZK` . 

- **C.3** Table system: `Iop.HolographicProtocol` , `AtPublicFiles.AtFiles.objOwn` , `AtPublicFiles.AtFiles. FullSizeMatrix` . 

- **C.4** Four conditions on the tables: `Iop.HolographicProtocol. ReadsR` , `Hiop.Seeded.ReadsG` , `AtPublicFiles.Premises. NoProverSide` , `OnBytes.WholeKey.PremW` . 

- **C.5** Table experiment: `Hiop.Seeded.EnvS` , `Hiop.Seeded. HostS.view` , `Hiop.Seeded.realHostS` , `AtPublicFiles. AtFiles.objOwn` . 

- **C.6** Within the opening budget: `Hiop.BoundsR.Fits` , `Hiop. Seeded.EnvS.Adm` . 

- **C.8** Columns are uniform: `AtPublicFiles.Table.table_law` , `AtPublicFiles.Table.strong_codingHonest` . 

- **C.9** Pads mask the answers: `AtPublicFiles.Table.answer_ bijective` , `AtPublicFiles.Table.masked_cells` , `Hiop. rec_answers_uniform` , `AtPublicFiles.Table.rowsOf_ laws` . 

- **C.10** One table: `AtPublicFiles.Reads.hidE_filesAt` , `AtPublicFiles.Table.hides_full` , `Hiop.rec_hides_of_ rows` , `Hiop.rec_hides_law` . 

- **C.11** Column counts: `AtPublicFiles.Reads.reads_filesAt_ scope` . 

- **C.12** Tables and runs before compilation: `Hiop.Seeded. whole_all_seeds` , `Hiop.Seeded.form_laws` , `Hiop.BoundsR. err` . 

- **C.13** Table game: `Hiop.Seeded.gameReal` , `Hiop.Seeded. gameSim` , `Hiop.Seeded.Kit` , `Hiop.Seeded.runAt` , `Hiop. BoundsR.Fits` . 

- **C.14** Tables and runs in the table game: `Hiop.Seeded. lifeSeededAt` , `Hiop.Seeded.errN` , `Hiop.Seeded.SimsG. GoodAt` . 

- **C.18** Compiled run of a prover: `OnBytes.WholeKey.shown` , `OnBytes.WholeKey.HonestSide` , `TypedOracleCompiler. Compile.Batch.Cms.proveTA` . 

- **C.19** Coins of a compiled run: `OnBytes.WholeKey.dress` , `OnBytes.WholeKey.RunLaw` , `OnBytes.WholeKey.SaltsLaw` , `OnBytes.WholeKey.CoinNums` . 

- **C.20** Whole-key argument: `OnBytes.WholeKey. wholeKeySchemeOverTypedOracles` , `OnBytes.WholeKey. argW` , `OnBytes.WholeKey.commitAt` , `OnBytes.WholeKey. proveW` , `OnBytes.WholeKey.` 

- `wholeKeySchemeOverTypedOracles_V` . 

- **C.21** Five conditions on the provers: `AtPublicFiles. Premises.FiveLeft` , `OnBytes.WholeKey.ExtEnc` , `OnBytes. WholeKey.Sims.Good` , `OnBytes.WholeKey.StopsSmall` , `OnBytes.WholeKey.RunVals` , `OnBytes.WholeKey.CoinsFit` . 

93 

- **C.22** Eight conditions on nonces and positions: `AtPublicFiles.Premises.NoProverSide` , `OnBytes. WholeKey.ReadsFree` , `OnBytes.WholeKey.NoncesBack` , `OnBytes.WholeKey.SeedPlain` , `Iop.` 

- `InteractiveOracleProof.IsLocal` . 

- **C.23** Three conditions on formats: `AtPublicFiles.Premises. NoProverSide` , `Iop.RoundSplits.Fmt` , `Iop.RAnn.Coherent` , `TypedOracleCompiler.Iop.QueryLayout.OKG` , `TypedOracleCompiler.Iop.WireRB.OKG` , 

- `TypedOracleCompiler.Compile.Batch.Cms.OKA` . 

- **C.24** Hashing error: `Rom.costTotal` , `OnBytes.WholeKey.mW` , `OnBytes.WholeKey.eW` , `OnBytes.WholeKey.bitsW` , `OnBytes. WholeKey.osW` . 

- **C.25** Reduction of index privacy to twenty conditions: `OnBytes.WholeKey.wholeKeyPrivacy` , `OnBytes.WholeKey. Spec.wholeKeyPrivacy` , `OnBytes.WholeKey.PremW` , `OnBytes.WholeKey.zetaW` . 

- **C.26** The conditions that mention no prover hold: `AtPublicFiles.Premises.noProverSide_description` , `AtPublicFiles.Reads.reads_filesAt_scope` , `AtPublicFiles.Reads.hidE_filesAt` . 

- **C.27** Bound on the error after compilation: `AtPublicFiles. LinkLaid.zeta_of_twelve` , `AtPublicFiles.ZetaAt.zetaW_ le_caps` , `AtPublicFiles.ZetaAt.twelve_description` , `AtPublicFiles.ZetaAt.boundZ` . 

- **C.28** Constant-message provers satisfy the five conditions: `AtPublicFiles.Premises.fiveLeft_fixed` , `AtPublicFiles. Premises.hasSilentMemberC_description` , `OnBytes. WholeKey.silent_good` , `OnBytes.WholeKey.extEnc_ silent` . 

- **C.29** Committed tables and link proofs after compilation: `AtPublicFiles.LinkLaid.linkLaidWithBound` , `AtPublicFiles.LinkLaid.linkLaidEight` , `AtPublicFiles. LinkLaid.premW_laid` , `AtPublicFiles.LinkLaid. linkLaidAtOnce` , `AtPublicFiles.LinkLaid.bitsE_listed` . 

- **C.30** Reach of the corollary: `AtPublicFiles.LinkLaid.laid_ hashed_accepted` , `AtPublicFiles.LinkLaid.laid_` 

`accepted` . 

##### **Appendix D** 

- **D.1** The default configuration, encoding: `VerifierAndGoals. storedPublicFiles` , `VerifierAndGoals.policyFile` , `VerifierAndGoals.FreeLines` , `Inst.Model.KLFile` , `OnBytes.Flat.klFileOf` . 

- **D.3** The integer model: `IntegerTransformer.model` , `IntegerTransformer.trace` , `VerifiableInference. hasModelSizes` , `VerifiableInference.lookupTables` , `VerifiableInference.weightsAreValid` , `VerifiableInference.ModelWeights` , `Policy. modelLogits` . 

- **D.6** Sub-protocols: `VerifiableInference.Desc.KindName` , `VerifiableInference.Desc.Block` , `TypedOracleCompiler. Run.PQ.roundS` , `TypedOracleCompiler.Iop.GrW` . 

- **D.7** Test on a committed table: `Desc.K14.rounds` , `Desc.K14. readsAtLoc` , `VerifiableInference.Desc.KindName.K14` . 

- **D.8** Two rounds at the default configuration: `OnBytes. Params.record_radius_inJohnson` , `Cb.Gl.Bld.J.cnt_32_ 446693374_2852142980_132` , `Desc.K14.Law.bad_le_efReal` , `Desc.K14.Law.real_test64` , `Desc.K14.Law.ownPt_below` , `Desc.K14.Law.k14_rounds_real` , `Desc.K14.Law.teeth` . 

- **D.9** The rounds of the three IOPs: `VerifiableInference. Desc.Key.descriptionFromSource` , `Desc.K15.linkDescV` , `ProofFile.onlineDescription` , `VerifiableInference. Desc.First.firstChain` , `VerifiableInference.Desc. KindName` . 

- **D.11** Oracles, seeds, and the compiled verifier of one IOP: `TypedOracleCompiler.Compile.Batch.typedOracleFamily` , `Rom.OracleCoding` , `TypedOracleCompiler.Compile.Batch. OneProtocolProof` , `TypedOracleCompiler.Compile.Batch. recomputeChain` , `TypedOracleCompiler.Compile.Batch. reqCheck` , `TypedOracleCompiler.Compile.Batch. proofHasStrictFormat` , `TypedOracleCompiler.Compile. Batch.checkPrunedOpenings` , `TypedOracleCompiler. Compile.Batch.verifyOneProtocol` , `TypedOracleCompiler. Compile.Batch.Cms.outOfDomainSeedIs` . 

## **G Further limitations** 

These limitations continue Section 8.1. The _judgement_ and an _honest run_ are defined in Definition H.1. 

#### **The implementation.** 

_Limitation_ G.1 (Implementation unverified). The C++ and CUDA programs and the Python check script are tested, not formally verified, as is usual: no theorem concerns them, for soundness, weight hiding or completeness, and no security level is claimed for them. They are only partly tested, with bad results among the tests. The released programs run a scored output head under every policy; that it is the head of Definition 3.4 is tested, not proved (Limitation 8.2, Protocol match unproved), and at one setting that no admissible configuration has its cost differs ( `docs/limits-all-rows.tsv` , SO-6.3.1.1). Between Theorem 5.1 and the C++ verifier’s answer on a real file lie several steps, none closed by a theorem ( `lean/GAPS.md` , `docs/limits.md` ). 

_Remark_ G.2 (Stricter counts). For the IOPs of Construction 4.13, at every admissible _Z_ with Chk( _Z_ ) = 1, the error 2<sup>_−_128</sup> of the Range test’s round is machine-checked (Proposition 5.14). The figures that the C++ verifier prints for single rounds follow another accounting, which no theorem relates to these IOPs (Limitation 8.2, Protocol match unproved). 

In that accounting the Range test’s round, which has no proof of work, and the positions in a link proof use a smaller error than a stricter count would give, and a batch’s shared rounds are counted once. That the smaller errors suffice there is not proved; which count of an own round applies to a single request is not known. By a union bound over all columns of the matrix of cells that is tested, not of Θ, the Range test’s round would have 113 _._ 44 bits at the default configuration, calculated and taken from no run, and so miss the target of 128 (Table 15); with a factor for the number of requests, some shared rounds miss it too. 

94 

Table 15: Figures in bits. _Programs_ : the programs’ own accounting at the default configuration, calculated and taken from no run, and no security level. 113 _._ 44 and “6 fewer” are printed by the released verifier’s planning call. “About 122 _._ 5”, with the rounds’ proof of work counted, is reckoned by hand from a list of rounds stored in the repository, which the released verifier’s planning call prints again line for line (the repository’s test `verifier.kit_rounds` <u>).</u> No figure is a level of the released <u>programs.</u> 

|Figure|Source|What it counts|
|---|---|---|
|113_._44|programs|the Range test’s round, by a union bound over all columns of the tested matrix|
|about 122_._2|this paper|the Range test’s round, 128_−_log254, if a factor for a list of 54 codewords were<br>owed|
|about 122_._5|programs|25 rounds that sample a segment’s positions, with their proof of work counted<br>and with that factor|
|6 fewer|programs|an own round, if it is counted jointly for 64 requests|



#### **The relation.** 

_Limitation_ G.3 (Greedy outputs and a fixed window only). Sampled outputs are not covered. By design a proof covers token IDs, not text, in a window whose length the configuration fixes, 1,024 tokens at the default configuration: every request fills a whole window and costs a full window of proving. An output cut short verifies as a shorter output, and under a clipped-gap policy a provider can still end one early ( `docs/verifying.md` ). 

_Limitation_ G.4 (Prefixes count). Nothing is stated about where the model would stop: any prefix of its output is a greedy output, so the client must check that the output is as long as expected, since Theorem 5.1 states nothing on length. The empty prefix is one too. By step 4 of Construction 4.15, _VZ_ rejects a request whose claimed output is empty, since build is undefined there (Definition 4.7); no theorem states this by itself; Serv _µ_ leaves such requests out (Definition 3.11). Every non-empty prefix of the model’s own output meets every admissible policy. 

#### **Security model.** 

_Limitation_ G.5 (Fiat–Shamir instantiation). That BLAKE2b, on which the programs build the hash function that Theorem 5.1 samples, may be treated as a random oracle, sampled after the configuration is fixed, is assumed; no proof can close this, as is usual. The bound states nothing of the concrete functions. One constant beneath the Lean verifier is a value of the programs’ own hash function; that nothing beneath the verifier reads it rests on a reading of the code, not on a theorem ( `lean/THEOREM.md` ). Fiat–Shamir attacks on protocols of this kind have not been assessed, among them attacks on GKR-style protocols that use a concrete hash function [KRS25] and on protocols whose instances a program generates [Fen26] ( `docs/limits.md` , `lean/LIMITS.md` ). 

_Limitation_ G.6 (Classical adversaries). As is usual, in all machine-checked statements the committer and the prover query the oracle classically, so nothing is claimed against a quantum adversary. 

#### **Evidence.** 

_Limitation_ G.7 (No independent review). The tests, runs, comparisons and readings were made and reviewed within this project, so reviews are not independent, and no written proof is verified or independently reviewed. The technical report’s written proofs for soundness and for zero knowledge are not settled, and readings of them within this project have found errors (Appendix E.1; `docs/limits.md` ). Each count, outcome, figure or condition of Section 7 and Appendix H is its producers’ account unless another checker is named. To our knowledge no person has checked the written proofs, the code, or the logs of any run; of the Lean statements, six definitions of properties were shaped with a person, which is not a check of them (Section “Who wrote and checked this”); the Lean statements apart from the six top theorems and the final statement’s six fields no person has checked, and the definitions that lie beneath these other statements alone no person has read in full; Lean’s kernel checked the proofs. 

_Limitation_ G.8 (Few runs). The figures of Table 3 rest on three runs each, each row’s made close together in time, whose rates differ, row by row, by 1.5 to 1.6 % in the three rows at exact match and by 1.5 to 4.7 % over all six (the most over the least), with no cause shown. The first row’s _±_ 0.92 is the standard deviation of three runs made side by side: three runs of the same kind on other programs, on another day on other machines, had a standard deviation of 3.9; whether the programs, the machines, the day or the rule of the steady rate made that difference is not shown. Each row’s spread reflects the machine and the seed, that of the like three runs also the order of the kinds of run on 

95 

a machine; it does not cover other prompts, days, kinds of machine or programs. Many measurements not on the released programs, and most counts of tests, are single runs ( `docs/limits.md` ). 

_Limitation_ G.9 (Quality unmeasured). The model of the theorems is the integer model (Definition 3.1), a program in integer arithmetic with the shape of Llama 4 Maverick, not the floating-point model; how far the two differ is stated by no theorem. That the released programs prove this same program is tested where compared, not proved (Remark D.4, Model specification); the model is defined at some inputs, weights and policies for which they make no proof. Its task quality has not been measured: no task benchmark was run ( `docs/limits.md` ). 

#### **Visible by design.** 

_Remark_ G.10 (What a client of the programs learns). By design, clients and the public still see some things. From an honest, fault-free run of the implementation, with the provider’s operating conditions kept and a randomly sampled secret seed, a client learns at least: 

1. whether each request was served or failed; 

2. the architecture, shape, ranges and fitted values; 

3. each next token; 

4. batch membership and timing; 

5. failures and stops; 

6. roughly how much was served; 

7. the OOD answers. 

The leakage (Definition 2.13) holds items 3 and 6, item 1 for served requests, and which revealed requests share a batch. Item 2 is part of _Z_ . Failed requests, timing and item 5 are outside the experiments. Item 7 is for the simulator to produce: an OOD answer is computed from the weights and a pad, and with a fresh uniform pad, at a seed chosen before the table is sampled, its distribution does not depend on the weights (Theorem 6.3). No statement covers the OOD answers as the programs make them (Remark 6.4, No prover message is covered). 

That nothing else is visible is not established: the release’s longer list (its page “What is not hidden”) is not shown to be complete, or to match the released programs. 

_Limitation_ G.11 (Outcome leakage). By design a client learns whether each request is served or fails, and nothing bounds what many answers reveal together. Whether an instance is served, that is, whether ( _M, x, y_ ) lies in Serv _µ_ (Definition 3.11), depends on the weights and is not hidden: in the results on the Lean provers the simulator is given it, one bit per request, before compilation and in the statement at every request after it (Theorem 6.37; in a corollary, within one label per request: served, failed or other tokens), and where a request is not served the prover shows nothing before compilation and returns the empty proof after it; in the statement on served requests (Theorem 6.35) it is given the requests and beside these nothing of the weights, and only served requests are asked (Section 6.5). A bit says whether the claimed output is the model’s result on that request, so over many requests the bits tell what the model answers. That a request is refused is unchecked, as the label is: an empty proof proves nothing, and a prover can return it at a request that it could serve; that a service says “failed” rightly is not proved either. For the implementation one bit is not a cap: no statement bounds how much is revealed, and rarer causes of a failure can also depend on hidden values ( `docs/what-is-not-hidden.md` ). 

#### **Operating the released programs: eleven limitations.** 

_Limitation_ G.12 (Further commitments). The judgement of Section 7.5 takes in a further commitment to the same weights, which renewal or a second GPU requires, on thin ground. Among its five conditions: each commitment must sample its own secret seed, on a machine that was not cloned or restored, take nothing from another, and keep the same configuration files (all five: Appendix H.5). No test has searched two commitments together for leaks. With hidden weights, nothing that a client can check shows that two commitments fix the same weights ( `docs/limits.md` ). 

_Limitation_ G.13 (Mask faults unnoticed). The first of the two main risks to weight hiding in the programs is a mask silently lost to a bug or a hardware fault; the second is Limitation 8.4 (Opening budget). A lost mask can go unnoticed: a file with an unmasked value can still verify, and no verifier can see it. A hardware fault or a compiler fault lies outside an honest run (Definition H.1), and so outside the judgement of Section 7.5; a bug in the sources lies inside both. Four compiler faults are known, at least three of which can act silently. One of them was seen to give wrong values in the test of the prover’s own product routine, never in a mask or a file, though no build looks for a wrong value there. It showed under an older version of the GPU compiler than the build requires, which built the GPU 

96 

programs of many of the tests and runs that the release’s pages report as not on the released programs; whether the required version has it is not known ( `docs/how-hiding-is-checked.md` , `docs/limits.md` ). No one has looked for a way for a client to cause such a fault. 

_Limitation_ G.14 (Published values are permanent). One lost mask can give away at least one exact equation in hidden values, in a published file, to everyone and for good. Without its mask an OOD answer gives one exact linear equation on a whole row of weights, permanently public, and so does every OOD answer if the secret seed gets out at any time. 

_Remark_ G.15 (Pseudorandom masks). All machine-checked statements of Section 6 suppose fresh randomness. The released programs derive their masks and other random values, the pads of the OOD answers among them, from a secret seed by a keyed function, the cipher ChaCha20 [RFC8439], which no machine-checked statement covers. For them index privacy is computational at best (Limitation G.14, Published values are permanent). 

_Limitation_ G.16 (Operating conditions). Weight hiding in the programs rests on operating conditions of the provider, most of which no program can check. Meeting them does not by itself show that the weights are hidden, and breaking one can leak weights permanently and silently. All of them apply: the release’s thirteen conditions and their further requirements. One of them is a check that the provider must make on every batch. The prover checks none of its proofs with the verifier, and a proof spoiled by a bug or a hardware fault can fail to verify and hold unmasked values, by readings of the code; so no response may leave the provider before the C++ verifier has accepted every proof of its batch. Of what ships, by readings of the code, the middleware of the vLLM integration, from which clients fetch their responses, hands on no response that has a proof before the C++ verifier has accepted every proof of its batch; `driver/measure_rate.py` reports no rate, and an example script shows no answer of the model’s, before the C++ verifier has accepted the proofs; a front end of a provider’s own must make that check too. At a rejection the middleware overwrites the batch’s proofs, answers its requests as failed, and stops handing out proofs under that batch’s acceptance policy until the operator lifts this hold; where a batch cannot be judged, it stops likewise, at the second try in a row or at the tenth at the latest, and deletes nothing. That check misses a lost mask in a proof that verifies (Limitation G.13). Narrow exceptions to what the middleware holds back and to what it does at a rejection, none open to a client, are told in `integrations/LIMITS.md` . Nothing is hidden from anyone who can read the provider’s disks, memory or GPU, and the shipped vLLM integration breaks at least one condition; in it vLLM and the prover run under one account, which can read and change the weight files, the secret key directory and the request directory ( `docs/limits.md` ). By a reading of the code, whoever can write the secret key directory, or name a directory of their own as the directory of weights, may learn about the weights from how the prover’s starts end; the commitment covers no line of the text file in the secret key directory, the line on the weights among them. 

_Limitation_ G.17 (Tensors of eight or sixteen bits). Nothing stated here on weight hiding in the programs, the judgement of Section 7.5 included, covers a weight file in which any tensor is stored in 8 or 16 bits. The programs accept such files without a warning; it is for the provider to know how each tensor is stored. The examples’ public 4-bit conversion has no such tensor ( `docs/limits.md` ). 

_Limitation_ G.18 (One trial with such a file). No test of hiding on the released programs has used a weight file with a tensor of 8 or 16 bits (Limitation G.17, Tensors of eight or sixteen bits); in the one trial not on the released programs, a party that could write such a file but not read it learned the magnitudes of two weights of their choice ( `docs/limits.md` ). 

_Limitation_ G.19 (Overlapping batches). By default, by a reading of the code, the prover takes a batch while one other, which it has proved, has not been released. In the vLLM integration’s middleware every stored response waits until somebody asks for its batch, up to 1,000 responses by default; a rejection there withholds one batch, and a hold (Limitation G.16) keeps back all of them and deletes none. No proved statement covers batches that overlap in time, so none reaches a request that is sent again to the service, at once or later. No attack through the overlap is known; the only search for one was a brief reading of the code, with nothing run ( `docs/limits.md` ). 

_Limitation_ G.20 (Other requests in a batch). For the programs, hiding a batch’s other requests is a design goal: not proved, never compared in a run of the prover, and outside the judgement of Section 7.5, which concerns the weights. Any client can submit a request that cannot be proved; for the implementation no statement on hiding then covers the batch’s other proofs ( `docs/what-is-not-hidden.md` , `docs/limits.md` ). 

_Limitation_ G.21 (Clipped gap reveals logits). Under a clipped-gap policy, which ordinary serving uses, each answer also reveals by design something about the model’s logits, and so about its weights, even if the masks hold; no statement bounds how much. Many requests can measure hidden logit gaps, up to the clip: outputs cut one token apart at a limit are such questions. Nothing bounds what many answers, time or a stop reveal ( `docs/what-is-not-hidden.md` ). 

97 

_Limitation_ G.22 (Unmarked public-seed mode). A commitment does not say in which mode it was made: the release’s mode for a public seed in place of the secret seed, in which nothing is hidden, neither the weights nor a batch’s other requests, leaves no mark in what is published, so neither a client nor the verifier can detect it ( `docs/what-is-not-hidden.md` ). 

_Limitation_ G.23 (Second verifier text). The Lean library holds the text of a second verifier, written after the C++ verifier’s source text; the release does not build it as a program. It is not _VZ_ . Definitions from that text also stand in _VZ_ ’s own term ( `lean/GAPS.md` , gap 58); the second verifier as a whole does not. Some runs that this paper tells set it, built as a program, beside the C++ verifier (Appendix H.1): the two are not independent, so their agreement adds no evidence, and they are known to differ ( `lean/OnBytes/LIMITS.md` , `docs/limits.md` ). 

**Pages of the release.** Table 16 names, for each limitation, pages of the release that hold finer facts. 

Table 16: Limitations and pages of the release that hold finer facts. 

|Limitation|Pages|
|---|---|
|Limitation 8.1 (Lean verifier only)|`docs/limits.md`|
|Limitation 8.2 (Protocol match unproved)|`lean/GAPS.md`|
|Limitation 8.3 (Zero knowledge of definitions<br>in Lean)|`docs/limits.md`, `docs/what-is-not-hidden.md`|
|Limitation 8.4 (Opening budget)<br>Limitation 8.5 (Provider-chosen configuration)<br>Limitation 8.6 (Scales can disable blocks)|`docs/limits.md`, `docs/limits-all-rows.tsv`, `docs/protocol.md`<br>`docs/verifying.md`, `docs/limits-all-rows.tsv`, `lean/GAPS.md`<br>`lean/GAPS.md`, `docs/limits-further.md`, `docs/limits-all-rows.tsv`|
|Limitation 8.7 (Unattributed commitment)|`docs/limits.md`, `docs/limits-further.md`|
|Limitation 8.8 (Clipped gap is weak)|`docs/limits.md`, `docs/limits-all-rows.tsv`,<br>`integrations/vllm/integer_model/OPERATING.md`, `lean/LIMITS.md`|
|Limitation 8.9 (Completeness on served<br>|`lean/THEOREM.md`, `docs/limits.md`|
|requests only)||
|Limitation 8.10 (Released commitment)<br>Limitation G.1 (Implementation unverified)|`docs/limits.md`, `docs/limits-all-rows.tsv`, `guide/what-is-proved.md`<br>`lean/GAPS.md`, `docs/limits.md`, `docs/verifying.md`, `lean/TERMS.md`|
|Limitation G.3 (Greedy outputs and a fixed<br>window only)|`docs/verifying.md`|
|Limitation G.4 (Prefixes count)|`lean/THEOREM.md`, `lean/GAPS.md`|
|Limitation G.5 (Fiat–Shamir instantiation)|`docs/limits.md`, `lean/LIMITS.md`|
|Limitation G.6 (Classical adversaries)|`docs/limits.md`|
|Limitation G.7 (No independent review)|`docs/limits.md`, `lean/CHECK-RESULTS.md`, `docs/limits-all-rows.tsv`,<br>`docs/figures.tsv`|
|Limitation G.8 (Few runs)|`docs/limits.md`, `docs/limits-all-rows.tsv`, `integrations/LIMITS.md`|
|Limitation G.9 (Quality unmeasured)|`docs/limits.md`, `docs/results.md`|
|Limitation G.11 (Outcome leakage)|`docs/what-is-not-hidden.md`|
|Limitation G.12 (Further commitments)|`docs/limits.md`|
|Limitation G.13 (Mask faults unnoticed)|`docs/how-hiding-is-checked.md`, `docs/limits.md`,<br>`docs/limits-all-rows.tsv`, `integrations/LIMITS.md`|
|Limitation G.14 (Published values are<br>|`docs/limits.md`|
|permanent)||
|Limitation G.16 (Operating conditions)|`docs/limits.md`, `docs/operators-duties.md`, `docs/what-is-not-hidden.md`,<br>`integrations/LIMITS.md`|
|Limitation G.17 (Tensors of eight or sixteen<br>bits)|`docs/limits.md`|
|Limitation G.18 (One trial with such a file)|`docs/limits.md`|
|Limitation G.19 (Overlapping batches)|`docs/limits.md`, `docs/limits-all-rows.tsv`|
|Limitation G.20 (Other requests in a batch)|`docs/what-is-not-hidden.md`, `docs/limits.md`|
|Limitation G.21 (Clipped gap reveals logits)|`docs/what-is-not-hidden.md`|
|Limitation G.22 (Unmarked public-seed mode)|`docs/what-is-not-hidden.md`|
|Limitation G.23 (Second verifier text)|`lean/OnBytes/LIMITS.md`, `docs/limits.md`|
|“Who wrote and checked this”|`lean/README.md`, `lean/CREDITS.md`, `lean/LIMITS.md`, `docs/limits.md`|



## **H The artifact and the evaluation in detail** 

Under the name of each table of Section 7, this appendix gives the conditions, the provenance and the limits of its results. 

98 

### **H.1 Artifact and programs** 

**In short.** The runs that this subsection reports as runs on the released programs were made on programs built from sources that differ from these in comments, names and the wording of messages alone; of the programs that these sources themselves build it reports three small runs and one at the default configuration’s size (“Which programs ran”, below); the sources of `cp_verify` do not differ. In the test runs that cover all 173 public tests, most of which start none of the three programs, 170 passed, one passed in part, two were not run and none failed. 

On synthetic weights, the released `cp_verify` and the Lean verifier were given the same sets of files at the default configuration’s size, as reported: both accepted all 47 honest proofs, and neither accepted one of the 57 proof files from malicious provers; at 3 of the 57 the Lean command’s parser refused the file, which is not a verdict. The Lean verifier also accepted 41 further proofs, on synthetic weights, as reported of the timed runs. One proof made at the default configuration’s size, on synthetic weights, by the programs that these sources build, as reported, was accepted by `cp_verify` and by the Lean verifier. 

`examples/maverick/run.sh` ran at the default configuration’s size on the public weights, without building, downloading or installing: its five stored values came out the same, byte for byte, and `cp_verify` ended with `YES` . In the timed runs on those weights, at the default configuration as fitted to them, all 1,851 requests were served, and all 1,851 proofs, each of the one token after the prompt, were accepted by the runs’ own calls of `cp_verify` . The first and the fifth of the five tests of weight hiding ended as written down beforehand, the first also at the default configuration, once on synthetic weights and once, as fitted to them, on the public weights. 

**What is left.** The 57 proof files from malicious provers were made beforehand by test programs that this repository cannot build, all at one commitment and for three requests: no malicious prover was run against the released programs, and no rejection at the lookup checks themselves has been seen. At the reduced configuration `cp_verify` accepted 228 replayed calls on altered stored tables: that is right if each table still decodes to the honest weights alone, which rests on counts of the changed columns: no decoder was run. Neither test of weight hiding shows that the masks hide, and the other three of the five were made on other programs; no run shows what the programs leave in host or GPU memory, and no audit with the provider’s secrets has run on a commitment to a real model’s weight files. 

Two runs at 64 requests on synthetic weights failed, and no cause is shown. The tests and scripts of those test runs differ from this repository’s, a few in code; of the public tests that need a GPU or the CUDA compiler only two have been run on this repository’s own files. The programs that ran and the runs’ records are not in this repository, so a reader cannot check how they are tied to these sources. Nothing in this subsection is a proof about the compiled programs. 

**Which programs ran.** The version of the release is the one line of the file `VERSION` at the repository’s top; this PDF ships with it. 

The release contains no model weights and no proof about the public weights. 

The _released programs_ are `cp_commit` , `cp_prove` and `cp_verify` (the committer, the prover and the C++ verifier) as the release’s sources build them (the production build). The programs that made the runs that this paper tells as runs on the released programs, or on a test program built from these sources, were built from sources that differ from these in comments, in the names of identifiers and files, in seven quoted texts that hold such a name and in the wording of eight messages (seven that `cp_commit` or `cp_prove` prints, one that a build script prints), and in nothing else. 

This paper tells a program built from those as built from these sources, a run on it as a run on the released programs or on a test program built from these sources, and a reading of those as a reading of these sources; how far that tie is shown, and where it ends, is told under “What a reader can check”, below. This rule speaks of `cp_commit` , `cp_prove` and `cp_verify` and of their test programs. How the tests’ own files, the serving scripts and the other scripts of those runs differ from this repository’s (they differ in code too) is told under “What these runs do not show”, at “The public tests”. 

Of the `cp_commit` and `cp_prove` that this repository’s sources build, this paper tells four runs, three of them here alone: the first and the fifth test of weight hiding, once each, at the reduced configuration, on synthetic weights, each ending as written down beforehand; and a comparison with the middle build told under “What a reader can check”, on synthetic weights: under one test seed the test programs of both sides gave the same files, byte for byte, at the reduced configuration, and so did both production builds of `cp_commit` , called with `-public-weights` , at the reduced configuration. The fourth made one commitment and one proof at the default configuration’s size, on synthetic weights: its item under the calls of the Lean verifier, below, tells it. The counts of runs elsewhere in this paper leave these four out. 

This paper tells of them no timed run, no other run at the default configuration’s size, no run of `examples/maverick/run.sh` and no public test that starts one of the two. The times, speeds and peaks of memory that this paper gives as the released programs’ were measured on the programs of the runs, and are told of the released programs on the comparison alone. 

99 

No program file is in this repository: a reader builds them, and no run told in this paper was made on a reader’s or a provider’s own build. The runs told in this paper are of these kinds, told below in turn: runs on the released programs (the timed runs are among these); runs on other programs, test programs among them; and, told apart, the calls of the Lean verifier. Rows are those of `docs/limits-all-rows.tsv` . 

_On the released programs_ is said only of a run in the list below; _in the timed runs_ , of those of them that are told under that heading; _not on the released programs_ (<sup>_†_</sup> ), or _on other programs_ , of every other run. _Test programs_ are programs in which a fault, a test switch or a test seed is built in (those of the second and the third test of hiding bear<sup>_‡_</sup> ); a test program may or may not be built from the released sources. Where this paper says test programs, the reduced configuration and the default configuration, the release’s pages say “on test builds”, “small size” and “full size”; of a run on the public weights this paper says the default configuration’s size, or that configuration as fitted to them. 

**What ran on the released programs.** The programs that ran were built from the sources told above, and are tied to those by two digests: the verifier’s build word, and the digest in each prover’s build record ( `cp_commit.build` , `cp_prove.build` ). The verifier’s build word is this repository’s too; the provers’ digest is not: comments, names and the wording of messages turn it. Those records are not in this repository, so a reader cannot check that tie. Where a run’s record holds no digest, its programs’ checksums are those of the build whose record holds both. Pages lie outside both digests, and so do nearly all tests and most scripts. Of the 469 files under the provers’ digest 33 are not sources, six scripts and a Makefile among them (what a reader can check: below). 

A test program leaves no build record. That the test programs of these runs came from these sources rests only on the logs of the test runs that made them, which are not in this repository; each checked a checksum of the sources before it built. 

Weights are synthetic, and each run was made once, unless a line says otherwise. The list tells what ran on them. Of the public tests only some rows start one of the three: how many, and how many run test programs instead, is told under “What these runs do not show”. The Lean verifier is none of them: its calls have a head of their own, below. A run that neither the list nor “The public tests, in full” names is to be taken as not made on the released programs, but for one test that is not in this repository, which by a reading of its commands ran them and passed (row RES-39), and the runs of the check by the vLLM integration’s middleware told under the timed runs, and the four runs told under “Which programs ran”. 

- _The public tests_ ( `SUITE.md` : 173 rows), on one H100 a test run, or a CPU machine; reduced configurations, synthetic weights, but for ten rows that give `cp_verify` a parameter file or a stored example of the default configuration’s size, also on synthetic weights, and a few that run a single kernel, sub-proof or matrix at a shape of the default configuration, also on synthetic data (among them `gpu.base_kernels_test` ); no row runs `cp_commit` or `cp_prove` at the default configuration’s size. 

   - On programs that each test run built from these sources, 170 of the 173 rows passed, one passed in part, two were not run and none failed. Apart from those test runs, on this repository’s files alone (without `lean/` ; some pages, tables, `paper/paper.pdf` , one list and licence and notice files differ), 90 of the 105 rows of the kind `cpu` passed and two passed in part; the other 13 need a set-up row that needs a GPU, and the runner’s `cpu` mode leaves such rows out. Two of the rows that need a GPU or the CUDA compiler have been run on this repository’s own files, once, as far as their files were compared, and passed. In full: below the list. 

- _`cp_verify`_ , on the four stored examples, two at the reduced configuration and two at the default configuration’s size; 20 altered sets of the two at the reduced configuration, ten of each; 57 proof files from malicious provers at the default configuration’s size, made by test programs that this repository cannot build, and replayed on the released `cp_verify` , beside 47 honest proofs at the default configuration’s size and 5 controls; 15 bad parameter files; 13 small proofs on synthetic weights, from a test program that cheats, which are not in this repository: 2 of true claims and 11 false proofs; 2,720 kept calls (a kept call is a command line and its files, stored with the ending that other programs gave it, and replayed), among them 228 calls, at the small example’s setting, on altered stored tables; planning calls at 15 configurations, of which one was read against the figures of the release’s pages (12 of the 15 are in this repository), and three more planning calls told here; and 21 made-up settings, each a set of files in which one file is changed. 

All four stored examples accepted (two with `YES` , two with `WITHIN LIMITS` ), and all 20 altered sets rejected. All 47 honest proofs accepted. All 57 proof files from malicious provers rejected, and all 5 controls (4 `NO: reply-differs` , 1 `NO: model-statement` ). All 15 bad parameter files refused (10 `NO: parameter-range (policy)` , 4 `NO: parameter-group-shape (policy)` , 1 `NO: parameter-table-shape (policy)` ). Of the 13 small proofs the 2 of true claims were accepted ( `YES` ) and the 11 false proofs rejected ( `NO: proof-refused` ), as expected. 

Of the 2,720 kept calls 2,672 ended as recorded, in exit code and last line, the 228 among them, which `cp_verify` accepted. Each of the 228 is on an altered table that, by separate programs’ counts of the changed columns, lies far inside the distance at which it decodes to the honest weights alone, so that the accepted request and output 

100 

are the honest run’s own. That rests on those counts: no decoder was run, nothing of it is machine-checked, and BLAKE2b-256 is taken as collision-free. 

The counts measure from the table of the run with nothing altered; that this table holds the honest codewords was checked at all positions under one of the four test seeds only, under which 24 of the 57 sets of files behind the 228 stand (these 57 are other files than the 57 proof files from malicious provers). In 72 others of these calls `cp_verify` passed a check of one part on a key file or a copy’s file whose own table was altered; a passed check of one part is not an acceptance (row SO-15.7). 

The other 48 read parameter files that the released programs refuse, and end with `NO: parameter-range (policy)` : they show nothing beyond that refusal. No call was more accepting than recorded. All 21 made-up settings were refused, the one with `answers` 2 with `NO: parameter-range (policy)` . The planning calls at the 15 configurations all ended with exit code 4 ( `PLAN ONLY` ): a planning call judges no proof. The three others told here ended with exit codes 1, 1 and 2, as written down beforehand; on which files they were made this paper does not tell. The one that was read against the release’s pages printed the figures that they give as its own. 

- _Tests of weight hiding_ (Section 7.5 and Appendix H.5). The first and the fifth of the five tests (row LIM-63.2, lines 1 and 5), under exact match, on all three of the released programs, each commitment under a seed that `cp_commit` drew. Both at reduced configurations, the first three times and the fifth once. The first also at the default configuration, once on synthetic weights and once, as fitted to them, on the public weights. Both audits also under a clipped-gap policy, at the reduced configuration, on two proofs. One of the two audits, the audit of a commitment, also at the default configuration’s size, in the timed runs. 

Each ended as written down beforehand. The first: all published OOD answers were the row’s weighted sum plus the public factor times the row’s own pad; no pad came twice, and none was 0. The fifth: both audits (checks, with the provider’s secrets, of a commitment and of a kept batch) accepted the honest files, and with a false file in turn for each of three kept files the answer was no each time. The figures: row LIM-63.2, lines 1 and 5. Under the clipped-gap policy both audits accepted the honest files, and `cp_verify` accepted both proofs as within the policy’s limits. 

- _The example script_ , `examples/maverick/run.sh` , at the default configuration’s size, on the public weights; all the stages of `run.sh all` , in their order, on one H100; the build stage was skipped: the programs were built beforehand from these sources; the download stage fetched nothing: the five weight files were laid there beforehand at their checksums; and the toolkit was installed beforehand by that page’s typed lines, so building, downloading and installing the tools were not exercised. 

   - Exit code 0. The five values in `examples/maverick/expected.txt` came out the same, byte for byte, and the commitment ID was the stored one; `cp_verify` ended with `YES` . By the run’s own record, which two readers read apart, it changed no file of the repository. 

- _The timed runs_ , told under their own heading, below. 

**The public tests, in full.** This part gives in full what the list’s item on the public tests gives in short. _The four test runs._ For 172 of the 173 rows, on programs that each test run built from these sources: 169 passed, one passed in part ( `tools.confine` : of its check “a device node” the part on the character device 0:0 was not run on that machine), two were not run ( `verifier.clang_build` : no `clang++` on the machine; `tools.lean_verify.full_size` , the one row that would give a proof to a Lean verifier: the Lean library was not built), none failed; all on synthetic weights, in four test runs, which ran the tests by name. 

The 173rd row, `verifier.rejected_proofs` , ran in a test run of its own on a verifier built from these sources, on a CPU machine, and passed. Of the 172 rows nine start `cp_commit` or `cp_prove` in the production build and 20 others start the production `cp_verify` alone; the rest run test programs, copies, the tests’ own programs or scripts, only build, or only read files. The four test runs ran in a tree that holds files that this repository does not hold, and that differs from it in some tests and scripts (described under “What these runs do not show”, at “The public tests”). 

_The rows that need a GPU or the CUDA compiler._ Nearly every public row that needs a GPU or the CUDA compiler compiles the provers’ sources, and so reads files that differ as “Which programs ran” describes; beyond that: `gpu.forest_lookup_test` and `build.forest_lookup_test` (their test, in code), `gpu.coins_own_gpu_test` and `build.coins_own_gpu_test` (one printed message of their test), `gpu.quick_start` (the page that it reads), and `gpu.serving_first_run` and `gpu.coin_contexts.services.not_full` (they run `driver/spool.py` , in which comments and docstrings alone differ); each of them ran, in the four test runs, under a runner that differs as described under “What these runs do not show”, at “The public tests”. 

The tests’ own files that differ in comments, names and printed texts alone are described there too. `build.forest_lookup_test` and `gpu.forest_lookup_test` have been run once on this repository’s own files, as far as their files were compared, by their rows’ own commands, apart from the four test runs, and passed; from which build the checking program of `gpu.forest_lookup_test` came is not shown. No other row that needs a GPU or the CUDA 

101 

compiler has been run on this repository’s own files. The builds for the comparison described under “What a reader can check” were made apart from these rows. 

_This repository’s files alone._ Apart from the four test runs, most of the rows to which `SUITE.md` gives the kind `cpu` or `long` (107 of the 173: the rows that themselves need neither a GPU nor the CUDA compiler) were run on a machine with no GPU, each row confined by the runner. On this repository’s files alone, without `lean/` , the runner was asked in one round for all 105 rows of the kind `cpu` : 92 were run to their end, and 90 of them passed. That round ran on files that differ from this repository’s in 47 paths: 38 pages, 4 tables, `paper/paper.pdf` , the list `tools/lean-verify/native-modules.txt` and 3 licence and notice files. No source, test or script is among them. 

The 90 are all the rows of the kind `cpu` except `weight_sums` , `tools.confine` and the 13 described below; `verifier.clang_build` , which the four test runs could not run, is one of them. `weight_sums` and `tools.confine` passed in part. Of `weight_sums` 13 steps were skipped without two Python packages; run with both installed it passed, with no step skipped. Of `tools.confine` one check was not run, “as root, another account’s file in /dev/shm”, since the test already ran inside a row that the runner had confined as root. Sixteen rows were also run on files that differ from this repository’s in 30 paths, of the same kinds, and all passed; `integer_model.cpu` (below) is one of them. Which rows read a path that differs has not been traced. Neither the round nor these sixteen ran on this repository’s files as they are. 

_Rows left out or not run._ On this repository’s files alone the runner left out 13 rows of the kind `cpu` whose name begins `verifier.` (those that need `setup.target.other_parameters_worlds` or its `.work` ): each needs a set-up row that needs the CUDA compiler and a GPU. The rows that were run to their end and these 13 make up all 105 rows of the kind `cpu` . Of the two rows of the kind `long` , `integer_model.cpu` passed on this repository’s files alone, in the environment that `SUITE.md` names for it, and `tools.lean_verify.full_size` was not run: that tree held no `lean/` . Its command, run once apart from the test runner, is described below, with the calls of the Lean verifier. 

_A row that can fail on a busy machine._ `driver.measure_rate` has failed twice, once on this repository’s files alone and once in another tree with the same `driver/` , each time on a machine under load: one of its timing tests found no sample from its stand-in for `nvidia-smi` . Run again it passed both times, and in the round described above it passed at once. That the load is the cause is not shown. 

_What no row runs._ No row runs `tools/accepts/accepts.sh` , the real 

`tools/lean-verify/LeanVerifyWalk.lean` ( `tools.lean_verify.script` runs a stand-in for it), `integrations/vllm/integer_model/tests/lean_vectors/run.py` , or `make_vectors.py` and `CompareVectors.lean` beside it, whose runs the integer-model package’s pages describe. 

#### **What these runs do not show.** 

- _The public tests._ Most rows run the tests’ own programs or scripts, or only build, and start none of the three released programs. By a reading of the commands of the 172 rows, not a trace, nine rows start `cp_commit` or `cp_prove` in the production build, and 32 others start `cp_verify` : ten of these a copy of it that the folder of a test program’s build holds, and two a sanitizer build of it. 

Of the public tests eight rows run `cp_commit` or `cp_prove` as test programs built from these sources. Those are not the released programs, and such rows are told as of test programs. Two of the eight are also among the nine. Thirteen further rows read or judge files that two of these eight made. In each GPU test run the two digests were found among the records of its production and verifier builds; test programs are tied to no digest. One production build is tied to none: `gpu.quick_start` builds its own programs in a copy of the tree. 

Each test run ran named rows ( `--only` ), so the commands that `SUITE.md` prints were not used. The tree that the rows ran in holds 473 files that this repository does not hold, nearly all of them tests, and no `lean/` . Of this repository’s 1,218 files outside `lean/` , `paper/paper.pdf` apart, 880 stand the same, byte for byte, in that tree. The others are 97 pages and tables, three of them tables that tests read; nine files that stand at another path; 129 files of the provers’ sources, which differ as “Which programs ran” tells; 12 files of the provers’ tests, in comments, names and printed texts as reported, and one of them, `prover/tests/forest_lookup_test.cu` , in code; 52 other tests; the runner, `run_suite.py` , in a comment; eight files of the serving scripts and their tests; one file of `tools/lean-verify/` , in which a comment alone differs; one list, `tools/lean-verify/native-modules.txt` , which names the Lean modules that the command runs compiled: it was drawn anew; of the rows that were run none takes a module’s name from it ( `tools.lean_verify.script` lays a stand-in list of its own); one script each of `tools/accepts/` and `examples/rejected-proofs/` ; eight tests and four sources of the integer-model package; four files under `docs/format/` and `tools/lean-verify/` , in comments, docstrings and two printed headings; three licence and notice files, one of which that tree lacks; and seven others: three `figures.txt` , `examples/maverick/run.sh` , `examples/quickstart-gpu.sh` , `scales/make_calibration_set.py` and `toy/mask_rules.py` . 

By a comparison of tokens, of the 52 other tests 35 differ in comments and docstrings alone, one in a name too, 11 in string literals too, two are text or data, and three differ in code ( `integrations/tests/test_ask_and_check.py` , 

102 

`test_end_to_end.py` and `test_spool.py` beside it); of the eight files of the serving scripts and their tests two differ in comments and docstrings alone and six in code ( `integrations/proof_middleware.py` and five tests or helpers; the runs of the middleware’s check, told under the timed runs, ran another text of that script, and in what it differs is told there); the scripts of `tools/accepts/` and `examples/rejected-proofs/` differ in code; of the integer-model package’s files and the seven others none differs in code. 

A string literal is not always only printed: some are values that a test compares, or lines that are written into a file. Where a test or a script differs in code, what passed in those test runs is the text that ran there, not this repository’s. `prover/tests/forest_lookup_test.cu` is the test of the rows `build.forest_lookup_test` and `gpu.forest_lookup_test` . Apart from those test runs, `tools/accepts/` , the released script and its table of calls, by their checksums, ran at the reduced configuration alone over the released Lean files: all 4 calls of Lean ended as expected, and all 17 controls were refused. It runs single parts’ checks, not the Lean verifier. 

A row that reads pages read that tree’s. A row that reads the stored examples read them as they lie in this repository. Three rows ran a second time in one test run, and passed both times. Two rows passed with some of their cases skipped, as their entries in `SUITE.md` allow ( `integrations.unit` , `integer_model.cpu` ). The two rows `setup.target.other_parameters_worlds` and `setup.target.other_parameters_worlds.work` made 17 sets of files under other parameters (the tests call them worlds); the set-up leaves 3 out. Of each row its verdict line was read, not all that it printed. 

The row `gpu.frame_readings` , one of the 169, compares the prover’s and the verifier’s readings of the public files. In the four test runs it ran over more folders than `SUITE.md` names; over the eight that it names it passed, at reduced configurations, in a test run of its own, on another machine, on the tree that the rows ran in. At the default configuration’s size, on four folders (two different configurations), on synthetic weights, it ran in a test run of its own, in a form that `SUITE.md` does not have, on one H100, and ended as expected. What failed in other runs of these tests: row T45. 

- _`cp_verify` ._ The kept calls say nothing of a byte that is hashed on a path that none of them takes: a change there would not show. At the kept refusals the replay is blind: those calls end just as a changed digest would end them. Of the runs on the 57 files, each run’s exit status, the number of its lines on standard output and its last two lines, cut at 120 characters, were kept; a refused run prints no build word. 

   - Twenty-two of the 57 proof files from malicious provers were tried once each, and all of the twenty-two claim the true output. Of the 57, 24 claim a false output and 33 the true output with a lie inside, as reported. All 57 proof files from malicious provers stand at one commitment and at exact match, for three requests. The stored files were checked, not made, and which programs made them is not traced. The altered sets are honest files with a few bytes changed, not the work of a malicious prover. 

How narrow the tests with proof files from malicious provers are, and that the falsifications of those twenty-two were not aimed at the lookup checks: `guide/known-limits.md` . Of the other 35, at least three, by their makers’ labels (the 24 made for false outputs have records that name no kind), aimed at a lookup (a key in no row; a count at another row) or at a digit’s range; these three were rejected at the check that the proof’s masked message fits the linear claim, before any check of a lookup or a range was reached, so no rejection at the lookup checks themselves has been seen (as reported; row T1.1). 

- _Tests of weight hiding._ Neither test shows that the masks hide (each test’s bounds: row LIM-63.2, lines 1 and 5). No run or comparison told in this subsection shows what the programs leave in host or GPU memory (rows H39.1 and HW-748). On the released programs, of the audits with the provider’s secrets, the audit of a commitment alone ran at the default configuration’s size: in the timed runs, behind each of four honest commitments, one of them under a clipped-gap policy: by the runs’ own account it accepted all four. At the default configuration’s size the fifth’s two audits have run together only on other programs (below). An audit’s yes to honest files is no negative test. At the reduced configuration the false files stood for kept files: no altered public file was put to the audits. 

With the provider’s secrets no audit has run on a commitment to a real model’s weight files: by a reading of the code, these programs do not let it run there (row HW-747). Under the clipped-gap policy the audits at the reduced configuration show that the masks are there as derived: that is no search for a leak; in that run the batch’s two requests have one and the same output, and no false proof was put to `cp_verify` (row H13.8). Of the five tests behind the judgement on weight hiding these two ran on the released programs ( `docs/limits.md` ). The other three, and most of the readings behind that judgement, were made on other programs, so it reaches the released programs on less evidence, and it does not reach code that differs from the released programs’. 

- _The example script._ One run, of the one example request of `examples/maverick/README.md` , on one card, with no batch. It neither built the programs nor downloaded the weights. The commitment is the published one, made with `-public-weights` and this repository’s public seed: nothing under a secret seed was tried. A program typed that page’s lines, and a reader’s machine may differ in CPUs, memory and disk. The script that ran differs from this repository’s in comments, in its usage text and in some messages, and in nothing else. 

103 

**The calls of the Lean verifier.** The command `lean-verify` , which evaluates the Lean verifier, is not one of the released programs (Appendix H.3; `lean/THEOREM.md` , “What has been run”). It ran over the released Lean files. It was called on proofs at the default configuration’s size, which are inputs made beforehand; 109 calls, on synthetic weights: 47 honest proofs (42, as reported, made by programs built from other sources; the 2 stored examples at the default configuration’s size, whose maker is not traced; 3 by test programs), 57 proof files from malicious provers, which test programs made, and 5 controls, each an honest proof beside a claim that is not its own. 

All 47 honest proofs accepted, 45 at exact match and 2 under the stored clipped-gap policy, where an accept says less. None of the 57 proof files from malicious provers accepted: 54 rejected, and at 3 the command’s parser refused the file before Lean was called, which is no verdict and no rejection. All 5 controls rejected. 

- _`lean-verify` , 41 calls more_ : proofs at the default configuration’s size; 14 requests, 7 of them also among those of the 47, at 3 commitments; 38 proofs for one next token, 3 for 1,001 tokens. 41 proofs, as reported of the timed runs, all on synthetic weights and none of them among the 42 told above, have been put to the Lean verifier, over the released Lean files; it accepted each ( `lean/CHECK-RESULTS.md` ). 

- _`lean-verify` , nine calls more_ : a small stored example beside altered parameter files. Each ended at the parser with no verdict, as written down beforehand. 

- _The command of the public test_ _`tools.lean_verify.full_size` , run once apart from the test runner, beside the 109 calls_ : the proof of the stored example `examples/full-size/` (on synthetic weights), which is one of the 2 stored examples among the 47; over the released Lean files. Exit code 0 and “VERDICT: ACCEPT (exact match)” ( `lean/CHECK-RESULTS.md` ). Through the runner that row has never reached Lean in a run that this paper tells; whether it can end in an accept there, under the runner’s own HOME and confinement, is untested ( `SUITE.md` names the obstacles). 

- _`cp_commit` ,_ _`cp_prove` ,_ _`cp_verify` and_ _`lean-verify` , one behind the other_ : one commitment, under a seed that `cp_commit` drew, and one proof at the default configuration’s size (the second call of `cp_prove` ; the first was refused for a fault in the testers’ command), for 1,001 tokens, on synthetic weights, made, as reported, by the `cp_commit` and `cp_prove` that this repository’s sources build; judged over the released Lean files. `cp_verify` ended with `YES` , and the Lean verifier accepted the proof: exit code 0 and “VERDICT: ACCEPT (exact match)”, as written down beforehand. It is one proof of one request: it says nothing of other requests, of proof files from malicious provers or of a real model’s weights. 

_What these calls do not show._ These calls were made over the released Lean files. This part speaks of the 109 calls, the 41, the nine and the run of the public test’s command; the call on the one proof at the default configuration’s size in the item on `cp_commit` , `cp_prove` , `cp_verify` and `lean-verify` above is described at its end. The command and the parser were held by checksums before and after the calls; every file of the command and of the parser outside `lean/` that the traced calls beside them opened has this repository’s checksum (none of the run’s own calls was traced: `lean/GAPS.md` , 14, “Which programs (k)”). 

The outcomes were known beforehand, from other calls, over other Lean files, and those written down beforehand were these. The program `cp_verify` was not called in these 109 calls: the script that made the calls calls the command alone, and traces of one small call and of one call at the default configuration’s size, neither of them among the 109, show no start of that program. Its source file is compiled into the parser’s unit. The same 109 sets of files were also put to the released `cp_verify` , as reported: it accepted the 47 honest proofs and rejected the 57 proof files from malicious provers and the 5 controls. 

The sets were paired afterwards, by the label, the policy and the first 16 hex digits of the proof file’s checksum; the other files of a set were not paired. The 42 told above are, as reported, 41 of the 363 at exact match and the one under a clipped-gap policy that `guide/what-is-proved.md` tells. They were taken by rules fixed beforehand and have little variety: as reported, the 42 are of 38 requests and 2 shapes (38 proofs for one next token, 4 for 1,001 tokens), at 4 commitments. 

All proofs of these calls at the default configuration’s size are at one model size: each set’s three configuration files are those of one of the two stored examples at that size, which part in three lines of the parameter file alone. Who made a file, of the 109 or of the 41, on which weights, by which rules it was taken, and that it is honest or from a malicious prover, is as reported: it rests only on the word of those who made and chose them. None of the 57 proof files from malicious provers and none of the 5 controls was put to the command under the clipped-gap policy. 

All calls ran on one machine, beside other work. By the command’s text the natively compiled part and the parser’s unit are each compiled by one compiler; what these two compilers print as their versions was read off that machine after the calls. That the command and the parser read the files that the traced calls opened, and no others, is not shown. At those of the 109 calls and of the 41 more that reached Lean the command loaded the compiled files of the released Lean library. 

By the word of those who made them and of those who keep them, a build of these Lean files made them from nothing, in several calls of Lake; that they were compiled from these files is Lake’s own finding, not the run’s. The 

104 

checksums of the library’s own compiled files, taken before and after the 109 calls, the 41 more and the nine more, are the same. The compiled files of the packages that the library imports, which that build did not make, were not held by checksum at the calls. 

This paper tells no comparison of two builds of these Lean files. The natively compiled part the command compiled itself, in a call of its own before the 109, but for three objects of a package that lay in the tree before that call. The natively compiled objects’ checksums, taken before and after the calls, are the same; each call links the library anew from them, and that library’s checksum was printed only at the call that compiled them. In full: `lean/GAPS.md` , 14. 

`lean-verify` evaluates the Lean verifier, natively compiled in part: that is a run, not a proof. Lean reads each file’s parsed form, made by the command’s parser, which no theorem covers. The parser’s compiled code had the same checksums at each of the 109 calls that reached it and at a control compiled apart; at the run of the public test’s command (its item above), at another path, its checksum is another, and the same once names are struck. 

The call on the one proof in the item on `cp_commit` , `cp_prove` , `cp_verify` and `lean-verify` above was held less closely. As reported, the compiled Lean files of its folder had the checksum of the build described above before and after that folder’s first call, which was another call; no checksum was taken after this one. That the Lean files, the command and the parser of that folder are the released ones rests on the testers’ word alone. It ran on another machine. 

**The timed runs.** These are the runs behind the figures of Section 7.2 and Appendix H.2 that carry no mark, made on the released programs. They were made within three hours of one day, on H100 GPUs: six kinds of run at the default configuration’s size, made 3 times each, each kind’s three on three machines (at the default configuration, on the public weights as fitted to them: on the public weights (1) a commitment with one request, and services (2) at 24, (3) at 64, and (4) at 8 and 1 requests a batch; on synthetic weights (5) a commitment, its audit and services at 24; (6) at that configuration with a clipped-gap policy: on the public weights, sessions with every output token checked and like sessions that check none). 

Each of the nine machines that served on the public files at the default configuration as fitted to them made its own commitment first; three of the nine commitments were named beforehand as the runs behind the figures. The audit in these runs is a check of a commitment with the provider’s secrets. One run each was made of the making of an example under a clipped-gap policy, with an audit of its commitment, and of one service and one call at 64 on synthetic weights, both on one commitment made for them on that machine, which was not audited; those last two failed. 

On synthetic weights every proof was accepted, and every altered copy and call rejected. On the public weights, in the four kinds of run at the default configuration as fitted to them, all 1,851 requests were served, and all 1,851 proofs were accepted by the runs’ own calls of `cp_verify` behind the timed stretches (1,845 in the runs behind the figures; six more, one on each of six further commitments); in the three runs of the services at 24 requests a batch, the only runs on these weights with that test, all 249 altered copies and calls were rejected. 

`cp_verify` rejected every one (exit code 1, a last line NO); six of the 240 fell at the test on a file’s first eight bytes, three of the calls fell at the comparison of the proof’s next token with the caller’s and three at that of the commitment’s ID, and for 234 copies and three calls the records keep only a name that covers tests of form and of substance alike (for two of the 234 a name that places the fall in a chain inside a sub-proof, and no nearer), so at which test each fell is not known. 

Each of these 1,851 proofs is of the one token after the prompt, under exact match. So are the 720 proofs of the services on synthetic weights; four more there, one on each of four commitments, cover 1,001 positions each: 1,000 tokens of the window are checked and the token behind it is stated; so does the clipped-gap example’s one proof, under that policy. 

In the sixth kind of run all 720 requests were served and all 720 proofs accepted by the runs’ own calls of `cp_verify` behind the timed stretches: the 360 of the sessions with whole outputs checked check every output token of their request, and passed within the clipped-gap policy’s limits, not by exact match; their control’s 360 check the last position alone, and passed by exact match. The requests of both were 48 stored windows of 1,024 tokens, some sent more than once. 

The requests were, on the public weights, in the four kinds of run at the default configuration as fitted to them, by each run’s own line, a set of 96 windows of 1,024 tokens, and at 24 and 64 a batch some were sent more than once; on synthetic weights 96 windows of made-up tokens (row RES-6). The altered copies are honest files with a few bytes changed, not the work of a malicious prover. The audits that were part of these runs are told under the tests of weight hiding (Appendix H.5). 

Whether a difference from a figure of another run comes from the programs is not shown: such runs were on other machines. The scripts `driver/spool.py` and `integrations/spool.py` that ran differ from this repository’s in comments and docstrings alone. The check by the vLLM integration’s middleware also ran in runs of its own, in none of the timed runs, with the released `cp_verify` , on stored proofs at the default configuration’s size on synthetic weights, as reported, called directly, with no inference engine and no HTTP server around it. 

By itself, on a batch of 24, all 24 were released, which is no negative test: no false proof was put to it. Beside a serving `cp_prove` , in one run, it read stored proofs, not that prover’s own (row RES-32). By the checksums that 

105 

these runs’ records print, the scripts that ran differ from this repository’s in comments, docstrings and the texts of messages of `integrations/proof_middleware.py` alone; for one run the constant `CHECK_THREADS` was also set to 4 (row RES-32; a run behind none of this paper’s figures). 

#### **What came out badly, or otherwise than expected.** 

- On the released programs: 48 of the 2,720 kept calls of `cp_verify` end otherwise than recorded, all with exit code 1 and `NO: parameter-range (policy)` : they read parameter files that the released programs refuse. 

- In the timed runs: 

      - Two runs at 64 requests failed, both on synthetic weights: a service of five batches of 64 in its first batch, and one call of the prover on 64 requests at once. None of their 320 and 64 requests got a proof, the GPU’s memory was nearly full in both (81,053 of 81,559 MiB at the largest sample, which is a reading of the samples, not a line of the program’s), and no cause is shown; one run each, so no failure rate comes of it. No batch of 64 has been proved on synthetic weights: every attempt on record failed, eight in all, six of them on other programs. On the public weights such services ended well (rows T50 and RES-11). 

      - On the public weights the services at 64 requests a batch came to their end with 930 MiB of the card’s 81,559 MiB free at the fullest (80,629 MiB in use); in like runs on other programs: 866 MiB (row RES-11). 

      - With the start-up counted, one of the three sessions with whole outputs checked, at 24 requests a batch under a clipped-gap policy, stays under 100 tokens per second (99.33; the others 103.94 and 103.79), and so does one of the three sessions of their control (98.12; 102.47 and 102.54): both on one machine, the slower in both sets. No cause is shown. Both ran at the default configuration as fitted to the public weights, but for the three numbers of the clipped-gap policy in the parameter file, on stored outputs with no inference engine; the rates with whole outputs checked, accepted under that weaker policy, cannot be set against the project’s target of 100 tokens per second or the rates at exact match (row RES-12). 

- On other programs<sup>_†_</sup> , in another set of runs that were timed, three of each of five kinds of run at the default configuration (on the public weights: as fitted to them), no run reaches 100 tokens per second with the start-up counted: 96.32, 97.21 and 96.43 on the public 4-bit weights (start-ups of 78.2 to 81.0 seconds), 97.90 ± 0.74 on synthetic weights (97.10 to 98.56); in one of the runs on the public weights other work of the project stood on the host through one service’s whole start-up, and the CPU pressure rose to 15.8 % of a session’s length (row RES-33). 

- On other programs<sup>_†_</sup> , in the first test’s run at the default configuration’s size on the public weights, which was not made to be timed, `cp_commit` took 3,143.5 seconds, about 38 % longer than a like run on other programs (2,277.0 seconds): a test script that is not in this repository bound the run to CPUs of the other NUMA node than its GPU’s. In the same test on the released programs, not made to be timed either, bound to CPUs of the GPU’s own node, on another machine of the same kind, it took 2,328.2 seconds; programs and machine differ too, so whether the binding is the cause has not been tried. 

   - Another run like the one of 3,143.5 seconds, also on other programs, took 3,169.9 seconds, against 2,369.76 to 2,374.11 in three runs of the same programs on the public weights, on other machines (row RES-46); on which NUMA node that run’s GPU sat is not known. In the run on synthetic weights, also on other programs, on CPUs of the GPU’s node, it took 2,248.7 seconds (a like run on other programs: 2,261.5) (row RES-9). 

- On other programs no run reached 100 tokens per second with the start-up counted, at the default configuration (on the public weights: as fitted to them) (row RES-33), two commitments took far longer than their like (row RES-9), and one proof was longer than any of the timed runs’, with no cause shown (row RES-6). 

- In the timed runs, and in one set of three runs on other programs, the audit with the provider’s secrets took longer in each run than in any of another set of three on other programs, on other machines: a small shift upward is neither shown nor ruled out, and no cause is named (the times in the timed runs, and the range of the others: row RES-15). 

- On other programs: some tests notice little of a fault in `HasherT` , the class that feeds and finishes BLAKE2b (its source: `docs/sources-notes.tsv` ; what that costs the self-checks: row E2.4). Faults were planted in a like class. Of three, 12, 10 and 12 of 50 CPU rows noticed each in turn. Of three more, planted for the GPU’s code alone, the comparison of host and GPU noticed all three, three tests noticed one, and two tests noticed none. 

- On other programs, in other runs of public tests or of tests not in this repository: a GPU test that is not in this repository failed on a wrong count in the test itself, with all its cases agreeing, and a runner marked two such tests as failed, against wrong counts in its own table (row H1.4); one CPU row, `integrations.unit` , failed for a cause in the test machine, and passed with that cause removed. 

106 

- Among the public tests: `driver.measure_rate` has failed on a busy machine, and `weight_sums` and `tools.confine` passed in part (above). 

**What this paper tells of other programs.** _Other programs_ are programs that are not the released ones: built from sources that differ from this repository’s in more than “Which programs ran” tells, and are not part of it, or from these sources with other options (test programs). What a run found on them does not carry over to the released programs; what a reading found carries over only as far as the differences were read (row HW-679). A fault found on them may be in the released programs too. 

The kinds of result below were got on other programs. Most bear the mark “not on the released programs” (<sup>_†_</sup> ) or, in Table 6, that of test programs (<sup>_‡_</sup> ); a run told with no mark, and named neither in the list, nor in “The public tests, in full”, nor among the calls of the Lean verifier, nor among the timed runs, is to be taken as one of them, but for the test of row RES-39, the runs of the middleware’s check and the four runs told under “Which programs ran”. 

A leak of weights found there is to be taken as holding of the released programs unless a page says otherwise. The calls of the Lean verifier, and any run of `tools/accepts/` , that this subsection tells outside its list are told apart: over which Lean files each ran stands at its sentence or at the head of the Lean verifier’s bullet. The judgement on weight hiding is given of the released programs: two of its five tests have run on them, the fifth at the reduced configuration alone (Section 7.5). It takes results on other programs as evidence, not as shown for the released programs; other code, changed or built otherwise, it does not reach. 

Many of the tests and readings that this paper reports were not made on the released programs, but on programs whose sources differ from the released ones; some, the tests with malicious provers among them, on programs that also held test-only code; and some on programs that differ from these in more code. The compiled programs differ too; the machine code that differs is not shown equivalent and has not been read. 

   - Runs with malicious or faulty programs, and searches for attacks. No malicious prover was built or run against the released programs. The 57 proof files from malicious provers of the list are files that test programs made beforehand; here `cp_verify` alone ran on them. This repository cannot build such a prover: the code is not in it. It does hold the functions that such a prover would use, which, by a reading, do nothing here (that their calls leave no instruction in the host code is not shown), and test code that can inject a fault in every build but the production one (row E29.1). 

      - No public test builds or runs a prover under a switch that makes it malicious; some run a prover that argues for something false, with code of the test’s own. `verifier.forest_twin_test` has no malicious prover. Eleven of the twelve stored false files of `examples.protocol_toy_false_files` were made by builds that this repository cannot make. 

   - Tests of weight hiding, but for the list’s: the second and the third of the five, on test programs built from other sources, which hold the code that injects a fault; the fourth, on programs of other sources that hold test-only code; the fifth’s two audits together at the default configuration’s size; and the audits by value that stand beside the five. What each found, and what went unnoticed: row LIM-63.2 and `docs/limits.md` . 

   - Other replays of altered, crossed or spliced honest files (what they found: row SO-15.7). 

   - Nine planning calls at reduced configurations, eight on parameter files made up for the call and one on the unchanged files of a small example of this repository: four were admitted, three refused, and two aborted at an internal check; none read a key, a copy’s file or a proof (row SO-6.3.2). The released programs refuse a file with `answers` other than 1, with `NO: parameter-range (policy)` (read in the code; run at 2). 

   - Every figure marked<sup>_†_</sup> . A<sup>_†_</sup> time, rate or memory peak is an estimate at most. 

   - Every run of the Lean verifier that this paper tells and this subsection does not, and the runs of `cp_verify` side by side with a stand-alone program built from the text of the second verifier (Limitation G.23, Second verifier text) (row L14), which this repository does not build as one. 

   - The vLLM integration’s runs, but for the check told under the timed runs (Section 7.4; `integrations/LIMITS.md` ). 

- Most readings of the code: a reading is of other sources, unless its line says otherwise. 

- Which refusals no test in this repository shows: `SUITE.md` . Which faults of hiding no test in this repository shows: 

- `docs/limits.md` . 

**What a reader can check.** `cp_verify` prints a build word ( `build=` ), the SHA-256 of a list of checksums of at least every file of this repository that it includes. A reader can compute this repository’s word from the sources, and build a `cp_verify` that prints it; the word stands in `examples/README.md` . It covers sources alone, tells neither the compiler nor the flags, and does not cover the rest of the provers’ sources, and a printed word shows only what a program says of itself. 

107 

_This repository’s sources and the programs of the timed runs._ The sources under the provers’ digest were compared, token by token: but for the comments, the names, the seven quoted texts and the wording of eight messages nothing differs. The built programs were compared in two steps, by way of a middle build, production and test, of sources that differ from this repository’s in the wording of those messages and in one line of comment alone. What either step leaves out, the tie leaves out; what is told below as not compared, not held or not covered holds of both. First step: the middle build was compared, by tools made for it, over objdump, readelf and cuobjdump, with the programs of the timed runs and with one test build of the sources of those runs, without being run. 

The programs that the test runs built for themselves were not in the comparison. `cp_verify` , `weight_sums` and `make_model` came out the same, byte for byte. The GPU code of `cp_commit` and `cp_prove` is the same, byte for byte: every function and its constant banks; every line of PTX too, but for the names that differ. In their host code every instruction of every function is the same, at the same place and of the same length, and each address in it points at the same thing. About half of the functions differ in raw bytes (by a count that is no part of the comparison; its program was read a second time and run anew, and it noticed a planted byte), in address fields alone, since names of other lengths move the data behind them. 

The data that the programs read (constants, tables, texts) is the same, but for the names that differ, where a program holds them (a kernel’s or a type’s name), and for texts that hold one: some printed texts begin with a routine’s name, and follow it, and one names a source file. The files of the two programs have other checksums. Not compared: the note that holds the build’s identifier, `.gnu.hash` , `.eh_frame_hdr` , and the relocation tables beyond their targets. Not held: up to 31 zeros of padding behind a string or table, the program and section headers, and the entry point. 

Of the files whose comments differ, four are compiled into nothing: two build scripts, a settings file and a header that nothing includes; of the seven quoted texts three stand in no program: two of `prover/common/hash_tree_ref.h` , and one that only the compiler prints. Code behind a switch that is off in both builds is not covered. Bytes were planted in the production `cp_commit` and `cp_prove` , and the comparison noticed each; their test programs were compared by the same tools with no plant. A second reader derived every figure of the comparison’s result anew, with a program of its own. Second step: a production and a test build of this repository’s sources were compared in the same way with the middle build, without being run. 

`cp_verify` , `weight_sums` and `make_model` came out the same, byte for byte. The GPU code of `cp_commit` and `cp_prove` is the same, and the container of the GPU code is the same, byte for byte. In their host code every instruction of every function is the same, and each address in it points at the same thing. About three functions in ten differ in raw bytes, in address fields alone, since texts of other lengths move the data behind them. Of the data that the programs read, the quoted texts of those messages alone differ: eight texts, of the seven messages that the programs print, one of which is two texts. The eighth message stands in a build script, in no program. 

As far as a reading of these sources goes, each is only printed, and none is a key, a label that goes into a hash or a part of anything hashed. One of them is the name under which a refusal is printed, and the test that looks for that name differs in the same way. Two test files of this repository differ too at this step: `prover/hash_commitment/tests/hf_params_test.cpp` , in texts that follow the messages, and `prover/tests/forest_lookup_test.cu` , in code; they lie outside the provers’ digest and are in no compared program. At this step, by a look beside the comparison, the entry point and the program headers are the same in all four pairs of programs ( `cp_commit` and `cp_prove` , production and test), but for two headers of each `cp_prove` , which differ by 32 bytes, the difference in length of its read-only data. 

Not held at this step: where inside the container of the GPU code a pointer of the host code points. Faults were planted at this step too, 69 in all, and each was noticed: 60 on the host and GPU sides of the production `cp_commit` and `cp_prove` , 8 at the check of the texts that differ (2 for each of the four programs; none of these turns a byte of code), and one at the count of raw bytes. At 2 controls with nothing planted the tools stayed silent. The host and GPU code of the test programs was compared by the same tools with no plant. 

The result and the tools were each read a second time, by readers looking for a fault, with nothing run: they found no way for the comparison to pass wrongly, beyond what is told here as not compared or not held. Neither comparison ran anything. No reader of either comparison is an outside party. Apart from the comparison, under one test seed the test programs of both sides of the first step gave the same files, byte for byte, at the reduced and at the default configuration, on synthetic weights, and so did both production builds of `cp_commit` , called with `-public-weights` , at the reduced configuration. 

At the default configuration there was no control and no second run of one side; at the reduced configuration each of six controls came out different, as a control must. Proofs of the production builds are drawn, so nothing is claimed of them by bytes. Equal bytes under one seed show that the two sides computed the same on those inputs; they are no test of weight hiding. The like run on both sides of the second step is the comparison with the middle build that “Which programs ran” tells: it was at the reduced configuration alone; two runs of one side gave equal bytes, and each of seven controls came out different. 

_Programs outside the comparison._ Both steps cover `cp_commit` and `cp_prove` , production and test programs, 

108 

`cp_verify` , `weight_sums` and `make_model` , and no other program: not the two helper programs that the audit tool ( `tools/audit/` ) builds for its own use, the programs of `blocks/` , the tests’ own programs or the parser of `lean-verify` . The audits with the provider’s secrets are runs of the audit tool. Its 14 scripts and the 2 sources of its helpers lie under neither digest. 

These 16 files, and the 118 sources that the helpers compile beside them (a list that a program drew from the `#include` lines, not a compiler’s), have the same checksums in this repository as in the tree that the rows ran in; from which tree the audits’ own copy of the tool came is not shown. One file that the tool imports from outside its folder, `docs/format/format_rules.py` , differs from that tree’s: by a comparison of tokens, in docstrings, a comment and two string literals, both headings that it prints into a file. 

_What a reader cannot check._ The files of the programs that ran and the runs’ records, the provers’ build records among them, are not in this repository, so a reader cannot check these results against them. 

_Two builds of the sources of the runs._ Two builds of those sources were compared file by file, without running them: each, by the account of those who built them, from a fresh copy with nothing kept from another build, on one machine, at one path, with the same compilers. `cp_verify` , `weight_sums` and `make_model` came out the same, byte for byte. `cp_commit` and `cp_prove` , production and test builds alike, differ only in the name of a temporary file of the compiler, which, by a reading, holds a number that differs from build to build and stays in the program’s list of symbols: a few bytes (6 and 11 in this pair; 5 and 12 in the test programs), at the same sizes. All their GPU code is the same, byte for byte. 

So the files of a second build of these two have another checksum. `cp_verify` , whose sources do not differ from this repository’s, was also built on a second machine (Ubuntu 24.04, g++ 13.3.0), with the repository’s own script: the same file, byte for byte. A third build of those sources, in the production build, in a fresh container on a GPU node, at the same path, with nvcc 13.2.86 installed at another place, was compared with the programs of the timed runs: `cp_verify` came out the same, byte for byte; in `cp_commit` and `cp_prove` no function differs, on the host or on the GPU, by a comparison function by function, but other parts of the files do. 

Each of the two differs from its counterpart in the same eleven sections of the host file, among them the container of the GPU code and the section that says which libraries to load and where to look for them, and in one note inside that container; none of these differences has been traced to its cause. That pair does not show what another build path does. Whether it was made on another machine than the programs of the timed runs, and which compilers made those, is not in the records that were compared. No test build was in that comparison. That comparison ran none of the three programs. It noticed a byte planted in a host function and bytes planted in the GPU images; the unpacking itself is not tested, and nothing was planted in the PTX. 

_How other sources differ._ How far the sources of other programs differ from these is told with each kind (row RES-39; the head of `docs/limits.md` ). 

_Test code in the production build._ That a production build holds no test code is checked by a search for strings, which does not read code (row E17.2); at least 7 of the 93 lines of `prover/tests/serving_build_words.txt` , which it reads, stand nowhere in the sources of `prover/` , `blocks/` and `verifier/` outside their tests, so their absence shows nothing; that the search ran is shown by a build’s records, not by a log. Some code that no run can reach is in these sources (rows E29 and E29.1). The readings behind that were not independent ( `docs/limits.md` ). 

_Libraries and counts._ Each run used the CUDA libraries, the driver and the C and C++ libraries of its machine, and says nothing of others. Counts of lines, files, kernels and functions in this paper may be off: no such count has been made on these sources, but for those that a row tells of the released programs (rows RES-39, RES-54, H39.1 to H39.4.1, HW-213 and HW-218). 

_Weight hiding and soundness._ What is proved, of what, and what is only tested is told on `guide/what-is-proved.md` and `guide/weight-hiding.md` . Nothing in this subsection is a proof about the compiled programs. 

### **H.2 Cost (Table 3)** 

Each row gathers three runs, each job’s three on three machines. Each machine that served on the public weights at the default configuration as fitted to them made its own commitment first, nine in all; three of the nine were named beforehand as the runs behind the figures. Worse figures of these quantities, not on the released programs, stand in the release’s table of limits (rows RES-13, RES-19, RES-24, RES-32 to RES-34 and RES-46). 

- The first row’s six executions of the prover, in the timed runs, gave between 110.07 and 112.04 tokens/second; two on one machine and commitment, one directly after the other, differed by at most 0.26 % (111.64 and 111.35 tokens/second). Three runs of the same kind on other programs, on other machines on another day, gave 105.3 ± 3.9 tokens/second (100.85 to 108.06); among their six executions two such differed by 6.3 %<sup>_†_</sup> of the larger (111.55 and 104.56 tokens/second). No cause is shown for either; the released programs did not run on those machines, nor the other programs on the timed runs’. A gap between rows that is no larger is uninformative. Apart from the clipped-gap row and its control, which ran in pairs on one machine and commitment, the rows 

109 

ran at other times or on other machines and are not comparable. These pairs show no cost of checking whole outputs; a cost smaller than the pairs’ own spread would not show. 

- Of the failed call and the failed service on 64 requests in the timed runs: the cause is not shown; the largest memory sample of each was 81,053 MiB of 81,559 MiB, which is a reading of the samples, not a line of the program’s. 

- Not on the released programs<sup>_†_</sup> six commitments to the same files, each under its own sampled seed, took 2,307.44 to 2,409.48 seconds. They are none of the nine commitments of the timed runs. 

- Not on the released programs<sup>_†_</sup> , in one run on the public weights, the largest single proof had 18,811,964 bytes, more than any proof of the timed runs, whose largest has 18,804,532 (18.81 MB, rounded up); no cause is shown (row RES-6). 

- Which proof the timed verifying calls took is not recorded. 

- For one commitment the provider keeps 2.15 GB (in the timed runs). While serving at 24 requests a batch, the prover’s own process held at most 257.1 GB of host memory on the public weight files and 265.5 GB on synthetic weights, and its container at most 269.7 GB on the public weight files and 278.1 to 278.3 GB on synthetic weights, file cache not counted; at 64 a batch on the public files the container reached 278.3 GB; not on the released programs<sup>_†_</sup> , on programs whose prover runs the verifier on every response before it hands it out (the released prover runs none), a session at 24 requests a batch on the public weights reached 290.7 GB in its container, file cache not counted, and 257.6 GB in the prover’s own process (largest sample of 3 runs); on synthetic weights 299.1 GB and 266.1 GB; with the middleware’s check running beside a serving prover the container reached 298.4 GB, against 278.2 GB without (one run on the released programs, none of the timed runs, on synthetic weights); in the three timed runs committing held at most 286.9 GB, and its container at most 299.6 GB; in one run of the example script, not on the released programs<sup>_†_</sup> , host memory in use reached 309.6 GB while the committer ran. The serving job held at most 4.69 GB of working disk (in the timed runs), the weight files and the commitment’s directories apart. How host memory behaves over a longer service is not known. 

- Two figures of Section 1.3, from the timed runs on the public weights at the default configuration as fitted to them ( _n_ = 3). Each commitment had one copy: its key file and its copy’s file had 6.24 MB together (6,235,160 to 6,237,760 bytes); by Definition 4.7 a further or renewed copy would have a file of its own; no run here made one. The rate with start-up counted is the 122,880 prefill tokens of five batches of 24 over the time from the program’s start to the last release. That time takes in a start-up of 74.8, 73.1 and 72.6 seconds, run by run, and a first batch longer than a steady gap between releases, so the rate depends on the number of batches: the longer the service, the nearer it comes to the steady rate; in the timed runs no service of more than five batches of 24, or six of any size, was run. In the timed runs the start-up at 64 requests a batch took 76.5 to 78.9 seconds. 

- By a reading of other sources, not by a run, each call on a proof also makes the full check of the key file and of the copy’s file, besides the proof’s own check, so 21.14 ± 0.03 seconds called with its option for one thread, on the prover’s machine (n = 3) takes both in, at every call. 

- Whether the CPU cores were reserved to a run is not shown: by each run’s own record the job ran on a set of 24 of its host’s 96 CPUs, under a hard limit of 26 CPUs, and every program printed that it was pinned to 24 CPUs of its GPU’s NUMA node; beyond that the host’s other seven GPUs were in use when the machine was assigned (reported), the records do not tell what else ran on the host. What other jobs on a host did during a run is recorded only in part ( `docs/limits-all-rows.tsv` ). 

“Masking on” is a run’s mode: by the runs’ own summaries the committer sampled the secret seed and no test option was set, so that, by readings of other sources (Section 1.3; Appendix H.1), values are masked. No run here or in Section 7.4 tested it (Section 7.5). The release’s `docs/figures.tsv` gives the conditions and further figures. 

### **H.3 The executable form on the prover’s proofs (Table 4)** 

For the 109 calls told at this section’s end and the second set of 41, by Theorem 5.5 each decision is that of _VZ_ on the proof as the parser of these calls cut it out of the files; for all other calls, that of the Lean verifier that the Lean files of that call define, on the proof as the parser of that call cut it out; in both cases under the assumption that Lean’s evaluator, compiler and runtime, a helper library with its C compiler, and the file that drives the evaluation are correct: the kernel checks proofs, not evaluations. Some runs evaluated the executable form directly, in a separate build of the executable form, tied to _VZ_ by name and statement, not by bytes; the `lean-verify` command was not used for them. That build’s Lean files are not shown to be the released ones; it alone made the decisions that Table 4 marks as by a separate build of the executable form, among them its side of the 43 comparisons and of the four matched alterations. No theorem covers any of these parsers. 

110 

- This evidence is thin in kind: at exact match, the only policy of Theorem 5.1, the 363 proofs made by one prover built from other sources<sup>_†_</sup> (as reported: those handed over, by a rule fixed before the runs, of the 723 it made) are 360 with one output token: 96 requests of one shape from one list, each proved under the same three commitments to, as reported, one set of synthetic weights, 24 of them twice under each, and three of the first stored instance; none of the 363 was blind, the C++ verifier having accepted each first in those runs, as reported (said of whole runs, no file checked by checksum); and that this prover made them is as reported: nothing in a file names the prover. 

- The eighteen proofs at exact match by other provers (12 by one set of such programs, as reported; 6 by others) are of five instances: seven of one; nine of three others, each a prompt of 1,024 tokens with one output token, under three commitments to, as reported, the same weights; two beside the malicious proofs of the true claim below, one of them of one of those three instances. 

   - Under the clipped gap, where an acceptance states less, the four by other provers (1 by the set that made the 12, as reported), and one by the prover of the 363<sup>_†_</sup> , as reported, are of one further instance, that one at a new commitment and not blind; under that policy `lean-verify` has met honest proofs alone, and a separate build of the executable form two proofs of false claims, both rejected. Four proofs by other programs come from test programs, one of them a malicious program set, as reported, to alter no value. 

   - Not counted here are calls of `lean-verify` that we stopped on purpose before a decision (nearly all tests of interruption; all on the first stored example’s proof, which was accepted whenever a call on the same files ran to its end) and one honest proof that, as reported, a separate build of the executable form alone accepted, at exact match. 

- All 26 come, as reported, from one malicious test program, each set in advance to one false claim, none from a prover built to search for a false proof that passes. The 24 share one window and commitment, and every rejection occurs at the step that reads the proof file: no evaluation shows the policy’s own test rejecting anything. 

- Of 22 further proof files that a malicious test prover made for the true claim (the default configuration’s size, synthetic weights, two requests under the commitment of the 24; neither the prover nor the files ship, so a reader cannot repeat this), neither verifier accepted any: the released C++ verifier rejected all 22; `lean-verify` , with sources and scripts as in Section 7.1, rejected 20, at the same step (within that step, by one evaluation of a test that does not ship and two readings of the order of checks, no theorem, at a later check than the 24’s; the two sets differ in other ways too, so it is not known whether that comes from the kind of lie), and gave no verdict at 2, whose files its parser refused, which is no rejection. 

- A second set of 41 honest proofs, from the timed runs’ own files on synthetic weights, has been put to `lean-verify` , in calls counted apart from the 109: each of three runs’ files holds proofs numbered 000 to 119 and one more; taken were that one and 000, 009 and so on, to 108 in two runs (13 each) and to 099 in the third (12): 3 + 13 + 13 + 12 = 41. It accepted each, at exact match, one call each. These are other files than the 41 of the 363 at exact match that are among the 109 calls, and than the 41 that were put both to `lean-verify` and to a C++ verifier (told below this list); of neither is one a proof of the timed runs. 

- That evidence is thin too: synthetic weights; not blind; one machine; one size of model, the default configuration’s; two shapes (38 with a prompt of 1,024 tokens and one output token, 3 with a prompt of 24 tokens and 1,001 output tokens); the count and that the files are those runs’ are as reported; and, like the 109 calls, the calls were made over the released Lean files. The command had accepted each of them before, over Lean files that differ from the released ones. No file is among the 47 honest proofs of the 109 calls; 7 of the 14 requests are. 

From the release a reader can repeat two acceptances and two rejections, though untried with Lean and its packages fetched as the release instructs ( `tools/lean-verify/README.md` ); of the other evaluations, those through `lean-verify` were driven by the form of `lean-verify` that each run had, with scripts as in Section 7.1, on inputs that are not in the release; the rest were driven by files that are not in the release and have not been reviewed ( `lean/THEOREM.md` ). 

Of the files put to `lean-verify` and to a C++ verifier, two groups. (1) The form of the command marked<sup>_†_</sup> in Section 7.1 judged 41 files over Lean sources that differ only in comments, in 25 files, from those of the three calls that were timed (Section 7.3); it judged 9 of the 41 again over the sources of these three calls, with the same decisions, and the nine proofs of three instances over these sources only. (2) The command of other calls, over Lean files that differ in code from the released ones, compiled beforehand (that the compiled files were built from those sources is not shown), judged those 41 and the file that its parser refuses, the 22 malicious proofs of the true claim with their two honest proofs and two controls (also judged in runs marked<sup>_†_</sup> ), and the 364 with three controls. The command `lean-verify` was called 109 times, at the default configuration’s size and on synthetic weights, on files that other runs had all judged: it accepted all 47 honest proofs put to it (45 at exact match, 2 under a clipped-gap policy), 42 of them of these 364, made by that prover<sup>_†_</sup> , and the others the two stored proofs and proofs by test programs, as reported; of the 57 proof files from malicious provers of Section 7.6 it accepted none (54 rejected; at 3 its parser refused the file, which is no verdict); and it rejected 5 controls ( `lean/THEOREM.md` , “What has been run”). These calls were made over 

111 

the released Lean files (Remark B.17: one file was added after these calls). The command and the parser were held by checksums before and after the calls; every file of the command and of the parser outside `lean/` that the traced calls beside them opened has the released file’s checksum (none of the run’s own calls was traced: `lean/GAPS.md` , 14, “Which programs (k)”). At those of the 109 calls that reached Lean the command loaded this library’s compiled Lean files. These were made beforehand; that they are made from the released Lean files is Lake’s finding at the build in several calls that Remark B.17 tells, not the run’s. Their checksums, taken before and after the 109 calls, are the same. Those of the packages were not held at the calls. The command evaluates the Lean verifier natively compiled in part (Appendix H.1); that part was not made by that build. The command compiled it itself, in a call of its own before the 109 (but for three objects of a package, which lay in the tree before that call). The program `cp_verify` was not called in the reference run (the 109 calls of this section). Apart from the 109 and the second set of 41, no call of this section was made over the Lean files of the 109. 

### **H.4 Runs by policy (Table 5)** 

Under neither policy has the prover run as a service beside the engine: it was started after the outputs were served. Serving a trained model with an ordinary engine and proving its outputs in one service has never run end to end; ordinary serving and proving have run end to end only in scripted demonstrations at a reduced configuration on a synthetic model. The release’s `integrations/LIMITS.md` and `docs/results.md` report further runs. 

### **H.5 Tests of weight hiding (Table 6)** 

**Definition H.1 (Weight hiding, judgement, honest run).** _Weight hiding_ is the goal that clients and the public learn nothing about the weights beyond a declared list. The _judgement_ of Section 7.5 is a grade that we give to a goal of the programs, from the design, what is proved in Lean for the protocol, tests and code review, on a scale of four: proved; likely holds; cannot say; does not hold. _Likely holds_ means that in honest runs, with the provider’s operating conditions and the judgement’s own conditions kept, no way to a leak beyond the list is known, while nothing is proved for the programs. A run is _honest_ if nobody has changed the provider’s programs and no injected, compiler or hardware fault has occurred; a bug in the published sources is not excluded. 

Outside the judgement of Section 7.5 are a real model’s weights, and what a fit to them writes (a parameter file, a file of scales): only one of that judgement’s five tests has used any, once on the released programs and three times on other programs, and it does not ask what they reveal. 

On the released programs two of the five tests have run (the first at the reduced and at the default configuration, there once on synthetic weights and once, as fitted to them, on the public weights; the fifth once, at the reduced one), and both audits once more under a clipped-gap policy (below): every published out-of-domain answer that the first recomputed was the row’s sum plus its own mask, and the shipped audits accepted the honest files. A reading of the committer’s and the prover’s sources, once and fast, with nothing built or run, found no leak beyond the declared list (Remark G.10, What a client of the programs learns); how many of the files so read differ in the released sources is told below. Recovering weights from published files has barely been tried, and not on the released programs. Every “stopped” was observed in a test program or, in the fourth test, in a program that held test-only code: no self-check on randomness (a program’s recomputation of masked values) has been observed to report a failure in the released programs, and it is assumed that the test programs and the released programs apply the masks in the same way. 

- The first and the fifth ran on the released programs, the first also at the default configuration. The second, the third and the fourth ran on test programs or on other programs alone (Section 7.5), the third and the fourth on more than one set of such programs, with the same outcomes, the bad ones too. For the released programs that judgement has less evidence than there is for the programs on which most of its tests ran (Appendix H.1). A provider’s own build from the released sources comes under it only as a correct build: every build tested here is our own. The second, the third and the fourth each have one fixed set of losses; a repeat of a test is no new witness. The test programs of the second and the third do not ship. 

What was read is the sources of programs other than the released ones: those of the committer and the prover were read whole, once and fast, in a text of 384 files, without the C++ verifier’s own files, which both programs of that text include. The released sources differ in code in 68 of the 384, the file with the routine that feeds BLAKE2b its blocks and finishes the hash among them, and in 136 more in comments, names and the wording of messages alone; two more of the 384 are not in the released sources; in the 68 the differences in code were read for hiding, by difference alone, apart from renamed names and reworded messages, and the files were not read whole again; that routine’s released code has been read for hiding twice, by readers working apart. For many single faults the one net that the whole reading found is a verifier’s run on a response before anyone else holds it; the prover makes no such run, and that reading did not take in the serving scripts (Limitation G.16). 

112 

- At the default configuration’s size two of the five tests have run, the first and the fifth; of the fifth, both audits there on other programs alone, and the audit of a commitment also in the timed runs. One test of hiding has run on the public weights: the first, once on the released programs and three times on other programs<sup>_†_</sup> , at the default configuration as fitted to them. In the run on the released programs, on a commitment that the committer made under a sampled secret seed, all 4,096 OOD answers of its first table and its one copy (Definition 4.6) were as prescribed, no pad occurred twice or was 0, and the separate program found all 8 faults that it planted for itself as controls. The 8 faults show that the program’s count on the pads can ring, not that each of its three counts (equal pads; equal single field values; pads that are 0) could have: no fault tells them apart, and that the program counts all three is read, not tested. This does not show that the masks hide, or what becomes of a response that does not verify (the run was honest); the test reads under 2 % of the bytes of the key file and of the copy’s file, and of the five tests this one alone has run on such weights. 

   - At that size the shipped audit accepted, in the timed runs, four honest commitments of the released programs’ to synthetic weights (with the provider’s secrets the audit tools accept commitments to synthetic weights alone), one of them under a clipped-gap policy, which is no negative test. On other programs both audits ran at that size too, on synthetic weights: in one run they accepted the honest files and, in trial calls, rejected four of four false secrets. On the released programs the fifth has run once under exact match, at the reduced configuration, and the first three times at reduced configurations, on synthetic weights (1,536 answers of two commitments; 512 of one; 512 of one to a small synthetic model in files of the public weights’ form, at a configuration of its own, whose parameter file that judgement does not reach), and twice at the default configuration: on a commitment that their committer made to the public weights, at that configuration as fitted to them, and on one to synthetic weights. On the released programs, of the audits with the provider’s secrets, the audit of a commitment alone ran at the default configuration’s size, in the timed runs (Appendix H.1). On the released programs, beside the first test, in each of its runs at the default configuration (on the public weights: as fitted to them), the audit tool’s program, run without the provider’s secrets over copies of what is published alone (the key file, the copy’s file and four proofs), reported nothing on them and ended with an error at each of three copies with one bit changed; a changed bit is not a lost mask, and this is not a test of hiding. At reduced configurations it read four honest sets of such files and reported nothing; no altered copy was put to it there. Here and in Table 6 “as prescribed” means by the rules that the test carries; these were typed mainly from the release’s pages, in a text that is not shown to be the released one. 

- None of the five tests shows what an OOD answer reveals or an unnoticed loss exposes; far more samples of the kind of the nine are covered by no self-check; the audits cover no served proof and do not remove the risk of a lost mask. 

- That judgement reaches a clipped-gap policy (in a parameter file that differs from the default configuration’s, or from that of a reduced configuration at which it is given, in that policy’s three numbers alone, and at the reduced configuration at which both audits ran under such a policy: In sum, below), on less evidence: audits apart, no test of hiding ran under such a policy. Both audits ran under one: once on the released programs, at the reduced configuration, on synthetic weights, on two proofs that each cover 32 output tokens. Every random value that they compare by value (in the batch’s audit just under a fifth of the bytes) was what the secret seed gives, they rejected three false secrets, and the C++ verifier accepted both proofs as within the policy’s limits. By the audits’ own table such a proof has no kind of record that a proof at exact match lacks, so this is the exact-match check passed on such proofs, not a check of anything that only such a policy has. It shows that the masks are there as derived, and it is not a search for a leak ( `docs/limits-all-rows.tsv` ). The audit of a commitment ran once more on the released programs, at the same reduced configuration and under such a policy, in the making of a small example, and accepted the honest commitment. The reach rests on the design (the released programs run one scored output head under every policy: Limitation G.1), on one reading aimed at that policy’s path, made in the code of other programs and not repeated in the released code, and on these audits. Lean’s theorem on zero knowledge is for exact match: under a clipped-gap policy no Lean theorem covers what a whole proof reveals (what the answers themselves tell: Limitation G.21). Also untested: batches that overlap in time, the prover’s default; several clients’ proofs together; out-of-range prompts. 

- One finding, in which no leak was shown, as far as its account goes: the number of events in a reply streamed by vLLM ( `integrations/LIMITS.md` ). 

In sum: that judgement is given at the default configuration and at the reduced configurations of its tests, but for that of the small synthetic model in files of the public weights’ form (above). It reaches, on less evidence, a configuration that differs from one of these in a clipped-gap policy’s three numbers alone, and the reduced configuration at which both audits ran under such a policy, which also has a scale file of its own. It covers no other configuration, so none fitted to a real model’s weights, no weight file with a tensor of 8 or 16 bits (Limitation G.17, Tensors of eight or sixteen bits), and not the mode for a public seed, which hides nothing (Limitation G.22, Unmarked public-seed mode). A provider that serves for long enough, or on a second GPU, needs a further commitment to the same weights 

113 

(Limitation G.12, Further commitments). The judgement of Section 7.5 takes that in, on thin ground, whether the commitments follow one another or serve at once on several GPUs, on five conditions. 

1. Each commitment comes from its own call of the release’s committing program, on a machine that was not cloned or restored, and samples its own secret seed. 

2. Each commitment takes nothing from another. 

3. The configuration files stay the same. 

4. Each commitment stays within its limit (Limitation 8.4, Opening budget). 

5. Nothing hidden may choose which commitment serves a batch. One exception: the commitment may be picked by load or queue length. That can depend on hidden values, and the pick gives away something of time and load, which the release declares as visible. 

Each commitment also meets that judgement’s other conditions. So two commitments to the same weights under different configuration files, for instance under two acceptance policies, may each be inside alone, but not together. 

By design, one commitment already has several tables of the same weights (Definition 4.6), each with its own randomness and OOD answers. Brief readings of parts of the host code, all of them of the code of other programs, found no way in which two commitments tell more than one; none of them covered the GPU kernels, except the two kernels that add the pads to a table and to its columns. Not on the released programs, a count over two small commitments found no derived random value twice, a check for gross faults only. 

No test has searched two commitments together for leaks, and Lean has no statement on hiding for more than one commitment. So the thin ground is the design, these readings and that count. 

A further commitment resets only the count of opened columns: each proved request’s share of any bound on hiding, and what the served-or-failed answers give, run on across commitments. That judgement on weight hiding does not cover a commitment that is in doubt: one of which a count, a key or a directory may not be what the provider’s record says (the list: `docs/operators-duties.md` ). The one that replaces it may be inside alone, on conditions of its own, but of these weights as a whole that judgement says nothing. It does not cover what the one before may already have shown. 

The release’s `docs/what-is-not-hidden.md` gives the declared list (Definition H.1), and `docs/limits.md` the provider’s operating conditions (Limitation G.16, Operating conditions), that judgement’s own conditions and its grounds; `docs/how-hiding-is-checked.md` and `docs/limits.md` give further tests and bad results. 

### **H.6 Negative tests (Table 7)** 

No malicious program has been built from the released sources or run against the released prover and committer. The malicious provers and committers of Table 7 are test programs, not the released ones, some of them edited for a test; they do not ship, so the rows that use them cannot be repeated from the release. No reported test uses a searching prover, or any malicious prover on real weights. Most tests are weak, at reduced sizes and on synthetic weights: evidence, not a guarantee, resting on no Lean theorem about the programs. 

### **H.7 The integer model** 

On 544 of 544 windows the range checks of the integer-model package (not on the released programs) found none failing at the public weights. Besides the runs of Section 7.2 and Section 7.4, this count outside Lean is the only evidence reported here that instances of a trained model can be served (Remark D.5, No real weights shown). 

114 

