commit:
	/usr/bin/time -l target/release/cp_commit --weights data/weights.bin --out data/commit.bin

prove:
	/usr/bin/time -l target/release/cp_prove --spec data/olmoe_small.json --weights data/weights.bin --trace data/trace.json --prompt data/prompt.json --output data/output.json --fill data/fill.json --out data/proof.bin

verify:
	/usr/bin/time -l target/release/cp_verify --spec data/olmoe_small.json --commit data/commit.bin --proof data/proof.bin
