commit:
	/usr/bin/time -l target/release/cp_commit --weights data/weights.bin --out data/commit.bin

prove:
	/usr/bin/time -l target/release/cp_prove --spec data/olmoe_small.json --weights data/weights.bin --trace data/trace.json --prompt data/prompt.json --output data/output.json --fill data/fill.json --out data/proof.bin

verify:
	/usr/bin/time -l target/release/cp_verify --spec data/olmoe_small.json --commit data/commit.bin --proof data/proof.bin

full-run:
	# 1. rebuild (cp_commit's CLI changed)
	cargo build --release --bin cp_commit --bin cp_prove --bin cp_verify

	# 2. spec + weights (only if config_small.json or the python side changed)
	.venv/bin/python python/gen_spec.py --config olmoe/config_small.json --out data/olmoe_small.json
	.venv/bin/python python/load_weights.py --dir olmoe --spec data/olmoe_small.json --out data/weights.bin

	# 3. trace + request files (writes trace/prompt/output/fill in one shot)
	.venv/bin/python python/run_real.py --spec data/olmoe_small.json --weights data/weights.bin --prompt "1 2 3" --window 64 --out data/trace.json

	# 4. commit -> prove -> verify
	make commit
	make prove
	make verify
