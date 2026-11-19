prepare-small-model:
	.venv/bin/python python/load_weights.py --dir olmoe --spec data/olmoe_small.json --out data/weights.bin
	.venv/bin/python python/run_real.py --spec data/olmoe_small.json --weights data/weights.bin --prompt "1 2 3" --window 64 --out data/trace.json

small-model:
	# 2. commit (27 segments)
	target/release/cp_commit --spec data/olmoe_small.json --weights data/weights.bin --config data/config.json --seed0 1 --seed1 2 --out data/commit.bin

	# 3. prove
	target/release/cp_prove --trace data/trace.json --prompt data/prompt.json --output data/output.json --fill data/fill.json --out data/proof.json

	# 4. verify (should print "accept")
	target/release/cp_verify --config data/config.json --commit data/commit.bin --proof data/proof.json --prompt data/prompt.json --output data/output.json --spec data/olmoe_small.json
