SPEC   := data/olmoe_small.json
CONFIG := olmoe/config_small.json
WEIGHTS ?= data/weights.bin
PROMPT ?= "1 2 3"
WINDOW ?= 64

commit:
	/usr/bin/time -l target/release/cp_commit --weights $(WEIGHTS) --out data/commit.bin

# Per-role requant calibration: writes data/shifts.json and the spec.
calibrate:
	.venv/bin/python python/calibrate.py --config $(CONFIG) --weights $(WEIGHTS) \
		--prompt $(PROMPT) --window $(WINDOW) \
		--out-spec $(SPEC) --out-shifts data/shifts.json

trace:
	.venv/bin/python python/run_real.py --spec $(SPEC) --weights $(WEIGHTS) \
		--prompt $(PROMPT) --window $(WINDOW) --out data/trace.json

prove:
	/usr/bin/time -l target/release/cp_prove --spec $(SPEC) --weights $(WEIGHTS) \
		--trace data/trace.json --prompt data/prompt.json --output data/output.json \
		--fill data/fill.json --out data/proof.bin

verify:
	/usr/bin/time -l target/release/cp_verify --spec $(SPEC) --weights $(WEIGHTS) \
		--commit data/commit.bin --proof data/proof.bin

# Same, with the field-width guard and the GPU kernels on.
prove-check:
	FLASHPROVE_GPU=1 FLASHPROVE_CHECK_BITS=31 $(MAKE) prove

full-run:
	cargo build --release --bin cp_commit --bin cp_prove --bin cp_verify
	.venv/bin/python python/load_weights.py --dir olmoe --spec $(SPEC) --out $(WEIGHTS)
	$(MAKE) calibrate
	$(MAKE) trace
	$(MAKE) commit
	$(MAKE) prove
	$(MAKE) verify
