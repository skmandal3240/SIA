.PHONY: ci lint validate smoke eval status privacy shell-p2 reasoner-p3

PYTHON := python3

FILES := $(shell find sia-lab -name '*.py' -type f)

SHELL_TESTS := sia-lab/shell/tests/test_shell.py

ci: lint validate smoke eval status

lint:
	@echo "==> lint: checking Python scripts"
	$(PYTHON) -m py_compile $(FILES)
	$(PYTHON) -m ruff check $(FILES) || echo "ruff not installed; skipping style check"
	$(PYTHON) $(SHELL_TESTS)

validate:
	@echo "==> validate: dataset + manifest schemas"
	$(PYTHON) -c "import json; json.load(open('sia-lab/posttrain/device_actions.json'))"
	$(PYTHON) sia-lab/infra/quantize.py --bits 8
	$(PYTHON) -c "import json; json.load(open('sia-lab/infra/outputs/quantized/manifest.json'))"

smoke:
	@echo "==> smoke: dry-run all pipelines"
	$(PYTHON) sia-lab/posttrain/sft.py
	$(PYTHON) sia-lab/infra/quantize.py --bits 8
	$(PYTHON) sia-lab/infra/benchmark.py
	$(PYTHON) sia-lab/reasoner/reasoner.py
	$(PYTHON) sia-lab/reasoner/tiny_overfit.py

shell-p2:
	@echo "==> P2 shell: compile + smoke"
	$(PYTHON) -m py_compile sia-lab/shell/*.py
	$(PYTHON) sia-lab/shell/smoke_p2.py

reasoner-p3:
	@echo "==> P3 reasoner: self-check + overfit + eval"
	$(PYTHON) sia-lab/reasoner/reasoner.py
	$(PYTHON) sia-lab/reasoner/tiny_overfit.py
	$(PYTHON) sia-lab/reasoner/p3_eval.py

eval:
	@echo "==> eval: memory + benchmark harness"
	$(PYTHON) sia-lab/memory/tokencake.py
	$(PYTHON) sia-lab/memory/episodic.py
	$(PYTHON) sia-lab/memory/graphrag.py
	$(PYTHON) sia-lab/eval/multi_hop.py
	$(PYTHON) sia-lab/eval/governor.py

status:
	@echo "==> status: artifact inventory"
	@for f in sia-lab/posttrain/data/device_actions_train.json sia-lab/posttrain/data/device_actions_val.json sia-lab/infra/outputs/quantized/manifest.json PROJECT/models/*.gguf sia-lab/posttrain/outputs/p1_merged/sia-p1.gguf; do \
		if [ -f $$f ]; then echo "  present : $$f"; else echo "  missing : $$f"; fi; \
	done

privacy:
	@echo "==> privacy: network egress test"
	$(PYTHON) sia-lab/safety/privacy.py
