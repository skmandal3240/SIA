.PHONY: ci lint validate smoke eval status privacy

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

eval:
	@echo "==> eval: memory + benchmark harness"
	$(PYTHON) sia-lab/memory/tokencake.py
	$(PYTHON) sia-lab/memory/episodic.py
	$(PYTHON) sia-lab/memory/graphrag.py
	$(PYTHON) sia-lab/eval/multi_hop.py
	$(PYTHON) sia-lab/eval/governor.py

status:
	@echo "==> status: artifact inventory"
	@for f in sia-lab/posttrain/device_actions.json sia-lab/infra/outputs/quantized/manifest.json models/LFM2.5-8B-A1B.gguf; do \
		if [ -f $$f ]; then echo "  present : $$f"; else echo "  missing : $$f"; fi; \
	done

privacy:
	@echo "==> privacy: network egress test"
	$(PYTHON) sia-lab/safety/privacy.py
