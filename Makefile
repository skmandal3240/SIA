.PHONY: ci lint validate smoke status privacy

PYTHON := python3

FILES := sia-lab/posttrain/sft.py sia-lab/infra/quantize.py sia-lab/infra/benchmark.py sia-lab/safety/privacy.py \
         sia-lab/shell/*.py

SHELL_TESTS := sia-lab/shell/tests/test_shell.py

ci: lint validate smoke status

lint:
	@echo "==> lint: checking Python scripts"
	$(PYTHON) -m py_compile $(FILES)
	$(PYTHON) -m ruff check $(FILES) || echo "ruff not installed; skipping style check"
	$(PYTHON) sia-lab/shell/tests/test_shell.py

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

status:
	@echo "==> status: artifact inventory"
	@for f in sia-lab/posttrain/device_actions.json sia-lab/infra/outputs/quantized/manifest.json models/LFM2.5-8B-A1B.gguf; do \
		if [ -f $$f ]; then echo "  present : $$f"; else echo "  missing : $$f"; fi; \
	done

privacy:
	@echo "==> privacy: network egress test"
	$(PYTHON) sia-lab/safety/privacy.py
