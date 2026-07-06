.PHONY: ci lint validate smoke eval status privacy shell-p2 reasoner-p3 v1-status p4-memory

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

swarm-p5:
	@echo "==> P5 swarm: operational + distillation smoke"
	$(PYTHON) sia-lab/swarm/swarm.py
	$(PYTHON) sia-lab/swarm/distill.py
	$(PYTHON) sia-lab/swarm/p5_eval.py

p4-memory:
	@echo "==> P4 memory: smoke test"
	$(PYTHON) sia-lab/memory/tokencake.py
	$(PYTHON) sia-lab/memory/episodic.py
	$(PYTHON) sia-lab/memory/graphrag.py
	$(PYTHON) sia-lab/shell/smoke_p4.py

v1-status:
	@echo "==> V1 remaining tasks status"
	$(PYTHON) sia-lab/posttrain/generate_dataset.py --n 100
	$(PYTHON) sia-lab/memory/tokencake.py
	$(PYTHON) sia-lab/memory/episodic.py
	$(PYTHON) sia-lab/memory/graphrag.py
	$(PYTHON) sia-lab/swarm/swarm.py
	$(PYTHON) sia-lab/safety/audit.py
	$(PYTHON) sia-lab/infra/ota.py
	$(PYTHON) sia-lab/safety/privacy.py

eval:
	@echo "==> eval: memory + benchmark harness"
	$(PYTHON) sia-lab/memory/tokencake.py
	$(PYTHON) sia-lab/memory/episodic.py
	$(PYTHON) sia-lab/memory/graphrag.py
	$(PYTHON) sia-lab/eval/multi_hop.py
	$(PYTHON) sia-lab/eval/governor.py
