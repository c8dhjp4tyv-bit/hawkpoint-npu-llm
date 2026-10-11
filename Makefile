# Makefile for HawkPoint NPU LLM

PYTHON ?= python3
ROOT_DIR := $(shell pwd)
MODEL_DIR ?= $(ROOT_DIR)/npu_llm/models/SmolLM2-135M-Instruct-xdna1-w8a16

.PHONY: help test doctor inspect benchmark eval chat api openwebui docker-build docker-up docker-down lint clean

help:
	@echo "HawkPoint NPU LLM - Development and Operational Targets"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Runtime & Services:"
	@echo "  api          Launch OpenAI-compatible HTTP API server (localhost:8000)"
	@echo "  openwebui    Launch API server and Open WebUI frontend (localhost:3000)"
	@echo "  chat         Launch interactive terminal chat client"
	@echo ""
	@echo "Diagnostics & Tooling:"
	@echo "  doctor       Run preflight host environment diagnostics (XDNA, XRT, drivers)"
	@echo "  inspect      Inspect and validate converted model weights and architecture"
	@echo "  benchmark    Run automated API latency and TTFT benchmark suite"
	@echo "  eval         Run automated model accuracy and perplexity evaluation"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test         Run all automated unit and integration test suites"
	@echo "  lint         Validate syntax and bytecode compilation"
	@echo "  clean        Remove compiled Python cache files"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  docker-build Build the container image"
	@echo "  docker-up    Start full compose stack (API + Open WebUI)"
	@echo "  docker-down  Stop compose stack"

test:
	$(PYTHON) npu_llm/tests/test_inspect_model.py
	$(PYTHON) tests/test_doctor.py
	$(PYTHON) npu_llm/tests/test_chat_cli.py
	$(PYTHON) tests/test_benchmark_api.py
	$(PYTHON) tests/test_examples.py
	$(PYTHON) npu_llm/tests/test_visualizer.py
	$(PYTHON) tests/offline_benchmark_harness.py
	$(PYTHON) tests/test_offline_benchmark_harness.py
	$(PYTHON) tests/test_launcher.py
	$(PYTHON) npu_llm/tests/test_eval_model.py
	$(PYTHON) tests/test_release_gates.py
	$(PYTHON) tests/test_api_server.py
	$(PYTHON) tests/test_review_regressions.py
	@echo "All tests passed successfully!"

doctor:
	$(PYTHON) scripts/doctor.py

inspect:
	@if [ -d "$(MODEL_DIR)" ]; then \
		$(PYTHON) npu_llm/tools/inspect_model.py "$(MODEL_DIR)"; \
	else \
		echo "Model not found. Specify path: make inspect MODEL_DIR=/path/to/model"; exit 1; \
	fi

benchmark:
	$(PYTHON) scripts/benchmark_api.py --requests 10 --concurrency 2

eval:
	$(PYTHON) npu_llm/tools/eval_model.py --self-test

chat:
	$(PYTHON) npu_llm/chat.py

api:
	$(PYTHON) launcher.py api

openwebui:
	$(PYTHON) launcher.py openwebui

docker-build:
	docker build -t hawkpoint-npu-api:latest .

docker-up:
	docker compose --profile full up -d

docker-down:
	docker compose down

lint:
	$(PYTHON) -m compileall -q .
	@echo "Compilation check passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
