# Makefile for HawkPoint NPU LLM

PYTHON ?= python3
ROOT_DIR := $(shell pwd)
MODELS_DIR ?= $(ROOT_DIR)/npu_llm/models

.PHONY: help test doctor inspect benchmark chat api openwebui docker-build docker-up docker-down lint clean

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
	$(PYTHON) tests/test_release_gates.py
	$(PYTHON) tests/test_api_server.py
	@echo "All tests passed successfully!"

doctor:
	$(PYTHON) scripts/doctor.py

inspect:
	@if [ -d "$(MODELS_DIR)/SmolLM2-135M-Instruct-xdna1-w8a16" ]; then \
		$(PYTHON) npu_llm/tools/inspect_model.py "$(MODELS_DIR)/SmolLM2-135M-Instruct-xdna1-w8a16"; \
	else \
		echo "Default model not found. Specify path: make inspect MODELS_DIR=/path/to/model"; \
	fi

benchmark:
	$(PYTHON) scripts/benchmark_api.py --requests 10 --concurrency 2

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
