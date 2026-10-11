#!/usr/bin/env python3
"""Exercise review fixes at their API, transport, and launcher boundaries."""

from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import MagicMock, mock_open, patch
from urllib.error import HTTPError
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import launcher
from examples.api_client_example import complete_text, list_models, stream_chat
from npu_llm.chat import list_api_models, stream_chat_api
from npu_llm.http_client import open_api_request, validate_api_url
from npu_llm.tools import eval_model
from npu_llm.tools.inspect_model import verify_model_files
from scripts import doctor
from scripts.benchmark_api import execute_single_request, run_benchmark
from test_api_server import API_KEY, OptionsDecoder, fetch, start_server, stop_server


def test_evaluator_against_real_api():
    server, thread, base = start_server({"smollm2-135m-xdna1": OptionsDecoder()})
    try:
        item = {"id": "live", "category": "integration", "prompt": "Hello", "expected": "ok", "max_tokens": 5}
        report = eval_model.run_evaluation(base, API_KEY, "smollm2-135m-xdna1", [item])
        assert report.accuracy == 1.0
        assert report.mean_cross_entropy == 0.25
        assert report.total_prompt_tokens == 7
        assert report.total_completion_tokens == 2
        assert report.mean_per_token_latency_ms > 0
        assert report.items[0].scored_tokens == 2
        assert report.items[0].negative_log_likelihood == 0.5
    finally:
        stop_server(server, thread)


def test_all_multi_choice_usage_and_metrics():
    server, thread, base = start_server({"smollm2-135m-xdna1": OptionsDecoder()})
    try:
        for endpoint in ["/v1/chat/completions", "/v1/completions"]:
            for stream in [False, True]:
                payload = {"model": "smollm2-135m-xdna1", "n": 2, "max_tokens": 5, "stream": stream}
                payload.update({"prompt": "Hello"} if endpoint == "/v1/completions" else {"messages": [{"role": "user", "content": "Hello"}]})
                if stream:
                    payload["stream_options"] = {"include_usage": True}
                status, body, _ = fetch(base + endpoint, payload)
                assert status == 200
                data = ([json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {") and '"usage"' in line][-1] if stream else json.loads(body))
                usage = data["usage"]
                assert usage["prompt_tokens"] == 14, (endpoint, stream, usage)
                assert usage["completion_tokens"] == 4
                assert usage["total_tokens"] == 18
                assert usage["prompt_tokens_details"]["cached_tokens"] == 10
        status, body, _ = fetch(base + "/metrics")
        assert status == 200
        assert 'hawkpoint_prompt_tokens_total{model="smollm2-135m-xdna1"} 56' in body
    finally:
        stop_server(server, thread)


def test_missing_loss_aggregation_and_gate():
    assert eval_model.compute_cross_entropy_and_perplexity([None, True, float("nan"), 0.5]) == (None, None)
    item = eval_model.EvalItemResult("a", "same", "Q", "A", "A", True, [], [], None, None, 1, 1)
    valid = replace(item, item_id="b", cross_entropy=2.0, perplexity=7.389, scored_tokens=1, negative_log_likelihood=2.0)
    with patch.object(eval_model, "evaluate_single_item", side_effect=[item, valid]):
        mixed = eval_model.run_evaluation("unused", None, "model", [{}, {}])
    assert mixed.mean_cross_entropy == 2.0
    assert mixed.categories["same"]["cross_entropy"] == 2.0
    with patch.object(eval_model, "evaluate_single_item", return_value=item):
        missing = eval_model.run_evaluation("unused", None, "model", [{}])
    assert missing.perplexity is None
    assert missing.categories["same"]["perplexity"] is None
    assert "N/A" in eval_model.format_text_report(missing)
    assert "N/A" in eval_model.format_markdown_report(missing)
    assert json.loads(json.dumps(asdict(missing), allow_nan=False))["perplexity"] is None
    with patch.object(eval_model, "run_evaluation", return_value=missing):
        assert eval_model.main(["--max-perplexity", "100", "--format", "json"]) == 1


def test_token_weighted_loss_and_perplexity_gate():
    item = {"id": "weighted", "category": "same", "prompt": "Q", "expected": "A"}
    scored = []
    for logprobs in [[-10.0, None, True, float("nan")], [-0.01] * 100]:
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"text": "A", "logprobs": {"token_logprobs": logprobs}}],
            # Reported generation counts must not weight loss aggregates.
            "usage": {"completion_tokens": 999},
        }).encode()
        with patch.object(eval_model, "open_api_request", return_value=response):
            scored.append(eval_model.evaluate_single_item("http://localhost", None, "model", item))
    assert [result.scored_tokens for result in scored] == [1, 100]
    assert [result.negative_log_likelihood for result in scored] == [10.0, 1.0]
    with patch.object(eval_model, "evaluate_single_item", side_effect=scored):
        report = eval_model.run_evaluation("unused", None, "model", [{}, {}])
    expected_loss = 11.0 / 101
    assert math.isclose(report.mean_cross_entropy, expected_loss)
    assert math.isclose(report.perplexity, math.exp(expected_loss))
    assert math.isclose(report.categories["same"]["cross_entropy"], expected_loss)
    assert math.isclose(report.categories["same"]["perplexity"], math.exp(expected_loss))
    with patch.object(eval_model, "run_evaluation", return_value=report):
        assert eval_model.main(["--max-perplexity", "2", "--format", "json"]) == 0
        assert eval_model.main(["--max-perplexity", "1.05", "--format", "json"]) == 1


def test_benchmark_stream_completion_and_exact_token_counts():
    content = 'data: {"choices":[{"delta":{"content":"several tokens in one chunk"}}]}\n\n'
    usage = 'data: {"usage":{"completion_tokens":3,"prompt_tokens":2}}\n\n'
    done = 'data: [DONE]\n\n'
    results = []
    for wire, success, known, tokens in [
        (content, False, False, 0),
        (content + usage, False, False, 0),
        (content + done, True, False, 0),
        (content + usage + done, True, True, 3),
    ]:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.__iter__.return_value = iter(wire.encode().splitlines(keepends=True))
        with patch("scripts.benchmark_api.open_api_request", return_value=response):
            result = execute_single_request("http://localhost", None, "model", "Q", 1000)
        assert (result.status_code == 200 and result.error is None) == success
        assert result.token_count_known == known
        assert result.generated_tokens == tokens
        if not success:
            assert "missing [DONE]" in result.error
        if not known:
            assert result.tokens_per_second == 0
        results.append(result)
    with patch("scripts.benchmark_api.execute_single_request", side_effect=results), patch("scripts.benchmark_api.fetch_metrics", return_value={}):
        report = run_benchmark("http://localhost", "model", num_requests=4)
    assert report["summary"]["successful_requests"] == 2
    assert report["summary"]["failed_requests"] == 2
    assert report["summary"]["total_generated_tokens"] == 3
    assert report["summary"]["unknown_token_count_requests"] == 1
    assert report["tokens_per_second"]["min"] > 0


def test_manifest_inventory_validation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tokenizer.json").write_text("{}")
        for inventory in [None, {}, [], {"weight.bin": None}, {"weight.bin": {}}, {"tokenizer.json": {"size": 2, "sha256": "0" * 64}}, {"../escape.bin": {"size": 0, "sha256": "0" * 64}}]:
            result = verify_model_files(root, {"files": inventory}, True)
            assert not result["valid"]
            assert result["inventory_errors"]


def test_launcher_credentials_and_exit_status():
    for mode in ["chat", "benchmark", "eval"]:
        for key in [None, "configured-key"]:
            env = {} if key is None else {"HAWKPOINT_API_KEY": key}
            with patch.dict("os.environ", env, clear=True), patch("sys.argv", ["launcher.py", mode]), patch("launcher.subprocess.run", return_value=MagicMock(returncode=7)) as run, patch("launcher.secrets.token_urlsafe") as random:
                try:
                    launcher.main()
                except SystemExit as exc:
                    assert exc.code == 7
                else:
                    raise AssertionError("launcher did not propagate failure")
                random.assert_not_called()
                assert "--api-key" not in run.call_args.args[0]
                assert run.call_args.kwargs["env"].get("HAWKPOINT_API_KEY") == key
    with patch.dict("os.environ", {}, clear=True), patch("launcher.glob.glob", return_value=[]), patch("launcher.os.getgid", return_value=1000):
        env = launcher.openwebui_environment("key")
        assert env["HAWKPOINT_ACCEL_GID"] == env["HAWKPOINT_DRI_GID"] == "1000"
    with patch("sys.argv", ["launcher.py", "inspect"]), patch("launcher.subprocess.run") as run:
        try:
            launcher.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("missing model directory was accepted")
        run.assert_not_called()


def test_doctor_failure_precedence_and_conflicts():
    with patch("scripts.doctor.platform.system", return_value="Windows"), patch("scripts.doctor.platform.machine", return_value="ARM64"), patch("builtins.open", mock_open(read_data="flags : sse\n")):
        assert doctor.check_os_and_cpu()["status"] == doctor.CheckStatus.FAIL
    with tempfile.TemporaryDirectory() as tmp, patch("scripts.doctor.discover_models", side_effect=RuntimeError("duplicate model id")):
        result = doctor.check_models_directory(Path(tmp))
        assert result["status"] == doctor.CheckStatus.FAIL
        assert "duplicate" in result["issues"][0]
        assert result["remediation"]
    with patch("scripts.doctor.socket.socket") as sock, patch("scripts.doctor.urllib.request.urlopen", side_effect=OSError("another service")):
        sock.return_value.connect.return_value = None
        result = doctor.check_network_and_services(port=12345)
        assert result["status"] == doctor.CheckStatus.FAIL
        assert "12345" in result["issues"][0]


def test_transport_policy_and_actual_redirects():
    for url in ["http://localhost:8000", "http://127.0.0.2:8000", "http://[::1]:8000", "https://example.com"]:
        validate_api_url(url, "key")
    for url in ["http://example.com", "http://localhost.example.com", "http://192.168.1.1"]:
        try:
            validate_api_url(url, "key")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe bearer destination accepted: {url}")

    class Redirect(BaseHTTPRequestHandler):
        hits = []
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == "/sink":
                self.hits.append(self.headers.get("Authorization"))
                self.send_response(200)
            else:
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/sink")
            self.end_headers()
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(307)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/sink")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for call in [lambda: open_api_request(Request(base), "key"), lambda: list_models(base, "key"), lambda: complete_text(base, "Q", api_key="key"), lambda: stream_chat(base, "Q", api_key="key")]:
            try:
                call()
            except HTTPError as exc:
                assert exc.code in [302, 307]
            else:
                raise AssertionError("authenticated redirect was followed")
        assert list_api_models(base, "key") == []
        result = execute_single_request(base, "key", "model", "Q", 5)
        assert result.error
        assert not Redirect.hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_stream_errors_and_unknown_benchmark_tokens():
    record = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: {"error":{"message":"worker failed"}}\n\n'
    with patch("npu_llm.chat.open_api_request", return_value=io.BytesIO(record)):
        generation = stream_chat_api("http://localhost", None, "model", [], 5)
        assert next(generation) == ("partial", None)
        try:
            next(generation)
        except RuntimeError as exc:
            assert "API stream failed: worker failed" in str(exc)
        else:
            raise AssertionError("stream error was swallowed")
    for usage in [None, {}]:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = json.dumps({"choices": [], "usage": usage}).encode()
        with patch("scripts.benchmark_api.open_api_request", return_value=response):
            result = execute_single_request("http://localhost", None, "model", "Q", 1000, stream=False)
        assert result.generated_tokens == 0
        assert result.tokens_per_second == 0
        assert not result.token_count_known


def test_schema_and_container_command():
    from npu_llm.api_server import OPENAPI_SPEC
    assert json.loads((ROOT / "docs/openapi.json").read_text()) == OPENAPI_SPEC
    props = OPENAPI_SPEC["paths"]["/v1/completions"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]
    array = props["prompt"]["oneOf"][1]
    assert array["minItems"] == array["maxItems"] == 1
    assert "suffix" not in props
    import subprocess
    command = json.loads(next(line[4:] for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("CMD ")))
    assert command[command.index("--host") + 1] == "127.0.0.1"
    result = subprocess.run([sys.executable, *command, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} review regression tests")
