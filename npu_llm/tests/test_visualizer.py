#!/usr/bin/env python3
"""Tests for the IRON graph visualizer tool."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.visualize_graph import (  # noqa: E402
    build_decoder_layer_model,
    build_qwen_model,
    build_smollm_model,
    export_graph,
    generate_dot,
    generate_svg,
    get_graph_model,
    main,
)


def test_models_structure():
    """Verify that all graph models define the required nodes and channels."""
    for design in ("smollm", "qwen", "decoder"):
        model = get_graph_model(design)
        assert model.nodes, f"{design} has no nodes"
        assert model.channels, f"{design} has no channels"
        assert any(n.tile_type == "shim" for n in model.nodes)
        assert any(n.tile_type == "core" for n in model.nodes)

    smollm = build_smollm_model()
    assert len([n for n in smollm.nodes if n.tile_type == "core"]) == 7  # 6 GEMV + 1 Hub
    assert any(n.name == "MemTile_Join" for n in smollm.nodes)

    qwen = build_qwen_model()
    assert len([n for n in qwen.nodes if n.tile_type == "core"]) == 7
    assert any("Logits Stream" in " ".join(n.operations) for n in qwen.nodes)

    decoder = build_decoder_layer_model()
    assert any(n.name == "Tile_QKV" for n in decoder.nodes)


def test_generate_dot_output():
    """Ensure generated DOT markup contains required subgraphs, nodes, and syntax."""
    dot = generate_dot("smollm")
    assert dot.startswith("digraph \"smollm_engine\" {")
    assert "cluster_shim" in dot
    assert "cluster_memtile" in dot
    assert "cluster_core" in dot
    assert "MemTile_Join" in dot
    assert "Hub_Tile" in dot
    assert "weights" in dot
    assert "broadcast" in dot
    assert dot.rstrip().endswith("}")

    qwen_dot = generate_dot("qwen")
    assert "qwen_engine" in qwen_dot
    assert "logits" in qwen_dot


def test_generate_svg_output():
    """Ensure generated SVG is valid standalone markup with responsive attributes."""
    svg = generate_svg("smollm")
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "viewBox=" in svg
    assert "SmolLM2 135M Single-Dispatch" in svg
    assert "Tile(1,1)" in svg
    assert "Attention Hub" in svg
    assert "<defs>" in svg
    assert "<rect" in svg
    assert "<path" in svg

    qwen_svg = generate_svg("qwen")
    assert "Qwen2.5 0.5B" in qwen_svg


def test_export_graph_and_cli():
    """Verify export_graph writes to disk and CLI executes without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        svg_file = Path(tmp_dir) / "test.svg"
        dot_file = Path(tmp_dir) / "test.dot"

        export_graph("smollm", "svg", svg_file)
        assert svg_file.is_file()
        assert "<svg" in svg_file.read_text(encoding="utf-8")

        export_graph("smollm", "dot", dot_file)
        assert dot_file.is_file()
        assert "digraph" in dot_file.read_text(encoding="utf-8")

        # Test CLI invocation
        cli_out = Path(tmp_dir) / "cli.svg"
        ret = main(["--design", "qwen", "--format", "svg", "--output", str(cli_out)])
        assert ret == 0
        assert cli_out.is_file()


def test_invalid_design_error():
    """Ensure invalid design names raise ValueError."""
    try:
        get_graph_model("invalid_model")
    except ValueError as exc:
        assert "unknown design" in str(exc)
    else:
        raise AssertionError("get_graph_model did not raise for invalid design")


def run_all_tests():
    test_models_structure()
    test_generate_dot_output()
    test_generate_svg_output()
    test_export_graph_and_cli()
    test_invalid_design_error()
    print("PASS 5 visualizer tests")


if __name__ == "__main__":
    run_all_tests()
