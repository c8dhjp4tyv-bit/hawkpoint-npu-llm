#!/usr/bin/env python3
"""IRON graph visualizer: export tile placements and ObjectFifo dataflows to DOT and SVG.

Generates visual representations of XDNA1 AIE2 array architectures,
including the SmolLM single-dispatch engine, Qwen2.5 engine, and chunked decoders.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import html
from pathlib import Path
import sys


@dataclass(frozen=True)
class TileNode:
    col: int
    row: int
    tile_type: str  # "shim", "memtile", "core"
    name: str
    role: str
    operations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FifoChannel:
    name: str
    src: tuple[int, int]  # (col, row)
    dst: tuple[int, int]  # (col, row)
    shape: str
    depth: int
    description: str


@dataclass
class GraphModel:
    name: str
    title: str
    description: str
    cols: int = 4
    rows: int = 4  # rows 0..3 (row 0: shim, row 1: memtile, rows 2..3: core)
    nodes: list[TileNode] = field(default_factory=list)
    channels: list[FifoChannel] = field(default_factory=list)


def build_smollm_model() -> GraphModel:
    """SmolLM 135M row-split single-dispatch decoder engine."""
    model = GraphModel(
        name="smollm_engine",
        title="SmolLM2 135M Single-Dispatch Decoder Engine",
        description="6 GEMV tiles (256-elem slices), 1 MemTile join, 1 Hub tile (RoPE/Attention/KV), DDR streaming.",
    )
    # Row 0: Shim DMA
    model.nodes.extend([
        TileNode(0, 0, "shim", "Shim_0_0", "DDR Shim DMA Col 0", ["Weights 0 & 1 stream"]),
        TileNode(1, 0, "shim", "Shim_1_0", "DDR Shim DMA Col 1", ["Weights 2 & 3 stream"]),
        TileNode(2, 0, "shim", "Shim_2_0", "DDR Shim DMA Col 2", ["Weights 4 & 5 stream"]),
        TileNode(3, 0, "shim", "Shim_3_0", "DDR Shim DMA Col 3", ["K/V Cache In", "KvNew Out", "Hidden Out"]),
    ])
    # Row 1: MemTile
    model.nodes.extend([
        TileNode(1, 1, "memtile", "MemTile_Join", "6-Way Slice Join", ["Joins 6x256 slices into 1536-elem vector"]),
    ])
    # Rows 2 & 3: Compute Tiles
    model.nodes.extend([
        TileNode(0, 2, "core", "GEMV_0", "GEMV Tile 0", ["RMSNorm, QKV rows 0..159", "O/MLP/Down slice 0", "Final norm -> Hidden Out"]),
        TileNode(0, 3, "core", "GEMV_1", "GEMV Tile 1", ["RMSNorm, QKV rows 160..319", "O/MLP/Down slice 1"]),
        TileNode(1, 2, "core", "GEMV_2", "GEMV Tile 2", ["RMSNorm, QKV rows 320..479", "O/MLP/Down slice 2"]),
        TileNode(1, 3, "core", "GEMV_3", "GEMV Tile 3", ["RMSNorm, QKV rows 480..639", "O/MLP/Down slice 3"]),
        TileNode(2, 2, "core", "GEMV_4", "GEMV Tile 4", ["RMSNorm, QKV rows 640..799", "O/MLP/Down slice 4"]),
        TileNode(2, 3, "core", "GEMV_5", "GEMV Tile 5", ["RMSNorm, QKV rows 800..959", "O/MLP/Down slice 5"]),
        TileNode(3, 2, "core", "Hub_Tile", "Attention Hub", ["RoPE, 9 Q / 3 KV Heads", "Vectorized Softmax", "K/V Cache Append", "1536-Elem Broadcast"]),
    ])
    # Fifo channels
    gemv_tiles = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)]
    for i, (col, row) in enumerate(gemv_tiles):
        shim_col = i // 2
        model.channels.append(FifoChannel(f"weights{i}", (shim_col, 0), (col, row), "bf16[1536]", 2, f"Weight Stream {i}"))
        model.channels.append(FifoChannel(f"slice{i}", (col, row), (1, 1), "bf16[256]", 2, f"Result Slice {i}"))
        model.channels.append(FifoChannel("broadcast", (3, 2), (col, row), "bf16[1536]", 2, "Activation Broadcast"))
    model.channels.append(FifoChannel("joined", (1, 1), (3, 2), "bf16[1536]", 2, "Joined 1536-vector"))
    model.channels.append(FifoChannel("cache", (3, 0), (3, 2), "bf16[256]", 2, "K/V History from DDR"))
    model.channels.append(FifoChannel("kv_new", (3, 2), (3, 0), "bf16[128]", 1, "Appended K/V to DDR"))
    model.channels.append(FifoChannel("hidden_out", (0, 2), (3, 0), "bf16[576]", 1, "Final Normalized State"))
    return model


def build_qwen_model() -> GraphModel:
    """Qwen2.5 0.5B single-dispatch decoder engine."""
    model = GraphModel(
        name="qwen_engine",
        title="Qwen2.5 0.5B Single-Dispatch Decoder Engine",
        description="6 GEMV tiles (uneven rows + LM head), 1 MemTile join, 1 Hub tile (14 Q / 2 KV heads), NPU LM-head streaming.",
    )
    # Row 0: Shim DMA
    model.nodes.extend([
        TileNode(0, 0, "shim", "Shim_0_0", "DDR Shim DMA Col 0", ["Weights 0 & 1", "Logits Stream 0 & 1"]),
        TileNode(1, 0, "shim", "Shim_1_0", "DDR Shim DMA Col 1", ["Weights 2 & 3", "Logits Stream 2 & 3"]),
        TileNode(2, 0, "shim", "Shim_2_0", "DDR Shim DMA Col 2", ["Weights 4 & 5", "Logits Stream 4 & 5"]),
        TileNode(3, 0, "shim", "Shim_3_0", "DDR Shim DMA Col 3", ["K/V Cache In", "KvNew Out"]),
    ])
    # Row 1: MemTile
    model.nodes.extend([
        TileNode(1, 1, "memtile", "MemTile_Join", "6-Way Slice Join", ["Joins 6x896 slices into 5376-elem vector"]),
    ])
    # Rows 2 & 3: Compute Tiles
    lm_start_table = (0, 25344, 50688, 76032, 101376, 126656)
    lm_groups_table = (396, 396, 396, 396, 395, 395)
    lm_group_sz = 64

    gemv_tiles = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)]
    for i, (col, row) in enumerate(gemv_tiles):
        lm_start = lm_start_table[i]
        lm_len = lm_groups_table[i] * lm_group_sz
        lm_end = lm_start + lm_len - 1
        model.nodes.append(
            TileNode(col, row, "core", f"GEMV_{i}", f"GEMV Tile {i}", [
                "RMSNorm, QKV rows",
                f"O/MLP/Down slice {i}",
                f"LM Head {lm_start}..{lm_end} (FP32)",
            ])
        )
    model.nodes.append(
        TileNode(3, 2, "core", "Hub_Tile", "Attention Hub", [
            "QKV Bias Add, RoPE",
            "14 Q-Heads / 2 KV-Heads Attention",
            "Vectorized Softmax",
            "5376-Elem Broadcast",
        ])
    )
    for i, (col, row) in enumerate(gemv_tiles):
        shim_col = i // 2
        lm_len = lm_groups_table[i] * lm_group_sz
        model.channels.append(FifoChannel(f"weights{i}", (shim_col, 0), (col, row), "bf16[1536]", 2, f"Weight Stream {i}"))
        model.channels.append(FifoChannel(f"slice{i}", (col, row), (1, 1), "bf16[896]", 2, f"Result Slice {i}"))
        model.channels.append(FifoChannel("broadcast", (3, 2), (col, row), "bf16[5376]", 2, "Activation Broadcast"))
        model.channels.append(FifoChannel(f"logits{i}", (col, row), (shim_col, 0), f"fp32[{lm_len}]", 1, f"LM Head Logits {i}"))
    model.channels.append(FifoChannel("joined", (1, 1), (3, 2), "bf16[5376]", 2, "Joined 5376-vector"))
    model.channels.append(FifoChannel("cache", (3, 0), (3, 2), "bf16[256]", 2, "K/V History from DDR"))
    model.channels.append(FifoChannel("kv_new", (3, 2), (3, 0), "bf16[128]", 1, "Appended K/V to DDR"))
    return model


def build_decoder_layer_model() -> GraphModel:
    """Chunked two-layer pipelined decoder graph."""
    model = GraphModel(
        name="decoder_layer",
        title="Chunked 2-Layer Pipelined Decoder",
        description="Layer-by-layer pipelined execution with multi-tile projections and sequential dispatch.",
    )
    model.nodes.extend([
        TileNode(0, 0, "shim", "Shim_0_0", "DDR Shim DMA Col 0", ["Weight Stream", "Input Activation In"]),
        TileNode(1, 0, "shim", "Shim_1_0", "DDR Shim DMA Col 1", ["K/V Cache In/Out"]),
        TileNode(2, 0, "shim", "Shim_2_0", "DDR Shim DMA Col 2", ["Output Activation Out"]),
        TileNode(0, 2, "core", "Tile_QKV", "QKV Projection Tile", ["RMSNorm", "QKV Projections", "RoPE"]),
        TileNode(1, 2, "core", "Tile_Attn", "Attention Tile", ["Multi-Head Attention", "K/V Cache Append"]),
        TileNode(2, 2, "core", "Tile_MLP", "MLP FeedForward Tile", ["Gate/Up Projections", "SwiGLU", "Down Projection"]),
    ])
    model.channels.extend([
        FifoChannel("act_in", (0, 0), (0, 2), "bf16[576]", 2, "Input Activation"),
        FifoChannel("qkv_out", (0, 2), (1, 2), "bf16[960]", 2, "QKV Vectors"),
        FifoChannel("cache_io", (1, 0), (1, 2), "bf16[256]", 2, "K/V Cache Line"),
        FifoChannel("attn_out", (1, 2), (2, 2), "bf16[576]", 2, "Attended State"),
        FifoChannel("act_out", (2, 2), (2, 0), "bf16[576]", 2, "Layer Output"),
    ])
    return model


def get_graph_model(design_name: str) -> GraphModel:
    """Retrieve the GraphModel for a named architecture."""
    lookup = {
        "smollm": build_smollm_model,
        "qwen": build_qwen_model,
        "decoder": build_decoder_layer_model,
    }
    factory = lookup.get(design_name.lower())
    if not factory:
        raise ValueError(f"unknown design: {design_name!r}. Expected one of: {list(lookup)}")
    return factory()


def generate_dot(design_name: str) -> str:
    """Generate Graphviz DOT markup for the architecture."""
    model = get_graph_model(design_name)
    lines = [
        f'digraph "{model.name}" {{',
        '  graph [rankdir=TB, compound=true, bgcolor="#0f172a", fontname="Inter,sans-serif", fontcolor="#f8fafc", pad=0.5, nodesep=0.5, ranksep=0.8];',
        '  node [shape=box, style="rounded,filled", fontname="Inter,sans-serif", margin="0.2,0.15", penwidth=1.5];',
        '  edge [fontname="Inter,sans-serif", fontsize=10, penwidth=1.8];',
        f'  label = "{model.title}\\n{model.description}";',
        '  labelloc = "t";',
        '  fontsize = 16;',
        '',
    ]

    # Cluster Shim DMA (Row 0)
    lines.append('  subgraph cluster_shim {')
    lines.append('    label = "Row 0: Shim DMA (External DDR Interfaces)";')
    lines.append('    style = "rounded,dashed";')
    lines.append('    color = "#334155";')
    lines.append('    fontcolor = "#94a3b8";')
    lines.append('    rank = same;')
    for node in model.nodes:
        if node.tile_type == "shim":
            op_text = "\\n".join(node.operations)
            lines.append(f'    "{node.name}" [label="Tile({node.col}, {node.row})\\n{node.role}\\n{op_text}", fillcolor="#064e3b", color="#10b981", fontcolor="#ecfdf5"];')
    lines.append('  }')
    lines.append('')

    # Cluster MemTiles (Row 1)
    lines.append('  subgraph cluster_memtile {')
    lines.append('    label = "Row 1: Memory Tiles (MemTile Shared Memory)";')
    lines.append('    style = "rounded,dashed";')
    lines.append('    color = "#334155";')
    lines.append('    fontcolor = "#94a3b8";')
    lines.append('    rank = same;')
    for node in model.nodes:
        if node.tile_type == "memtile":
            op_text = "\\n".join(node.operations)
            lines.append(f'    "{node.name}" [label="Tile({node.col}, {node.row})\\n{node.role}\\n{op_text}", fillcolor="#78350f", color="#f59e0b", fontcolor="#fef3c7"];')
    lines.append('  }')
    lines.append('')

    # Cluster Compute Cores (Rows 2 & 3)
    lines.append('  subgraph cluster_core {')
    lines.append('    label = "Rows 2-3: AIE2 Core Compute Tiles";')
    lines.append('    style = "rounded,dashed";')
    lines.append('    color = "#334155";')
    lines.append('    fontcolor = "#94a3b8";')
    for node in model.nodes:
        if node.tile_type == "core":
            op_text = "\\n".join(node.operations)
            if "Hub" in node.name:
                fill, border, text = "#581c87", "#a855f7", "#faf5ff"
            else:
                fill, border, text = "#1e3a8a", "#3b82f6", "#eff6ff"
            lines.append(f'    "{node.name}" [label="Tile({node.col}, {node.row})\\n{node.role}\\n{op_text}", fillcolor="{fill}", color="{border}", fontcolor="{text}"];')
    lines.append('  }')
    lines.append('')

    # Edges
    node_by_pos = {(n.col, n.row): n for n in model.nodes}
    # Group channels to prevent duplicate edges
    edge_map = {}
    for ch in model.channels:
        src_node = node_by_pos.get(ch.src)
        dst_node = node_by_pos.get(ch.dst)
        if src_node and dst_node:
            pair = (src_node.name, dst_node.name)
            edge_map.setdefault(pair, []).append(ch)

    for (src, dst), channels in edge_map.items():
        if len(channels) == 1:
            ch = channels[0]
            label = f"{ch.name} (d={ch.depth})"
        else:
            names = ", ".join(c.name for c in channels[:2])
            if len(channels) > 2:
                names += f", +{len(channels)-2}"
            label = f"{names} (d={channels[0].depth})"

        color = "#38bdf8"
        if "weight" in label:
            color = "#34d399"
        elif "broadcast" in label:
            color = "#c084fc"
        elif "slice" in label or "joined" in label:
            color = "#fbbf24"
        elif "cache" in label or "kv" in label:
            color = "#f43f5e"

        lines.append(f'  "{src}" -> "{dst}" [label="{label}", color="{color}", fontcolor="{color}"];')

    lines.append('}')
    return "\n".join(lines)


def generate_svg(design_name: str) -> str:
    """Generate a clean, standalone, responsive SVG diagram."""
    model = get_graph_model(design_name)

    width = 960
    height = 700
    margin_x = 40
    margin_y = 70

    # Grid tile layout:
    # 4 columns (col 0..3), rows 0..3 (Y flipped so row 0 is bottom or top)
    # In AIE2 hardware architecture:
    # Row 0: Shim DMA (bottom / physical interface)
    # Row 1: MemTile
    # Rows 2..3: Compute Cores
    # To read naturally from top to bottom (compute flow):
    # Let's arrange Row 0 (Shim) at top, Row 1 (MemTile) in middle, Rows 2 & 3 below, or
    # Physical array view: Row 3 at top, Row 2, Row 1, Row 0 at bottom.
    # Physical view is standard for hardware engineers.
    cell_w = 200
    cell_h = 110
    col_gap = 25
    row_gap = 35

    def get_coords(col: int, row: int) -> tuple[float, float]:
        # Physical coordinates: row 0 is bottom (Y=520), row 1 (Y=380), row 2 (Y=240), row 3 (Y=100)
        x = margin_x + col * (cell_w + col_gap)
        # Flip row so row 3 is topmost, row 0 is bottom
        y = margin_y + (3 - row) * (cell_h + row_gap)
        return x, y

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#090d16; font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#0f172a"/><stop offset="100%" stop-color="#020617"/></linearGradient>',
        '    <linearGradient id="gemvGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#1e3a8a"/><stop offset="100%" stop-color="#0f2156"/></linearGradient>',
        '    <linearGradient id="hubGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#581c87"/><stop offset="100%" stop-color="#3b0764"/></linearGradient>',
        '    <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#78350f"/><stop offset="100%" stop-color="#451a03"/></linearGradient>',
        '    <linearGradient id="shimGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#064e3b"/><stop offset="100%" stop-color="#022c22"/></linearGradient>',
        '    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#38bdf8"/></marker>',
        '    <marker id="arrow-gold" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#fbbf24"/></marker>',
        '    <marker id="arrow-purple" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#c084fc"/></marker>',
        '    <marker id="arrow-green" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><polygon points="0 1, 8 4, 0 7" fill="#34d399"/></marker>',
        '    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%"><feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.5"/></filter>',
        '  </defs>',
        '  <rect width="100%" height="100%" fill="url(#bgGrad)"/>',
    ]

    # Header
    svg_parts.append(f'  <text x="{margin_x}" y="36" fill="#f8fafc" font-size="20" font-weight="700">{html.escape(model.title)}</text>')
    svg_parts.append(f'  <text x="{margin_x}" y="56" fill="#94a3b8" font-size="12">{html.escape(model.description)}</text>')

    # Row backgrounds & labels
    row_labels = [
        (0, "Row 0: Shim DMA (DDR / Host Interface)", "#064e3b22"),
        (1, "Row 1: Memory Tiles (MemTile Join & Buffers)", "#78350f22"),
        (2, "Row 2: AIE2 Core Tiles (GEMV / Hub)", "#1e3a8a22"),
        (3, "Row 3: AIE2 Core Tiles (GEMV Tiles)", "#1e3a8a22"),
    ]
    for r, label, bg_color in row_labels:
        _, y = get_coords(0, r)
        svg_parts.append(f'  <rect x="{margin_x - 15}" y="{y - 12}" width="{width - margin_x * 2 + 30}" height="{cell_h + 24}" rx="8" fill="{bg_color}" stroke="#334155" stroke-width="0.8" stroke-dasharray="4 4"/>')
        svg_parts.append(f'  <text x="{margin_x - 5}" y="{y + 4}" fill="#64748b" font-size="11" font-weight="600">{html.escape(label)}</text>')

    # Draw Nodes
    for node in model.nodes:
        x, y = get_coords(node.col, node.row)
        if node.tile_type == "shim":
            grad, border, title_color = "url(#shimGrad)", "#10b981", "#6ee7b7"
        elif node.tile_type == "memtile":
            grad, border, title_color = "url(#memGrad)", "#f59e0b", "#fde68a"
        elif "Hub" in node.name:
            grad, border, title_color = "url(#hubGrad)", "#a855f7", "#e9d5ff"
        else:
            grad, border, title_color = "url(#gemvGrad)", "#3b82f6", "#bfdbfe"

        svg_parts.append(f'  <g filter="url(#shadow)">')
        svg_parts.append(f'    <rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="8" fill="{grad}" stroke="{border}" stroke-width="1.5"/>')
        svg_parts.append(f'    <rect x="{x + 8}" y="{y + 8}" width="65" height="18" rx="4" fill="#00000055"/>')
        svg_parts.append(f'    <text x="{x + 14}" y="{y + 21}" fill="#94a3b8" font-size="10" font-weight="700">Tile({node.col},{node.row})</text>')
        svg_parts.append(f'    <text x="{x + 80}" y="{y + 21}" fill="{title_color}" font-size="12" font-weight="700">{html.escape(node.role)}</text>')

        op_y = y + 42
        for op in node.operations[:3]:
            svg_parts.append(f'    <circle cx="{x + 14}" cy="{op_y - 3}" r="2.5" fill="{border}"/>')
            svg_parts.append(f'    <text x="{x + 22}" y="{op_y}" fill="#cbd5e1" font-size="10">{html.escape(op)}</text>')
            op_y += 18
        svg_parts.append('  </g>')

    # Channel flows rendered from model.channels
    vector_dim = "5376" if model.name == "qwen_engine" else "1536"
    node_positions = {(node.col, node.row) for node in model.nodes}
    for channel in model.channels:
        if channel.src not in node_positions or channel.dst not in node_positions:
            continue
        src_x, src_y = get_coords(*channel.src)
        dst_x, dst_y = get_coords(*channel.dst)
        dx, dy = dst_x - src_x, dst_y - src_y
        if abs(dx) >= abs(dy):
            start_x = src_x + (cell_w if dx > 0 else 0)
            start_y = src_y + cell_h / 2
            end_x = dst_x + (0 if dx > 0 else cell_w)
            end_y = dst_y + cell_h / 2
        else:
            start_x = src_x + cell_w / 2
            start_y = src_y + (cell_h if dy > 0 else 0)
            end_x = dst_x + cell_w / 2
            end_y = dst_y + (0 if dy > 0 else cell_h)

        if channel.name == "joined" or channel.name.startswith("slice"):
            stroke, marker = "#fbbf24", "arrow-gold"
        elif channel.name == "broadcast":
            stroke, marker = "#c084fc", "arrow-purple"
        else:
            stroke, marker = "#38bdf8", "arrow"
        svg_parts.append(
            f'  <path d="M {start_x} {start_y} L {end_x} {end_y}" fill="none" '
            f'stroke="{stroke}" stroke-width="2.5" marker-end="url(#{marker})"/>'
        )

    if (1, 1) in node_positions and (3, 2) in node_positions:
        mt_x, mt_y = get_coords(1, 1)
        hub_x, hub_y = get_coords(3, 2)
        svg_parts.append(f'  <text x="{mt_x + cell_w + 15}" y="{mt_y + cell_h/2 - 10}" fill="#fbbf24" font-size="11" font-weight="700">joined [{vector_dim}]</text>')
        svg_parts.append(f'  <text x="{hub_x - 140}" y="{hub_y - 30}" fill="#c084fc" font-size="11" font-weight="700">broadcast [{vector_dim}]</text>')

    # Legend at bottom
    leg_y = height - 30
    svg_parts.append(f'  <rect x="{margin_x}" y="{leg_y - 8}" width="{width - margin_x*2}" height="28" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>')
    svg_parts.append(f'  <circle cx="{margin_x + 20}" cy="{leg_y + 6}" r="5" fill="#3b82f6"/>')
    svg_parts.append(f'  <text x="{margin_x + 30}" y="{leg_y + 10}" fill="#94a3b8" font-size="11">GEMV Tiles</text>')
    svg_parts.append(f'  <circle cx="{margin_x + 120}" cy="{leg_y + 6}" r="5" fill="#a855f7"/>')
    svg_parts.append(f'  <text x="{margin_x + 130}" y="{leg_y + 10}" fill="#94a3b8" font-size="11">Attention Hub</text>')
    svg_parts.append(f'  <circle cx="{margin_x + 230}" cy="{leg_y + 6}" r="5" fill="#f59e0b"/>')
    svg_parts.append(f'  <text x="{margin_x + 240}" y="{leg_y + 10}" fill="#94a3b8" font-size="11">MemTile Join</text>')
    svg_parts.append(f'  <circle cx="{margin_x + 340}" cy="{leg_y + 6}" r="5" fill="#10b981"/>')
    svg_parts.append(f'  <text x="{margin_x + 350}" y="{leg_y + 10}" fill="#94a3b8" font-size="11">Shim DMA (DDR)</text>')
    svg_parts.append(f'  <line x1="{margin_x + 470}" y1="{leg_y + 6}" x2="{margin_x + 495}" y2="{leg_y + 6}" stroke="#fbbf24" stroke-width="2"/>')
    svg_parts.append(f'  <text x="{margin_x + 505}" y="{leg_y + 10}" fill="#94a3b8" font-size="11">Joined Slices</text>')
    svg_parts.append(f'  <line x1="{margin_x + 600}" y1="{leg_y + 6}" x2="{margin_x + 625}" y2="{leg_y + 6}" stroke="#c084fc" stroke-width="2"/>')
    svg_parts.append(f'  <text x="{margin_x + 635}" y="{leg_y + 10}" fill="#94a3b8" font-size="11">Broadcast Vector</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def export_graph(design_name: str, fmt: str, output_path: str | Path | None = None) -> str:
    """Export graph representation in requested format."""
    fmt = fmt.lower()
    if fmt == "dot":
        content = generate_dot(design_name)
    elif fmt == "svg":
        content = generate_svg(design_name)
    elif fmt == "all":
        dot_content = generate_dot(design_name)
        svg_content = generate_svg(design_name)
        if output_path:
            p = Path(output_path)
            p.with_suffix(".dot").write_text(dot_content, encoding="utf-8")
            p.with_suffix(".svg").write_text(svg_content, encoding="utf-8")
        return f"{dot_content}\n\n<!-- SVG Output -->\n{svg_content}"
    else:
        raise ValueError(f"unsupported format: {fmt!r}")

    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize IRON AIE2 Array and ObjectFifo Dataflows."
    )
    parser.add_argument(
        "--design",
        choices=["smollm", "qwen", "decoder"],
        default="smollm",
        help="Architecture design to visualize (default: smollm)",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "dot", "all"],
        default="svg",
        help="Output format (default: svg)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (prints to stdout if omitted)",
    )
    args = parser.parse_args(argv)

    content = export_graph(args.design, args.format, args.output)
    if not args.output:
        print(content)
    else:
        print(f"Exported {args.design} ({args.format}) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
