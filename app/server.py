"""HTTP API for the pipe-network route audit."""

from __future__ import annotations

import os
from typing import Any, Dict

from .solver import AuditError, Edge, RouteStep, audit, audit_transfer


def _serialize(nodes, edges: list[Edge], result) -> Dict[str, Any]:
    edge_objs = [
        {
            "id": e.eid,
            "u": e.u,
            "v": e.v,
            "length": e.length,
            "index": e.index,
            "classification": result.classification[e.index],
            "duplicated": e.index in result.canonical_set,
            "copies": result.multiplicity[e.index],
        }
        for e in edges
    ]
    route = [
        {
            "seq": i + 1,
            "edgeIndex": st.edge_index,
            "edgeId": st.edge_id,
            "from": st.frm,
            "to": st.to,
            "length": st.length,
            "copy": st.duplicate_no,
        }
        for i, st in enumerate(result.route)
    ]
    # positions at which each edge occurs in the route, for highlighting
    positions: Dict[int, list] = {i: [] for i in range(len(edges))}
    for i, st in enumerate(result.route):
        positions[st.edge_index].append(i + 1)

    return {
        "ok": True,
        "mode": "normal",
        "nodes": nodes,
        "edges": edge_objs,
        "start": result.start,
        "oddVertices": list(result.odd_vertices),
        "totalLength": result.total_length,
        "addedLength": result.added_length,
        "optimalCount": result.optimal_count,
        "canonicalVector": result.bit_vector,
        "canonicalEdges": [
            edges[i].eid for i in sorted(result.canonical_set)
        ],
        "route": route,
        "positions": {str(k): v for k, v in positions.items()},
        "eulerian": result.is_eulerian,
    }


def _serialize_transfer(nodes, edges: list[Edge], result) -> Dict[str, Any]:
    edge_objs = [
        {
            "id": e.eid,
            "u": e.u,
            "v": e.v,
            "length": e.length,
            "index": e.index,
        }
        for e in edges
    ]
    route = []
    for i, st in enumerate(result.route):
        route.append(
            {
                "seq": i + 1,
                "edgeIndex": st.edge_index,
                "edgeId": st.edge_id,
                "from": st.frm,
                "to": st.to,
                "length": st.length,
                "transfer": st.transfer_cost,
                "rule": list(st.rule) if st.rule is not None else None,
            }
        )
    # positions at which each transfer rule (unordered edge-index pair) is
    # used in the canonical walk, and each edge occurs, for highlighting
    rule_positions: Dict[str, list] = {}
    edge_positions: Dict[int, list] = {i: [] for i in range(len(edges))}
    for i, st in enumerate(result.route):
        edge_positions[st.edge_index].append(i + 1)
        if st.rule is not None:
            a, b = st.rule
            key = f"{a}-{b}"
            rule_positions.setdefault(key, []).append(i + 1)

    rules_out = [
        {
            "row": rl.row,
            "edgeA": edges[rl.edge_a].eid,
            "edgeB": edges[rl.edge_b].eid,
            "a": rl.edge_a,
            "b": rl.edge_b,
            "cost": rl.cost,  # null == forbidden
            "forbidden": rl.cost is None,
            "positions": rule_positions.get(
                f"{min(rl.edge_a, rl.edge_b)}-{max(rl.edge_a, rl.edge_b)}", []
            ),
        }
        for rl in result.rules
    ]

    return {
        "ok": True,
        "mode": "transfer",
        "nodes": nodes,
        "edges": edge_objs,
        "start": result.start,
        "rules": rules_out,
        "walkingLength": result.total_length,
        "transferTime": result.transfer_time,
        "totalTime": result.total_time,
        "optimalCount": result.optimal_count,
        "route": route,
        "positions": {str(k): v for k, v in edge_positions.items()},
        "rulePositions": rule_positions,
    }


def run_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure logic entry: validate + audit, return a response dict."""
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    start = payload.get("start")
    transfer_mode = payload.get("transferMode") is True
    raw_rules = payload.get("transfers", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    if not isinstance(raw_rules, list):
        raw_rules = []
    try:
        if transfer_mode:
            result = audit_transfer(nodes, edges, start, raw_rules)
            return _serialize_transfer(result.nodes, result.edges, result)
        result = audit(nodes, edges, start)
    except AuditError as exc:
        return {
            "ok": False,
            "error": exc.message,
            "fields": list(exc.fields),
            "locations": exc.locations,
        }
    return _serialize(result.nodes, result.edges, result)


def create_app():
    from flask import Flask, jsonify, request, send_from_directory

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="")

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/audit")
    def api_audit():
        payload = request.get_json(silent=True) or {}
        return jsonify(run_audit(payload))

    return app


app = create_app()
