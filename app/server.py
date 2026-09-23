"""HTTP API for the pipe-network route audit."""

from __future__ import annotations

import os
from typing import Any, Dict

from .solver import AuditError, Edge, audit


def _serialize(nodes, edges: list[Edge], result) -> Dict[str, Any]:
    transfer_mode = result.transfer_mode
    edge_objs = [
        {
            "id": e.eid,
            "u": e.u,
            "v": e.v,
            "length": e.length,
            "index": e.index,
            "classification": result.classification.get(e.index, ""),
            "duplicated": (
                result.multiplicity[e.index] > 1 if transfer_mode
                else e.index in result.canonical_set
            ),
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
            "transfer": st.transfer,
            "ruleRef": (
                [st.rule_key[0], st.rule_key[1]] if st.rule_key else None
            ),
        }
        for i, st in enumerate(result.route)
    ]
    # positions at which each edge occurs in the route, for highlighting
    positions: Dict[int, list] = {i: [] for i in range(len(edges))}
    for i, st in enumerate(result.route):
        positions[st.edge_index].append(i + 1)

    payload: Dict[str, Any] = {
        "ok": True,
        "mode": "transfer" if transfer_mode else "normal",
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

    if transfer_mode:
        rule_objs = []
        for rule in result.rules:
            pair = tuple(sorted((rule.edge_a, rule.edge_b)))
            hits = result.rule_hits.get((rule.node, pair[0], pair[1]), ())
            rule_objs.append(
                {
                    "ref": rule.row,
                    "node": rule.node,
                    "edgeA": pair[0],
                    "edgeB": pair[1],
                    "cost": rule.cost,  # null == forbidden
                    "forbidden": rule.cost is None,
                    "positions": list(hits),
                }
            )
        payload.update(
            {
                "walkLength": result.walk_length,
                "transferCost": result.transfer_cost,
                "totalTime": result.walk_length + result.transfer_cost,
                "rules": rule_objs,
            }
        )
    return payload


def run_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure logic entry: validate + audit, return a response dict."""
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    start = payload.get("start")
    transfer_mode = bool(payload.get("transferMode", False))
    rules = payload.get("rules", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    try:
        result = audit(nodes, edges, start, transfer_mode, rules)
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
