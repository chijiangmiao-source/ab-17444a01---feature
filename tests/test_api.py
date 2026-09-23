"""Tests for the HTTP-facing serialization layer (pure, no server needed)."""

from app.server import run_audit


K4 = {
    "nodes": ["A", "B", "C", "D"],
    "edges": [
        {"id": "e1", "u": "A", "v": "B", "length": 1},
        {"id": "e2", "u": "A", "v": "C", "length": 1},
        {"id": "e3", "u": "A", "v": "D", "length": 1},
        {"id": "e4", "u": "B", "v": "C", "length": 1},
        {"id": "e5", "u": "B", "v": "D", "length": 1},
        {"id": "e6", "u": "C", "v": "D", "length": 1},
    ],
    "start": "A",
}


def test_success_payload():
    r = run_audit(K4)
    assert r["ok"] is True
    assert r["optimalCount"] == 3
    assert r["addedLength"] == 2
    assert r["totalLength"] == 6
    assert r["canonicalVector"] == "001100"
    assert r["canonicalEdges"] == ["e3", "e4"]
    assert len(r["route"]) == 8
    # positions cover exactly duplicated canonical edges' extra copies
    counts = {e["id"]: len(r["positions"][str(e["index"])]) for e in r["edges"]}
    duplicated = set(r["canonicalEdges"])
    for eid, n in counts.items():
        assert n == (2 if eid in duplicated else 1)
    # route closes at start
    assert r["route"][0]["from"] == "A"
    assert r["route"][-1]["to"] == "A"


def test_eulerian_payload():
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "a", "u": "A", "v": "B", "length": 3},
            {"id": "b", "u": "B", "v": "C", "length": 4},
            {"id": "c", "u": "C", "v": "A", "length": 5},
        ],
        "start": "B",
    }
    r = run_audit(payload)
    assert r["ok"]
    assert r["eulerian"] is True
    assert r["addedLength"] == 0
    assert r["optimalCount"] == 1
    assert r["canonicalVector"] == "000"
    assert r["canonicalEdges"] == []
    assert all(e["classification"] == "never" for e in r["edges"])


def test_failure_payload_locations():
    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "Z", "length": 1}],
                   "start": "A"})
    assert r["ok"] is False
    assert "edges" in r["fields"]
    assert r["locations"][0]["row"] == 0

    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": -2}],
                   "start": "A"})
    assert not r["ok"] and "正整数" in r["error"]

    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": 1}],
                   "start": "Q"})
    assert not r["ok"] and r["fields"] == ["start"]

    r = run_audit({"nodes": ["A", "B", "C"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": 1}],
                   "start": "A"})
    assert not r["ok"] and "不连通" in r["error"]


def test_malformed_payload_is_safe():
    assert run_audit({})["ok"] is False
    assert run_audit({"nodes": "ab", "edges": None})["ok"] is False


# ---------------------------------------------------------------------------
# Transfer-time mode
# ---------------------------------------------------------------------------

DETOUR = {
    "nodes": ["A", "B", "C", "X"],
    "edges": [
        {"id": "e1", "u": "A", "v": "B", "length": 1},
        {"id": "e2", "u": "B", "v": "C", "length": 1},
        {"id": "e3", "u": "C", "v": "A", "length": 1},
        {"id": "e4", "u": "A", "v": "X", "length": 1},
        {"id": "e5", "u": "X", "v": "C", "length": 1},
    ],
    "start": "A",
}


def test_transfer_payload_costly_switch_detour():
    payload = {**DETOUR, "transferMode": True,
               "rules": [{"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": 100}]}
    r = run_audit(payload)
    assert r["ok"] is True
    assert r["mode"] == "transfer"
    assert r["walkLength"] == 7
    assert r["transferCost"] == 0
    assert r["totalTime"] == 7
    assert r["addedLength"] == 2  # walked 7 vs original 5
    assert r["optimalCount"] >= 1
    # route closes and covers every pipe at least once
    assert r["route"][0]["from"] == "A" and r["route"][-1]["to"] == "A"
    covered = {st["edgeId"] for st in r["route"]}
    assert covered == {f"e{i}" for i in range(1, 6)}
    # rule echo with its hit positions; the costly switch is never used
    rule0 = r["rules"][0]
    assert rule0["node"] == "B" and rule0["forbidden"] is False
    assert rule0["positions"] == []
    # every route step carries the transfer accounting fields
    assert all("transfer" in st and "ruleRef" in st for st in r["route"])


def test_transfer_forbidden_rule_absent_from_route():
    payload = {**DETOUR, "transferMode": True,
               "rules": [{"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": -1}]}
    r = run_audit(payload)
    assert r["ok"]
    rule0 = r["rules"][0]
    assert rule0["forbidden"] is True and rule0["cost"] is None
    assert rule0["positions"] == []
    for st in r["route"]:
        assert not (st["ruleRef"] == ["e1", "e2"] and st["from"] == "B")
    assert r["totalTime"] == r["walkLength"] + r["transferCost"]


def test_transfer_rule_hit_highlight_positions():
    nodes = ["A", "B", "C"]
    edges = [
        {"id": "e1", "u": "A", "v": "B", "length": 2},
        {"id": "e2", "u": "B", "v": "C", "length": 3},
    ]
    r = run_audit({"nodes": nodes, "edges": edges, "start": "A",
                   "transferMode": True,
                   "rules": [{"node": "B", "edgeA": "e1", "edgeB": "e2",
                              "cost": 4}]})
    assert r["ok"]
    assert r["walkLength"] == 10 and r["transferCost"] == 8
    rule0 = r["rules"][0]
    # the tour e1,e2,e2,e1 pays the B-switch when entering step 2 and step 4
    assert rule0["positions"] == [2, 4]
    hits = [st for st in r["route"] if st["seq"] in (2, 4)]
    assert all(st["transfer"] == 4 for st in hits)


def test_transfer_infeasible_is_error():
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "e1", "u": "A", "v": "B", "length": 1},
            {"id": "e2", "u": "B", "v": "C", "length": 1},
        ],
        "start": "A", "transferMode": True,
        "rules": [{"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": "禁行"}],
    }
    r = run_audit(payload)
    assert r["ok"] is False
    assert "无可行闭游" in r["error"]
    assert "rules" in r["fields"]


def test_transfer_rule_reference_errors():
    base = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "e1", "u": "A", "v": "B", "length": 1},
            {"id": "e2", "u": "B", "v": "C", "length": 1},
        ],
        "start": "A", "transferMode": True,
    }
    # unknown pipe
    r = run_audit({**base, "rules": [
        {"node": "B", "edgeA": "e1", "edgeB": "nope", "cost": 1}]})
    assert not r["ok"] and "rules" in r["fields"]
    assert r["locations"][0]["row"] == 0

    # non-adjacent pair at the node
    r = run_audit({**base, "rules": [
        {"node": "A", "edgeA": "e1", "edgeB": "e2", "cost": 1}]})
    assert not r["ok"] and "相邻" in r["error"]

    # bad cost
    r = run_audit({**base, "rules": [
        {"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": "x!"}]})
    assert not r["ok"] and "非负整数" in r["error"]

    # malformed rules container
    r = run_audit({**base, "rules": {"node": "B"}})
    assert not r["ok"]


def test_transfer_edge_limit_16():
    nodes = [chr(65 + i) for i in range(8)]
    chain = "ABCDEFGH"
    edges = [{"id": f"e{i}", "u": a, "v": b, "length": 1}
             for i, (a, b) in enumerate(zip(chain, chain[1:]))]
    for i in range(10):
        edges.append({"id": f"q{i}", "u": nodes[i % 8],
                      "v": nodes[(i + 3) % 8], "length": 1})
    assert len(edges) == 17
    r = run_audit({"nodes": nodes, "edges": edges, "start": "A",
                   "transferMode": True, "rules": []})
    assert not r["ok"] and "16" in r["error"]
    # normal mode remains compatible with the larger network
    assert run_audit({"nodes": nodes, "edges": edges, "start": "A"})["ok"]


def test_normal_mode_fields_unchanged():
    r = run_audit(K4)
    # mode is additive; all historical fields keep their meaning
    assert r["mode"] == "normal"
    for key in ("totalLength", "addedLength", "optimalCount", "canonicalVector",
                "canonicalEdges", "route", "positions", "eulerian", "oddVertices"):
        assert key in r
    assert "walkLength" not in r and "rules" not in r
