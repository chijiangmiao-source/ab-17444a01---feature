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


TRI = {
    "nodes": ["A", "B", "C"],
    "edges": [
        {"id": "a", "u": "A", "v": "B", "length": 1},
        {"id": "b", "u": "B", "v": "C", "length": 1},
        {"id": "c", "u": "C", "v": "A", "length": 1},
    ],
    "start": "A",
}


def test_transfer_high_cost_detour_payload():
    payload = dict(TRI, transferMode=True,
                   transfers=[{"edgeA": "a", "edgeB": "b", "cost": 100}])
    r = run_audit(payload)
    assert r["ok"] is True
    assert r["mode"] == "transfer"
    # one lap costs 103; reversing walk costs 6 length and zero transfer
    assert r["walkingLength"] == 6
    assert r["transferTime"] == 0
    assert r["totalTime"] == 6
    assert r["optimalCount"] == 2
    # the expensive rule is never hit
    assert r["rulePositions"].get("0-1", []) == []
    # every route step is contiguous and closes
    assert r["route"][0]["from"] == "A"
    assert r["route"][-1]["to"] == "A"
    for prev, nxt in zip(r["route"], r["route"][1:]):
        assert prev["to"] == nxt["from"]
        if nxt["rule"] is not None:
            assert nxt["transfer"] >= 0


def test_transfer_free_rules_match_one_lap():
    payload = dict(TRI, transferMode=True,
                   transfers=[{"edgeA": "a", "edgeB": "b", "cost": ""}])
    r = run_audit(payload)
    assert r["ok"] and r["totalTime"] == 3
    assert r["walkingLength"] == 3 and r["transferTime"] == 0
    assert len(r["route"]) == 3
    # zero-cost rule echo: rule object present, positions reported
    assert r["rules"][0]["cost"] == 0 and r["rules"][0]["forbidden"] is False


def test_transfer_forbidden_no_feasible_walk():
    # path graph: forbidding the only distinct-edge transfer at B makes a
    # covering closed walk impossible
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "a", "u": "A", "v": "B", "length": 2},
            {"id": "b", "u": "B", "v": "C", "length": 3},
        ],
        "start": "A",
        "transferMode": True,
        "transfers": [{"edgeA": "a", "edgeB": "b", "cost": "x"}],
    }
    r = run_audit(payload)
    assert r["ok"] is False
    assert "无可行闭游" in r["error"]
    assert "transfers" in r["fields"]


def test_transfer_rule_reference_and_adjacency_errors():
    payload = dict(TRI, transferMode=True,
                   transfers=[{"edgeA": "a", "edgeB": "zz", "cost": 1}])
    r = run_audit(payload)
    assert not r["ok"] and "不存在" in r["error"]
    assert r["locations"][0]["field"] == "transfers"
    assert r["locations"][0]["row"] == 0

    path = {
        "nodes": ["A", "B", "C", "D"],
        "edges": [
            {"id": "a", "u": "A", "v": "B", "length": 1},
            {"id": "b", "u": "B", "v": "C", "length": 1},
            {"id": "c", "u": "C", "v": "D", "length": 1},
        ],
        "start": "A",
        "transferMode": True,
        "transfers": [{"edgeA": "a", "edgeB": "c", "cost": 1}],
    }
    r = run_audit(path)
    assert not r["ok"] and "不相邻" in r["error"]

    bad_cost = dict(TRI, transferMode=True,
                    transfers=[{"edgeA": "a", "edgeB": "b", "cost": -2}])
    assert run_audit(bad_cost)["ok"] is False


def test_transfer_edge_limit():
    payload = {
        "nodes": ["A", "B"],
        "edges": [{"id": f"e{i}", "u": "A", "v": "B", "length": 1}
                  for i in range(17)],
        "start": "A",
        "transferMode": True,
        "transfers": [],
    }
    r = run_audit(payload)
    assert r["ok"] is False and "16" in r["error"]


def test_normal_mode_fields_unchanged():
    # existing fields still present and identical; mode is purely additive
    r = run_audit(K4)
    for key in ("ok", "nodes", "edges", "start", "oddVertices", "totalLength",
                "addedLength", "optimalCount", "canonicalVector",
                "canonicalEdges", "route", "positions", "eulerian"):
        assert key in r
    assert r["mode"] == "normal"
    # absent flag / untruthy flag never routes to transfer mode
    r2 = run_audit(dict(K4, transfers=[{"edgeA": "e1", "edgeB": "e2", "cost": 5}]))
    assert r2["mode"] == "normal" and r2["optimalCount"] == 3
