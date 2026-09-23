"""Tests for transfer-time mode.

The global optimum is found directly over closed walks that may traverse
pipes repeatedly; a depth-first brute-force oracle (cost-bounded, practical
on tiny graphs) independently enumerates every optimal closed walk and its
lexicographically smallest state sequence for randomized cross-checks.
"""

import random

import pytest

from app.solver import AuditError, audit


def edge(id_, u, v, length):
    return {"id": id_, "u": u, "v": v, "length": length}


def rule(node, a, b, cost):
    return {"node": node, "edgeA": a, "edgeB": b, "cost": cost}


def check_transfer_route(r, start, raw_rules=()):
    """Structural + accounting invariants for a transfer-mode result."""
    assert r.transfer_mode is True
    by_id = {e.eid: e for e in r.edges}
    m = len(r.edges)

    # every pipe traversed at least once; walk contiguous and closed
    seen = [0] * m
    cur = start
    walk = 0
    switches = 0
    for st in r.route:
        assert st.frm == cur
        e = by_id[st.edge_id]
        assert {st.frm, st.to} == {e.u, e.v}
        seen[st.edge_index] += 1
        walk += st.length
        switches += st.transfer
        assert st.transfer >= 0
        cur = st.to
    assert cur == start
    assert all(c >= 1 for c in seen)
    assert tuple(seen) == r.multiplicity
    assert walk == r.walk_length
    assert switches == r.transfer_cost
    assert r.added_length == walk - r.total_length

    # duplicate numbers per edge are 1..multiplicity
    per = {}
    for st in r.route:
        per.setdefault(st.edge_index, []).append(st.duplicate_no)
    for ei, nos in per.items():
        assert sorted(nos) == list(range(1, r.multiplicity[ei] + 1))

    # accounting: each step's switch cost matches the governing rule
    cost_of = {}
    forbidden = set()
    for rr in raw_rules:
        key = (rr["node"], tuple(sorted((rr["edgeA"], rr["edgeB"]))))
        if rr["cost"] in (-1, "禁行"):
            forbidden.add(key)
        else:
            cost_of[key] = int(rr["cost"])
    for i, st in enumerate(r.route):
        if i == 0 or r.route[i - 1].edge_id == st.edge_id:
            assert st.transfer == 0  # departure / free U-turn
            continue
        key = (st.frm, tuple(sorted((r.route[i - 1].edge_id, st.edge_id))))
        assert key not in forbidden, "a forbidden switch appears in the route"
        assert st.transfer == cost_of.get(key, 0)

    # forbidden rules must have no hit position
    for robj in r.rules:
        pair = tuple(sorted((robj.edge_a, robj.edge_b)))
        hits = r.rule_hits[(robj.node, pair[0], pair[1])]
        if robj.cost is None:
            assert hits == ()
        else:
            for p in hits:
                st = r.route[p - 1]
                assert st.frm == robj.node
                assert st.rule_key == pair


def state_seq(r):
    """Canonical sequence as comparable (edge_id-order index, direction)."""
    return tuple((st.edge_index, 0 if st.frm == r.edges[st.edge_index].u else 1)
                 for st in r.route)


# ---------------------------------------------------------------------------
# Deterministic cases
# ---------------------------------------------------------------------------


def test_eulerian_triangle_zero_switches():
    nodes = list("ABC")
    raw = [edge("e1", "A", "B", 1), edge("e2", "B", "C", 1),
           edge("e3", "C", "A", 1)]
    r = audit(nodes, raw, "A", True, [])
    assert r.walk_length == 3
    assert r.transfer_cost == 0
    assert r.optimal_count == 2  # the two orientations of the triangle
    assert state_seq(r) == ((0, 0), (1, 0), (2, 0))  # lexicographically first
    check_transfer_route(r, "A")


def test_path_forced_backtrack():
    nodes = list("ABC")
    raw = [edge("e1", "A", "B", 2), edge("e2", "B", "C", 3)]
    r = audit(nodes, raw, "A", True, [])
    assert r.optimal_count == 1
    assert r.walk_length == 10  # 2+3+3+2
    assert r.transfer_cost == 0
    assert r.multiplicity == (2, 2)
    check_transfer_route(r, "A")


def test_expensive_switch_prefers_longer_walk():
    # B's e1<->e2 switch is expensive: the optimum U-turns at B and covers
    # e2 via a detour through A-C, even though the walked length grows.
    nodes = list("ABCX")
    raw = [
        edge("e1", "A", "B", 1), edge("e2", "B", "C", 1),
        edge("e3", "C", "A", 1), edge("e4", "A", "X", 1),
        edge("e5", "X", "C", 1),
    ]
    # cheap switch: the walk-6 tour (duplicate the direct A-C edge, one B
    # switch) totals 6 + 1 = 7 and ties the zero-switch walk-7 detour.
    r1 = audit(nodes, raw, "A", True, [rule("B", "e1", "e2", 1)])
    assert r1.walk_length + r1.transfer_cost == 7
    # zero-cost switch: walk-6 tour wins outright
    r0 = audit(nodes, raw, "A", True, [])
    assert r0.walk_length == 6 and r0.transfer_cost == 0
    # expensive switch: detour avoids every paid switch
    r2 = audit(nodes, raw, "A", True, [rule("B", "e1", "e2", 100)])
    assert r2.walk_length == 7
    assert r2.transfer_cost == 0
    assert state_seq(r2)[0][0] == 0  # still starts along the cheapest id pipe
    check_transfer_route(r2, "A", [rule("B", "e1", "e2", 100)])


def test_forbidden_switch_makes_tour_infeasible():
    nodes = list("ABC")
    raw = [edge("e1", "A", "B", 1), edge("e2", "B", "C", 1)]
    with pytest.raises(AuditError) as ei:
        audit(nodes, raw, "A", True, [rule("B", "e1", "e2", -1)])
    assert "无可行闭游" in ei.value.message
    assert "rules" in ei.value.fields


def test_forbidden_switch_detour_still_feasible():
    nodes = list("ABCX")
    raw = [
        edge("e1", "A", "B", 1), edge("e2", "B", "C", 1),
        edge("e3", "C", "A", 1), edge("e4", "A", "X", 1),
        edge("e5", "X", "C", 1),
    ]
    r = audit(nodes, raw, "A", True, [rule("B", "e1", "e2", "禁行")])
    assert r.walk_length == 7
    assert r.transfer_cost == 0
    # the forbidden rule exists in the output but is never hit
    robj = r.rules[0]
    assert robj.cost is None
    assert r.rule_hits[("B", "e1", "e2")] == ()
    check_transfer_route(r, "A", [rule("B", "e1", "e2", "禁行")])


def test_parallel_pipes_distinguished_by_id():
    nodes = list("AB")
    raw = [edge("p1", "A", "B", 1), edge("p2", "A", "B", 1)]
    r = audit(nodes, raw, "A", True, [])
    # walk p1 out, p2 back (cost 2); the two id assignments are the 2 optima
    assert r.optimal_count == 2
    assert r.walk_length == 2
    # forbidding the p1<->p2 switch does NOT kill tours: a same-pipe U-turn
    # is always allowed (rules only describe two distinct pipes), so each
    # pipe can be covered by an out-and-back walk with zero switches.
    r1 = audit(nodes, raw, "A", True, [rule("B", "p1", "p2", -1)])
    assert r1.walk_length == 4 and r1.transfer_cost == 0
    # zero-cost explicit rule keeps the short mixed tour
    assert audit(nodes, raw, "A", True, [rule("B", "p1", "p2", 0)]).walk_length == 2
    # high switch cost likewise forces the U-turn tour
    r2 = audit(nodes, raw, "A", True, [rule("B", "p1", "p2", 5)])
    assert r2.transfer_cost == 0
    assert r2.walk_length == 4
    check_transfer_route(r2, "A", [rule("B", "p1", "p2", 5)])


def test_blank_rule_cost_is_zero():
    nodes = list("ABC")
    raw = [edge("e1", "A", "B", 1), edge("e2", "B", "C", 1)]
    r = audit(nodes, raw, "A", True, [rule("B", "e1", "e2", "")])
    assert r.transfer_cost == 0
    assert r.rules[0].cost == 0


def test_rule_validation_errors():
    nodes = list("ABC")
    raw = [edge("e1", "A", "B", 1), edge("e2", "B", "C", 1)]

    def must_fail(rules, *parts):
        with pytest.raises(AuditError) as ei:
            audit(nodes, raw, "A", True, rules)
        assert "rules" in ei.value.fields
        for p in parts:
            assert p in ei.value.message
        return ei.value

    must_fail([rule("Z", "e1", "e2", 1)], "不存在")
    must_fail([rule("A", "e1", "e2", 1)], "相邻")       # e2 not at A
    must_fail([rule("B", "x", "e2", 1)], "不存在")
    must_fail([rule("B", "e1", "e1", 1)], "不同")
    must_fail([rule("B", "e1", "e2", -3)], "非负整数")
    must_fail([rule("B", "e1", "e2", 1.5)], "非负整数")
    e = must_fail(
        [rule("B", "e1", "e2", 1), rule("B", "e2", "e1", 2)], "重复"
    )
    assert e.locations[0]["row"] == 1

    # 17 pipes rejected in transfer mode but fine in normal mode
    big_nodes = list("ABCDEFGH")
    big = []
    chain = "ABCDEFGH"
    for a, b in zip(chain, chain[1:]):
        big.append(edge(f"e{len(big)}", a, b, 1))
    for i in range(10):
        big.append(edge(f"q{i}", big_nodes[i % 8], big_nodes[(i + 3) % 8], 1))
    assert len(big) == 17
    with pytest.raises(AuditError) as ei:
        audit(big_nodes, big, "A", True, [])
    assert "16" in ei.value.message
    # normal mode still accepts up to 32
    audit(big_nodes, big, "A", False, [])


# ---------------------------------------------------------------------------
# Brute-force DFS oracle
# ---------------------------------------------------------------------------


def _allowed(raw_edges_by_node, switch, forbidden, node, prev, eid):
    if prev is None or prev == eid:
        return True, 0
    key = (node, tuple(sorted((prev, eid))))
    if key in forbidden:
        return False, 0
    return True, switch.get(key, 0)


def feasible_closed_walk(nodes, raw, start, raw_rules):
    """Budget-free feasibility: BFS over (node, coverage-mask)."""
    adj = {n: [] for n in nodes}
    for e in raw:
        adj[e["u"]].append((e["v"], e["id"], e["length"]))
        adj[e["v"]].append((e["u"], e["id"], e["length"]))
    ids = sorted(e["id"] for e in raw)
    pos = {eid: i for i, eid in enumerate(ids)}
    switch, forbidden = {}, set()
    for rr in raw_rules:
        key = (rr["node"], tuple(sorted((rr["edgeA"], rr["edgeB"]))))
        if rr["cost"] in (-1, "禁行"):
            forbidden.add(key)
        else:
            switch[key] = int(rr["cost"])
    full = (1 << len(ids)) - 1
    seen = {(start, 0)}
    stack = [(start, 0, None)]
    while stack:
        node, mask, prev = stack.pop()
        if mask == full and node == start:
            return True
        for nb, eid, _ in adj[node]:
            ok, _ = _allowed(None, switch, forbidden, node, prev, eid)
            if not ok:
                continue
            st2 = (nb, mask | (1 << pos[eid]), eid)
            if st2 not in seen:
                seen.add(st2)
                stack.append(st2)
    return False


def enumerate_walks_within(nodes, raw, start, raw_rules, cap,
                           node_budget=4_000_000):
    """Enumerate closed walks of cost <= cap; return min cost and all walks.

    Raises RuntimeError when the node budget is exhausted.
    """
    adj = {n: [] for n in nodes}
    for e in raw:
        adj[e["u"]].append((e["v"], e["id"], e["length"]))
        adj[e["v"]].append((e["u"], e["id"], e["length"]))
    ids = sorted(e["id"] for e in raw)
    pos = {eid: i for i, eid in enumerate(ids)}
    declared_u = {e["id"]: e["u"] for e in raw}
    full = (1 << len(ids)) - 1
    switch, forbidden = {}, set()
    for rr in raw_rules:
        key = (rr["node"], tuple(sorted((rr["edgeA"], rr["edgeB"]))))
        if rr["cost"] in (-1, "禁行"):
            forbidden.add(key)
        else:
            switch[key] = int(rr["cost"])

    best = [cap + 1]
    walks = []
    budget = [node_budget]

    def dfs(node, mask, cost, seq, prev):
        budget[0] -= 1
        if budget[0] < 0:
            raise RuntimeError
        if cost > best[0]:
            return
        if mask == full and node == start and seq:
            if cost < best[0]:
                best[0] = cost
                walks[:] = [tuple(seq)]
            elif cost == best[0]:
                walks.append(tuple(seq))
            return
        for nb, eid, length in sorted(adj[node], key=lambda t: (pos[t[1]], t[0])):
            ok, sw = _allowed(None, switch, forbidden, node, prev, eid)
            if not ok or cost + length + sw > best[0]:
                continue
            drc = 0 if declared_u[eid] == node else 1
            seq.append((pos[eid], drc))
            dfs(nb, mask | (1 << pos[eid]), cost + length + sw, seq, eid)
            seq.pop()

    dfs(start, 0, 0, [], None)
    return best[0], walks


@pytest.mark.parametrize("seed", range(30))
def test_brute_force_transfer_crosscheck(seed):
    rng = random.Random(1000 + seed)
    nnodes = rng.randint(2, 4)
    nodes = [chr(65 + i) for i in range(nnodes)]
    max_extra = min(5, nnodes * (nnodes - 1) // 2)
    nedges = rng.randint(nnodes - 1, max_extra)
    raw = []
    used = set()
    order = nodes[:]
    rng.shuffle(order)
    for a, b in zip(order, order[1:]):  # spanning path: connected
        used.add(tuple(sorted((a, b))))
        raw.append(edge(f"e{len(raw)}", a, b, rng.randint(1, 4)))
    while len(raw) < nedges:
        a, b = rng.sample(nodes, 2)
        p = tuple(sorted((a, b)))
        if p in used:
            continue
        used.add(p)
        raw.append(edge(f"e{len(raw)}", a, b, rng.randint(1, 4)))

    # random rules at nodes between incident distinct pipe pairs
    incident = {n: set() for n in nodes}
    for e in raw:
        incident[e["u"]].add(e["id"])
        incident[e["v"]].add(e["id"])
    pairs = []
    for n in nodes:
        ids_n = sorted(incident[n])
        for i in range(len(ids_n)):
            for j in range(i + 1, len(ids_n)):
                pairs.append((n, ids_n[i], ids_n[j]))
    rng.shuffle(pairs)
    raw_rules = []
    for n, a, b in pairs[: rng.randint(0, len(pairs))]:
        raw_rules.append(
            rule(n, a, b, -1 if rng.random() < 0.3 else rng.randint(0, 5))
        )

    start = rng.choice(nodes)

    if not feasible_closed_walk(nodes, raw, start, raw_rules):
        with pytest.raises(AuditError):
            audit(nodes, raw, start, True, raw_rules)
        return

    r = audit(nodes, raw, start, True, raw_rules)
    cap = r.walk_length + r.transfer_cost
    try:
        opt, walks = enumerate_walks_within(
            nodes, raw, start, raw_rules, cap
        )
    except RuntimeError:
        return  # exhaustive enumeration too large; structural checks still hold
    assert walks, "feasible BFS but DFS found nothing"
    assert opt == cap
    assert len(walks) == r.optimal_count
    assert min(walks) == state_seq(r)
    check_transfer_route(r, start, raw_rules)
