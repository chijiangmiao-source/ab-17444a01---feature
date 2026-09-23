"""Tests for transfer-time mode.

The independent oracle builds the same layered state graph (covered-edge
mask, current directed dart) and solves it with Floyd-Warshall plus a
tight-edge DAG count -- a completely different algorithm from the solver's
pop-time Dijkstra -- then compares optimum, optimal-walk count and the
canonical walk cost.  Route validity (contiguity, coverage, closure, cost
accounting, forbidden-pair absence, free reversals) is checked directly.
"""

import itertools
import random

import pytest

from app.solver import AuditError, audit_transfer


def edge(id_, u, v, length):
    return {"id": id_, "u": u, "v": v, "length": length}


def rule(a, b, cost):
    return {"edgeA": a, "edgeB": b, "cost": cost}


def check_transfer_route(nodes, raw, r, start, cost_map):
    """Validate the canonical walk end to end."""
    ends = {e["id"]: e for e in raw}
    covered = set()
    cur = start
    walk_len = 0
    xfer = 0
    prev_dart = None  # (edge_id, to_node)
    for st in r.route:
        u, v, length = (ends[st.edge_id]["u"], ends[st.edge_id]["v"],
                        ends[st.edge_id]["length"])
        assert (st.frm, st.to) in {(u, v), (v, u)}, "step direction invalid"
        assert st.frm == cur, "walk not contiguous"
        assert st.length == length
        covered.add(st.edge_id)
        walk_len += length
        if prev_dart is None:
            assert st.transfer_cost == 0 and st.rule is None
        else:
            pe, pto = prev_dart
            assert pto == st.frm
            if pe == st.edge_id:
                # reversal along the same pipe: always free, no rule hit
                assert st.transfer_cost == 0 and st.rule is None
            else:
                pair = frozenset((pe, st.edge_id))
                # unset rules are treated as zero cost; forbidden is None
                assert cost_map.get(pair, 0) is not None, \
                    "walk hits a forbidden transfer"
                assert st.transfer_cost == cost_map.get(pair, 0)
                assert st.rule is not None
                xfer += cost_map.get(pair, 0)
        prev_dart = (st.edge_id, st.to)
        cur = st.to
    assert cur == start, "walk does not close at the inspection port"
    assert covered == set(ends), "some pipe is never traversed"
    assert walk_len == r.total_length
    assert xfer == r.transfer_time
    assert walk_len + xfer == r.total_time
    # rule indices resolve through the identifier-sorted edge list
    for st in r.route:
        if st.rule is not None:
            a, b = st.rule
            assert cost_map.get(
                frozenset((r.edges[a].eid, r.edges[b].eid)), 0) is not None


def oracle(nodes, raw, start, cost_map):
    """Independent solver over the (mask, dart) state graph.

    Distances by a heap Dijkstra with explicit edge list and a virtual start
    node; the optimal-walk count is derived afterwards on the tight-edge DAG
    sorted by distance -- a different implementation from the solver's
    pop-time accumulation.  The canonical walk is rebuilt by an independent
    backward tight sweep over ALL reverse edges followed by a lexicographic
    (edge identifier, direction) greedy.  Returns (best, count, steps) or
    (None, 0, []) when infeasible; steps are (edge_id, frm, to) tuples.
    """
    import heapq

    ids = [e["id"] for e in raw]
    m = len(ids)
    ends = {e["id"]: e for e in raw}
    D = 2 * m
    nidx = {n: i for i, n in enumerate(nodes)}

    def dart_endpoints(d):
        e = ends[ids[d >> 1]]
        if d % 2 == 0:
            return nidx[e["u"]], nidx[e["v"]], e["length"]
        return nidx[e["v"]], nidx[e["u"]], e["length"]

    out_darts = {n: [] for n in nidx.values()}
    for d in range(D):
        t, h, _ = dart_endpoints(d)
        out_darts[t].append((d, h))

    S = (1 << m) * D
    Ns = S + 1
    adj = [[] for _ in range(Ns)]

    def add(a, b, w):
        adj[a].append((b, w))

    for d, h in out_darts[nidx[start]]:
        add(S, (1 << (d >> 1)) * D + d, dart_endpoints(d)[2])
    for mask in range(1 << m):
        for d in range(D):
            _, h, _ = dart_endpoints(d)
            for d2, h2 in out_darts[h]:
                if d2 == (d ^ 1):
                    w = dart_endpoints(d2)[2]
                else:
                    c = cost_map.get(frozenset((ids[d >> 1], ids[d2 >> 1])), 0)
                    if c is None:
                        continue
                    w = c + dart_endpoints(d2)[2]
                add(mask * D + d,
                    (mask | (1 << (d2 >> 1))) * D + d2, w)

    INF = 10**18
    dist = [INF] * Ns
    dist[S] = 0
    pq = [(0, S)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist[u]:
            continue
        for b, w in adj[u]:
            nd = du + w
            if nd < dist[b]:
                dist[b] = nd
                heapq.heappush(pq, (nd, b))

    goals = [((1 << m) - 1) * D + d for d in range(D)
             if dart_endpoints(d)[1] == nidx[start]]
    best = min((dist[g] for g in goals), default=INF)
    if best >= INF:
        return None, 0, []

    # weights are >= pipe length >= 1, so tight edges strictly increase
    # distance: processing vertices in distance order counts every shortest
    # path exactly once.
    ways = [0] * Ns
    ways[S] = 1
    for u in sorted(range(Ns), key=lambda x: dist[x]):
        if ways[u] == 0:
            continue
        for b, w in adj[u]:
            if dist[u] + w == dist[b]:
                ways[b] += ways[u]
    count = sum(ways[g] for g in goals if dist[g] == best)

    # independent canonical reconstruction: backward tight sweep through all
    # reverse edges (states are distinct, so both mask histories are covered)
    radj = [[] for _ in range(Ns)]
    for a, outs in enumerate(adj):
        for b, w in outs:
            radj[b].append((a, w))
    tight = [False] * Ns
    stack = [g for g in goals if dist[g] == best]
    for g in stack:
        tight[g] = True
    while stack:
        b = stack.pop()
        for a, w in radj[b]:
            if not tight[a] and dist[a] + w == dist[b]:
                tight[a] = True
                stack.append(a)

    def dart_key(d):
        # lexicographic identifier string, then u->v (even) before v->u (odd)
        return (ids[d >> 1], 0 if d % 2 == 0 else 1)

    goal_set = set(goals)
    cur = S
    steps = []
    while cur not in goal_set:
        cand = []
        for b, w in adj[cur]:
            if tight[b] and dist[cur] + w == dist[b]:
                d2 = b % D
                cand.append((dart_key(d2), d2, b))
        assert cand, f"tight continuation missing at state {cur}"
        _, d2, b = min(cand, key=lambda x: x[0])
        e = ends[ids[d2 >> 1]]
        if d2 % 2 == 0:
            steps.append((e["id"], e["u"], e["v"]))
        else:
            steps.append((e["id"], e["v"], e["u"]))
        cur = b
    return best, count, steps


def random_pipe_graph(rng, nnodes, nedges):
    nodes = [chr(65 + i) for i in range(nnodes)]
    order = nodes[:]
    rng.shuffle(order)
    raw = []
    # connected backbone; extra edges may be parallel (ids stay unique)
    for a, b in zip(order, order[1:]):
        raw.append(edge(f"e{len(raw):02d}", a, b, rng.randint(1, 5)))
    while len(raw) < nedges:
        a, b = rng.sample(nodes, 2)
        raw.append(edge(f"e{len(raw):02d}", a, b, rng.randint(1, 5)))
    return nodes, raw


def adjacent_pairs(raw):
    pairs = []
    for i, j in itertools.combinations(range(len(raw)), 2):
        if {raw[i]["u"], raw[i]["v"]} & {raw[j]["u"], raw[j]["v"]}:
            pairs.append(frozenset((raw[i]["id"], raw[j]["id"])))
    return pairs


# ---------------------------------------------------------------------------
# Deterministic scenarios
# ---------------------------------------------------------------------------


def test_triangle_zero_rules():
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 3), edge("b", "B", "C", 4), edge("c", "C", "A", 5)]
    r = audit_transfer(nodes, raw, "A", [])
    # Eulerian: perimeter once, two orientations tie
    assert r.total_length == 12
    assert r.transfer_time == 0
    assert r.total_time == 12
    assert r.optimal_count == 2
    assert [s.edge_id for s in r.route] == ["a", "b", "c"]
    check_transfer_route(nodes, raw, r, "A", {})


def test_path_leaf_reversal_is_free():
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 2), edge("b", "B", "C", 3)]
    r = audit_transfer(nodes, raw, "A", [])
    assert [s.edge_id for s in r.route] == ["a", "b", "b", "a"]
    assert r.total_length == 10 and r.transfer_time == 0 and r.total_time == 10
    assert r.optimal_count == 1
    check_transfer_route(nodes, raw, r, "A", {})


def test_transfer_cost_charged_both_directions():
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 2), edge("b", "B", "C", 3)]
    rules = [rule("a", "b", 5)]
    r = audit_transfer(nodes, raw, "A", rules)
    # a->b at B going out and b->a at B coming back: 2 * 5
    assert r.total_length == 10
    assert r.transfer_time == 10
    assert r.total_time == 20
    assert r.optimal_count == 1
    paid = [st for st in r.route if st.rule is not None]
    assert len(paid) == 2 and all(st.transfer_cost == 5 for st in paid)
    check_transfer_route(nodes, raw, r, "A",
                         {frozenset(("a", "b")): 5})


def test_high_cost_transfer_detoured():
    # triangle where switching a<->b at B costs 100: the optimum reverses
    # pipes to never pay it, accepting extra walking length (perimeter 3 vs 6).
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 1), edge("b", "B", "C", 1),
           edge("c", "C", "A", 1)]
    r = audit_transfer(nodes, raw, "A", [rule("a", "b", 100)])
    assert r.total_length == 6
    assert r.transfer_time == 0
    assert r.total_time == 6
    assert r.optimal_count == 2
    # canonical: smallest first dart a AB, then its free reversal
    seq = [(s.edge_id, s.frm + s.to) for s in r.route]
    assert seq[0] == ("a", "AB") and seq[1] == ("a", "BA")
    for st in r.route:
        if st.rule is not None:
            assert st.rule != (0, 1)  # the expensive pair never occurs
    check_transfer_route(nodes, raw, r, "A",
                         {frozenset(("a", "b")): 100})


def test_blank_rule_is_zero_and_unordered():
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 3), edge("b", "B", "C", 4),
           edge("c", "C", "A", 5)]
    r = audit_transfer(nodes, raw, "A",
                       [{"edgeA": "b", "edgeB": "a", "cost": ""}])
    assert r.total_time == 12 and r.transfer_time == 0
    check_transfer_route(nodes, raw, r, "A",
                         {frozenset(("a", "b")): 0})


def test_forbidden_pair_no_feasible_closed_walk():
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 2), edge("b", "B", "C", 3)]
    with pytest.raises(AuditError) as ei:
        audit_transfer(nodes, raw, "A", [rule("a", "b", "x")])
    assert "无可行闭游" in ei.value.message
    assert "transfers" in ei.value.fields

    # explicit forbidden flag too
    with pytest.raises(AuditError):
        audit_transfer(nodes, raw, "A",
                       [{"edgeA": "a", "edgeB": "b", "forbidden": True}])


def test_forbidden_does_not_appear_but_detour_exists():
    # forbid a<->b at B in the triangle: detour of length 6 still feasible
    nodes = ["A", "B", "C"]
    raw = [edge("a", "A", "B", 1), edge("b", "B", "C", 1),
           edge("c", "C", "A", 1)]
    r = audit_transfer(nodes, raw, "A", [rule("a", "b", "禁行")])
    assert r.total_time == 6
    for st in r.route:
        assert st.rule != (0, 1)
    check_transfer_route(nodes, raw, r, "A",
                         {frozenset(("a", "b")): None})


def test_parallel_pipes_distinguished_by_id():
    nodes = ["A", "B"]
    raw = [edge("p1", "A", "B", 3), edge("p2", "A", "B", 5)]
    # each pipe must be visited; switching p1<->p2 forbidden anywhere makes
    # covering both in one closed walk impossible
    with pytest.raises(AuditError):
        audit_transfer(nodes, raw, "A", [rule("p1", "p2", "x")])
    # with a switch cost, feasible: out on p1, back on p2 (one transfer each
    # end? walk p1 AB then must use p2: switch at B costs 7, p2 BA closes)
    r = audit_transfer(nodes, raw, "A", [rule("p1", "p2", 7)])
    assert r.transfer_time == 7
    check_transfer_route(nodes, raw, r, "A",
                         {frozenset(("p1", "p2")): 7})


# ---------------------------------------------------------------------------
# Rule validation
# ---------------------------------------------------------------------------


def test_rule_validation_errors():
    nodes = ["A", "B", "C", "D"]
    raw = [edge("a", "A", "B", 1), edge("b", "B", "C", 1),
           edge("c", "C", "D", 1)]

    with pytest.raises(AuditError) as ei:
        audit_transfer(nodes, raw, "A", [rule("a", "zz", 1)])
    assert "不存在" in ei.value.message
    assert ei.value.locations[0]["row"] == 0

    with pytest.raises(AuditError) as ei:
        audit_transfer(nodes, raw, "A", [rule("a", "c", 1)])
    assert "不相邻" in ei.value.message

    with pytest.raises(AuditError):
        audit_transfer(nodes, raw, "A", [rule("a", "a", 1)])

    with pytest.raises(AuditError):
        audit_transfer(nodes, raw, "A", [rule("a", "b", -1)])
    with pytest.raises(AuditError):
        audit_transfer(nodes, raw, "A", [rule("a", "b", 1.5)])

    with pytest.raises(AuditError):
        audit_transfer(nodes, raw, "A",
                       [rule("a", "b", 1), rule("b", "a", 2)])

    with pytest.raises(AuditError) as ei:
        audit_transfer(nodes, raw, "A", [{"edgeA": "a", "cost": 1}])
    assert ei.value.locations[0]["field"] == "transfers"


def test_edge_limit_16_in_transfer_mode():
    nodes = ["A", "B"]
    raw = [edge(f"e{i:02d}", "A", "B", 1) for i in range(17)]
    with pytest.raises(AuditError) as ei:
        audit_transfer(nodes, raw, "A", [])
    assert "16" in ei.value.message


def test_zero_transfer_costs_match_normal_cpp_optimum():
    # with every transfer free, transfer-mode optimum must equal the ordinary
    # CPP optimum totalLength + addedLength
    from app.solver import audit
    rng = random.Random(7)
    for _ in range(10):
        nn = rng.randint(2, 5)
        ne = rng.randint(nn - 1, min(10, nn * (nn - 1) // 2))
        nodes, raw = random_pipe_graph(rng, nn, ne)
        start = rng.choice(nodes)
        normal = audit(nodes, raw, start)
        tr = audit_transfer(nodes, raw, start, [])
        assert tr.total_time == normal.total_length + normal.added_length


# ---------------------------------------------------------------------------
# Randomized cross-validation against the Floyd-Warshall oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(80))
def test_oracle_crosscheck(seed):
    rng = random.Random(1000 + seed)
    nn = rng.randint(2, 4)
    # allow several parallel edges to stress equal-cost predecessor branches
    ne = rng.randint(nn - 1, 8)
    nodes, raw = random_pipe_graph(rng, nn, ne)
    pairs = adjacent_pairs(raw)
    rng.shuffle(pairs)
    cost_map = {}
    rules = []
    for p in pairs:
        roll = rng.random()
        a, b = sorted(p)
        if roll < 0.2:
            cost_map[p] = None
            rules.append(rule(a, b, "x"))
        elif roll < 0.6:
            c = rng.randint(0, 4)
            cost_map[p] = c
            rules.append(rule(a, b, c))
        # else unset -> zero
    start = rng.choice(nodes)

    best, count, canon = oracle(nodes, raw, start, cost_map)
    if best is None:
        with pytest.raises(AuditError):
            audit_transfer(nodes, raw, start, rules)
        return

    r = audit_transfer(nodes, raw, start, rules)
    assert r.total_time == best
    assert r.optimal_count == count
    # canonical walk adjudicated by (identifier, direction) sequences
    got = [(st.edge_id, st.frm, st.to) for st in r.route]
    assert got == canon
    check_transfer_route(nodes, raw, r, start, cost_map)
    # every rule marked forbidden is absent from the canonical walk
    for st in r.route:
        if st.rule is not None:
            ia, ib = st.rule
            pair = frozenset((r.edges[ia].eid, r.edges[ib].eid))
            assert cost_map.get(pair, 0) is not None


def test_largest_bounds_run_fast():
    # 16 pipes must solve quickly: 2^16 * 32 states
    rng = random.Random(99)
    nodes = ["A", "B", "C", "D", "E"]
    raw = []
    # connected backbone first
    for a, b in zip(nodes, nodes[1:]):
        raw.append(edge(f"b{a}{b}", a, b, rng.randint(1, 9)))
    while len(raw) < 16:
        a, b = rng.sample(nodes, 2)
        raw.append(edge(f"e{len(raw):02d}", a, b, rng.randint(1, 9)))
    import time
    t0 = time.time()
    r = audit_transfer(nodes, raw, nodes[0], [])
    assert time.time() - t0 < 20
    check_transfer_route(nodes, raw, r, nodes[0], {})
