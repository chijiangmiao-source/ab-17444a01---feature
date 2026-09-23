"""Chinese postman / route inspection core.

Undirected multigraph (parallel edges allowed, self loops forbidden) with
positive integer edge lengths.  Finds the minimum added length needed for all
vertices to have even degree, the exact number of optimal duplicate sets, the
canonical duplicate set (0-preferred bit vector in edge order), per-edge
classification, and a closed Euler tour for the canonical augmentation.

Two modes
---------
Normal mode (``transfer_mode=False``) is the classic Chinese postman problem;
see the T-join exposition below.

Transfer-time mode (``transfer_mode=True``) additionally accepts, for every
node, unordered pairs of incident pipes that may be switched between there, a
non-negative integer switch time or a "forbidden" marker; an unspecified pair
costs zero.  The optimum is then found directly in the space of closed walks
that may traverse pipes repeatedly: a walk is a sequence of directed pipe
traversals, and its objective is the sum of traversed pipe lengths plus every
adjacent switch time.  The solver never fixes a minimum-augmentation edge set
first -- a longer walk can be cheaper when a switch is costly.  The mode is
restricted to at most 16 pipes.

Exact counting without enumeration
----------------------------------
A duplicate set is a T-join: in the subgraph formed by the duplicated edges
exactly the originally odd vertices T have odd degree.  T-join theorem:

  * minimum T-join weight = minimum weight of a perfect matching of T under
    the shortest-path metric;
  * every minimum T-join decomposes into edge-disjoint shortest paths whose
    endpoint pairs form such a minimum matching.

For each odd pair (i, j) let A[i][j] be the number of shortest i-j paths
(parallel edges count separately).  Choices for the pairs of a matching are
independent -- two shortest paths of pairs inside one minimum matching cannot
share an edge, because their edge-union would then be a strictly cheaper
T-join.  Hence the number of optimal sets for a matching M is the product of
A[i][j] over its pairs, and different matchings give different sets.  The
total count is obtained by a weighted subset DP without ever enumerating the
sets:

    C[S] = sum over min-cost partners j of the first vertex:
               A[i][j] * C[S \\ {i, j}]

For per-edge classification, let B_e[i][j] be the number of shortest i-j
paths that use edge e (forward/backward shortest-path-count product through
the edge).  A second DP G_e[S] counts optimal sets for subproblem S that
contain e, using B_e for the pair whose path carries e and A - B_e otherwise:

    G_e[S] = sum over min-cost partners j:
               B_e[i][j] * C[S'] + (A[i][j] - B_e[i][j]) * G_e[S']

Edge e is required when G_e[T] == C[T], optional for 0 < G_e[T] < C[T], and
never duplicated when G_e[T] == 0.

The canonical set is built greedily in edge order: bit p is 0 whenever an
optimum T-join still exists with the edges pinned so far, the newly forced
edges toggling the parity target.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

INF = 10**30
TOKEN_RE = re.compile(r"^[!-~]+$")  # printable non-space ASCII


class AuditError(ValueError):
    """Validation failure with page locations."""

    def __init__(
        self,
        message: str,
        fields: Sequence[str] = (),
        locations: Sequence[dict] = (),
    ):
        super().__init__(message)
        self.message = message
        self.fields = tuple(fields)
        self.locations = list(locations)


@dataclass(frozen=True)
class Edge:
    eid: str
    u: str
    v: str
    length: int
    index: int


@dataclass(frozen=True)
class RouteStep:
    edge_index: int
    edge_id: str
    frm: str
    to: str
    length: int
    duplicate_no: int  # which copy of this edge, 1-based, in traversal order
    transfer: int = 0  # switch time paid when entering this pipe
    rule_key: Optional[Tuple[str, str]] = None  # ordered edge ids of the rule


@dataclass(frozen=True)
class TransferRule:
    """A switch rule at one node between two unordered incident pipes."""

    node: str
    edge_a: str
    edge_b: str
    cost: Optional[int]  # None == forbidden
    row: int  # row index in the request, for error locations


@dataclass
class AuditResult:
    nodes: List[str]
    edges: List[Edge]
    start: str
    odd_vertices: Tuple[str, ...]
    components: Tuple[Tuple[str, ...], ...]
    total_length: int
    added_length: int
    optimal_count: int
    canonical_set: FrozenSet[int]
    bit_vector: str
    classification: Dict[int, str]  # required | optional | never
    multiplicity: Tuple[int, ...]
    route: Tuple[RouteStep, ...]
    transfer_mode: bool = False
    transfer_cost: int = 0
    walk_length: int = 0
    rules: Tuple[TransferRule, ...] = ()
    rule_hits: Dict[Tuple[str, ...], Tuple[int, ...]] = field(default_factory=dict)

    @property
    def is_eulerian(self) -> bool:
        return not self.odd_vertices


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_positive_int(value, eid: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    if isinstance(value, int):
        length = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        length = int(value.strip())
    else:
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    if length <= 0:
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    return length


def _as_nonneg_cost(value, where: str):
    """Parse a transfer cell: blank/missing -> 0, -1/'禁行' -> forbidden."""
    if value is None:
        return 0
    if isinstance(value, bool):
        raise AuditError(f"{where}耗时必须为非负整数或禁行", ("rules",))
    if isinstance(value, int):
        if value == -1:
            return None
        cost = value
    elif isinstance(value, str):
        s = value.strip()
        if s == "":
            return 0
        if s in ("-1", "禁行", "禁止", "forbidden", "x", "X"):
            return None
        if not re.fullmatch(r"\d+", s):
            raise AuditError(f"{where}耗时必须为非负整数或禁行", ("rules",))
        cost = int(s)
    else:
        raise AuditError(f"{where}耗时必须为非负整数或禁行", ("rules",))
    if cost < 0:
        raise AuditError(f"{where}耗时必须为非负整数或禁行", ("rules",))
    return cost


def validate_input(
    nodes: Sequence[str],
    raw_edges: Sequence[dict],
    start: Optional[str],
    transfer_mode: bool = False,
    raw_rules=None,
):
    clean_nodes: List[str] = []
    node_rows: Dict[str, int] = {}
    for i, raw in enumerate(nodes):
        name = ("" if raw is None else str(raw)).strip()
        if name == "":
            continue
        loc = [{"field": "nodes", "row": i}]
        if not TOKEN_RE.match(name):
            raise AuditError(
                f"节点 {name!r} 必须为非空白 ASCII 字符", ("nodes",), loc
            )
        if name in node_rows:
            raise AuditError(
                f"节点 {name!r} 重复",
                ("nodes",),
                loc + [{"field": "nodes", "row": node_rows[name]}],
            )
        node_rows[name] = i
        clean_nodes.append(name)

    if not (2 <= len(clean_nodes) <= 18):
        raise AuditError(
            f"唯一节点数量为 {len(clean_nodes)}，必须在 2 至 18 之间",
            ("nodes",),
            [{"field": "nodes"}],
        )

    if not raw_edges:
        raise AuditError("至少需要 1 条管段", ("edges",), [{"field": "edges"}])
    edge_limit = 16 if transfer_mode else 32
    if len(raw_edges) > edge_limit:
        raise AuditError(
            f"管段数量为 {len(raw_edges)}，"
            + ("转接耗时模式下不能超过 16" if transfer_mode else "不能超过 32"),
            ("edges",),
            [{"field": "edges"}],
        )

    edges: List[Edge] = []
    id_rows: Dict[str, int] = {}
    for i, re_ in enumerate(raw_edges):
        loc = [{"field": "edges", "row": i}]
        eid = str(re_.get("id", "") or "").strip()
        if not eid:
            raise AuditError(
                f"第 {i + 1} 条管段缺少唯一标识", ("edges",), loc
            )
        if not TOKEN_RE.match(eid):
            raise AuditError(
                f"管段标识 {eid!r} 必须为非空白 ASCII 字符", ("edges",), loc
            )
        if eid in id_rows:
            raise AuditError(
                f"管段标识 {eid!r} 重复",
                ("edges",),
                loc + [{"field": "edges", "row": id_rows[eid]}],
            )
        id_rows[eid] = i

        u = str(re_.get("u", "") or "").strip()
        v = str(re_.get("v", "") or "").strip()
        if u not in node_rows:
            raise AuditError(
                f"管段 {eid} 的端点 {u or '(空)'} 不是已声明节点",
                ("edges",),
                loc,
            )
        if v not in node_rows:
            raise AuditError(
                f"管段 {eid} 的端点 {v or '(空)'} 不是已声明节点",
                ("edges",),
                loc,
            )
        if u == v:
            raise AuditError(
                f"管段 {eid} 为自环（{u}），禁止自环", ("edges",), loc
            )

        length = _as_positive_int(re_.get("length", None), eid)
        if transfer_mode and length > 10**6:
            raise AuditError(
                f"转接耗时模式下管段长度不能超过 1000000（管段 {eid}）",
                ("edges",),
                loc,
            )
        edges.append(Edge(eid=eid, u=u, v=v, length=length, index=i))

    # The canonical bit vector is ordered by edge *identifier*, so reorder
    # the edges (and their indices) lexicographically now that validation of
    # rows/locations is done.
    edges.sort(key=lambda e: e.eid)
    edges = [
        Edge(eid=e.eid, u=e.u, v=e.v, length=e.length, index=i)
        for i, e in enumerate(edges)
    ]

    start_s = "" if start is None else str(start).strip()
    if not start_s:
        raise AuditError("请选择检修口", ("start",), [{"field": "start"}])
    if start_s not in node_rows:
        raise AuditError(
            f"检修口 {start_s!r} 不存在", ("start",), [{"field": "start"}]
        )

    rules = _validate_rules(
        raw_rules if transfer_mode else (),
        edges,
        node_rows,
    )

    return clean_nodes, edges, rules


def _validate_rules(raw_rules, edges, node_rows) -> Tuple[TransferRule, ...]:
    """Validate switch rules.

    A rule references one declared node and two distinct existing pipe
    identifiers that are both incident to that node (they may be parallel
    pipes).  Unspecified/zero cost is allowed; ``-1`` / "禁行" forbids the
    switch.  Duplicate rules on the same unordered pair at the same node are
    rejected.
    """
    if raw_rules is None:
        return ()
    if not isinstance(raw_rules, (list, tuple)):
        raise AuditError(
            "转接规则必须为列表", ("rules",), [{"field": "rules"}]
        )

    by_id = {e.eid: e for e in edges}
    seen: Dict[Tuple[str, str, str], int] = {}
    rules: List[TransferRule] = []
    for i, rr in enumerate(raw_rules):
        if not isinstance(rr, dict):
            raise AuditError(
                f"第 {i + 1} 条转接规则格式错误",
                ("rules",),
                [{"field": "rules", "row": i}],
            )
        loc = [{"field": "rules", "row": i}]
        node = str(rr.get("node", "") or "").strip()
        ea = str(rr.get("edgeA", rr.get("edge_a", "")) or "").strip()
        eb = str(rr.get("edgeB", rr.get("edge_b", "")) or "").strip()
        raw_cost = rr.get("cost", rr.get("time", None))
        cost_blank = raw_cost is None or (
            isinstance(raw_cost, str) and raw_cost.strip() == ""
        )
        # a wholly empty row is an unused editor row: skip it silently so
        # request row indices still line up with the editor table rows
        if not node and not ea and not eb and cost_blank:
            continue
        where = f"转接规则第 {i + 1} 行（{ea or '?'}↔{eb or '?'} @ {node or '?'}）"

        if not node:
            raise AuditError(f"{where} 缺少节点", ("rules",), loc)
        if node not in node_rows:
            raise AuditError(
                f"{where} 引用的节点 {node!r} 不存在", ("rules",), loc
            )
        if not ea or not eb:
            raise AuditError(f"{where} 缺少管段标识", ("rules",), loc)
        if ea not in by_id:
            raise AuditError(
                f"{where} 引用的管段 {ea!r} 不存在", ("rules",), loc
            )
        if eb not in by_id:
            raise AuditError(
                f"{where} 引用的管段 {eb!r} 不存在", ("rules",), loc
            )
        if ea == eb:
            raise AuditError(
                f"{where} 的两条管段必须不同", ("rules",), loc
            )
        ga, gb = by_id[ea], by_id[eb]
        incident_a = {ga.u, ga.v}
        incident_b = {gb.u, gb.v}
        if node not in incident_a or node not in incident_b:
            raise AuditError(
                f"{where} 中管段 {ea} 与 {eb} 未在节点 {node} 相邻，"
                "转接只能发生在两条管段的公共节点",
                ("rules",),
                loc,
            )

        cost = _as_nonneg_cost(rr.get("cost", rr.get("time", None)), where)
        key = (node, *sorted((ea, eb)))
        if key in seen:
            raise AuditError(
                f"{where} 与第 {seen[key] + 1} 行规则重复（节点与无序管段对相同）",
                ("rules",),
                loc + [{"field": "rules", "row": seen[key]}],
            )
        seen[key] = i
        rules.append(
            TransferRule(node=node, edge_a=ea, edge_b=eb, cost=cost, row=i)
        )

    # The layered DP stores costs in signed 64-bit arrays.  An optimal walk
    # has at most m first visits, each reached after an intra-layer detour of
    # at most 2m-1 arcs; keeping the unit costs bounded leaves ample margin.
    for rule in rules:
        if rule.cost is not None and rule.cost > 10**6:
            raise AuditError(
                f"转接耗时不能超过 1000000（第 {rule.row + 1} 行规则）",
                ("rules",),
                [{"field": "rules", "row": rule.row}],
            )
    return tuple(rules)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def build_nadj(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    forbidden: FrozenSet[int] = frozenset(),
):
    nadj: Dict[str, List[Tuple[str, int, int]]] = {n: [] for n in nodes}
    for e in edges:
        if e.index in forbidden:
            continue
        nadj[e.u].append((e.v, e.index, e.length))
        nadj[e.v].append((e.u, e.index, e.length))
    return nadj


def adjacency(nodes: Sequence[str], edges: Sequence[Edge]):
    adj: Dict[str, List[Tuple[str, int]]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.u].append((e.v, e.index))
        adj[e.v].append((e.u, e.index))
    return adj


def connected_components(nodes: Sequence[str], adj) -> List[List[str]]:
    comps: List[List[str]] = []
    unvisited = set(nodes)
    while unvisited:
        seed = next(iter(unvisited))
        stack = [seed]
        unvisited.discard(seed)
        comp: List[str] = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y, _ in adj[x]:
                if y in unvisited:
                    unvisited.discard(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


def dijkstra(nodes: Sequence[str], nadj, src: str):
    """Shortest distances from src."""
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        for w, _, length in nadj[u]:
            nd = du + length
            if w not in dist or nd < dist[w]:
                dist[w] = nd
                heapq.heappush(pq, (nd, w))
    return dist


def shortest_path_masks(
    nadj,
    src: str,
    dst: str,
    dist: dict,
) -> Tuple[int, ...]:
    """All shortest src->dst paths as integer edge-index masks.

    DFS over the shortest-path DAG; positive edge lengths make it acyclic
    (every predecessor has strictly smaller distance).
    """
    if src == dst:
        return (0,)
    pred: Dict[str, List[Tuple[str, int]]] = {}
    for x, dx in dist.items():
        if x == src:
            continue
        ps = []
        for y, ei, length in nadj[x]:
            if y in dist and dist[y] + length == dx:
                ps.append((y, ei))
        pred[x] = ps

    result: List[int] = []
    cur = 0
    seen_v = {dst}

    def dfs(x: str):
        nonlocal cur
        if x == src:
            result.append(cur)
            return
        for y, ei in pred.get(x, ()):
            if y in seen_v:
                continue
            seen_v.add(y)
            cur |= 1 << ei
            dfs(y)
            cur &= ~(1 << ei)
            seen_v.discard(y)

    dfs(dst)
    return tuple(result)


# ---------------------------------------------------------------------------
# Perfect matching DP over a subset of vertices
# ---------------------------------------------------------------------------


def matching_dp(dist_matrix, labels: Sequence[int]):
    """Minimum matching cost and number of matchings attaining it.

    dist_matrix[i][j] is the shortest-path distance; INF means unreachable.
    Returns (cost array by mask, count array by mask).
    """
    k = len(labels)
    size = 1 << k
    cost = [INF] * size
    count = [0] * size
    cost[0] = 0
    count[0] = 1
    for mask in range(1, size):
        if mask.bit_count() & 1:
            continue
        i = (mask & -mask).bit_length() - 1
        rest0 = mask ^ (1 << i)
        best = INF
        total = 0
        bits = rest0
        while bits:
            jb = bits & -bits
            j = jb.bit_length() - 1
            bits ^= jb
            d = dist_matrix[i][j]
            sub = rest0 ^ jb
            if d >= INF or cost[sub] >= INF:
                continue
            val = d + cost[sub]
            if val < best:
                best = val
                total = count[sub]
            elif val == best:
                total += count[sub]
        cost[mask] = best
        count[mask] = total
    return cost, count


# ---------------------------------------------------------------------------
# Euler circuit (Hierholzer) on the expanded multigraph
# ---------------------------------------------------------------------------


def euler_circuit(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    start: str,
    multiplicity: Sequence[int],
) -> List[RouteStep]:
    copies: List[Tuple[int, str, str, int]] = []
    for e in edges:
        for _ in range(multiplicity[e.index]):
            copies.append((e.index, e.u, e.v, e.length))

    adj: Dict[str, List[int]] = {n: [] for n in nodes}
    for ci, (ei, u, v, _) in enumerate(copies):
        adj[u].append(ci)
        adj[v].append(ci)

    used = [False] * len(copies)
    stack: List[Tuple[str, int]] = [(start, -1)]
    circuit: List[Tuple[str, int]] = []
    while stack:
        x, _ = stack[-1]
        chosen: Optional[int] = None
        for ci in adj[x]:  # incident lists in ascending copy order
            if not used[ci]:
                chosen = ci
                break
        if chosen is None:
            circuit.append(stack.pop())
        else:
            used[chosen] = True
            _, u, v, _ = copies[chosen]
            stack.append((v if x == u else u, chosen))

    circuit.reverse()
    walk_copies = [ci for _, ci in circuit[1:]]

    steps: List[RouteStep] = []
    dup_counter: Dict[int, int] = {}
    cur = start
    for ci in walk_copies:
        ei, u, v, length = copies[ci]
        frm, to = (u, v) if cur == u else (v, u)
        dup_counter[ei] = dup_counter.get(ei, 0) + 1
        steps.append(
            RouteStep(
                edge_index=ei,
                edge_id=edges[ei].eid,
                frm=frm,
                to=to,
                length=length,
                duplicate_no=dup_counter[ei],
            )
        )
        cur = to
    return steps


# ---------------------------------------------------------------------------
# Enumeration of the distinct optimal T-join edge sets
# ---------------------------------------------------------------------------


def enumerate_optimal_tjoins(
    odd: Tuple[str, ...],
    dist_from: Dict[str, dict],
    nadj,
    costdp,
    dist_matrix,
):
    """All distinct minimum T-join masks for the odd vertices.

    dp[mask] is the set of distinct edge masks of optimal T-joins pairing
    exactly the odd vertices in ``mask``.  Anchor the lowest-index vertex i
    and combine a shortest i-j path with an optimal solution of the remaining
    mask via symmetric difference; integer masks deduplicate automatically.
    The matching cost DP gates which partners j can occur in an optimum.
    """
    k = len(odd)
    size = 1 << k
    pair_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def paths_between(a: int, b: int) -> Tuple[int, ...]:
        key = (a, b)
        if key not in pair_cache:
            pair_cache[key] = shortest_path_masks(
                nadj, odd[a], odd[b], dist_from[odd[a]]
            )
        return pair_cache[key]

    dp: List[Optional[set]] = [None] * size
    dp[0] = {0}
    for mask in range(1, size):
        if mask.bit_count() & 1:
            continue
        ib = mask & -mask
        i = ib.bit_length() - 1
        rest0 = mask ^ ib
        target_cost = costdp[mask]
        result: set = set()
        bits = rest0
        while bits:
            jb = bits & -bits
            j = jb.bit_length() - 1
            bits ^= jb
            sub = rest0 ^ jb
            if dist_matrix[i][j] >= INF or costdp[sub] >= INF:
                continue
            if dist_matrix[i][j] + costdp[sub] != target_cost:
                continue
            sub_sets = dp[sub]
            for pmask in paths_between(i, j):
                for base in sub_sets:
                    result.add(pmask ^ base)
        dp[mask] = result
    return dp[size - 1]


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def audit(
    nodes: Sequence[str],
    raw_edges: Sequence[dict],
    start: Optional[str],
    transfer_mode: bool = False,
    raw_rules=None,
) -> AuditResult:
    nodes, edges, rules = validate_input(
        nodes, raw_edges, start, transfer_mode, raw_rules
    )
    adj = adjacency(nodes, edges)

    comps = connected_components(nodes, adj)
    if len(comps) > 1:
        raise AuditError(
            "管网不连通，存在多个连通分量："
            + "；".join("{" + ",".join(c) + "}" for c in comps),
            ("edges", "nodes"),
            [{"field": "edges"}],
        )

    if transfer_mode:
        return audit_transfers(nodes, edges, start, rules, comps)

    degree = {n: 0 for n in nodes}
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
    odd = tuple(sorted(n for n in nodes if degree[n] % 2 == 1))
    total_length = sum(e.length for e in edges)
    m = len(edges)

    if not odd:
        empty: FrozenSet[int] = frozenset()
        multiplicity = tuple(1 for _ in edges)
        route = euler_circuit(nodes, edges, start, multiplicity)
        return AuditResult(
            nodes=nodes,
            edges=edges,
            start=start,
            odd_vertices=odd,
            components=tuple(tuple(c) for c in comps),
            total_length=total_length,
            added_length=0,
            optimal_count=1,
            canonical_set=empty,
            bit_vector="0" * m,
            classification={i: "never" for i in range(m)},
            multiplicity=multiplicity,
            route=tuple(route),
        )

    # shortest distances from every vertex
    nadj = build_nadj(nodes, edges)
    dist_from: Dict[str, dict] = {s: dijkstra(nodes, nadj, s) for s in nodes}

    k = len(odd)
    D = [[INF] * k for _ in range(k)]
    for i, s in enumerate(odd):
        for j, t in enumerate(odd):
            if t in dist_from[s]:
                D[i][j] = dist_from[s][t]

    costdp, _ = matching_dp(D, list(range(k)))
    full = (1 << k) - 1
    optimum = costdp[full]

    # distinct optimal duplicate sets, exact
    opt_masks = enumerate_optimal_tjoins(odd, dist_from, nadj, costdp, D)
    total_count = len(opt_masks)

    # canonical: 0 preferred at the earliest edge index => smallest binary
    # number with edge 0 as most significant bit
    canonical_mask = min(
        opt_masks, key=lambda mm: sum(1 << (m - 1 - i) for i in range(m) if mm >> i & 1)
    )
    bit_vector = "".join("1" if canonical_mask >> i & 1 else "0" for i in range(m))

    in_all = (1 << m) - 1
    in_any = 0
    for mm in opt_masks:
        in_all &= mm
        in_any |= mm
    classification: Dict[int, str] = {}
    for i in range(m):
        if in_all >> i & 1:
            classification[i] = "required"
        elif in_any >> i & 1:
            classification[i] = "optional"
        else:
            classification[i] = "never"

    multiplicity = tuple(1 + (1 if canonical_mask >> i & 1 else 0) for i in range(m))
    route = euler_circuit(nodes, edges, start, multiplicity)

    return AuditResult(
        nodes=nodes,
        edges=edges,
        start=start,
        odd_vertices=odd,
        components=tuple(tuple(c) for c in comps),
        total_length=total_length,
        added_length=int(optimum),
        optimal_count=total_count,
        canonical_set=frozenset(
            i for i in range(m) if canonical_mask >> i & 1
        ),
        bit_vector=bit_vector,
        classification=classification,
        multiplicity=multiplicity,
        route=tuple(route),
    )


# ---------------------------------------------------------------------------
# Transfer-time mode: global optimum directly over closed walks
# ---------------------------------------------------------------------------
#
# A state is one directed pipe traversal d = 2*edge_index + dir (dir 0 is the
# declared u->v orientation, dir 1 v->u).  An arc d -> d' exists exactly when
# d ends at the node d' starts at; its weight is
#
#     length(pipe(d')) + switch_time(pipe(d), pipe(d') at that node)
#
# A closed walk from the maintenance port is a state sequence d1..dt with
# d1 starting at the port, dt ending at the port, every pipe covered at least
# once, and total cost = sum of arc weights.  Repeated traversals of covered
# pipes stay within the same coverage mask; the first traversal of a pipe
# raises the mask bit.  Positive pipe lengths keep every arc strictly
# positive, so within one mask layer the closure is solved by a multi-source
# Dijkstra, and mask layers are processed in numeric order (raising a bit
# always increases the mask).


def _transfer_arc_index(edges: Sequence[Edge], rules: Sequence[TransferRule]):
    """Precompute switch times keyed by (node, ordered edge-index pair).

    Unspecified distinct-pipe switches (and a U-turn back through the very
    pipe just traversed) cost zero; forbidden rules carry None.
    """
    costs: Dict[Tuple[str, int, int], Optional[int]] = {}
    idx_by_id = {e.eid: e.index for e in edges}
    for rule in rules:
        a, b = idx_by_id[rule.edge_a], idx_by_id[rule.edge_b]
        costs[(rule.node, min(a, b), max(a, b))] = rule.cost
    return costs


def _switch_cost(costs, node: str, a: int, b: int):
    if a == b:
        return 0  # U-turn on the same pipe, always free and allowed
    return costs.get((node, min(a, b), max(a, b)), 0)


def audit_transfers(nodes, edges, start, rules, comps) -> AuditResult:
    from array import array

    m = len(edges)
    S = 2 * m
    nmasks = 1 << m
    full = nmasks - 1
    total_length = sum(e.length for e in edges)
    INF64 = 10**18

    costs = _transfer_arc_index(edges, rules)

    def st_from(d):
        e = edges[d >> 1]
        return e.v if (d & 1) else e.u

    def st_to(d):
        e = edges[d >> 1]
        return e.u if (d & 1) else e.v

    def st_key(d):
        # pipe identifier order (edges sorted by id), then u->v before v->u
        return (d >> 1, d & 1)

    # incident edge indices per node, and the one directed state in which a
    # pipe leaves a given node
    incident = {n: [] for n in nodes}
    for e in edges:
        incident[e.u].append(e.index)
        incident[e.v].append(e.index)
    leaving: Dict[Tuple[int, str], int] = {}
    for e in edges:
        leaving[(e.index, e.u)] = 2 * e.index
        leaving[(e.index, e.v)] = 2 * e.index + 1

    # arcs[d] = list of (next_state, weight); weight is the entered pipe
    # length plus the switch time at the joining node
    arcs: List[List[Tuple[int, int]]] = [[] for _ in range(S)]
    for d in range(S):
        x = st_to(d)
        ei = d >> 1
        for ej in incident[x]:
            c = _switch_cost(costs, x, ei, ej)
            if c is None:
                continue  # forbidden switch
            arcs[d].append((leaving[(ej, x)], edges[ej].length + c))
        arcs[d].sort(key=lambda z: st_key(z[0]))

    # reversed arcs (static): rev[u] = [(predecessor d, weight)] for d -> u
    rev_arcs: List[List[Tuple[int, int]]] = [[] for _ in range(S)]
    for d, outs in enumerate(arcs):
        for nd, w in outs:
            rev_arcs[nd].append((d, w))

    # ------------------------------------------------------------------
    # Forward pass: best cost / exact number of optimal walks for each
    # (coverage mask, current directed state).  Intra-layer closure is a
    # multi-source Dijkstra on the arcs staying on covered pipes; with at
    # most 32 states a simple O(S^2) selection Dijkstra is fastest.
    # ------------------------------------------------------------------
    dist = array("q", [INF64]) * (nmasks * S)
    cnt = [0] * (nmasks * S)

    def active_states(M):
        """Both directed states of every pipe covered by mask M."""
        b = M
        while b:
            lb = b & -b
            ei = lb.bit_length() - 1
            yield 2 * ei
            yield 2 * ei + 1
            b ^= lb

    for e in edges:  # seed: first traversal leaves the maintenance port
        if e.u == start:
            p = (1 << e.index) * S + 2 * e.index
            dist[p] = e.length
            cnt[p] = 1
        if e.v == start:
            p = (1 << e.index) * S + 2 * e.index + 1
            dist[p] = e.length
            cnt[p] = 1

    for M in range(1, nmasks):
        base = M * S
        active = tuple(active_states(M))
        done = bytearray(S)
        for _ in range(len(active)):
            u = -1
            best = INF64
            for d in active:
                if not done[d] and dist[base + d] < best:
                    best, u = dist[base + d], d
            if u < 0:
                break
            done[u] = 1
            du = dist[base + u]
            cu = cnt[base + u]
            for nd, w in arcs[u]:
                if not ((M >> (nd >> 1)) & 1) or done[nd]:
                    continue
                p = base + nd
                v = du + w
                if v < dist[p]:
                    dist[p] = v
                    cnt[p] = cu
                elif v == dist[p]:
                    cnt[p] += cu

        # raise coverage onto a pipe traversed for the first time
        for d in active:
            if done[d]:  # finalized reachable state
                du = dist[base + d]
                ways = cnt[base + d]
                for nd, w in arcs[d]:
                    ej = nd >> 1
                    if (M >> ej) & 1:
                        continue
                    p = (M | (1 << ej)) * S + nd
                    v = du + w
                    if v < dist[p]:
                        dist[p] = v
                        cnt[p] = ways
                    elif v == dist[p]:
                        cnt[p] += ways

    fbase = full * S
    end_states = tuple(d for d in range(S) if st_to(d) == start)
    finish = [(dist[fbase + d], cnt[fbase + d]) for d in end_states]
    finish = [t for t in finish if t[0] < INF64]
    if not finish:
        raise AuditError(
            "无可行闭游：当前禁行规则下，无法从检修口出发经过全部管段后返回"
            "（请检查相关节点处的转接是否被禁行阻断）",
            ("rules",),
            [{"field": "rules"}],
        )
    optimum = min(w for w, _ in finish)
    optimal_count = sum(c for w, c in finish if w == optimum)

    # ------------------------------------------------------------------
    # Backward pass: H[M][d] = cheapest suffix from state d (its pipe just
    # traversed, coverage M) to a closed finish; bottom-up over masks with
    # the same intra-layer closure on reversed covered-pipe arcs.
    # ------------------------------------------------------------------
    H = array("q", [INF64]) * (nmasks * S)
    for M in range(full, 0, -1):
        base = M * S
        active = tuple(active_states(M))
        h = [INF64] * S
        if M == full:
            for d in end_states:
                h[d] = 0
        for d in active:
            bd = h[d]
            for nd, w in arcs[d]:
                ej = nd >> 1
                if (M >> ej) & 1:
                    continue
                hv = H[(M | (1 << ej)) * S + nd]
                if hv < INF64 and w + hv < bd:
                    bd = w + hv
            h[d] = bd
        if all(h[d] >= INF64 for d in active):
            continue  # cannot finish from this coverage set
        # multi-source Dijkstra on reversed intra-layer arcs
        done = bytearray(S)
        for _ in range(len(active)):
            u = -1
            best = INF64
            for d in active:
                if not done[d] and h[d] < best:
                    best, u = h[d], d
            if u < 0:
                break
            done[u] = 1
            hu = h[u]
            for pa, w in rev_arcs[u]:
                if not ((M >> (pa >> 1)) & 1) or done[pa]:
                    continue
                v = hu + w
                if v < h[pa]:
                    h[pa] = v
        for d in active:
            H[base + d] = h[d]

    # ------------------------------------------------------------------
    # Canonical walk: at every step the lexicographically smallest next
    # (pipe id, direction) that still admits an optimal suffix.
    # ------------------------------------------------------------------
    first = None
    first_best = INF64
    for e in edges:
        for d in (2 * e.index, 2 * e.index + 1):
            if st_from(d) != start:
                continue
            hv = H[(1 << e.index) * S + d]
            if hv < INF64 and e.length + hv < first_best:
                first_best = e.length + hv
                first = d
    if first is None or first_best != optimum:
        raise AuditError(
            "无可行闭游：当前禁行规则下，无法从检修口出发经过全部管段后返回",
            ("rules",),
            [{"field": "rules"}],
        )

    seq: List[int] = [first]
    d = first
    M = 1 << (d >> 1)
    remaining = H[M * S + d]
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > nmasks * S:
            raise AuditError("内部错误：规范路线恢复失败", ("rules",))
        chosen = None
        chosen_w = None
        for nd, w in arcs[d]:
            ej = nd >> 1
            M2 = M if (M >> ej) & 1 else M | (1 << ej)
            hv = H[M2 * S + nd]
            if hv < INF64 and w + hv == remaining:
                if chosen is None or st_key(nd) < st_key(chosen):
                    chosen, chosen_w = nd, w
        if chosen is None:
            raise AuditError("内部错误：规范路线恢复失败", ("rules",))
        seq.append(chosen)
        M |= 1 << (chosen >> 1)
        remaining -= chosen_w
        d = chosen

    # ------------------------------------------------------------------
    # Assemble steps, totals and per-rule hit positions
    # ------------------------------------------------------------------
    steps: List[RouteStep] = []
    occ = [0] * m
    walk_length = 0
    transfer_cost = 0
    prev = None
    for d in seq:
        ei = d >> 1
        e = edges[ei]
        frm, to = (e.u, e.v) if not (d & 1) else (e.v, e.u)
        if prev is None:
            switch = 0
        else:
            for nd, w in arcs[prev]:
                if nd == d:
                    switch = w - e.length
                    break
            else:  # pragma: no cover - reconstruction guarantees the arc
                raise AuditError("内部错误：规范路线恢复失败", ("rules",))
        rkey = (
            tuple(sorted((edges[prev >> 1].eid, e.eid)))
            if prev is not None and (prev >> 1) != ei
            else None
        )
        occ[ei] += 1
        walk_length += e.length
        transfer_cost += switch
        steps.append(
            RouteStep(
                edge_index=ei,
                edge_id=e.eid,
                frm=frm,
                to=to,
                length=e.length,
                duplicate_no=occ[ei],
                transfer=switch,
                rule_key=rkey,
            )
        )
        prev = d

    multiplicity = tuple(occ)
    rule_hits: Dict[Tuple[str, ...], Tuple[int, ...]] = {}
    for rule in rules:
        pair = tuple(sorted((rule.edge_a, rule.edge_b)))
        rule_hits[(rule.node, pair[0], pair[1])] = tuple(
            i + 1
            for i, st in enumerate(steps)
            if st.rule_key == pair and st.frm == rule.node
        )

    degree = {n: 0 for n in nodes}
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
    odd = tuple(sorted(n for n in nodes if degree[n] % 2 == 1))

    return AuditResult(
        nodes=nodes,
        edges=edges,
        start=start,
        odd_vertices=odd,
        components=tuple(tuple(c) for c in comps),
        total_length=total_length,
        added_length=walk_length - total_length,
        optimal_count=optimal_count,
        canonical_set=frozenset(),
        bit_vector="",
        classification={},
        multiplicity=multiplicity,
        route=tuple(steps),
        transfer_mode=True,
        transfer_cost=transfer_cost,
        walk_length=walk_length,
        rules=rules,
        rule_hits=rule_hits,
    )
