"""Chinese postman / route inspection core.

Undirected multigraph (parallel edges allowed, self loops forbidden) with
positive integer edge lengths.  Finds the minimum added length needed for all
vertices to have even degree, the exact number of optimal duplicate sets, the
canonical duplicate set (0-preferred bit vector in edge order), per-edge
classification, and a closed Euler tour for the canonical augmentation.

Transfer-time mode (``transfer_mode=True``)
-------------------------------------------
At a shared node the robot may continue from one pipe onto another; such a
transfer carries a user supplied non-negative integer cost (unset = 0), or is
forbidden.  Pairs of *distinct* edges only; reversing straight back along the
same edge is always free (leaf pipes can only be inspected that way).  The
objective is then

    total time = sum of traversed pipe lengths + sum of per-transfer costs

over closed walks based at the inspection port that use every pipe at least
once and may reuse pipes freely.  This is NOT solved by fixing the ordinary
minimum-augmentation set first: the walk space is searched directly.

Each undirected edge becomes two directed darts (2i = u->v, 2i+1 = v->u).  A
walk state is (covered-edge mask, current dart); moving onto dart d2 costs the
transfer cost plus the length of d2 (taking d2 = d ^ 1 is the free reversal
plus the length of walking the same pipe back).  Layered Dijkstra over this
state graph finds the global optimum and the exact number of optimal walks;
non-negative costs make every shortest state path simple.  Tight-edge
back-marking followed by a lexicographic memo picks the canonical walk,
adjudicated by (edge identifier, direction) sequences.

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

    @property
    def is_eulerian(self) -> bool:
        return not self.odd_vertices


@dataclass(frozen=True)
class TransferRule:
    edge_a: int
    edge_b: int
    cost: Optional[int]  # None == forbidden, otherwise non-negative integer
    row: int


@dataclass(frozen=True)
class TransferStep:
    edge_index: int
    edge_id: str
    frm: str
    to: str
    length: int
    transfer_cost: int  # cost paid to enter this pipe from the previous one
    rule: Optional[Tuple[int, int]]  # unordered edge-index pair of the transfer


@dataclass
class TransferResult:
    nodes: List[str]
    edges: List[Edge]
    start: str
    rules: Tuple[TransferRule, ...]
    total_length: int          # sum of lengths over the canonical walk
    transfer_time: int         # sum of transfer costs over the canonical walk
    total_time: int
    optimal_count: int
    route: Tuple[TransferStep, ...]


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


def validate_input(
    nodes: Sequence[str], raw_edges: Sequence[dict], start: Optional[str]
) -> Tuple[List[str], List[Edge]]:
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
    if len(raw_edges) > 32:
        raise AuditError(
            f"管段数量为 {len(raw_edges)}，不能超过 32",
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

    return clean_nodes, edges


def _as_nonneg_int(value) -> Optional[int]:
    """Blank -> 0.  A bare 'x' style sentinel is handled by the caller."""
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError()
    if isinstance(value, int):
        iv = value
    elif isinstance(value, str):
        s = value.strip()
        if s == "":
            return 0
        if not re.fullmatch(r"\d+", s):
            raise ValueError()
        iv = int(s)
    else:
        raise ValueError()
    if iv < 0:
        raise ValueError()
    return iv


def validate_transfer_rules(
    edges: Sequence[Edge],
    raw_rules,
    enabled: bool,
) -> Tuple[Dict[FrozenSet[int], Optional[int]], Tuple[TransferRule, ...]]:
    """Parse the unordered transfer rules.

    Blank rows are ignored; blank cost means zero cost.  A cost of the
    forbidden sentinel (string ``x``/``禁行`` or boolean ``true`` field
    ``forbidden``) blocks the transfer.  Returns (cost map, cleaned rules),
    where the map value is None for forbidden pairs.
    """
    id_to_index = {e.eid: e.index for e in edges}

    def loc(i, *fields):
        return [{"field": "transfers", "row": i, "sub": f} for f in fields]

    if not isinstance(raw_rules, list):
        raw_rules = []
    if enabled and len(edges) > 16:
        raise AuditError(
            f"转接耗时模式下管段数量为 {len(edges)}，不能超过 16",
            ("edges",),
            [{"field": "edges"}],
        )

    cost_map: Dict[FrozenSet[int], Optional[int]] = {}
    cleaned: List[TransferRule] = []
    seen_pairs: Dict[FrozenSet[int], int] = {}

    for i, rr in enumerate(raw_rules):
        if not isinstance(rr, dict):
            continue
        a = str(rr.get("edgeA", "") or "").strip()
        b = str(rr.get("edgeB", "") or "").strip()
        if not a and not b:
            continue  # blank row

        if not a or not b:
            raise AuditError(
                f"第 {i + 1} 条转接规则必须同时指定两条管段",
                ("transfers",),
                loc(i, "edgeA", "edgeB"),
            )
        if a not in id_to_index:
            raise AuditError(
                f"转接规则引用了不存在的管段 {a!r}",
                ("transfers",),
                loc(i, "edgeA"),
            )
        if b not in id_to_index:
            raise AuditError(
                f"转接规则引用了不存在的管段 {b!r}",
                ("transfers",),
                loc(i, "edgeB"),
            )
        ia, ib = id_to_index[a], id_to_index[b]
        if ia == ib:
            raise AuditError(
                f"转接规则 {a!r} 与自身配对：只能填写同一节点处两条不同管段"
                "（同一条管段原路返回始终零耗时）",
                ("transfers",),
                loc(i, "edgeA", "edgeB"),
            )
        ea, eb = edges[ia], edges[ib]
        shared = {ea.u, ea.v} & {eb.u, eb.v}
        if not shared:
            raise AuditError(
                f"管段 {ea.eid} 与 {eb.eid} 不相邻（没有公共节点），无法在此转接",
                ("transfers",),
                loc(i, "edgeA", "edgeB"),
            )

        forbidden = rr.get("forbidden", False)
        raw_cost = rr.get("cost", "")
        if isinstance(forbidden, str):
            forbidden = forbidden.strip().lower() in ("1", "true", "yes", "禁行", "x")
        raw_s = raw_cost.strip() if isinstance(raw_cost, str) else raw_cost
        is_block_token = isinstance(raw_s, str) and raw_s in ("x", "X", "禁", "禁行", "-", "×")

        if forbidden is True or is_block_token:
            cost: Optional[int] = None
        else:
            try:
                cost = _as_nonneg_int(raw_cost)
            except ValueError:
                raise AuditError(
                    f"转接 {ea.eid}↔{eb.eid} 的耗时必须为非负整数，或填写“禁行”",
                    ("transfers",),
                    loc(i, "cost"),
                )

        pair = frozenset((ia, ib))
        if pair in seen_pairs:
            raise AuditError(
                f"管段 {ea.eid} 与 {eb.eid} 的转接规则重复",
                ("transfers",),
                loc(i, "edgeA")
                + [{"field": "transfers", "row": seen_pairs[pair], "sub": "edgeA"}],
            )
        seen_pairs[pair] = i
        cost_map[pair] = cost
        cleaned.append(
            TransferRule(edge_a=ia, edge_b=ib, cost=cost, row=i)
        )

    return cost_map, tuple(cleaned)


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
    nodes: Sequence[str], raw_edges: Sequence[dict], start: Optional[str]
) -> AuditResult:
    nodes, edges = validate_input(nodes, raw_edges, start)
    adj = adjacency(nodes, edges)

    comps = connected_components(nodes, adj)
    if len(comps) > 1:
        raise AuditError(
            "管网不连通，存在多个连通分量："
            + "；".join("{" + ",".join(c) + "}" for c in comps),
            ("edges", "nodes"),
            [{"field": "edges"}],
        )

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
# Transfer-time mode: shortest covering closed walk in the directed line graph
# ---------------------------------------------------------------------------


def audit_transfer(
    nodes: Sequence[str],
    raw_edges: Sequence[dict],
    start: Optional[str],
    raw_rules: Sequence[dict],
) -> TransferResult:
    """Global optimum over closed walks that may reuse any pipe.

    The ordinary minimum-augmentation machinery is deliberately not used:
    routing is solved directly in the walk space via layered Dijkstra on
    (covered-edge mask, current directed dart).
    """
    nodes, edges = validate_input(nodes, raw_edges, start)
    m = len(edges)
    if m > 16:
        raise AuditError(
            f"转接耗时模式下管段数量为 {m}，不能超过 16",
            ("edges",),
            [{"field": "edges"}],
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

    cost_map, rules = validate_transfer_rules(edges, raw_rules, True)

    nidx = {n: i for i, n in enumerate(nodes)}
    start_i = nidx[start]
    D = 2 * m

    # Dart d: edge d>>1 traversed u->v when d even, v->u when d odd.
    dtail = [0] * D
    dhead = [0] * D
    dlen = [0] * D
    out_darts: List[List[int]] = [[] for _ in nodes]
    for i, e in enumerate(edges):
        u, v = nidx[e.u], nidx[e.v]
        dtail[2 * i], dhead[2 * i], dlen[2 * i] = u, v, e.length
        dtail[2 * i + 1], dhead[2 * i + 1], dlen[2 * i + 1] = v, u, e.length
        out_darts[u].append(2 * i)
        out_darts[v].append(2 * i + 1)
    for lst in out_darts:
        lst.sort()

    # Successors of a dart arriving at a node: reverse along the same edge is
    # always free; every other incident edge pays its transfer rule (unset
    # pairs cost zero; forbidden pairs are absent).
    succ: List[Tuple[Tuple[int, int], ...]] = [()] * D
    for d in range(D):
        nxt = []
        for d2 in out_darts[dhead[d]]:
            if d2 == (d ^ 1):
                nxt.append((d2, 0))
                continue
            c = cost_map.get(frozenset((d >> 1, d2 >> 1)), 0)
            if c is None:
                continue
            nxt.append((d2, c))
        succ[d] = tuple(sorted(nxt))

    full = (1 << m) - 1
    nstates = (1 << m) * D
    dist = [INF] * nstates
    counts = [0] * nstates
    settled = bytearray(nstates)

    heap = []
    for d in out_darts[start_i]:
        sid = (1 << (d >> 1)) * D + d
        dist[sid] = dlen[d]
        counts[sid] = 1  # one first move per dart
        heapq.heappush(heap, (dlen[d], sid))

    # Pop-time relaxation: every transition weighs at least the entered pipe
    # length (>= 1), so all predecessors of a state are settled before it and
    # its shortest-path count is final when it is popped.
    while heap:
        du, sid = heapq.heappop(heap)
        if du != dist[sid] or settled[sid]:
            continue
        settled[sid] = 1
        mask, d = divmod(sid, D)
        for d2, tc in succ[d]:
            nmask = mask | (1 << (d2 >> 1))
            nsid = nmask * D + d2
            nd = du + tc + dlen[d2]
            if nd < dist[nsid]:
                dist[nsid] = nd
                counts[nsid] = counts[sid]
                heapq.heappush(heap, (nd, nsid))
            elif nd == dist[nsid]:
                counts[nsid] += counts[sid]

    goal_sids = [
        full * D + d for d in range(D) if dhead[d] == start_i
    ]
    best = min((dist[g] for g in goal_sids), default=INF)
    if best >= INF:
        raise AuditError(
            "按当前禁行规则，不存在能够覆盖全部管段并返回检修口的可行闭合行走"
            "（无可行闭游）",
            ("transfers",),
            [{"field": "transfers"}],
        )
    optimal_goals = [g for g in goal_sids if dist[g] == best]
    optimal_count = sum(counts[g] for g in optimal_goals)

    # Mark every state that can still reach an optimal goal on a tight edge.
    # A state may have several equally tight predecessors (the single stored
    # predecessor would miss branches), so propagate backwards through ALL of
    # them.  Predecessors of dart d are the darts arriving at d's tail; the
    # bit of d's edge was either already covered or gets newly added.
    rev: List[Tuple[Tuple[int, int], ...]] = [()] * D
    for d in range(D):
        pre = []
        w = dtail[d]
        for x in out_darts[w]:
            d1 = x ^ 1  # reverse: a dart leaving w becomes one arriving at w
            if d1 == (d ^ 1):
                pre.append((d1, 0))
                continue
            c = cost_map.get(frozenset((d1 >> 1, d >> 1)), 0)
            if c is None:
                continue
            pre.append((d1, c))
        rev[d] = tuple(pre)

    tight = bytearray(nstates)
    stack = list(optimal_goals)
    for g in optimal_goals:
        tight[g] = 1
    dbit = [1 << (d >> 1) for d in range(D)]
    while stack:
        cur = stack.pop()
        mask, d = divmod(cur, D)
        bit = dbit[d]
        pre_masks = (mask, mask ^ bit) if (mask & bit) else (mask,)
        for d1, tc in rev[d]:
            for pmask in pre_masks:  # edge d repeated vs newly used
                p = pmask * D + d1
                if not tight[p] and dist[p] + tc + dlen[d] == dist[cur]:
                    tight[p] = 1
                    stack.append(p)

    seed_states = [(1 << (d >> 1)) * D + d for d in out_darts[start_i]]
    cur = min((sid for sid in seed_states if tight[sid]), key=lambda s: s % D)

    # Canonical walk: at every state take the smallest dart (edge identifier
    # order, then u->v before v->u) that keeps a tight continuation.  Because
    # successor darts are distinct, the first dart alone adjudicates the
    # lexicographic (identifier, direction) sequence -- no suffix ties exist.
    dart_walk: List[int] = []
    while True:
        mask, d = divmod(cur, D)
        dart_walk.append(d)
        if mask == full and dhead[d] == start_i:
            break
        chosen = None
        for d2, tc in succ[d]:
            nsid = (mask | (1 << (d2 >> 1))) * D + d2
            if tight[nsid] and dist[cur] + tc + dlen[d2] == dist[nsid]:
                chosen = (d2, nsid, tc)
                break
        if chosen is None:  # pragma: no cover - defensive, tight guarantees it
            raise AuditError("规范路线重建失败", ("transfers",))
        cur = chosen[1]

    steps: List[TransferStep] = []
    for k, dd in enumerate(dart_walk):
        i = dd >> 1
        e = edges[i]
        frm, to = (e.u, e.v) if (dd & 1) == 0 else (e.v, e.u)
        if k == 0:
            tc, pair = 0, None
        else:
            pdd = dart_walk[k - 1]
            if dd == (pdd ^ 1):
                tc, pair = 0, None  # free reversal along the same pipe
            else:
                pi = pdd >> 1
                tc = cost_map.get(frozenset((pi, i)), 0)
                pair = (pi, i) if pi < i else (i, pi)
        steps.append(
            TransferStep(
                edge_index=i,
                edge_id=e.eid,
                frm=frm,
                to=to,
                length=e.length,
                transfer_cost=tc,
                rule=pair,
            )
        )

    walking_length = sum(dlen[d] for d in dart_walk)
    transfer_time = sum(st.transfer_cost for st in steps)
    if walking_length + transfer_time != best:  # pragma: no cover - defensive
        raise AuditError("转接路线核算不一致", ("transfers",))

    return TransferResult(
        nodes=nodes,
        edges=edges,
        start=start,
        rules=rules,
        total_length=walking_length,
        transfer_time=transfer_time,
        total_time=best,
        optimal_count=optimal_count,
        route=tuple(steps),
    )
