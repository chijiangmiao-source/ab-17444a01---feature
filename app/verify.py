"""One-shot verification service.

Runs, in order:
  1. the code test suite (pytest);
  2. domain boundary checks against the solver directly
       - odd-degree network: exact co-optimal count / classification,
       - Eulerian network: zero augmentation boundary;
  3. an HTTP smoke test against the running web service
       - GET /health,
       - POST /api/audit success case,
       - POST /api/audit validation failure case.

BASE_URL may point at an already running instance (compose sets it to the
web service).  When unset, the script starts gunicorn locally on an
ephemeral port and tears it down afterwards.

Exits 0 only when every stage passes; the failed stage is reported.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


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

TRIANGLE = {
    "nodes": ["A", "B", "C"],
    "edges": [
        {"id": "a", "u": "A", "v": "B", "length": 3},
        {"id": "b", "u": "B", "v": "C", "length": 4},
        {"id": "c", "u": "C", "v": "A", "length": 5},
    ],
    "start": "B",
}

BAD = {
    "nodes": ["A", "B"],
    "edges": [{"id": "x", "u": "A", "v": "A", "length": 1}],
    "start": "A",
}

# Transfer-time mode fixtures ------------------------------------------------

DETOUR_NODES = ["A", "B", "C", "X"]
DETOUR_EDGES = [
    {"id": "e1", "u": "A", "v": "B", "length": 1},
    {"id": "e2", "u": "B", "v": "C", "length": 1},
    {"id": "e3", "u": "C", "v": "A", "length": 1},
    {"id": "e4", "u": "A", "v": "X", "length": 1},
    {"id": "e5", "u": "X", "v": "C", "length": 1},
]
# B 点 e1<->e2 转接极贵：最优为掉头绕行（行走 7、转接 0），
# 而非更短行走 + 高转接的直连方案。
DETOUR_COSTLY = {
    "nodes": DETOUR_NODES, "edges": DETOUR_EDGES, "start": "A",
    "transferMode": True,
    "rules": [{"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": 100}],
}
# 同样的转接禁行：路径图 A-B-C 无任何替代回路，闭游不可行。
PATH_NODES = ["A", "B", "C"]
PATH_EDGES = [
    {"id": "e1", "u": "A", "v": "B", "length": 1},
    {"id": "e2", "u": "B", "v": "C", "length": 1},
]
PATH_FORBID = {
    "nodes": PATH_NODES, "edges": PATH_EDGES, "start": "A",
    "transferMode": True,
    "rules": [{"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": -1}],
}


def stage(name):
    print(f"\n=== {name} ===", flush=True)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


# ---------------------------------------------------------------------------
# Stage 1: pytest
# ---------------------------------------------------------------------------


def run_pytest() -> bool:
    stage("1/3 代码测试 (pytest)")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Stage 2: solver domain boundaries
# ---------------------------------------------------------------------------


def run_domain_checks() -> bool:
    stage("2/3 奇度同优/欧拉零增程 & 转接模式边界（高耗时绕行、禁行无解）")
    from app.solver import audit

    # odd network: K4 unit weights -> 4 odd vertices, 3 distinct optima
    r = audit(K4["nodes"], K4["edges"], K4["start"])
    check(tuple(r.odd_vertices) == ("A", "B", "C", "D"), "K4 有四个奇度节点")
    check(r.added_length == 2, "K4 最小增程为 2")
    check(r.optimal_count == 3, "K4 同优集合数量恰为 3")
    check(r.bit_vector == "001100", "K4 规范位向量为 001100（0 优先）")
    check(
        set(r.classification.values()) == {"optional"},
        "K4 每条边均为可重复（无必重复/从不重复）",
    )
    check(
        len(r.route) == sum(r.multiplicity)
        and r.route[0].frm == "A"
        and r.route[-1].to == "A",
        "K4 规范路线闭合且副本数吻合",
    )

    # optional-vs-required mix: two equal shortest routes + a long edge
    mix_nodes = ["A", "B", "X", "Y"]
    mix_edges = [
        {"id": "p1", "u": "A", "v": "X", "length": 1},
        {"id": "p2", "u": "X", "v": "B", "length": 1},
        {"id": "p3", "u": "A", "v": "Y", "length": 1},
        {"id": "p4", "u": "Y", "v": "B", "length": 1},
        {"id": "long", "u": "A", "v": "B", "length": 3},
    ]
    r2 = audit(mix_nodes, mix_edges, "A")
    check(r2.optimal_count == 2, "双桥结构同优集合数量为 2")
    # edges are identifier-sorted: long(0), p1(1), p2(2), p3(3), p4(4)
    check(all(r2.classification[i] == "optional" for i in (1, 2, 3, 4)),
          "两条等长路径上的边为可重复")
    check(r2.classification[0] == "never", "长边从不重复")
    # candidate sets {p1,p2}=01100 and {p3,p4}=00011 -> 0-pref = 00011
    check(r2.bit_vector == "00011", "双桥规范位向量 00011")

    # Eulerian boundary: triangle
    r3 = audit(TRIANGLE["nodes"], TRIANGLE["edges"], TRIANGLE["start"])
    check(r3.is_eulerian, "三角形为欧拉管网")
    check(r3.added_length == 0, "欧拉管网零增程")
    check(r3.optimal_count == 1, "欧拉管网同优集合数量为 1（空集）")
    check(set(r3.canonical_set) == set(), "规范重复集合为空")
    check(r3.bit_vector == "000", "位向量全 0")
    check(
        all(v == "never" for v in r3.classification.values()),
        "所有边归属为从不重复",
    )
    check(len(r3.route) == 3 and r3.route[-1].to == "B",
          "欧拉回路从检修口出发并返回")

    # ---- transfer-time mode boundaries ----
    rt = audit(
        DETOUR_COSTLY["nodes"], DETOUR_COSTLY["edges"],
        DETOUR_COSTLY["start"], True, DETOUR_COSTLY["rules"],
    )
    check(rt.transfer_mode, "转接模式标志置位")
    check(rt.walk_length == 7 and rt.transfer_cost == 0,
          "高耗时转接：最优为行走 7、转接 0 的掉头绕行（而非更短的高耗时走法）")
    check(rt.optimal_count >= 1, "高耗时绕行存在最优路线")
    check(
        rt.route[0].frm == "A" and rt.route[-1].to == "A"
        and len(rt.route) == sum(rt.multiplicity),
        "高耗时绕行规范路线闭合、副本数吻合",
    )
    # the costly rule is declared but never used in the canonical walk
    check(rt.rule_hits[("B", "e1", "e2")] == (),
          "高耗时规则在规范路线中零命中")

    rz = audit(
        DETOUR_COSTLY["nodes"], DETOUR_COSTLY["edges"],
        DETOUR_COSTLY["start"], True, [],
    )
    check(rz.walk_length == 6 and rz.transfer_cost == 0,
          "无规则基线：行走 6（直连重复），证明耗时规则真正改变最优走法")

    from app.solver import AuditError
    try:
        audit(PATH_FORBID["nodes"], PATH_FORBID["edges"],
              PATH_FORBID["start"], True, PATH_FORBID["rules"])
        raise AssertionError("禁行无解用例应当抛出 AuditError")
    except AuditError as exc:
        check("无可行闭游" in exc.message and "rules" in exc.fields,
              "禁行转接导致无可行闭游并定位到规则表")

    # bad rule references keep the structured error contract
    for bad_rules, needle in (
        ([{"node": "B", "edgeA": "e1", "edgeB": "x", "cost": 1}], "不存在"),
        ([{"node": "A", "edgeA": "e1", "edgeB": "e2", "cost": 1}], "相邻"),
        ([{"node": "B", "edgeA": "e1", "edgeB": "e2", "cost": "abc"}], "非负整数"),
    ):
        try:
            audit(PATH_FORBID["nodes"], PATH_FORBID["edges"], "A", True, bad_rules)
            raise AssertionError(f"非法规则未被拒绝: {bad_rules}")
        except AuditError as exc:
            check(needle in exc.message and "rules" in exc.fields,
                  f"非法规则（{needle}）被拒绝并定位")
    return True


# ---------------------------------------------------------------------------
# Stage 3: HTTP smoke
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(base: str, proc, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    return False


def _post(base: str, path: str, body: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def run_http_smoke() -> bool:
    stage("3/3 HTTP 冒烟（普通回归 / 转接高耗时绕行 / 禁行无解 / 非法输入）")
    base = os.environ.get("BASE_URL", "").rstrip("/")
    proc = None
    if not base:
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        print(f"  启动临时 gunicorn: {base}")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "gunicorn",
                "-w", "1", "-b", f"127.0.0.1:{port}",
                "--timeout", "30", "app.server:app",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        if not _wait_ready(base, proc):
            print("  FAIL: 服务未在限定时间内就绪")
            return False
        print("  PASS: GET /health -> 200")

        status, body = _post(base, "/api/audit", K4)
        check(status == 200 and body.get("ok") is True,
              "POST /api/audit 奇度管网审计成功")
        check(body.get("optimalCount") == 3, "HTTP 返回同优数量为 3")
        check(body.get("addedLength") == 2, "HTTP 返回最小增程为 2")
        check(body.get("canonicalVector") == "001100",
              "HTTP 返回规范位向量 001100")
        check(len(body.get("route", [])) == 8,
              "HTTP 返回 8 步闭合路线（6 原边 + 2 重复副本）")

        status, body = _post(base, "/api/audit", TRIANGLE)
        check(body.get("ok") and body.get("addedLength") == 0,
              "HTTP 欧拉管网零增程")

        status, body = _post(base, "/api/audit", BAD)
        check(status == 200 and body.get("ok") is False,
              "非法输入返回 ok=false（HTTP 层仍为 200）")
        check("edges" in body.get("fields", []), "自环错误定位到管段表")
        check(any(l.get("row") == 0 for l in body.get("locations", [])),
              "错误位置精确到第 1 行")

        # transfer-time mode: costly switch detour
        status, body = _post(base, "/api/audit", DETOUR_COSTLY)
        check(status == 200 and body.get("ok") is True
              and body.get("mode") == "transfer",
              "转接模式审计成功")
        check(body.get("walkLength") == 7 and body.get("transferCost") == 0
              and body.get("totalTime") == 7,
              "高耗时绕行：行走 7 / 转接 0 / 总耗时 7")
        check(body.get("rules", [{}])[0].get("positions") == [],
              "高耗时禁避规则无命中位置")
        check(
            body.get("route", [{}])[0].get("from") == "A"
            and body.get("route", [{}])[-1].get("to") == "A",
            "转接模式路线闭合",
        )

        # transfer-time mode: forbidden switch, no feasible tour
        status, body = _post(base, "/api/audit", PATH_FORBID)
        check(status == 200 and body.get("ok") is False,
              "禁行无解返回 ok=false")
        check("无可行闭游" in body.get("error", "")
              and "rules" in body.get("fields", []),
              "禁行无解定位到转接规则表")

        # normal-mode regression over HTTP (historical fields intact)
        status, body = _post(base, "/api/audit", TRIANGLE)
        check(body.get("mode") == "normal" and body.get("addedLength") == 0
              and body.get("canonicalVector") == "000",
              "普通模式审计字段保持兼容")
        return True
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    failures = []
    for name, fn in (
        ("代码测试", run_pytest),
        ("同优分类/零增程边界", run_domain_checks),
        ("HTTP 冒烟", run_http_smoke),
    ):
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 - report any failure
            ok = False
            print(f"  FAIL: {name} 抛出异常: {exc!r}")
        if not ok:
            failures.append(name)

    print("\n========================================")
    if failures:
        print("VERIFY 失败：" + "、".join(failures))
        return 1
    print("VERIFY 全部通过：测试 / 奇度同优分类 / 欧拉零增程 / 转接耗时模式 / HTTP 冒烟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
