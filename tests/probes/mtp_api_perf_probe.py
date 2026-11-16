#!/usr/bin/env python3
"""mtp_api_perf_probe —— 走 OpenAI 兼容 API 的 MTP 开/关 A/B：单发/并发 × 小/大上下文。

@case d2#1 基线 A/B 表（MTP 关 vs 开，decode tok/s），含 shape/batch/TP/是否 graph/时钟（D12）。

与 tests/probes/mtp_decode_breakdown_probe.sh（cli 模式、单请求、分段计时）互补：
本探针只给端到端读数，但覆盖 cli 探针覆盖不了的两维——并发与上下文长度。
同条件交替：每个 round 内 off 与 on 相邻起停（同一台机、同一份权重、同一 prompt）。

用法：ROUNDS=1 CONC=4 python3 tests/probes/mtp_api_perf_probe.py
覆盖：BIN MODEL K ROUNDS CONC SMALL_CTX LARGE_CTX MAXT OK EXTRA XINFER_PROBE_DIR
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ANSI = re.compile(r"\x1b\[[0-9;]*[a-z]")
BIN = os.environ.get("BIN", "/workspace/xxinfer/target/release/xinfer")
MODEL = os.environ.get("MODEL", "unsloth/Qwen3.8-27B-NVFP4")
K = int(os.environ.get("K", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "1"))
CONC = int(os.environ.get("CONC", "2"))
SMALL_CTX = int(os.environ.get("SMALL_CTX", "24"))
LARGE_CTX = int(os.environ.get("MTP_LARGE_CTX", "768"))
MAXT = int(os.environ.get("MAXT", "256"))
MML = os.environ.get("MML", "1024")
DTYPE = os.environ.get("DTYPE", "f16")
DEVS = os.environ.get("DEVS", "0,1")
OK = os.environ.get("OK", "0")            # 图：真模型上 MTP+开图在捕获期 OOM，默认关
SEQS = int(os.environ.get("SEQS", "2"))   # 调度器并发上限，必须 ≥ CONC，否则请求排队
MFRAC = os.environ.get("MFRAC", "0.35")   # 混合状态（GDN conv/ssm）预算占比，并发时这块先吃地方
OUT = os.path.join(os.environ.get("XINFER_PROBE_DIR", "/workspace/probe-out"), "mtp-api-perf")


def env_for(mtp_on):
    e = dict(os.environ)
    e.update(HF_HUB_OFFLINE="1", XINFER_ONESHOT_DISABLE="1", XINFER_MTP_RESERVE_MB="128")
    if OK == "0":
        e["XINFER_DISABLE_CUDA_GRAPH"] = "1"
    return e


def args_for(mtp_on, mml, seqs):
    a = [BIN, "--m", MODEL, "--dtype", DTYPE, "--d", DEVS, "--max-model-len", str(mml),
          "--max-num-seqs", str(seqs), "--mamba-fraction", MFRAC,
          "--prefill-chunk-size", str(mml)]
    if OK == "0":
        a += ["--disable-cuda-graph"]
    if mtp_on:
        a += ["--num-speculative-tokens", str(K)]
    return a


WORDS = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper"]


def prompt_for(tokens):
    """~tokens 个 token：编号事实 + 中间那条的 needle 问题（实测 ~23 token/行）。"""
    if tokens <= 64:
        return ("Explain in one short paragraph why a cache-bounded GPU spends most of its "
                "decode time moving weights instead of computing them.")
    n = max(1, tokens // 23)
    facts = "\n".join(f"Fact {i}: the code word for item {i} is {WORDS[i % 10]}-{i * 7 % 1000}."
                      for i in range(n))
    i = n // 2
    return (f"{facts}\n\nWhat is the code word for item {i}? Answer with the code word only, "
            f"then write at least 200 more words about caching, bandwidth and decode latency.")


def stream_once(base, prompt, max_tokens, timeout, out):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_k": 1, "stream": True, "enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            sent = time.perf_counter()
            stamps, text = [], []
            for raw in resp:
                line = ANSI.sub("", raw.decode("utf-8", errors="replace")).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    stamps.append(time.perf_counter())
                    text.append(delta)
        out.append((sent, stamps, "".join(text)))
    except Exception as exc:                                    # noqa: BLE001 —— 记录进结果，不吞掉
        out.append((None, [], f"ERROR {type(exc).__name__}: {exc}"))


def fire(base, prompt, concurrency, timeout, max_tokens=None):
    """并发打 concurrency 路流式请求；聚合吞吐 = 总 chunk 数 ÷ 首 chunk 到末 chunk。"""
    out = []
    th = [threading.Thread(target=stream_once,
                           args=(base, prompt, max_tokens or MAXT, timeout, out))
          for _ in range(concurrency)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join(timeout)
    wall = time.perf_counter() - t0
    good = [(s, st) for (s, st, _) in out if st]
    if not good:
        return {"error": str(out[0][2]) if out else "no results", "wall_s": wall}
    chunks = sum(len(st) for _, st in good)
    first = min(st[0] for _, st in good)   # 首个 chunk，不是发送时刻：让 agg 与 per-req 同口径
    last = max(st[-1] for _, st in good)
    per = [len(st) / ((st[-1] - st[0]) or 1e-9) for _, st in good if len(st) > 2]
    ttft = [1e3 * (st[0] - s) for s, st in good]
    return {"agg_tok_per_s": chunks / ((last - first) or 1e-9),
            "per_req_tok_per_s": (sum(per) / len(per)) if per else 0.0,
            "ttft_p50_ms": sorted(ttft)[len(ttft) // 2], "ttft_max_ms": max(ttft),
            "chunks": chunks, "wall_s": wall, "streams": len(good)}


def serve(mtp_on, port, mml, seqs):
    """起一台服务；日志落盘，否则起不来时什么线索都没有。"""
    tag = os.path.join(OUT, f"srv_mtp{int(mtp_on)}_mml{mml}_seqs{seqs}.log")
    log = open(tag, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(args_for(mtp_on, mml, seqs) + ["--server", f"127.0.0.1:{port}"],
                            env=env_for(mtp_on), stdout=log, stderr=subprocess.STDOUT)


def wait_ready(base, proc, timeout=600):
    """服务父进程会 daemonize（自己退 0，真服务活在脱离的子里），所以不能拿 proc.poll() 当失败信号。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2).close()
            return True
        except (urllib.error.URLError, OSError):
            if proc.poll() not in (None, 0):
                return False
            time.sleep(1)
    return False


def stop():
    """服务父进程会 daemonize，只能按进程名收。"""
    subprocess.call(["pkill", "-x", "xinfer"])
    time.sleep(3)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for r in range(1, ROUNDS + 1):
        for ctx in (SMALL_CTX, LARGE_CTX):
            mml = ctx + MAXT + 64
            for c in (1, CONC):
                for mtp_on in (False, True):
                    with socket.socket() as s:
                        s.bind(("127.0.0.1", 0))
                        port = s.getsockname()[1]
                    base = f"http://127.0.0.1:{port}"
                    # 每格一台服务：并行度必须等于这一格的并发，否则槽位/预算按别的值算。
                    proc = serve(mtp_on, port, mml, c)
                    ready = wait_ready(base, proc)
                    p = prompt_for(ctx)
                    if not ready:
                        rows.append({"round": r, "mtp": mtp_on, "ctx_tokens": ctx,
                                     "concurrency": c, "error": f"server not ready rc={proc.poll()}"})
                        stop()
                        continue
                    fire(base, "Say OK.", 1, 60, 8)       # 预热：另一条 prompt，前缀缓存帮不上忙
                    res = fire(base, p, c, 600, MAXT)
                    res.update(round=r, mtp=mtp_on, ctx_tokens=ctx, concurrency=c,
                               out_tokens=MAXT, k=K if mtp_on else 0, mml=mml)
                    rows.append(res)
                    print(f"[probe] r{r} mtp={int(mtp_on)} ctx={ctx} c={c} mml={mml} seqs={c} -> "
                          f"{res.get('agg_tok_per_s', res.get('error'))}", flush=True)
                    stop()
    with open(os.path.join(OUT, "results.jsonl"), "a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
