"""读 gdn 探针产物（python3.6 兼容）：打印每条记录的 name/dims/maxabs。"""
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/xinfer_gdn_probe.bin"
raw = open(path, "rb").read()
U64 = struct.Struct("<Q")
F32 = struct.Struct("<f")
off = 0
if raw[:4] == b"XINF":
    off = 8  # 跳过魔数
rows = []
while off < len(raw):
    (n,) = U64.unpack_from(raw, off); off += 8
    if n == 0 or off + n > len(raw):
        rows.append(("?? 解析越界", [], 0.0, 0.0)); break
    name = raw[off:off + n].decode("utf-8", "replace"); off += n
    (nd,) = U64.unpack_from(raw, off); off += 8
    dims = []
    for _ in range(nd):
        (d,) = U64.unpack_from(raw, off); off += 8
        dims.append(d)
    (cnt,) = U64.unpack_from(raw, off); off += 8
    vals = [F32.unpack_from(raw, off + 4 * i)[0] for i in range(cnt)]
    off += 4 * cnt
    mx = max(abs(v) for v in vals) if cnt else 0.0
    mean = sum(abs(v) for v in vals) / cnt if cnt else 0.0
    rows.append((name, dims, mx, mean))
for r in rows:
    print("%-26s dims=%-16s maxabs=%.4e absmean=%.4e" % (r[0], str(r[1]), r[2], r[3]))
