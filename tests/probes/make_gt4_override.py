import io

s = io.open("/workspace/gt3.py", encoding="utf-8").read()
inject = (
    "spec.loader.exec_module(P)\n"
    "_extra = os.environ.get(\"EXTRA\", \"\").split()\n"
    "_orig = P.args_for\n"
    "def _ov(m, s_, q):\n"
    "    r = _orig(m, s_, q); out = []\n"
    "    i = 0\n"
    "    while i < len(r):\n"
    "        if r[i] in _extra:\n"
    "            i += 2 if i + 1 < len(r) and not r[i + 1].startswith(\"--\") else 1\n"
    "            continue\n"
    "        out.append(r[i]); i += 1\n"
    "    return out + _extra\n"
    "P.args_for = _ov\n"
)
s = s.replace("spec.loader.exec_module(P)\n", inject, 1)
io.open("/workspace/gt4.py", "w", encoding="utf-8").write(s)
print("ok", s.count("_extra"))
