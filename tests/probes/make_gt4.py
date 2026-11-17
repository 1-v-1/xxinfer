import io

s = io.open("/workspace/gt3.py", encoding="utf-8").read()
# 在 exec_module 之后注入：把 EXTRA 里的额外 CLI 追加到 args_for 的结果上。
inject = (
    "spec.loader.exec_module(P)\n"
    "_extra = os.environ.get(\"EXTRA\", \"\").split()\n"
    "_orig = P.args_for\n"
    "P.args_for = lambda *a, **k: _orig(*a, **k) + _extra\n"
)
s = s.replace("spec.loader.exec_module(P)\n", inject, 1)
s = s.replace('"墙钟 %.2f s" % wall', '"extra=%s" % _extra', 1)
io.open("/workspace/gt4.py", "w", encoding="utf-8").write(s)
print("injected:", s.count("_extra"), "lines:", s.count(chr(10)))
