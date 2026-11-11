#!/bin/bash
# 逐项核对用户报错清单里的 19 个错误点是否都已修复。
cd "$(dirname "$0")"
pass=0; fail=0
chk() { # chk <描述> <期望内容> <文件> <sed表达式>
  local desc="$1" want="$2" file="$3" expr="$4"
  local got
  got=$(sed -n "$expr" "$file" 2>/dev/null)
  if echo "$got" | grep -qF "$want"; then
    echo "  ✅ $desc"; pass=$((pass+1))
  else
    echo "  ❌ $desc"; echo "       期望包含: $want"; echo "       实际: $got"; fail=$((fail+1))
  fi
}

echo "===== 根因 1: openai-protocol 1.18 (Option<Value>) ====="
chk "mcp/manager.rs:398 包 Some" "parameters: Some(tool.input_schema)" src/mcp/manager.rs "398p"
chk "tools/mod.rs build() 包 Some" "parameters: Some(self.parameters)" src/tools/mod.rs "/parameters: Some(self.parameters)/p"
chk "tools/helpers.rs 用 parameters_schema()" "tool.function.parameters_schema().clone()" src/tools/helpers.rs "/parameters_schema/p"
chk "server/parser.rs 用 parameters_schema()" ".parameters_schema()" src/server/parser.rs "/parameters_schema/p"
chk "guidance_grammar.rs:606 用 parameters_schema()" "sanitize_schema_for_llguidance(tool.function.parameters_schema())" src/utils/guidance_grammar.rs "606,607p"
n=$(grep -c "parameters_schema()" src/utils/guidance_grammar.rs)
if [ "$n" -ge 8 ]; then echo "  ✅ guidance_grammar.rs 共 $n 处改用 parameters_schema()"; pass=$((pass+1));
else echo "  ❌ guidance_grammar.rs 只改了 $n 处（应为 8 处）"; fail=$((fail+1)); fi
if grep -q "\.function\.parameters\.get\|function\.parameters\.get" src/utils/guidance_grammar.rs; then
  echo "  ❌ guidance_grammar.rs 仍有裸 .parameters.get()"; fail=$((fail+1))
else echo "  ✅ guidance_grammar.rs 无裸 .parameters.get()"; pass=$((pass+1)); fi

echo
echo "===== 根因 2: c_char 平台适配 (ppc64le = powerpc64) ====="
chk "runner/mod.rs 用 c_char 数组" "[0 as core::ffi::c_char; 128]" src/runner/mod.rs "/0 as core::ffi::c_char/p"
chk "multi_node.rs 用 c_char 切片" "&[core::ffi::c_char; 128]" src/utils/multi_node.rs "/core::ffi::c_char; 128/p"
chk "multi_node.rs 用 c_char 数组" "[0 as core::ffi::c_char; 128]" src/utils/multi_node.rs "/0 as core::ffi::c_char/p"
chk "transfer/mod.rs 用 Vec<c_char>" "Vec<core::ffi::c_char>" src/transfer/mod.rs "/CudaIpcMemHandle(Vec<core::ffi::c_char>/p"
chk "cuda_remote.rs 用 [c_char; 64]" "[c_char; 64] = handle.0.clone()" src/transfer/cuda_remote.rs "/raw_array/p"

echo
echo "===== 残留的 c_char 相关 aarch64 特判（应已全部消除）====="
leftover=$(grep -rn 'target_arch = "aarch64"' src/ --include="*.rs" \
  | grep -E "c_char|Vec<u8>|Vec<i8>|\[u8; |\[i8; |raw_id|raw_array|Id::uninit|parameters" || true)
if [ -z "$leftover" ]; then
  echo "  ✅ src/ 中已无 c_char 相关的 aarch64 特判"; pass=$((pass+1))
else
  echo "  ❌ 仍有 c_char 相关的 aarch64 特判:"; echo "$leftover"; fail=$((fail+1))
fi
# 仅日志文案分支，与 c_char 无关，确认它仍在
if grep -q 'target_os = "macos", target_arch = "aarch64"' src/utils/mod.rs; then
  echo "  ✅ utils/mod.rs 的 aarch64 分支是日志文案分支（与 c_char 无关，保留）"; pass=$((pass+1))
else
  echo "  ❌ utils/mod.rs 的日志分支不见了"; fail=$((fail+1))
fi

echo
echo "===== Cargo.toml 版本 ====="
chk "openai-protocol 已升到 1.18" 'openai-protocol = "1.18"' Cargo.toml "/openai-protocol/p"

echo
echo "==================================="
echo "  通过: $pass    失败: $fail"
echo "==================================="
[ "$fail" -eq 0 ]
