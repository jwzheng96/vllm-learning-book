#!/usr/bin/env bash
# 一条命令搞定本地预览：重新组装 _web + 启动 mkdocs serve。
#
# 为什么需要这个脚本：
#   mkdocs.yml 的 docs_dir 是 _web，而 _web 是 build_site.sh 的产物。
#   mkdocs serve 对「外部脚本覆盖写 _web」感知不灵敏（启动后把文件树读进
#   内存，后续外部覆盖经常触发不了 livereload），导致改完源 md 跑了
#   build_site.sh 之后，浏览器还是看到旧页面。本脚本保证每次启动前
#   _web 是最新的，并且 serve 是新进程。
#
# 用法：
#   bash scripts/serve.sh              # 默认 127.0.0.1:8899
#   bash scripts/serve.sh 9000         # 自定义端口
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8899}"
ADDR="127.0.0.1:${PORT}"

cd "$ROOT"

# 1. 如果端口已被 mkdocs 占用，先停掉（避免旧进程吐旧缓存）
OLD_PID="$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null || true)"
if [ -n "$OLD_PID" ]; then
  echo "→ 端口 ${PORT} 已被 pid=${OLD_PID} 占用，先停掉..."
  kill "$OLD_PID" 2>/dev/null || true
  sleep 1
fi

# 2. 重新组装 _web（docs_dir），确保 serve 读到的是最新源
echo "→ 重新组装 _web/ ..."
bash "$ROOT/scripts/build_site.sh"

# 3. 启动 mkdocs serve（--clean 清掉 mkdocs 自己的构建缓存）
echo "→ 启动 mkdocs serve 于 http://${ADDR} ..."
exec mkdocs serve -a "$ADDR" --clean
