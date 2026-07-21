#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 MODEL_ID TENSOR_PARALLEL_SIZE" >&2
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

model_id="$1"
tensor_parallel_size="$2"

if [[ ! "$model_id" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]]; then
  echo "MODEL_ID must match ^[A-Za-z0-9][A-Za-z0-9._/-]*$" >&2
  exit 2
fi

case "$tensor_parallel_size" in
  1|2|4|8) ;;
  *)
    echo "TENSOR_PARALLEL_SIZE must be one of: 1, 2, 4, 8" >&2
    exit 2
    ;;
esac

if [ -z "${VLLM_API_KEY:-}" ]; then
  echo "VLLM_API_KEY is required" >&2
  exit 2
fi
if [[ "$VLLM_API_KEY" == *$'\n'* || "$VLLM_API_KEY" == *$'\r'* ]]; then
  echo "VLLM_API_KEY must not contain a newline" >&2
  exit 2
fi

for required_command in curl git nvidia-smi python3 vllm; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required command not found: $required_command" >&2
    exit 1
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_sha="$(git -C vllm rev-parse HEAD)"
tutorial_sha="$(git rev-parse HEAD)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${GPU_VALIDATION_OUTPUT_ROOT:-artifacts/gpu-validation}"
output_dir="$output_root/${run_stamp}-${source_sha}"
mkdir -p "$output_dir"

server_pid=""
server_raw_log="$(mktemp)"
status="failed"
base_url="http://127.0.0.1:8000"

redact_file() {
  local input_path="$1"
  local output_path="$2"
  VLLM_REDACT_VALUE="$VLLM_API_KEY" python3 - "$input_path" "$output_path" <<'PY'
import os
import sys

source, destination = sys.argv[1:]
secret = os.environ["VLLM_REDACT_VALUE"]
with open(source, "rb") as handle:
    data = handle.read()
if secret:
    data = data.replace(secret.encode(), b"<redacted>")
with open(destination, "wb") as handle:
    handle.write(data)
PY
}

write_result() {
  local finished_at="$1"
  python3 - "$output_dir/result.json" "$status" "$source_sha" "$model_id" \
    "$tensor_parallel_size" "$started_at" "$finished_at" <<'PY'
import json
import sys

(
    destination,
    status,
    source_sha,
    model_id,
    tensor_parallel_size,
    started_at,
    finished_at,
) = sys.argv[1:]
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": status,
            "source_sha": source_sha,
            "model_id": model_id,
            "tensor_parallel_size": int(tensor_parallel_size),
            "started_at": started_at,
            "finished_at": finished_at,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}

cleanup() {
  local exit_code="$?"
  set +e
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid" 2>/dev/null
    fi
    wait "$server_pid" 2>/dev/null
  fi

  redact_file "$server_raw_log" "$output_dir/server.log"
  rm -f "$server_raw_log"
  nvidia-smi -q > "$output_dir/nvidia-smi-end.txt" 2>&1
  nvidia-smi topo -m > "$output_dir/nvidia-topology-end.txt" 2>&1

  if [ "$exit_code" -eq 0 ]; then
    status="passed"
  fi
  write_result "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "GPU validation evidence: $output_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if curl --silent --fail --max-time 2 "$base_url/health" >/dev/null 2>&1; then
  echo "port 8000 already serves a healthy endpoint; refusing to reuse it" >&2
  exit 1
fi

printf '%s\n' "$tutorial_sha" > "$output_dir/tutorial-sha.txt"
printf '%s\n' "$source_sha" > "$output_dir/source-sha.txt"
printf '%s\n' "$model_id" > "$output_dir/model-id.txt"
printf '%s\n' "$tensor_parallel_size" > "$output_dir/tensor-parallel-size.txt"
printf 'vllm serve %s --tensor-parallel-size %s --api-key <redacted>\n' \
  "$model_id" "$tensor_parallel_size" > "$output_dir/launch-command.txt"

nvidia-smi -q > "$output_dir/nvidia-smi-start.txt"
nvidia-smi topo -m > "$output_dir/nvidia-topology-start.txt"

python3 - "$output_dir/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys

try:
    import torch
except Exception as exc:  # evidence should survive a broken runtime
    torch_info = {"import_error": repr(exc)}
else:
    torch_info = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
    }

try:
    vllm_version = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    vllm_version = None

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "vllm_package_version": vllm_version,
            "torch": torch_info,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

python3 - "$model_id" "$output_dir/chat-request.json" \
  "$output_dir/stream-request.json" <<'PY'
import json
import sys

model_id, chat_path, stream_path = sys.argv[1:]
base = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "temperature": 0,
    "max_tokens": 8,
}
with open(chat_path, "w", encoding="utf-8") as handle:
    json.dump(base, handle, indent=2, sort_keys=True)
    handle.write("\n")
with open(stream_path, "w", encoding="utf-8") as handle:
    json.dump({**base, "stream": True}, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

vllm serve "$model_id" \
  --tensor-parallel-size "$tensor_parallel_size" \
  --api-key "$VLLM_API_KEY" > "$server_raw_log" 2>&1 &
server_pid="$!"

ready="false"
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming healthy" >&2
    redact_file "$server_raw_log" "$output_dir/server.log"
    tail -n 200 "$output_dir/server.log" >&2
    exit 1
  fi
  if curl --silent --fail --max-time 2 "$base_url/health" >/dev/null 2>&1; then
    ready="true"
    break
  fi
  sleep 1
done

if [ "$ready" != "true" ]; then
  echo "vLLM server did not become healthy within 300 seconds" >&2
  redact_file "$server_raw_log" "$output_dir/server.log"
  tail -n 200 "$output_dir/server.log" >&2
  exit 1
fi

capture_request() {
  local name="$1"
  shift
  local raw_headers
  local raw_body
  local http_code
  raw_headers="$(mktemp)"
  raw_body="$(mktemp)"

  set +e
  http_code="$(curl --silent --show-error --max-time 120 \
    --dump-header "$raw_headers" --output "$raw_body" \
    --write-out '%{http_code}' "$@")"
  local curl_rc="$?"
  set -e

  redact_file "$raw_headers" "$output_dir/${name}.headers"
  redact_file "$raw_body" "$output_dir/${name}.body"
  rm -f "$raw_headers" "$raw_body"
  printf '%s\n' "$http_code" > "$output_dir/${name}.status"

  if [ "$curl_rc" -ne 0 ] || [[ ! "$http_code" =~ ^2[0-9][0-9]$ ]]; then
    echo "$name request failed with curl=$curl_rc http=$http_code" >&2
    return 1
  fi
}

auth_header="Authorization: Bearer $VLLM_API_KEY"
capture_request models \
  --header "$auth_header" \
  "$base_url/v1/models"
capture_request chat \
  --header "$auth_header" \
  --header "Content-Type: application/json" \
  --request POST \
  --data-binary "@$output_dir/chat-request.json" \
  "$base_url/v1/chat/completions"
capture_request stream \
  --no-buffer \
  --header "$auth_header" \
  --header "Content-Type: application/json" \
  --request POST \
  --data-binary "@$output_dir/stream-request.json" \
  "$base_url/v1/chat/completions"
capture_request metrics \
  --header "$auth_header" \
  "$base_url/metrics"
