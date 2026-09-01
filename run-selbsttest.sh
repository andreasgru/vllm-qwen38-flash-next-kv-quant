#!/bin/bash
# Wegwerf-Container fuer die Kernel-Selbsttests (kein Server, kein Modell).
# Aufruf (im Repo-Verzeichnis): ./run-selbsttest.sh kvq-kerneltest.py test-fp8.log
#         QFN_DIR=/pfad/zum/repo ./run-selbsttest.sh kvq-nvfp4-test.py test-nvfp4.log
set -u
TEST="${1:?testdatei}"
LOG="${2:?logdatei}"
IMG="${QFN_VLLM_IMG:-vllm/vllm-openai:qwen38-flash-next}"
{
echo "=== SELBSTTEST $TEST $(date -u "+%F %T UTC") ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
docker run --rm --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e NVIDIA_DISABLE_REQUIRE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.0f \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e QFN_KVQ_NVFP4_WRITER="${QFN_KVQ_NVFP4_WRITER:-native}" \
  -e QFN_KVQ_V_SF_SWIZZLED="${QFN_KVQ_V_SF_SWIZZLED:-}" \
  -e QFN_KVQ_PROD_PAGES="${QFN_KVQ_PROD_PAGES:-1600,2848}" \
  -v "${QFN_DIR:-$PWD}":/qfn \
  --entrypoint bash "$IMG" -c "python3 /qfn/kvq-patch.py && python3 /qfn/$TEST"
echo "=== rc=$? ENDE $(date -u "+%F %T UTC") ==="
} > "$LOG" 2>&1
