# Quantised KV cache (fp8_e4m3 and nvfp4) for the QSA path of Qwen3.8-Flash-Next in vLLM

A container-side patch that lets vLLM run `Qwen3.8-Flash-Next` with `--kv-cache-dtype fp8_e4m3` or `--kv-cache-dtype nvfp4` on the QSA (Qwen sparse attention) layers. Stock vLLM refuses anything but bf16 there. Measured on one RTX PRO 6000 Blackwell (SM120, 96 GB): the KV pool grows 1.889× with fp8 and 3.059× with nvfp4 at identical VRAM.

**Status:** not upstream, no pull request yet. Built and validated on one machine and one architecture. Comments in the code are German. Everything below was measured, nothing is quoted from vendor material.

## What it does

Qwen3.8-Flash-Next has 48 layers, 36 gated-deltanet (GDN) layers and 12 QSA layers with 2 KV heads and head_dim 256. Only the 12 QSA layers hold a per-token KV cache. Their main KV is read by exactly one Triton kernel (`_qsa_sparse_paged_gqa_splitk_kernel`) and written by one inherited path (`reshape_and_cache_flash`). The write path can already emit fp8 and nvfp4, the KV spec already passes `kv_quant_mode`, and the per-layer scales exist as device buffers. What is missing is the read side, plus a set of guards that refuse anything but bf16.

The patch:

- adds a dequantisation branch to the read kernel (`KV_QUANT_MODE` is a `constexpr`, so the bf16 path compiles out unchanged),
- lifts the bf16 guards on this path, including one inherited from `FlashAttentionImpl` that rejects SM120 wholesale although QSA never calls FlashAttention kernels,
- reinterprets vLLM's `uint8` storage as `float8_e4m3fn` before the kernel (otherwise Triton decodes integers),
- for nvfp4 adds the page geometry (K and V in separate head slots, 144 bytes per cell of 256 values: 128 bytes of E2M1 data plus 16 fp8 block scales), an own cache-update override and the in-kernel E2M1 unpack with block-scale lookup,
- keeps the two indexer caches in bf16 by construction (`QSACompressedKeyCache` pins its dtype, `QSAKeyStateCache` packs int64 mRoPE positions into bf16 cells), so the block selector never sees quantised bytes.

Every anchor in `kvq-patch.py` is asserted with a count of exactly one against the image file system. If the image layout differs, the script refuses instead of patching half.

## Built against

| | |
|---|---|
| Image | `vllm/vllm-openai:qwen38-flash-next` = vLLM `0.1.dev20073+g8e685d198` |
| Module in the image | `vllm/models/qwen3_8_flash_next/` (upstream main names it `qwen4_exp/`) |
| Model | `RadixArk/Qwen3.8-Flash-Next-NVFP4`, TP1, MTP off |
| GPU | 1× RTX PRO 6000 Blackwell (SM120, 96 GB), driver 595.71 |

Upstream `main` still has the bf16-only guards on this path (checked 2026-09-02). The patch has not been ported to the `qwen4_exp` module name.

## How to apply

Inside the container, before the server starts:

```bash
python3 /qfn/kvq-patch.py        # exit 0 = applied or already applied
```

Then start vLLM with the dtype you want. Settings we run with:

```bash
# fp8, native 262k window (our default)
QFN_KVQ_NVFP4_WRITER=native QFN_KVQ_V_SF_SWIZZLED=1 QFN_KVQ_SF_MODE=2 \
  vllm serve ... --kv-cache-dtype fp8_e4m3

# nvfp4, for pools that have to reach 1M tokens
QFN_KVQ_NVFP4_WRITER=native QFN_KVQ_V_SF_SWIZZLED=1 QFN_KVQ_SF_MODE=2 QFN_KVQ_WIDE_BLOCK_N=32 \
  vllm serve ... --kv-cache-dtype nvfp4
```

Environment switches (all optional, defaults are the validated ones):

| Variable | Default | Meaning |
|---|---|---|
| `QFN_KVQ_SF_MODE` | `2` | nvfp4 read kernel variant. `2` = reordered loads plus narrow scale tile (validated). `0` = original, `3` to `5` = experimental and not validated. |
| `QFN_KVQ_WIDE_BLOCK_N` | unset | Width of the wide (prefill) tile. `32` is what took the nvfp4 prefill cost from 3.01× to 1.75× of fp8. |
| `QFN_KVQ_STAGES` | unset | Override the pipeline-stage rule. Unset keeps one stage on the wide tile (shared memory limit on SM120). |
| `QFN_KVQ_NVFP4_WRITER` | `native` | `native` = precompiled CUDA writer, `triton` = architecture-independent software writer. |
| `QFN_KVQ_V_SF_SWIZZLED` | unset | Scale layout of the V side. The native writer swizzles V block scales; the self-test measures which layout the image uses. Set `1` for the native writer. |

## Self-tests, before any server start

A server start with this model takes 10 to 20 minutes, and a wrong dequant branch would show up afterwards only as a quality number, which is the easiest kind of error to misread as "quantisation costs quality". The two self-tests separate the questions:

```bash
./run-selbsttest.sh kvq-kerneltest.py  test-fp8.log     # fp8 dequant branch vs float64 reference
./run-selbsttest.sh kvq-nvfp4-test.py  test-nvfp4.log   # nvfp4 layout probe (4 scale hypotheses) + kernel
```

Both compare the kernel on quantised inputs against a reference computed on the same dequantised values, so they measure the kernel, not the quantisation. Both passed with zero errors on the first run. The nvfp4 test additionally measures the scale layout instead of assuming it: the correct hypothesis lands at a relative error of about 0.09 (pure nvfp4 rounding), every wrong one at about 1.33.

## Results

All numbers from paired runs on the same day, same checkpoint, same start line, only `--kv-cache-dtype` varied. bf16 control run included.

### fp8_e4m3, scales fixed at 1.0

| | bf16 | fp8_e4m3 | |
|---|---:|---:|---:|
| KiB per token, main KV | 25.451 | **13.477** | ×1.889 |
| KV pool at identical VRAM | 535,178 tokens | **1,010,693** | ×1.889 |
| GSM8K, paired, n=100 | 97/100 | 96/100 | McNemar p = 1.0000, one discordant pair |
| Perplexity, three corpora, 1.49 M tokens | | +0.116 / −0.063 / −0.031 % | all CIs include 0; run-to-run self-difference +0.0995 % |
| Needle in a haystack, 65k to 235k, standard and hard sets | 50/50 | **70/70** | exhaustion 0.0 % |
| Decode, two prompt classes | 86.0 / 83.8 t/s | 85.3 / 82.5 t/s | −0.8 % / −1.6 % |
| Prefill, true cache miss, 12.8k tokens | 10,326 t/s | 9,724 t/s | −5.8 % |
| Saturation / underflow of cached K/V at scale 1.0 | | 0.0000 % / 0.0000 % | calibrated scales not needed on this card |

A check that costs nothing: the bf16 residual above the main KV is 25.451 − 24.00 = 1.451 KiB per token before and 13.477 − 12.00 = 1.477 after. Exactly one thing halved. Had a guard silently stayed active the number would sit near 25.45; had the GDN or indexer caches been quantised by accident it would sit below 12.

### nvfp4, scales fixed at 1.0

| | nvfp4 | vs bf16 | vs fp8 |
|---|---:|---:|---:|
| KiB per token, main KV | **8.320** | ×3.059 | ×1.62 |
| KV pool at identical VRAM | **1,637,114** tokens | ×3.059 | ×1.62 |
| GSM8K, paired, n=100 | p = 1.0000 | | zero discordant pairs vs fp8 |
| Needle in a haystack | 70/70 | | |
| Perplexity | **+0.37 to +0.59 % worse**, consistent, all CIs exclude 0 | | 3.4 to 6× above our run-to-run noise floor |
| Prefill | | | 1.75× slower at short context after the tile fix, 2.15× at 840k, 2.00× at 1M |
| Decode | flat, under 2 % | | |

nvfp4 is not free on this model. The perplexity penalty is real and does not surface on the task axes we ran. Under a 1M configuration (YaRN factor 4.0 over the native 262,144 window) the pool holds 1,812,324 tokens.

### 1M context with nvfp4 KV

Needle tests at 262k and 524k: 10/10 each. Needles near 1M: 6/6. Aggregation tasks near 1M: 6/12, where SGLang on the same KV precision class scored 8/8 on byte-identical instances (Fisher exact p = 0.0769, three independent instance sets, three times exactly 2/4). We treat that as suspicious and unresolved. More runs would settle it, a better argument would not.

## Operating rule we ended up with

| Context | KV dtype | Why |
|---|---|---|
| up to 262k | fp8_e4m3 | parity on every gate, 1.89× pool, decode and prefill within a few percent |
| above roughly 500k | nvfp4 | only way to reach a 1M pool on 96 GB; perplexity penalty accepted, prefill 2× of fp8 |
| bf16 | off | double the memory with no measured benefit |

## What is not tested

- One machine, one architecture (SM120). The AMD QSA mirror (`amd/qsa.py`) carries the same guards and was left untouched.
- MTP (speculative decoding) and batch scaling were not part of the gate chain.
- The nvfp4 read kernel modes `3` to `5` are experimental; mode `3` produced wrong values at `BLOCK_N=16` in the self-test and is disabled by default.
- The aggregation gap near 1M (above) is open.

## Related work

- SGLang: fp8 KV on the QSA path was attempted upstream on 2026-08-26 ([sgl-project/sglang#36545](https://github.com/sgl-project/sglang/issues/36545), fix in [#36644](https://github.com/sgl-project/sglang/pull/36644), open). MiaAI-Lab ships nvfp4 KV for QSA in SGLang as six container patches for dual DGX Spark.
- vLLM, fp8 on QSA: `alesha-pro/qwen38-flash-next-4x3090` publishes calibrated fp8 QSA KV as overlays for 4× RTX 3090 (2026-08-28). [vllm-project/vllm#54426](https://github.com/vllm-project/vllm/issues/54426) is an RFC with a working fp8 diff on GB10 (2026-08-30), corroborated on a second GB10 and merged into `blazux/qwen3.8-Flash-DGX` as patch 7. That work and this one were done independently and hit the same four integration points; the indexer cache is handled differently (see above).
- vLLM, nvfp4 on QSA: no other implementation known to us as of 2026-09-02. [vllm-project/vllm#54772](https://github.com/vllm-project/vllm/pull/54772) enables nvfp4 KV on SM120 in the FlashInfer backend only.

Measurement write-up as a thread: https://x.com/andreas__g/status/2094914909230272628

## License

Apache-2.0, same as vLLM. The patch embeds short anchor snippets from vLLM to locate its insertion points.
