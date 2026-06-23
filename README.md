# llama-quik

**llama-quik** is an experimental high-performance fork/patchset for `llama.cpp` focused on making very large Mixture-of-Experts (MoE) GGUF models usable on limited-VRAM systems.

The core idea is simple:

> Keep the full routed MoE expert pool in CPU RAM, profile which experts are actually used, then promote hot experts into GPU VRAM so warmed decode can run faster without requiring the entire model to fit in VRAM.

TLDR:

> Make huge MoE GGUF models usable and faster on hardware that cannot fit the full expert pool in VRAM.

`llama-quik` is aimed at large MoE GGUF models where dense/shared weights and runtime buffers can use the GPU, while the much larger routed expert pool is paged, cached, or promoted dynamically.

---

## Status

`llama-quik` is experimental.

Current backend status:

* CPU-MoE fallback works.
* MoE expert routing profiler works.
* GPU VRAM cache allocation works.
* Whole-layer expert promotion works.
* Hard backend per-expert GPU slot cache works for decode.
* CUDA graph capture is bypassed for dynamic expert promotion.
* Prompt/batch phase uses a safe fallback path.
* Warmed decode uses the GPU expert slot cache where available.
* Normal llama.cpp multi-GPU tensor split can still be used.
* The expert-slot cache is currently centered on the main CUDA device, not fully distributed across all GPUs.

In short:

```text
Supported today:
  multi-GPU model compute via llama.cpp
  single-main-GPU expert slot cache via llama-quik

Not fully implemented yet:
  distributed expert slot caches across every GPU
  mixed CPU/GPU expert execution inside the same layer for arbitrary batch shapes
  fully async multi-device expert promotion/eviction
```

---

## Why llama-quik exists

Large MoE models are different from dense models.

A dense model needs most of its weights available every token. An MoE model has many experts, but only a subset are routed for each token. That creates an opportunity:

```text
Dense model:
  every layer uses its full weights every token

MoE model:
  router selects a small set of experts per token
  many experts are cold or rarely used
```

`llama-quik` exploits that by profiling routing behavior and using GPU VRAM for the hot expert paths instead of trying to fit the entire expert pool on the GPU.

---

## How it works

When `--moe-expert-pager` is enabled, `llama-quik`:

1. Loads routed MoE expert tensors in CPU memory.
2. Keeps normal dense/shared model parts on GPU when llama.cpp can offload them.
3. Profiles which experts are selected during real prompts.
4. Allocates GPU expert slot cache memory.
5. JIT-promotes selected experts from CPU RAM into GPU VRAM slots.
6. Uses the GPU slot cache during decode.
7. Falls back safely when the cache path is not ready or not appropriate.

Conceptually:

```text
CPU RAM:
  full routed expert pool

GPU VRAM:
  dense/shared compute
  KV/runtime buffers
  hot expert slot cache

Runtime:
  selected expert ID -> GPU slot ID
```

---

## Requirements

`llama-quik` reduces VRAM pressure, not total memory requirements.

The full model still needs to fit in system RAM.

`llama-quik` is most useful when:

```text
System RAM:
  large enough to hold the full GGUF model and runtime memory

GPU VRAM:
  not large enough to hold the full model
  but large enough to cache hot experts and run dense/shared compute
```

Using `--no-mmap` can improve speed on some systems because model data is loaded into RAM instead of relying on demand paging from disk. However, `--no-mmap` also increases the importance of having enough available system RAM.

---

## Features

### MoE expert pager

Enable with:

```bash
--moe-expert-pager
```

This turns on the MoE expert pager system.

---

### Hard backend per-expert GPU slot cache

The hard backend is the current recommended backend.

It allocates compact GPU slot banks for selected expert tensors. Instead of copying whole MoE layers, it promotes individual selected experts into GPU slots.

This is more VRAM-efficient than whole-layer caching.

---

### Whole-layer fallback cache

A previous backend promoted complete hot MoE layers into GPU VRAM.

That path is still useful as a fallback or debugging mode because it is simpler and more conservative.

Use:

```bash
--moe-expert-whole-layer-cache
```

Current recommended default is the hard backend, not whole-layer mode.

---

### Auto-tuned server defaults

When `--moe-expert-pager` is enabled, `llama-quik` automatically applies several performance-oriented defaults:

```text
--no-mmap
--flash-attn on
--cache-ram 0
--ctx-checkpoints 0
--cache-type-k q8_0
--cache-type-v q8_0
-np 1
```

These defaults are chosen for single-user raw token/sec performance on large MoE models.

Disable automatic tuning with:

```bash
--moe-expert-pager-no-auto-tune
```

---

## Build

### Windows CUDA build

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j 8
```

Verify the new flags exist:

```cmd
build\bin\Release\llama-server.exe --help | findstr moe
```

---

### Linux CUDA build

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build --target llama-server -j 8
```

For older Pascal GPUs, use the appropriate CUDA architecture explicitly. For example:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=60

cmake --build build --target llama-server -j 8
```

Older Pascal GPUs should generally use CUDA 12.x. CUDA 13.x may not support them properly.

---

## Basic usage

Example using a large GGUF MoE model:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --host 127.0.0.1 \
  --port 8080
```

Log/debug mode:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --moe-expert-log \
  --host 127.0.0.1 \
  --port 8080
```

The backend is active when the logs show:

```text
backend=gpu-expert-slot-cache-active(jit-promote-cpu-to-gpu)
```

---

## Recommended commands

### General benchmark command

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --host 127.0.0.1 \
  --port 8080
```

### General debug command

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --moe-expert-log \
  --host 127.0.0.1 \
  --port 8080
```

### Explicit tuned command

Most of these settings are automatically applied by `--moe-expert-pager`, but they can also be specified manually:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -np 1 \
  -ngl 999 \
  --cache-ram 0 \
  --ctx-checkpoints 0 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --moe-expert-pager \
  --moe-expert-cache-experts 64 \
  --moe-expert-prefetch 8 \
  --moe-expert-cache-policy lfru \
  --no-mmap \
  --host 127.0.0.1 \
  --port 8080
```

### Conservative limited-VRAM command

Use this if auto VRAM sizing is unstable:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --moe-expert-log \
  --moe-expert-cache-vram-mib 8192 \
  --moe-expert-vram-reserve-mib 2048 \
  --moe-expert-cache-experts 48 \
  --moe-expert-prefetch 4 \
  --moe-expert-cache-policy lfru \
  --host 127.0.0.1 \
  --port 8080
```

### Aggressive limited-VRAM command

Use this only after confirming the conservative command is stable:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --moe-expert-log \
  --moe-expert-cache-vram-mib 10240 \
  --moe-expert-vram-reserve-mib 1536 \
  --moe-expert-cache-experts 64 \
  --moe-expert-prefetch 8 \
  --moe-expert-cache-policy lfru \
  --host 127.0.0.1 \
  --port 8080
```

---

## Multi-GPU usage

Normal llama.cpp multi-GPU tensor parallelism can still be used with:

```text
--split-mode
--tensor-split
--main-gpu
```

However, the current `llama-quik` expert slot cache is not yet a fully distributed multi-GPU cache.

Current design:

```text
All GPUs:
  normal llama.cpp tensor/model parallel work

Main GPU:
  llama-quik MoE expert slot cache
```

Future distributed design:

```text
GPU0:
  expert cache shard 0

GPU1:
  expert cache shard 1

GPU2:
  expert cache shard 2

GPU3:
  expert cache shard 3
```

For a multi-GPU single-response setup, start with an explicit VRAM cache cap so the main GPU has room for both normal llama.cpp work and the expert cache:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --split-mode row \
  --tensor-split 0.5,1,1,1 \
  --main-gpu 0 \
  --moe-expert-pager \
  --moe-expert-log \
  --moe-expert-cache-vram-mib 8192 \
  --moe-expert-vram-reserve-mib 2048 \
  --moe-expert-cache-experts 64 \
  --moe-expert-prefetch 8 \
  --moe-expert-cache-policy lfru \
  --host 127.0.0.1 \
  --port 8080
```

If row split is unstable or slower, try:

```bash
--split-mode layer
```

Do not describe the current expert cache as fully multi-GPU parallel yet. Normal model compute can use multiple GPUs, but the MoE expert pager cache itself is currently centered on the main CUDA device.

---

## Benchmarking

Do not judge performance from the first prompt alone.

The first prompt may include:

* routing profiler warmup
* GPU expert cache allocation
* JIT expert promotion
* cache build stalls

Recommended benchmark flow:

1. Start the server with `--moe-expert-pager --moe-expert-log`.
2. Send a warmup prompt.
3. Wait for backend activation.
4. Send the same prompt again.
5. Compare `eval time = ... tokens per second`.
6. Use the second or third prompt as the warmed result.

Useful warmup/test prompt:

```text
Write a 3 paragraph essay on LLMs.
```

The backend is active when logs show:

```text
backend=gpu-expert-slot-cache-active(jit-promote-cpu-to-gpu)
```

The most important timing line is:

```text
eval time = ... tokens per second
```

The less important timing line is:

```text
prompt eval time = ... tokens per second
```

Prompt processing may use safer fallback paths, while generated-token decode is where the pager is intended to help most.

---

## Comparing against normal llama.cpp

Normal tuned llama.cpp baseline:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -np 1 \
  -ngl 999 \
  --no-mmap \
  --flash-attn on \
  --cache-ram 0 \
  --ctx-checkpoints 0 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --host 127.0.0.1 \
  --port 8080
```

llama-quik hard backend:

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 2048 \
  -ngl 999 \
  --moe-expert-pager \
  --host 127.0.0.1 \
  --port 8080
```

For fair comparison:

* use the same model
* use the same context size
* use the same max tokens
* use the same sampling settings
* compare warmed second-prompt decode speed
* compare `eval time`, not just prompt processing speed

---

## Command-line options

### `--moe-expert-pager`

Enables the MoE expert pager.

This starts routed expert tensors in CPU memory, profiles expert routing, then activates the GPU expert cache after enough samples are collected.

---

### `--moe-expert-log`

Prints MoE pager summaries.

Useful for verifying backend activation, cache size, profiler status, and general behavior.

Recommended for testing:

```bash
--moe-expert-pager --moe-expert-log
```

Recommended for benchmarking after validation:

```bash
--moe-expert-pager
```

---

### `--moe-expert-cache-experts N`

Controls the number of expert slots per cached expert tensor/bank.

Example:

```bash
--moe-expert-cache-experts 64
```

Lower values use less VRAM but may increase misses and evictions.

Higher values can reduce misses but consume more VRAM.

Common values:

```text
48  = conservative
64  = recommended default
96  = aggressive
128 = very aggressive
```

---

### `--moe-expert-prefetch N`

Controls how many likely-needed experts the pager tries to prepare ahead of time.

Example:

```bash
--moe-expert-prefetch 8
```

Lower values reduce CPU->GPU copy pressure.

Higher values may warm the cache faster, but can waste bandwidth if predictions are poor.

Common values:

```text
4  = conservative
8  = recommended
16 = aggressive
```

---

### `--moe-expert-cache-policy POLICY`

Controls eviction policy.

Supported policies:

```text
lru
lfu
lfru
```

Meaning:

```text
lru:
  least recently used

lfu:
  least frequently used

lfru:
  hybrid frequency + recency policy
```

Recommended:

```bash
--moe-expert-cache-policy lfru
```

MoE routing often has both short-term locality and long-term hot experts, so `lfru` is usually the best default.

---

### `--moe-expert-cache-vram-mib N`

Caps how much VRAM the pager may use for the expert cache.

Example:

```bash
--moe-expert-cache-vram-mib 8192
```

If omitted or set to `0`, auto mode tries to use available VRAM after reserve.

For testing, explicit caps are safer.

---

### `--moe-expert-vram-reserve-mib N`

Leaves a safety reserve of free VRAM.

Example:

```bash
--moe-expert-vram-reserve-mib 2048
```

Recommended values:

```text
512  = aggressive
1024 = moderate
1536 = safer
2048 = conservative
```

Older GPUs and multi-GPU setups usually need a larger reserve.

---

### `--moe-expert-vram-max-mib N`

Sets a maximum auto-sized VRAM limit.

Useful if you want auto-sizing but do not want it to consume nearly all available VRAM.

---

### `--moe-expert-whole-layer-cache`

Uses the older whole-layer cache backend instead of the hard per-expert slot backend.

This is slower and less VRAM-efficient, but it is simpler and useful for debugging.

---

## Understanding the logs

### Profiler warming up

```text
ready=false
backend=gpu-cache-arena-reserved(cpu-moe-fallback)
```

The profiler is collecting routing data. Expect baseline-ish speed.

---

### Hard backend activated

```text
backend=gpu-expert-slot-cache-active(jit-promote-cpu-to-gpu)
```

The expert slot cache is active. Decode should be faster after the cache warms.

---

### Example activation line

```text
MoE expert hard backend active: allocated <N> per-expert slot tensors across <N> hot layers on CUDA<device> (<VRAM> MiB, <slots> slots/tensor)
```

Meaning:

```text
<N> cached expert tensor banks
<N> hot layers involved
<VRAM> MiB VRAM allocated
<slots> slots per cached tensor bank
```

---

## Observed performance behavior

Performance depends heavily on:

* model quantization
* CPU RAM bandwidth
* PCIe bandwidth
* GPU architecture
* prompt length
* output length
* routing locality
* cache churn
* number of cached slots
* VRAM cap/reserve
* whether the prompt is first or warmed

Approximate observed progression in one test configuration:

```text
Original / early baseline:
  ~1.26 tokens/sec

Tuned CPU-MoE fallback:
  ~5.58 tokens/sec

With Flash Attention:
  ~6.38 tokens/sec

With --no-mmap and tuned settings:
  ~11.33 tokens/sec

Best warmed llama-quik windows:
  ~13–14 tokens/sec
```

These numbers are not guaranteed. They are examples of observed behavior from one test configuration.

In longer responses, token/sec may start strong after the expert cache warms, then gradually fall later in the generation. This is likely caused by routing shifts, cache churn, evictions, or repeated CPU-to-GPU expert promotion.

---

## Known limitations

### First prompt is slower

The first prompt includes profiling and cache setup.

Use the second or third prompt for benchmarking.

---

### Prompt phase may use fallback

The current hard backend uses the safer path for prompt/batch processing and the GPU expert slot cache for decode.

This avoids batch-shape issues and prioritizes generated token/sec.

---

### Long generations may slow down

Very long generations can degrade if the cache churns or if routing shifts over time.

Future work will add detailed real hit/miss counters and better long-generation eviction behavior.

---

### Expert cache is not fully multi-GPU distributed yet

Normal llama.cpp multi-GPU tensor split works.

The expert slot cache itself is currently centered on the main CUDA device.

---

### CUDA graphs and dynamic promotion

Dynamic expert promotion is incompatible with CUDA graph capture in the affected graph.

`llama-quik` bypasses graph capture for the dynamic cache path.

---

## Potential fixes and optimization directions

`llama-quik` is still experimental, and two major optimization areas remain: long-generation slowdown and true distributed multi-GPU expert caching.

### Long-generation slowdown

Possible causes:

```text
routing pattern shifts as the response continues
hot experts change over time
cache slots churn between old and new expert sets
repeated CPU->GPU promotions add copy overhead
eviction policy is not yet tuned for long generations
some layers may need more slots than others
```

Potential future fixes:

```text
real backend hit/miss counters:
  measure actual hard-backend slot hits, misses, promotions, evictions, and copied bytes

per-layer cache statistics:
  show which MoE layers are stable and which layers churn

adaptive per-layer slot allocation:
  give more slots to layers with high traffic or high miss rates

expert pinning:
  permanently keep the most-used experts resident once they prove hot enough

warm-profile preloading:
  after the warmup prompt, pre-promote the top experts before the real benchmark prompt starts

long-context re-profiling:
  periodically refresh the hot-expert plan during very long generations

promotion throttling:
  avoid too many CPU->GPU copies in a short window if promotion traffic starts hurting decode speed

async promotion:
  move selected experts with safer stream/event scheduling so copies block decode less

better eviction scoring:
  combine recency, frequency, layer importance, and promotion cost instead of using a simple global policy
```

---

### Multi-GPU expert cache

The ideal future version would allow all GPUs to contribute not only to normal tensor-split model execution, but also to the MoE expert cache itself.

Potential approaches:

```text
shard by layer:
  assign different MoE layers' expert caches to different GPUs

shard by expert ID:
  assign expert ID ranges to different GPUs

replicate hottest experts:
  keep the most common experts on multiple GPUs to reduce cross-device traffic

device-local routing:
  send selected expert work to the GPU that owns the needed expert slot

cross-GPU result merge:
  gather expert outputs back into the normal llama.cpp graph after expert computation

topology-aware planning:
  choose cache placement based on PCIe/NVLink bandwidth and peer-to-peer support

multi-device cache telemetry:
  report hits, misses, promotions, and evictions per GPU
```

---

## Troubleshooting

### Startup is slower

Expected. The pager reserves VRAM, initializes cache metadata, profiles routing, and promotes experts.

For faster testing, cap the cache:

```bash
--moe-expert-cache-vram-mib 8192
```

---

### Too much VRAM usage

Increase reserve or cap cache:

```bash
--moe-expert-vram-reserve-mib 2048
--moe-expert-cache-vram-mib 8192
```

---

### Backend does not activate

Run with:

```bash
--moe-expert-log
```

Look for:

```text
backend=gpu-expert-slot-cache-active(jit-promote-cpu-to-gpu)
```

If the backend remains in fallback mode, try a longer warmup prompt or lower the cache VRAM cap/reserve settings.

---

## Roadmap

Planned improvements:

* Real backend hit/miss counters.
* Per-layer hard-backend cache hit rate reporting.
* Promotion/eviction telemetry.
* Better long-generation cache stability.
* Prewarming top experts after profiler warmup.
* Distributed multi-GPU expert slot caches.
* Async CPU->GPU promotion with safer stream/event scheduling.
* Smarter per-layer slot allocation.
* Optional saved profile per model hash.
* Better tuning presets for older CUDA GPUs.
* More correctness tests for expert ID remapping.

---

## Project philosophy

`llama-quik` prioritizes practical MoE inference on real hardware over perfect theoretical design.

The current implementation is intentionally incremental:

```text
v1:
  CPU-MoE fallback + profiler

v2:
  whole-layer GPU expert cache

v3:
  hard backend per-expert GPU slot cache

future:
  fully distributed multi-GPU expert cache
```

---

## Credits

`llama-quik` is based on `llama.cpp` and GGML.

This project is experimental and should be treated as a research/performance patchset until the backend is fully validated across more models, GPUs, and CUDA versions.

---

## Contributing

Pull requests, bug reports, testing notes, and performance results are welcome.

`llama-quik` is experimental and needs help from people testing different models, GPUs, CUDA versions, operating systems, and multi-GPU setups. If you find a crash, build issue, incorrect behavior, or performance regression, please open an issue with as much detail as possible.

Helpful contributions include:

* Linux and Windows build fixes.
* CUDA backend fixes.
* Multi-GPU expert cache experiments.
* Better long-generation cache stability.
* Real hit/miss/promotion/eviction telemetry.
* Better default settings for different GPU classes.
* Documentation improvements.
* Reproducible benchmarks against normal llama.cpp.
* Testing with different MoE GGUF models and quantizations.

This project is rough and experimental, so even small fixes, logs, and benchmark results are useful.

---

## Disclaimer

This is not an official `llama.cpp` feature.

`llama-quik` is an experimental patchset I built in my free time over a short period (a couple of hours). It is a rough prototype, not a polished product. It may have bugs, crash, perform poorly, or not work at all on some systems.

I may not come back to maintain or fix this project, so pull requests and community contributions may be the only way it continues improving.

Use at your own risk. Expect bugs, crashes, backend-specific issues, build issues, and large performance variation across models, GPUs, CUDA versions, operating systems, and driver setups.

Always compare against a normal tuned llama.cpp baseline before judging speedups.
