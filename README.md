# cuda-u

CUDA device information and precision benchmark for Linux. One static binary,
no dependencies beyond the NVIDIA driver — no Qt, no Python, no root.

MIT licensed, written from scratch.

## Install

Build from source (CUDA Toolkit 12.8+ required, 13.x tested):

```bash
git clone https://github.com/xzwgit/cuda-u.git
cd cuda-u
make
./cuda-u
```

No pre-built binaries are shipped: the tool compiles kernel code for
specific GPU architectures at build time, so building on the target host
with its own CUDA version avoids driver/toolkit mismatches and ensures
the correct `mma.sync` / `tcgen05` code paths are generated. The build
takes ~30 seconds on a typical server.

## What it measures

Each precision in both flavours where the hardware has them — the vector path
(CUDA cores) and the tensor path (`mma.sync` on Tensor Cores):

| | vector | tensor |
|---|---|---|
| FP64 / FP32 | fma chains | — |
| TF32 | — | m16n8k8 |
| BF16 / FP16 | packed hfma2 | m16n8k16 |
| FP8 (E4M3) | — | m16n8k32 (sm_89+) |
| INT8 | `__dp4a` | m16n8k32 s8 |
| FP4 (E2M1) | — | m16n8k32 kind::f8f6f4 (consumer Blackwell) |
| NVFP4 | — | block-scaled GEMM via cuBLASLt (consumer Blackwell) |
| INT4 | — | m16n8k32 s4 |

Plus: board vendor and chip codename, memory maker/type (NVAPI), physical
memory size (NVML — the number `nvidia-smi` prints), tensor-core count, power
limit, PCIe link, per-SM registers/shared memory, theoretical memory
bandwidth, clocks, and host↔device / device↔device copy bandwidth. Every cell
reads a value or `N/A` — never zero, blank or dashes.

Notes on the numbers, so they can be compared against spec sheets:

- rates are measured at the card's **actual** clock, so they usually exceed
  spec values computed at the rated boost clock;
- tensor FP16 uses **FP32 accumulation**; the FP16-accumulate rate is 2×;
- all tensor rates are **dense** (no 2:4 sparsity) — "AI TOPS" headlines are
  sparse, so halve them before comparing;
- **FP4** comes in two rows: `FP4 (E2M1)` is the tool's own mma.sync m16n8k32
  kernel, which shares the FP8 issue slot (reads ≈ FP8); `NVFP4` is the
  block-scaled format spec sheets quote, measured as a real NVFP4 GEMM
  through cuBLASLt (loaded lazily — no hard dependency), reaching the dense
  spec rate (~1.6 PFLOPS on an RTX 5090, ~9.3 PFLOPS on a B300). Cards
  without the hardware or without libcublasLt report `N/A`;
- the two FP4 rows are **not equally reachable**: `NVFP4` works on consumer
  *and* datacenter Blackwell (`cc >= 10.0`), whereas `FP4 (E2M1)` is
  **consumer-only**. The `mma.sync ... kind::f8f6f4` spelling that carries
  E2M1 operands does not exist for the `sm_100` family — ptxas rejects it
  outright (`Instruction 'mma with FP6/FP4 floating point type' not
  supported on .target 'sm_100f'`). Datacenter Blackwell feeds FP4 through
  `tcgen05.mma`, which this tool does not implement, so that row reports
  `N/A` on B200/B300;
- INT4 on Blackwell runs on a legacy compatibility path (~1/7 of INT8);
  Ampere/Ada run it at the full 2× INT8 rate;
- vector FP16/BF16 on GeForce parts are 1:1 with FP32 by design.

## Usage

```
cuda-u                      # full report for every GPU
cuda-u -d 1                 # only device 1
cuda-u -l                   # list devices
cuda-u -j                   # JSON on stdout (array when several GPUs)
cuda-u -j report.json       # JSON to file
cuda-u -n                   # static info only, no benchmarks
```

JSON carries raw values plus `supported` flags, e.g.:

```json
"fp16_tensor": {"value": 84.15, "unit": "Gflop/s", "supported": true}
```

## Sample output

Full report on an RTX 5090 (`./cuda-u`):

```
cuda-u 0.0.2 report - GPU 0: NVIDIA GeForce RTX 5090

---- Device ------------------------------------------------------------------
  Name               NVIDIA GeForce RTX 5090
  Board Vendor       N/A [0x10de]
  Codename           GB202
  Compute            12.0 (Blackwell)
  SMs / Cores        170 / 21760
  Tensor Cores       680 (4/SM)
  Clock              2407 MHz (memory 14001 MHz)
  Power Limit        575 W
  PCIe Link          Gen 5 x16 (now Gen 1 x16)
  Regs/SMEM per SM   65536 / 100 KB
  Driver / Runtime   580.105.08 / CUDA 13.0

---- Memory ------------------------------------------------------------------
  Total              32607 MiB
  Type / Maker       GDDR7 / Samsung
  Bus Width          512 bit
  Bandwidth (spec)   1792 GB/s
  L2 Cache           96 MiB (ECC off)

---- Bandwidth ---------------------------------------------------------------
  H2D pinned         54.13 GB/s
  H2D pageable       26.81 GB/s
  D2H pinned         56.16 GB/s
  D2H pageable       21.39 GB/s
  D2D                2273.34 GB/s

---- Compute -----------------------------------------------------------------
                          vector path       tensor path
------------------------------------------------------------------------------
  FP64                   1.67 Tflop/s               N/A
  FP32                 112.24 Tflop/s               N/A
  TF32                            N/A    126.45 Tflop/s
  BF16                 124.55 Tflop/s    252.96 Tflop/s
  FP16                 124.52 Tflop/s    252.87 Tflop/s
  FP8 (E4M3)                      N/A    505.61 Tflop/s
  INT8 (DP4A)           244.97 Tiop/s               N/A
  INT8                            N/A     998.73 Tiop/s
  FP4 (E2M1)                      N/A    503.45 Tflop/s
  NVFP4                           N/A   1611.60 Tflop/s
  INT4                            N/A     150.26 Tiop/s
------------------------------------------------------------------------------
```

On cards without FP8/FP4 tensor cores (e.g. RTX 3090) those rows read `N/A`;
partner boards name their vendor (`ASUSTeK Computer Inc. [0x1043]`), NVIDIA
reference boards show `N/A [0x10de]`.

## Build

```
make            # needs nvcc (CUDA 12.8+); targets sm_80..sm_120a + PTX
```

## License

MIT — see [LICENSE](LICENSE).
