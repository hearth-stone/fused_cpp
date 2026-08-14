# llama.cpp SDPA 接入文档

本文说明如何把抽离版 fp32 `flash2_neon_l3kv_packqkv_pbf16pv` 接入
llama.cpp。当前推荐路径是：

```text
standalone/fp32_packqkv/
```

这个目录不改现有 `csrc` SDPA 框架，不注册版本，不链接 Python、pybind、
libtorch 或 `fused_cpp._C`。它只保留一个纯 C/C++ 裸指针入口和一个 bench。
目录内的 `csrc/` 是该目标需要的 header-only 依赖拷贝，Makefile 只使用
`-I. -Icsrc -Icsrc/sdpa_microkernels`，不会 include repo 根目录下的
`../../csrc`。

## 编译

在 Arm Codex（`Arm-codex-internal` / `Arm-codex`）上：

```sh
cd /home/zhangxu/code/standalone/fp32_packqkv
make -j
```

只编 shared library：

```sh
make lib
```

产物：

```text
standalone/fp32_packqkv/libfused_cpp_sdpa_fp32_packqkv.so
```

动态依赖应只有 C++ runtime、libm、libgomp、libgcc、libc，不应出现
`libtorch_python`、`libtorch`、`python` 或 pybind。

## 导出符号

当前 `.so` 导出三个 llama.cpp-facing C ABI：

```c
int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t S,
    int64_t D,
    int64_t DV,
    int64_t q_nb0,
    int64_t q_nb1,
    int64_t q_nb2,
    int64_t q_nb3,
    int64_t k_nb0,
    int64_t k_nb1,
    int64_t k_nb2,
    int64_t k_nb3,
    int64_t v_nb0,
    int64_t v_nb1,
    int64_t v_nb2,
    int64_t v_nb3,
    int64_t o_nb0,
    int64_t o_nb1,
    int64_t o_nb2,
    int64_t o_nb3,
    float scale);

int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f16(
    const float* q,
    const float* k,
    const float* v,
    const uint16_t* mask,
    float* out,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t S,
    int64_t D,
    int64_t DV,
    /* q/k/v/out byte strides, same as no-mask entry */
    int64_t q_nb0, int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
    int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
    int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
    int64_t o_nb0, int64_t o_nb1, int64_t o_nb2, int64_t o_nb3,
    /* F16 mask shape/byte strides for [S,L,H_mask,B_mask] */
    int64_t mask_ne0, int64_t mask_ne1, int64_t mask_ne2, int64_t mask_ne3,
    int64_t mask_nb0, int64_t mask_nb1, int64_t mask_nb2, int64_t mask_nb3,
    float scale);

int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    float* out,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t S,
    int64_t D,
    int64_t DV,
    /* q/k/v/out byte strides, same as no-mask entry */
    int64_t q_nb0, int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
    int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
    int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
    int64_t o_nb0, int64_t o_nb1, int64_t o_nb2, int64_t o_nb3,
    /* F32 mask shape/byte strides for [S,L,H_mask,B_mask] */
    int64_t mask_ne0, int64_t mask_ne1, int64_t mask_ne2, int64_t mask_ne3,
    int64_t mask_nb0, int64_t mask_nb1, int64_t mask_nb2, int64_t mask_nb3,
    float scale);
```

`scale == 0.0f` 时内部使用 `1 / sqrt(D)`。

`mask_f16` / `mask_f32` 入口按 llama.cpp/ggml 的 additive mask 读取：

```text
mask layout: [S, L, H_mask, B_mask]
scores += mask[s, l, h % H_mask, b % B_mask]
```

当前 masked 路径不再 materialize fp32 `[B,H,L,S]`。kernel 在 QK tile 内
按 ggml stride 直接读取 mask；full 8x8 QK block 在写回 `scores` 前
把 mask 加到临时 QK block，partial block 在 QK 写回后立即加 mask。
全 `0` mask block 会跳过加法；全 off block（`-inf`，或有限 mask 值
`<= -1000`）会跳过 QK、直接写 `-inf` scores，并在当前 Q tile 对应 key
range 全部 masked 时跳过该段 PV。
普通 mixed F16 mask 仍需要为相关 score 转换 half；F32 mask 直接按 fp32
读取。

源码里还保留了 contiguous pointer helper 供 bench 或直接源码接入复用；
默认 shared library 通过 `fp32_packqkv_sdpa.version` 把它隐藏。

返回码：

```text
0   success
1   null pointer
2   invalid shape/config
3   unsupported alignment, e.g. S % 8 != 0 or DV % 8 != 0
100 std::exception
101 unknown exception
```

## llama.cpp 输入输出 layout

BGE-small 进入 `ggml_flash_attn_ext` 前的输入：

```text
Qcur: [head_dim, n_head,    n_tokens] = [64, 8, 512]
Kcur: [head_dim, n_head_kv, n_tokens] = [64, 8, 512]
Vcur: [head_dim, n_head_kv, n_tokens] = [64, 8, 512]
```

`build_attn` 组织后的逻辑形状：

```text
q: [head_dim, q_len,  n_head,    batch]
k: [head_dim, kv_len, n_head_kv, batch]
v: [v_dim,    kv_len, n_head_kv, batch]
```

对应 `fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp`：

```text
B  = batch
H  = n_head
L  = q_len
S  = kv_len
D  = head_dim
DV = v_dim
```

BGE-small 当前 `H == n_head_kv == 8`，这个 llama.cpp entry 暂按
`n_head == n_head_kv` 实现。GQA/MQA 需要额外传 `H_kv`，再把 query head
`h` 映射到 KV head `h / (H / H_kv)`。

## stride 调用

如果 ggml tensor 是逻辑：

```text
q:   [D,  L, H, B]
k:   [D,  S, H, B]
v:   [DV, S, H, B]
out: [DV, H, L, B]
```

并且 dim0 连续，则 stride 字节数通常是：

```c
const int64_t q_nb0 = sizeof(float);
const int64_t q_nb1 = D * sizeof(float);
const int64_t q_nb2 = D * L * sizeof(float);
const int64_t q_nb3 = D * L * H * sizeof(float);

const int64_t k_nb0 = sizeof(float);
const int64_t k_nb1 = D * sizeof(float);
const int64_t k_nb2 = D * S * sizeof(float);
const int64_t k_nb3 = D * S * H * sizeof(float);

const int64_t v_nb0 = sizeof(float);
const int64_t v_nb1 = DV * sizeof(float);
const int64_t v_nb2 = DV * S * sizeof(float);
const int64_t v_nb3 = DV * S * H * sizeof(float);

const int64_t o_nb0 = sizeof(float);
const int64_t o_nb1 = DV * sizeof(float);
const int64_t o_nb2 = DV * H * sizeof(float);
const int64_t o_nb3 = DV * H * L * sizeof(float);
```

BGE-small 调用：

```c
int rc = fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(
    qcur, kcur, vcur, out,
    1, 8, 512, 512, 64, 64,
    q_nb0, q_nb1, q_nb2, q_nb3,
    k_nb0, k_nb1, k_nb2, k_nb3,
    v_nb0, v_nb1, v_nb2, v_nb3,
    o_nb0, o_nb1, o_nb2, o_nb3,
    0.0f);
```

## 接入位置

建议先在 ggml CPU attention 中加一个很窄的 fast path：

1. 只处理 encoder full attention：`is_causal == false`。
2. 只处理 fp32。
3. 只处理 `n_head == n_head_kv`。
4. 要求 `S % 8 == 0 && DV % 8 == 0`。
5. 按 ggml tensor 的 `nb[]` 传入真实 byte stride。
6. `rc != 0` 时 fallback 到原 `ggml_flash_attn_ext`。

## 当前 Arm Codex 验证

```text
isolated directory build:
  cp -a standalone/fp32_packqkv /tmp/.../fp32_packqkv
  cd /tmp/.../fp32_packqkv && make -j && make check

make check:
  noncausal max_abs_diff=0
  causal    max_abs_diff=0

BGE-small fp32, B=1 H=8 L=S=512 D=DV=64, OMP_NUM_THREADS=1:
  Arm Codex isolated mean_ms ~= 11.00
  Arm Codex isolated GFLOP/s ~= 49.75
  AWS isolated mean_ms ~= 11.78
  AWS isolated GFLOP/s ~= 46.45

shared library:
  nm -D --defined-only libfused_cpp_sdpa_fp32_packqkv.so
    exports fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp
    exports fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f16
    exports fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f32
  ldd libfused_cpp_sdpa_fp32_packqkv.so
    no torch / libtorch_python / pybind / Python dependency

llama.cpp stride entry smoke test:
  rc=0
  max_abs_diff ~= 1.08e-7 vs Python reference

same-input comparison against current in-framework C++ version
`flash2_neon_l3kv_packqkv_pbf16pv`:
  shape (1,4,64,64,64,64):   max_abs=0
  shape (1,8,512,512,64,64): max_abs=0
```
