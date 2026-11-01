// One-token group-wise W8A16 GEMV in CUDA C++: y[n] = sum_k x[k] * q[n, k] * scale[n, k / group], fp32 accumulation.
//
// The same computation as the Triton kernel _w8a16_group_gemv_kernel in infer_opt/quant_kernels.py, written by hand
// to compare the two. A block of THREADS threads owns ROWS output rows. Every thread reads 16 contiguous int8 of a row
// (one 16-byte load, streaming: the weight is read once per token and should not displace x and the scales in the
// caches), the 16 matching x values (two 16-byte loads) and the scale of their group; it sums q * x over the 16 in
// registers and multiplies that partial sum by the scale, so the scale costs one load and one multiply per 16 weights.
// Each thread first issues the weight loads of UNROLL chunks for all its rows and only then computes, so UNROLL * ROWS
// 16-byte loads are in flight per thread. The partial sums go through warp shuffles and, with more than one warp,
// shared memory. Rows past N read row N - 1 instead of masking the loads and are not stored.
//
// Requirements (checked on the host): K and the group size multiples of 16, x and q 16-byte aligned, everything
// contiguous, x and scale of one dtype (bf16 or fp16).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float2 to_float2(unsigned int v, __nv_bfloat16) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v));
}
__device__ __forceinline__ float2 to_float2(unsigned int v, __half) {
  return __half22float2(*reinterpret_cast<__half2*>(&v));
}
__device__ __forceinline__ void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void store(__half* p, float v) { *p = __float2half(v); }

// sum over 16 weights of q[i] * x[i]: the four 32-bit words of w hold q[0..15], xa and xb hold x[0..7] and x[8..15]
template <typename T>
__device__ __forceinline__ float dot16(const int4& w, const uint4& xa, const uint4& xb) {
  const unsigned int xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  const int ws[4] = {w.x, w.y, w.z, w.w};
  float d = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 lo = to_float2(xs[2 * i], T{});
    const float2 hi = to_float2(xs[2 * i + 1], T{});
    const int b = ws[i];
    d = fmaf(static_cast<float>(static_cast<signed char>(b)), lo.x, d);
    d = fmaf(static_cast<float>(static_cast<signed char>(b >> 8)), lo.y, d);
    d = fmaf(static_cast<float>(static_cast<signed char>(b >> 16)), hi.x, d);
    d = fmaf(static_cast<float>(static_cast<signed char>(b >> 24)), hi.y, d);
  }
  return d;
}

template <typename T, int ROWS, int THREADS, int UNROLL>
__global__ void __launch_bounds__(THREADS)
w8a16_group_gemv_kernel(const T* __restrict__ x, const int8_t* __restrict__ q, const T* __restrict__ s,
                        T* __restrict__ y, int n, int k, int group) {
  const int row0 = blockIdx.x * ROWS;
  const int groups = k / group;
  const int chunks = k / 16;  // 16-byte chunks of a weight row
  const int4* q_row[ROWS];
  const T* s_row[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int row = min(row0 + r, n - 1);
    q_row[r] = reinterpret_cast<const int4*>(q + static_cast<int64_t>(row) * k);
    s_row[r] = s + static_cast<int64_t>(row) * groups;
  }
  const uint4* x16 = reinterpret_cast<const uint4*>(x);  // 8 x values per uint4

  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

  for (int c0 = threadIdx.x; c0 < chunks; c0 += THREADS * UNROLL) {
    int4 w[UNROLL][ROWS];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int c = c0 + u * THREADS;
      if (c < chunks) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) w[u][r] = __ldcs(q_row[r] + c);
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int c = c0 + u * THREADS;
      if (c < chunks) {
        const uint4 xa = __ldg(x16 + 2 * c);
        const uint4 xb = __ldg(x16 + 2 * c + 1);
        const int g = c * 16 / group;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = fmaf(dot16<T>(w[u][r], xa, xb), to_float(__ldg(s_row[r] + g)), acc[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
  }
  if constexpr (THREADS > 32) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float partial[ROWS][WARPS];
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) partial[r][warp] = acc[r];
    }
    __syncthreads();
    if (threadIdx.x < ROWS && row0 + threadIdx.x < n) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < WARPS; ++i) sum += partial[threadIdx.x][i];
      store(y + row0 + threadIdx.x, sum);
    }
  } else if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (row0 + r < n) store(y + row0 + r, acc[r]);
    }
  }
}

template <typename T, int ROWS, int THREADS, int UNROLL>
void launch(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& s, torch::Tensor& y, int n, int k,
            int group) {
  const dim3 grid((n + ROWS - 1) / ROWS);
  w8a16_group_gemv_kernel<T, ROWS, THREADS, UNROLL><<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(x.data_ptr()), q.data_ptr<int8_t>(), reinterpret_cast<const T*>(s.data_ptr()),
      reinterpret_cast<T*>(y.data_ptr()), n, k, group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
bool dispatch(int rows, int threads, int unroll, const torch::Tensor& x, const torch::Tensor& q,
              const torch::Tensor& s, torch::Tensor& y, int n, int k, int group) {
#define INFER_OPT_CASE(R, TH, U)                       \
  if (rows == R && threads == TH && unroll == U) {     \
    launch<T, R, TH, U>(x, q, s, y, n, k, group);      \
    return true;                                       \
  }
#define INFER_OPT_UNROLLS(R, TH) \
  INFER_OPT_CASE(R, TH, 1) INFER_OPT_CASE(R, TH, 2) INFER_OPT_CASE(R, TH, 4) INFER_OPT_CASE(R, TH, 8)
#define INFER_OPT_THREADS(R) \
  INFER_OPT_UNROLLS(R, 32) INFER_OPT_UNROLLS(R, 64) INFER_OPT_UNROLLS(R, 128) INFER_OPT_UNROLLS(R, 256)
  INFER_OPT_THREADS(1)
  INFER_OPT_THREADS(2)
  INFER_OPT_THREADS(4)
#undef INFER_OPT_THREADS
#undef INFER_OPT_UNROLLS
#undef INFER_OPT_CASE
  return false;
}

torch::Tensor w8a16_group_gemv(torch::Tensor x, torch::Tensor q, torch::Tensor s, int64_t rows, int64_t threads,
                               int64_t unroll) {
  TORCH_CHECK(x.is_cuda() && q.is_cuda() && s.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(q.dim() == 2 && q.scalar_type() == torch::kInt8 && q.is_contiguous(), "weight: contiguous int8 [N, K]");
  TORCH_CHECK(s.dim() == 2 && s.is_contiguous() && s.size(0) == q.size(0), "scale: contiguous [N, K / group]");
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == s.scalar_type(), "x: contiguous, the dtype of the scale");
  const int n = static_cast<int>(q.size(0));
  const int k = static_cast<int>(q.size(1));
  TORCH_CHECK(x.numel() == k, "x must hold one row of K elements");
  TORCH_CHECK(k % s.size(1) == 0, "the scale's columns must divide K");
  const int group = k / static_cast<int>(s.size(1));
  TORCH_CHECK(k % 16 == 0 && group % 16 == 0, "K and the group size must be multiples of 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "x and the weight must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(x.device());
  auto shape = x.sizes().vec();
  shape.back() = n;
  auto y = torch::empty(shape, x.options());
  bool ok;
  if (x.scalar_type() == torch::kBFloat16) {
    ok = dispatch<__nv_bfloat16>(rows, threads, unroll, x, q, s, y, n, k, group);
  } else {
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be bf16 or fp16");
    ok = dispatch<__half>(rows, threads, unroll, x, q, s, y, n, k, group);
  }
  TORCH_CHECK(ok, "no compiled configuration rows=", rows, " threads=", threads, " unroll=", unroll);
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_group_gemv", &w8a16_group_gemv, "one-token group-wise W8A16 GEMV");
}
