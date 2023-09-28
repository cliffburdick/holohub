#include "kernels.cuh"
#include <stdio.h>
#include "matx.h"

union ADCSamp{
  struct {
    int16_t i;
    int16_t q;
  } samp;

  uint32_t full;
};

__global__ void i16tofp32(const uint16_t *in, cuda::std::complex<float> *out, int num_samps);

__global__ void i16tofp32(const int16_t *in, cuda::std::complex<float> *out, int num_samps) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_samps) {
    ADCSamp tmp = *(reinterpret_cast<const ADCSamp *>(in) + tid);

    cuda::std::complex<float> ctmp = {
      __int2float_rd(tmp.samp.i), __int2float_rd(tmp.samp.q)
    };

    // if (tid < 100) {
    //   printf("%d %f %f %08x\n", tid, ctmp.real(), ctmp.imag(), tmp.full);
    // }

    out[tid] = ctmp;
  }
}

using ftype = float;
using ctype = cuda::std::complex<ftype>;


void process_input(int16_t *in, ftype *out, int num_samps, cudaStream_t stream) {
  constexpr int SAMP_PER_CLK = 4;
  constexpr int Fs = 2949120000/3;
  constexpr int samps = 1048576 * 18;
  constexpr int nfft = 16384;
  auto in_t             = matx::make_tensor<int16_t>(in, {samps*2});
  auto out_convert_t    = matx::make_tensor<ftype>({samps*2}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);
  auto out_tmp_t        = matx::make_tensor<ftype>({nfft}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);
  auto infc_t           = matx::make_tensor<ctype>((ctype*)out_convert_t.Data(), {samps});
  auto out_doublebuf_t  = matx::make_tensor<ftype>((ftype*)out, {nfft});

  (out_convert_t = in_t / 32768.0).run(stream);
  (out_tmp_t = pwelch(infc_t, matx::hanning<0>({nfft}), nfft, 16, nfft)).run(stream);
  (out_doublebuf_t = fftshift1D( 10.f*matx::log10(out_tmp_t))).run(stream);
}

__global__ void print_kernel(uint8_t *in, int len) {
  // int tid = blockIdx.x*blockDim.x + threadIdx.x;
  // if (tid < len) {
  //   printf("%05d: %02x\n", tid, in[tid]);
  // }
}

void launch_print(uint8_t *in, int len) {
  print_kernel<<<1, len>>>(in, len);
}