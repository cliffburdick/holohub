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


void process_input(int16_t *in, float *out, int num_samps, cudaStream_t stream) {
  constexpr int SAMP_PER_CLK = 4;
  constexpr int Fs = 2949120000/3;
  constexpr int samps = 16384;
  constexpr int nfft = 256;
  auto in_t = matx::make_tensor<int16_t, 1>(in, {samps*2});

  auto inf_t = matx::make_tensor<float, 1>((float*)out, {samps*2});
  auto out_t = matx::make_tensor<cuda::std::complex<float>, 1>((cuda::std::complex<float>*)out, {samps});

  auto out_tmp_t = matx::make_tensor<float, 1>({nfft}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);
  auto out_cplx_t = matx::make_tensor<cuda::std::complex<float>, 1>({samps}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);

  auto infc_t = matx::make_tensor<cuda::std::complex<float>, 1>((cuda::std::complex<float>*)out, {samps});
  auto out_doublebuf_t = matx::make_tensor<float, 1>((float*)out, {nfft});

  (inf_t = in_t / 32768.0).run(stream);
//matx::print(infc_t);
  // (out_cplx_t = out_t * matx::hanning<0>(out_t.Shape())).run(stream);


  // (out_cplx_t = matx::fft(out_cplx_t)).run(stream);
  // (out_doublebuf_t = 20.0f * fftshift1D( matx::log10(matx::abs(out_cplx_t)) - 
  //                                       log10(static_cast<float>(out_t.Size(0)) / 2.0f)
  //                                     )).run(stream);

  //auto Pxx  = make_tensor<float>({sat});
  (out_tmp_t = pwelch(infc_t, nfft, 64, nfft)).run(stream);

  (out_doublebuf_t = fftshift1D( matx::log10(out_tmp_t))).run(stream);
}

__global__ void print_kernel(uint8_t *in, int len) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < len) {
    printf("%05d: %02x\n", tid, in[tid]);
  }
}

void launch_print(uint8_t *in, int len) {
  print_kernel<<<1, len>>>(in, len);
}