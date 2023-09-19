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
  constexpr int samps = 1024;
  auto in_t = matx::make_tensor<int16_t, 1>(in, {samps*2});

  auto inf_t = matx::make_tensor<float, 1>((float*)out, {samps*2});
  auto out_t = matx::make_tensor<cuda::std::complex<float>, 1>((cuda::std::complex<float>*)out, {samps});

  auto out_tmp_t = matx::make_tensor<float, 1>({samps * 2}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);
  auto out_cplx_t = matx::make_tensor<cuda::std::complex<float>, 1>({samps}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);

  auto out_doublebuf_t = matx::make_tensor<float, 1>((float*)out, {samps});

  (inf_t = in_t / 32768.0).run(stream);

  (out_cplx_t = out_t * matx::hanning<0>(out_t.Shape())).run(stream);


  matx::fft(out_cplx_t, out_cplx_t, 0, stream);
  (out_doublebuf_t = 20.0f * fftshift1D( matx::log10(matx::abs(out_cplx_t)) - 
                                        log10(static_cast<float>(out_t.Size(0)) / 2.0f)
                                      )).run(stream);
//matx::print(out_doublebuf_t);
  // constexpr matx::index_t nperseg = 64;
  // constexpr matx::index_t nfft = 64;
  // constexpr matx::index_t noverlap = nperseg / 8;
  // constexpr matx::index_t nstep = nperseg - noverlap;

  // auto stackedMatrix = inf_t.OverlapView({nperseg}, {nstep});

  // auto fftStackedMatrix = matx::make_tensor<cuda::std::complex<float>>({(samps - noverlap) / nstep, nfft / 2 + 1}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);

  // // FFT along rows
  // matx::fft(fftStackedMatrix, stackedMatrix, 0, stream);
  // // Absolute value
  // (fftStackedMatrix = matx::conj(fftStackedMatrix) * fftStackedMatrix).run(stream);
  // // Get real part and transpose
  // auto Sxx = fftStackedMatrix.RealView().Permute({1, 0});  
  // matx::print(Sxx);
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