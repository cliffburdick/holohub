#include <cuda/std/complex>

void process_input(int16_t *in, float *out, int num_samps, cudaStream_t stream);
void launch_print(uint8_t *in, int len) ;