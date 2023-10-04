/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "adv_network_kernels.h"
#include <stdio.h>

/**
 * @brief Simple packet reorder kernel to demonstrate reordering a batch of packets into 
 *        contiguous memory
 *
 * @param out Output buffer
 * @param in Pointer to list of input packet pointers
 * @param pkt_len Length of each packet. All packets must be same length for this example
 * @param num_pkts Number of packets
 */
__global__ void simple_packet_reorder_kernel(void * __restrict__ out,
                                            const void *const *const __restrict__ in,
                                            uint16_t pkt_len,
                                            uint32_t num_pkts) {
  const int pkt_idx = blockIdx.x;
  const int len = pkt_len;
  const void *in_pkt = in[pkt_idx];

  if (pkt_idx < num_pkts) {
    for (int pos = threadIdx.x; pos < len / 4; pos += blockDim.x) {
      const uint32_t *in_ptr  = static_cast<const uint32_t*>(in_pkt) + pos;
      uint32_t *out_ptr       = (uint32_t*)((uint8_t *)out + pkt_idx * pkt_len)  + pos;
      // if (pkt_idx == 0 && pos < 32) {
      //   printf("%d %d %08x\n", pkt_idx, pos, *in_ptr);
      // }
      *out_ptr            = *in_ptr;
    }
  }
}

/**
 * @brief Wrapper to launch packet reorder kernel
 *
 * @param out Output buffer
 * @param in Pointer to list of input packet pointers
 * @param pkt_len Length of each packet in bytes. Must be a multiple of 4
 * @param num_pkts Number of packets
 * @param stream CUDA stream
 */
void simple_packet_reorder(void *out, const void *const *const in,
          uint16_t pkt_len, uint32_t num_pkts, cudaStream_t stream) {
  simple_packet_reorder_kernel<<<num_pkts, 128, 0, stream>>>(out, in, pkt_len, num_pkts);
}


__global__ void populate_packets(uint8_t **gpu_bufs,
  uint16_t pkt_len) {
    int pkt = blockIdx.x;

    for (int samp = threadIdx.x; samp < pkt_len / 4; samp += blockDim.x) {
      auto p = reinterpret_cast<uint32_t *>(gpu_bufs[pkt] + samp * sizeof(uint32_t));
      *p = (samp << 16) | (samp & 0xff);
    }
}

/**
 * @brief Populate each packet with a monotonically-increasing sequence
 *
 * @param gpu_bufs GPU packet pointer list from ANO "gpu_pkts"
 * @param pkt_len Length of each packet in bytes. Must be a multiple of 4
 * @param num_pkts Number of packets
 * @param stream CUDA stream
 */
void populate_packets(uint8_t **gpu_bufs,
  uint16_t pkt_len,
  uint32_t num_pkts,
  cudaStream_t stream) {
    populate_packets<<<num_pkts, 256, 0, stream>>>(gpu_bufs, pkt_len);
  }
