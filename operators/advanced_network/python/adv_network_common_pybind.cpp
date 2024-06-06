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

#include "../adv_network_common.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace holoscan::ops {

PYBIND11_MODULE(_advanced_network_common, m) {
  m.doc() = "Advanced networking operator utility functions";

  py::enum_<AdvNetStatus>(m, "AdvNetStatus")
      .value("SUCCESS", AdvNetStatus::SUCCESS)
      .value("NULL_PTR", AdvNetStatus::NULL_PTR)
      .value("NO_FREE_BURST_BUFFERS", AdvNetStatus::NO_FREE_BURST_BUFFERS)
      .value("NO_FREE_PACKET_BUFFERS", AdvNetStatus::NO_FREE_PACKET_BUFFERS);

  m.def("adv_net_create_burst_params",
        &adv_net_create_burst_params,
        py::return_value_policy::reference,
        "Create a shared pointer burst params structure");
  m.def("adv_net_free_pkt",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int>(&adv_net_free_pkt),
        "Free a single packet");
  m.def("adv_net_free_pkt",
        py::overload_cast<AdvNetBurstParams*, int>(&adv_net_free_pkt),
        "Free a single packet");        
  m.def("adv_net_get_seg_pkt_len",
        py::overload_cast<AdvNetBurstParams*, int, int>(&adv_net_get_seg_pkt_len),
        "Get length of one segments of the packet");
  m.def("adv_net_get_seg_pkt_len",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int, int>(&adv_net_get_seg_pkt_len),
        "Get length of one segments of the packet");
  m.def("adv_net_get_pkt_len",
        py::overload_cast<AdvNetBurstParams*, int>(&adv_net_get_pkt_len),
        "Get length of the packet");
  m.def("adv_net_get_pkt_len",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int>(&adv_net_get_pkt_len),
        "Get length of the packet");

  m.def("adv_net_free_all_seg_pkts",
        py::overload_cast<AdvNetBurstParams*, int>(&adv_net_free_all_seg_pkts),
        "Free all packets in a burst for one segment");
  m.def("adv_net_free_all_seg_pkts",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int>(&adv_net_free_all_seg_pkts),
        "Free all packets in a burst for one segment");
  m.def("adv_net_free_all_pkts_and_burst",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_free_all_pkts_and_burst),
        "Free all packets and burst structure");
  m.def("adv_net_free_all_pkts_and_burst",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_free_all_pkts_and_burst),
        "Free all packets and burst structure");
  m.def("adv_net_free_seg_pkts_and_burst",
        py::overload_cast<AdvNetBurstParams*, int>(&adv_net_free_seg_pkts_and_burst),
        "Free all packets and burst structure for one packet segment");
  m.def(
      "adv_net_free_seg_pkts_and_burst",
      py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int>(&adv_net_free_seg_pkts_and_burst),
      "Free all packets and burst structure for one packet segment");
  m.def("adv_net_tx_burst_available",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_tx_burst_available),
        "Return true if a TX burst is available for use");
  m.def("adv_net_tx_burst_available",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_tx_burst_available),
        "Return true if a TX burst is available for use");
  m.def("adv_net_get_tx_pkt_burst",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_get_tx_pkt_burst),
        "Get TX packet burst");
  m.def("adv_net_get_tx_pkt_burst",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_get_tx_pkt_burst),
        "Get TX packet burst");
  m.def("adv_net_shutdown", (&adv_net_shutdown), "Shut down the ANO manager");
  m.def("adv_net_print_stats", (&adv_net_print_stats), "Print statistics in in the ANO");
  // m.def("adv_net_set_cpu_udp_payload",
  //     [](AdvNetBurstParams *burst, int idx, long int data, int len) {
  //             return adv_net_set_cpu_udp_payload(burst, idx,
  //                     reinterpret_cast<void*>(data), len); },
  //             "Set UDP header parameters and copy payload");
  // m.def("adv_net_set_cpu_udp_payload",
  //     [](std::shared_ptr<AdvNetBurstParams> burst, int idx, long int data, int len) {
  //         return adv_net_set_cpu_udp_payload(burst, idx,
  //              reinterpret_cast<void*>(data), len); },
  //         "Set UDP header parameters and copy payload");

  m.def("adv_net_get_num_pkts",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_get_num_pkts),
        "Get number of packets in a burst");
  m.def("adv_net_get_num_pkts",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_get_num_pkts),
        "Get number of packets in a burst");
  m.def("adv_net_get_q_id",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_get_q_id),
        "Get queue ID of a burst");
  m.def("adv_net_get_q_id",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_get_q_id),
        "Get queue ID of a burst");
  m.def("adv_net_set_num_pkts",
        py::overload_cast<AdvNetBurstParams*, int64_t>(&adv_net_set_num_pkts),
        "Set number of packets in a burst");
  m.def("adv_net_set_num_pkts",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int64_t>(&adv_net_set_num_pkts),
        "Set number of packets in a burst");
  m.def("adv_net_set_hdr",
        py::overload_cast<AdvNetBurstParams*, uint16_t, uint16_t, int64_t, int>(&adv_net_set_hdr),
        "Set parameters of burst header");
  m.def("adv_net_set_hdr",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, uint16_t, uint16_t, int64_t, int>(
            &adv_net_set_hdr),
        "Set parameters of burst header");
  m.def("adv_net_free_tx_burst",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_free_tx_burst),
        "Free TX burst");
  m.def("adv_net_free_tx_burst",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_free_tx_burst),
        "Free TX burst");
  m.def("adv_net_free_rx_burst",
        py::overload_cast<AdvNetBurstParams*>(&adv_net_free_rx_burst),
        "Free RX burst");
  m.def("adv_net_free_rx_burst",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_free_rx_burst),
        "Free RX burst");
  m.def("adv_net_get_seg_pkt_ptr",
        py::overload_cast<AdvNetBurstParams*, int, int>(&adv_net_get_seg_pkt_ptr),
        "Get packet pointer for one segment");
  m.def("adv_net_get_seg_pkt_ptr",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int, int>(&adv_net_get_seg_pkt_ptr),
        "Get packet pointer for one segment");
  m.def("adv_net_get_pkt_ptr",
        py::overload_cast<AdvNetBurstParams*, int>(&adv_net_get_pkt_ptr),
        "Get packet pointer");
  m.def("adv_net_get_pkt_ptr",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int>(&adv_net_get_pkt_ptr),
        "Get packet pointer");
  m.def("adv_net_get_port_from_ifname",
        &adv_net_get_port_from_ifname,
        "Get port number from interface name");

  m.def("adv_net_set_eth_hdr",
        py::overload_cast<AdvNetBurstParams*, int, char*>(&adv_net_set_eth_hdr),
        "Set ethernet header fields");
  m.def("adv_net_set_eth_hdr",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int, char*>(&adv_net_set_eth_hdr),
        "Set ethernet header fields");      

  m.def("adv_net_set_ipv4_hdr",
        py::overload_cast<AdvNetBurstParams*, int, int, uint8_t, unsigned int, unsigned int>(&adv_net_set_ipv4_hdr),
        "Set ipv4 header fields");
  m.def("adv_net_set_ipv4_hdr",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int, int, uint8_t, unsigned int, unsigned int>(&adv_net_set_ipv4_hdr),
        "Set ipv4 header fields");

  m.def("adv_net_set_udp_hdr",
        py::overload_cast<AdvNetBurstParams*, int, int, uint16_t, uint16_t>(&adv_net_set_udp_hdr),
        "Set UDP header fields");
  m.def("adv_net_set_udp_hdr",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int, int, uint16_t, uint16_t>(&adv_net_set_udp_hdr),
        "Set UDP header fields");


//   m.def("adv_net_set_udp_payload", [](std::shared_ptr<AdvNetBurstParams> burst, int idx, long int data, int len) {
//                               return adv_net_set_udp_payload(burst, idx,
//                                     reinterpret_cast<void*>(data), len); },
//         "Set UDP payload");
        
  m.def("adv_net_set_udp_payload", 
                  [](AdvNetBurstParams *burst, int idx, long int data, int len) {
                              return adv_net_set_udp_payload(burst, idx,
                                    reinterpret_cast<void*>(data), len); },        
        "Set UDP payload");
    

//   m.def("adv_net_set_pkt_lens",
//       [](std::shared_ptr<AdvNetBurstParams> burst, int idx, py::list list) {
        
//                   return adv_net_set_pkt_lens(burst, idx,
//                         list.cast<std::vector<int>>()); },   
//        "Set packet lengths");
  m.def("adv_net_set_pkt_lens",
      [](AdvNetBurstParams *burst, int idx, std::vector<int> list) {
                  return adv_net_set_pkt_lens(burst, idx,
                        list); },         
        "Set packet lengths");

  m.def("adv_net_set_pkt_tx_time",
        py::overload_cast<AdvNetBurstParams*, int, uint64_t>(&adv_net_set_pkt_tx_time),
        "Set packet TX times");
  m.def("adv_net_set_pkt_tx_time",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>, int, uint64_t>(&adv_net_set_pkt_tx_time),
        "Set packet TX times");     

  m.def("adv_net_get_burst_tot_byte",
        py::overload_cast<std::shared_ptr<AdvNetBurstParams>>(&adv_net_get_burst_tot_byte),
        "Get total bytes in burst");

  m.def("adv_net_free_pkt_seg",
        py::overload_cast<AdvNetBurstParams*, int, int>(&adv_net_free_pkt_seg),
        "Free packet segment");

  m.def("adv_net_get_mac",
        py::overload_cast<int, char*>(&adv_net_get_mac),
        "Get MAC address of port");

  m.def("adv_net_address_to_port",
        py::overload_cast<const std::string& >(&adv_net_address_to_port),
        "Convert address to port number");

  m.def("adv_net_format_eth_addr",
        py::overload_cast<char*, std::string>(&adv_net_format_eth_addr),
        "Format ethernet address to bytes from string");

  // py::class_<AdvNetBurstHdrParams>(m, "AdvNetBurstHdrParams").def(py::init<>())
  //     .def_readwrite("num_pkts",  &AdvNetBurstHdrParams::num_pkts)
  //     .def_readwrite("port_id",   &AdvNetBurstHdrParams::port_id)
  //     .def_readwrite("q_id",      &AdvNetBurstHdrParams::q_id);

  // py::class_<AdvNetBurstHdr>(m, "AdvNetBurstHdr").def(py::init<>())
  //     .def_readwrite("hdr",  &AdvNetBurstHdr::hdr);

  // py::class_<AdvNetBurstParams>(m, "AdvNetBurstParams").def(py::init<>())
  //     .def_readwrite("hdr", &AdvNetBurstParams::hdr)
  //     .def_readwrite("cpu_pkts", &AdvNetBurstParams::cpu_pkts)
  //     .def_readwrite("gpu_pkts", &AdvNetBurstParams::gpu_pkts);

  py::class_<AdvNetBurstHdrParams>(m, "AdvNetBurstHdrParams").def(py::init<>());
  py::class_<AdvNetBurstHdr>(m, "AdvNetBurstHdr").def(py::init<>());
  py::class_<AdvNetBurstParams>(m, "AdvNetBurstParams").def(py::init<>());
  //  py::class_<AdvNetBurstParams, std::shared_ptr<AdvNetBurstParams>>
  //    (m, "AdvNetBurstParams").def(py::init<>());
}
};  // namespace holoscan::ops
