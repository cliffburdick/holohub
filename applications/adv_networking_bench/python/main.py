# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import cupy
import socket
import binascii
from holoscan.conditions import CountCondition, BooleanCondition
from holoscan.core import Application, Operator, OperatorSpec

from holohub.advanced_network_common import (
    AdvNetStatus,
    adv_net_create_burst_params,
    adv_net_get_num_pkts,
    adv_net_get_tx_pkt_burst,
    adv_net_set_udp_payload,
    adv_net_set_hdr,
    adv_net_tx_burst_available,
    adv_net_set_eth_hdr,
    adv_net_set_ipv4_hdr,
    adv_net_set_pkt_lens,
    adv_net_format_eth_addr,
)
from holohub.advanced_network_tx import AdvNetworkOpTx

logger = logging.getLogger("AdvancedNetworkingBench")
logging.basicConfig(level=logging.INFO)



class AdvancedNetworkingBenchRxOp(Operator):
    def __init__(self, fragment, *args, batch_size, payload_size, **kwargs):
        self.index = 1
        self.batch_size = kwargs['batch_size']
        self.max_packet_size = kwargs['max_packet_size']
        self.header_size = kwargs['header_size']

        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        Operator.initialize(self)

        nom_payload_size = self.max_packet_size - self.header_size

        self.full_batch_data_h = cupy.cuda.alloc_pinned_memory(self.batch_size * nom_payload_size)

    def setup(self, spec: OperatorSpec):
        spec.input("burst_in")

    def compute(self, op_input, op_output, context):
        ttl_bytes_in_cur_batch = 0

        auto burst_opt = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in")
        if not burst_opt:
            free_bufs()
            return

        msg = adv_net_create_burst_params()
        adv_net_set_hdr(msg, 0, 0, self.batch_size, 1)

        while not adv_net_tx_burst_available(msg):
            continue

        ret = adv_net_get_tx_pkt_burst(msg)
        if ret != AdvNetStatus.SUCCESS:
            logger.error(f"Error returned from adv_net_get_tx_pkt_burst: {ret}")
            return
        
        for num_pkt in range(adv_net_get_num_pkts(msg)):
            ret = adv_net_set_eth_hdr(msg, num_pkt, self.eth_dst_addr)
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_eth_hdr: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return
            
            ip_len = self.payload_size + self.header_size - (14 + 20)
            ret = adv_net_set_ipv4_hdr(msg, num_pkt, ip_len, 17, self.ip_src_addr, self.ip_dst_addr)
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_eth_hdr: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return  
            
            ret = adv_net_set_udp_payload(
                msg, num_pkt, self.buf.ptr + num_pkt * self.payload_size, self.payload_size
            )
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_udp_payload: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return
            
            ret = adv_net_set_pkt_lens(msg, num_pkt, [self.payload_size + self.header_size])
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_pkt_lens: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return            

        op_output.emit(msg, "msg_out", "AdvNetBurstParams*")


class AdvancedNetworkingBenchTxOp(Operator):
    def __init__(self, fragment, *args, batch_size, payload_size, **kwargs):
        self.index = 1
        self.batch_size = batch_size
        self.payload_size = payload_size
        self.eth_dst_addr = binascii.unhexlify(kwargs['eth_dst_addr'].replace(':', ''))
        self.header_size = kwargs['header_size']
        self.ip_src_addr = int.from_bytes(socket.inet_pton(socket.AF_INET, kwargs['ip_src_addr']), "big")
        self.ip_dst_addr = int.from_bytes(socket.inet_pton(socket.AF_INET, kwargs['ip_dst_addr']), "big")
        self.buf_size = self.batch_size * self.payload_size
        self.buf = cupy.cuda.alloc_pinned_memory(self.buf_size)

        
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        Operator.initialize(self)

    def setup(self, spec: OperatorSpec):
        spec.output("msg_out")

    def compute(self, op_input, op_output, context):
        msg = adv_net_create_burst_params()
        adv_net_set_hdr(msg, 0, 0, self.batch_size, 1)

        while not adv_net_tx_burst_available(msg):
            continue

        ret = adv_net_get_tx_pkt_burst(msg)
        if ret != AdvNetStatus.SUCCESS:
            logger.error(f"Error returned from adv_net_get_tx_pkt_burst: {ret}")
            return
        
        for num_pkt in range(adv_net_get_num_pkts(msg)):
            ret = adv_net_set_eth_hdr(msg, num_pkt, self.eth_dst_addr)
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_eth_hdr: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return
            
            ip_len = self.payload_size + self.header_size - (14 + 20)
            ret = adv_net_set_ipv4_hdr(msg, num_pkt, ip_len, 17, self.ip_src_addr, self.ip_dst_addr)
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_eth_hdr: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return  
            
            ret = adv_net_set_udp_payload(
                msg, num_pkt, self.buf.ptr + num_pkt * self.payload_size, self.payload_size
            )
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_udp_payload: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return
            
            ret = adv_net_set_pkt_lens(msg, num_pkt, [self.payload_size + self.header_size])
            if ret != AdvNetStatus.SUCCESS:
                logger.error(
                    f"Error returned from adv_net_set_pkt_lens: "
                    f"{ret} != {AdvNetStatus.SUCCESS}"
                )
                return            

        op_output.emit(msg, "msg_out", "AdvNetBurstParams*")


# Now define a simple application using the operators defined above
NUM_MSGS = 10


class App(Application):
    def compose(self):
        # Define the tx and rx operators, allowing the tx operator to execute 10 times
        if (
            "cfg" in self.kwargs("advanced_network")
            and "tx" in self.kwargs("advanced_network")["cfg"]["interfaces"][0]
        ):
            tx = AdvancedNetworkingBenchTxOp(
                self, BooleanCondition(self, True), name="tx", **self.kwargs("bench_tx")
            )
            adv_net_tx = AdvNetworkOpTx(self, name="adv_net_tx")
            self.add_flow(tx, adv_net_tx, {("msg_out", "burst_in")})
        else:
            logger.info("No TX config found")

        # if len(self.kwargs("network_rx")) > 0:
        #     basic_net_rx = BasicNetworkOpRx(self, name="basic_net_rx", **self.kwargs("network_rx"))
        #     rx = BasicNetworkPingRxOp(self, name="rx")
        #     self.add_flow(basic_net_rx, rx, {("burst_out", "msg_in")})
        # else:
        #     logger.info("No RX config found")


if __name__ == "__main__":
    config_path = sys.argv[1]
    app = App()
    app.config(config_path)
    print("CFG")
    app.run()
