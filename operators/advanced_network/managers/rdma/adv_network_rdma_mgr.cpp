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

#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "adv_network_rdma_mgr.h"

/* The ordering of most RDMA/CM setup follows the ordering specified here:
   https://man7.org/linux/man-pages/man7/rdma_cm.7.html
   The exception is that there is no standard way to pass around keys, so we use standard 
   sends and receives.
*/

namespace holoscan::ops {
  void RDMAMgr::set_config_and_initialize(const AdvNetConfigYaml &cfg) {
    if (!this->initialized_) {
      cfg_ = cfg;
      initialize();
    }    
  }

  bool RDMAMgr::get_ip_from_interface(const std::string_view &if_name, sockaddr_in &addr) {
    struct ifaddrs *ifaddr, *ifa;
    bool found = false;

    // Initialize the sockaddr_in structure
    memset(&addr, 0, sizeof(sockaddr_in));
    addr.sin_family = AF_INET;

    // Get the list of network interfaces
    if (getifaddrs(&ifaddr) == -1) {
        HOLOSCAN_LOG_FATAL("Failed to get a list of interfaces");
        return false;
    }

    // Loop through the list of interfaces
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) {
        continue;
      }

      if (ifa->ifa_addr->sa_family != AF_INET) {
        continue; // Only IPv4 for now
      }

      // Check if the interface name matches
      if (if_name == ifa->ifa_name) {
          struct sockaddr_in *in_addr = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
          addr.sin_addr = in_addr->sin_addr;

          found = true;
          break;
      }
    }

    freeifaddrs(ifaddr);
    return found;
  }

  inline bool RDMAMgr::ack_event(rdma_cm_event *cm_event) {
    int ret = rdma_ack_cm_event(cm_event);
    if (ret != 0) {
      HOLOSCAN_LOG_ERROR("Failed to acknowledge CM event: {}", ret);
      return false;
    }
    return true;
  }

  /**
   * Set up all parameters needed for a newly-connected client
  */
  int RDMAMgr::setup_client_params_for_server(rdma_server_params *sparams) {
    sparams->pd = ibv_alloc_pd(sparams->client_id->verbs);
    if (sparams->pd == nullptr) {
      HOLOSCAN_LOG_FATAL("Failed to allocate PD for client! {}", errno);
      return -1;
    }

    HOLOSCAN_LOG_INFO("Successfully created PD for client");

    sparams->iocc = ibv_create_comp_channel(sparams->client_id->verbs);
    if (sparams->iocc == nullptr) {
      HOLOSCAN_LOG_FATAL("Failed to create completion channel! {}", errno);
      return -1;
    }  

    HOLOSCAN_LOG_INFO("Successfully created IO completion channel");

    sparams->cq = ibv_create_cq(sparams->client_id->verbs, 
			MAX_CQ, nullptr, sparams->iocc,	0);
    if (sparams->cq == nullptr) {
      HOLOSCAN_LOG_FATAL("Failed to create queue! {}", errno);
      return -1;
    }          

    if (ibv_req_notify_cq(sparams->cq, 0) != 0) {
      HOLOSCAN_LOG_FATAL("Failed to request notifications on CQ! {}", errno);
      return -1;
    }
  }

  void RDMAMgr::run_server() {
    int ret;
    struct rdma_cm_event *cm_event = NULL;

    // Set the CPU inifity of our master/loop thread
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cfg_.common_.master_core_, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
      HOLOSCAN_LOG_FATAL("Failed to set master thread affinity");
      return;
    }

    struct rdma_cm_event *cm_event = nullptr;
    sockaddr_in server_addr;
    int ret;
    
    // Create a channel for events
    cm_event_channel_ = rdma_create_event_channel();    
    if (cm_event_channel_ == nullptr) {
      HOLOSCAN_LOG_FATAL("Failed to create a CM channel: {}", errno);
      return;
    }

    HOLOSCAN_LOG_INFO("Successfully created CM event channel");    

    ret = rdma_create_id(cm_event_channel_, &cm_server_id_, nullptr, RDMA_PS_TCP);
    if (ret != 0) {
      HOLOSCAN_LOG_FATAL("Failed to create RDMA server ID {}", errno);
      return;
    }

    HOLOSCAN_LOG_INFO("Created RDMA server ID successfully");

    ret = rdma_bind_addr(cm_server_id_, reinterpret_cast<struct sockaddr*>(server_addr));
    if (ret != 0) {
      HOLOSCAN_LOG_FATAL("Failed to bind for RDMA server: {}", errno);
      return;
    }

    ret = rdma_listen(cm_server_id_, MAX_RDMA_CONNECTIONS);
    if (ret != 0) {
      HOLOSCAN_LOG_FATAL("Failed to listen for RDMA server: {}", errno);
      return;
    }

    HOLOSCAN_LOG_INFO("RDMA server successfully started");    

    // Our master thread's job is to wait on connections from clients, set up all the needed
    // information for them (QPs, MRs, etc), and spawn client threads to monitor both TX and RX
    while (true) {
      ret = rdma_get_cm_event(cm_event_channel_, &cm_event);
      if (ret != 0) {
        HOLOSCAN_LOG_INFO("Failed to get CM event: {}", errno);
        continue;
      }

      if (cm_event->status != 0) {
        HOLOSCAN_LOG_ERROR("Error status received in CM event: {}", cm_event->status);
        if (!rdma_ack_cm_event(cm_event)) {
          return;
        }

        continue;
      }

      switch (cm_event->event) {
        case RDMA_CM_EVENT_CONNECT_REQUEST: {
          HOLOSCAN_LOG_INFO("Received new connection request for client ID {}", cm_event->id);
          rdma_server_params *sparams = new rdma_server_params{};
          sparams->client_id = cm_event->id;
          ack_event(cm_event);
          setup_client_params_for_server(sparams);
          break;
        }
        default:
          HOLOSCAN_LOG_INFO("Cannot handle event type {}", cm_event->event);
      }     
    }
  }


  void RDMAMgr::initialize() {
    if (cfg_.common_.rdma_.mode_ == RDMAMode::SERVER) {
      std::thread t(&RDMAMgr::init_server, this);
      t.join();      
    }
    else {
      init_client();
      std::thread t(&RDMAMgr::run_client, this);
      t.join();      
    }
  }

}