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
#include <mqueue.h>
#include "adv_network_rdma_mgr.h"

/* The ordering of most RDMA/CM setup follows the ordering specified here:
   https://man7.org/linux/man-pages/man7/rdma_cm.7.html
   The exception is that there is no standard way to pass around keys, so we use standard 
   sends and receives.
*/

namespace holoscan::ops {
  std::atomic<bool> rdma_force_quit = false;

  void RDMAMgr::set_config_and_initialize(const AdvNetConfigYaml &cfg) {
    if (!this->initialized_) {
      cfg_ = cfg;
      initialize();
    }    
  }

  // Common ANO functions

  void *RDMAMgr::get_cpu_pkt_ptr(AdvNetBurstParams *burst, int idx) {
    return burst->cpu_pkts[idx];
  }
  
  void *RDMAMgr::get_gpu_pkt_ptr(AdvNetBurstParams *burst, int idx) {
    return burst->gpu_pkts[idx];
  }

  uint16_t get_cpu_pkt_len(AdvNetBurstParams *burst, int idx) {
    return burst->cpu_pkts_len[idx];
  }

  uint16_t get_gpu_pkt_len(AdvNetBurstParams *burst, int idx) {
    return burst->gpu_pkts_len[idx];
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

  int RDMAMgr::mr_access_to_ibv(uint32_t access) {
    int ibv_access = 0;

    if (access & MEM_ACCESS_LOCAL) {
      ibv_access |= IBV_ACCESS_LOCAL_WRITE;
    }
    if (access & MEM_ACCESS_RDMA_READ) {
      ibv_access |= IBV_ACCESS_REMOTE_READ ;
    }
    if (access & MEM_ACCESS_RDMA_WRITE) {
      ibv_access |= IBV_ACCESS_REMOTE_WRITE;
    }

    return ibv_access;
  }

  int RDMAMgr::rdma_register_mr(const MemoryRegion &mr, void *ptr) {
    rdma_mr_params params;
    params.params_.mr = mr;

    const auto pd = pd_map_.find(mr.intf_);
    if (pd == pd_map_.end()) {
      HOLOSCAN_LOG_FATAL("Cannot find MR interface {} in PD mapping", mr.intf_);
      return -1;
    }

    int access = mr_access_to_ibv(mr.access_);
    params.mr_ = ibv_reg_mr(*pd, ptr, mr.params_.buf_size_ * mr.params_.num_bufs_, access);
    if (params.mr_ == nullptr) {
      HOLOSCAN_LOG_FATAL("Failed to register MR {}", mr.name_);
      return -1;
    }

    HOLOSCAN_LOG_INFO("Successfully registered MR {} with {} bytes", mr.name_, mr.params_.buf_size_ * mr.params_.num_bufs_);

    mrs_[mr.name_] = params;  

    return 0;  
  }

  int RDMAMgr::rdma_register_cfg_mrs() {
    HOLOSCAN_LOG_INFO("Registering memory regions");

    for (const auto &mr: cfg_.mrs_) {
      void *ptr;
      
      if (mr.owned_) {
        switch (mr.kind_) {
          case MemoryKind::HOST:
            ptr = malloc(mr.size_);
            break;
          case MemoryKind::HOST_PINNED:
            ptr = cudaHostAlloc(&ptr, mr.size_, 0);
            break;
          case MemoryKind::HUGE:
            ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (ptr == MAP_FAILED) {
              HOLOSCAN_LOG_FATAL("Failed to map hugepages memory!");
              return;
            }
            break;            
          case MemoryKind::DEVICE:
            ptr = malloc(mr.size_);
            break;      
          default:
            HOLOSCAN_LOG_ERROR("Unknown memory type {}!". mr.kind_);
            return -1;                  
        }

        if (ptr == nullptr) {
          HOLOSCAN_LOG_FATAL("Fatal to allocate {} of type {} for MR", mr.buf_size_ * mr.num_bufs_, params.kind_);
          return -1;
        }
      }

      int ret = register_mr(mr, ptr);
      if (ret < 0) {
        return ret;
      }
    }

    // Register all the MRs for exchanging keys
    for (int p = 0; p < cfg_.intfs.size(); p++) {
      const std::string name = std::string("lkey_") + std::to_string(p);
      MemoryRegion mr{
        name,
        cfg_intfs[p].name_,
        MemoryKind::HOST,
        MEM_ACCESS_LOCAL,
        sizeof(rdma_key_xchg),
        MAX_NUM_MR
      };

      lkey_mrs_.emplace_back(rdma_key_xchg{}); // Dummy entry where data will be written

      if (rdma_register_mr(mr, &lkey_mrs_[p]) < 0) {
        HOLOSCAN_LOG_FATAL("Failed to register key exchange MR for interface {}", p);
        return -1;
      }
    }

    return 0;
  }

  AdvNetStatus RDMAMgr::wait_on_key_xchg() {
    
  }

  /**
   * Set up all parameters needed for a newly-connected client
  */
  int RDMAMgr::setup_client_params_for_server(rdma_server_params *sparams, int if_idx) {
    // RX/TX queues should be symmetric with RDMA
    const auto num_queues = cfg_.ifs_[if_idx].rx_.queues_.size();
    for (int qi = 0; qi < num_queues; qi++) {
      rdma_qp_params qp_params;

      qp_params.rx_cq = ibv_create_cq(sparams->client_id->verbs, 
        MAX_CQ, nullptr, nullptr,	0);
      if (qp_params.rx_cq == nullptr) {
        HOLOSCAN_LOG_FATAL("Failed to create RX queue pair! {}", errno);
        return -1;
      }  

      qp_params.tx_cq = ibv_create_cq(sparams->client_id->verbs, 
        MAX_CQ, nullptr, nullptr,	0);
      if (qp_params.tx_cq == nullptr) {
        HOLOSCAN_LOG_FATAL("Failed to create TX queue pair! {}", errno);
        return -1;
      }              

      memset(&qp_params.qp_attr, 0, sizeof(qp_params.qp_attr));       
      qp_params.qp_attr.cap.max_recv_sge = 1; // No header-data split in RDMA right now
      qp_params.qp_attr.cap.max_recv_wr = MAX_OUSTANDING_WR;
      qp_params.qp_attr.cap.max_send_sge = 1;
      qp_params.qp_attr.cap.max_send_wr = MAX_OUSTANDING_WR;

      if (cfg_.ifs_[if_idx].rdma_.xmode_ == RDMATransportMode::RC) {
        qp_params.qp_attr.qp_type = IBV_QPT_RC;
      }
      else if (cfg_.ifs_[if_idx].rdma_.xmode_ == RDMATransportMode::UC) {
        qp_params.qp_attr.qp_type = IBV_QPT_UC;
      }
      else {
        HOLOSCAN_LOG_ERROR("RDMA transport mode {} not supported!", cfg_.ifs_[if_idx].rdma_.xmode_);
        return -1;
      }

      // Share the CQ between TX and RX
      qp_params.recv_cq = qp_params.rx_cq;
      qp_params.send_cq = qp_params.tx_cq;

      ret = rdma_create_qp(sparams->client_id, sparams->pd, &qp_params);
      if (ret != 0) {
	      HOLOSCAN_LOG_FATAL("Failed to create QP: {}", errno);
	      return -1;
      } 

      // Create POSIX message queues for talking to client
      struct mq_attr attr;

      attr.mq_flags = 0;
      attr.mq_maxmsg = 128;
      attr.mq_msgsize = sizeof(AdvNetBurstParams);
      attr.mq_curmsgs = 0;

      std::string q_name = "I" + std::tostring(port) + "_Q" + std::tostring(q);
      qp_params.rx_mq = mq_open((q_name + "_RX").c_str(), O_CREAT | O_WRONLY, 0644, &attr);
      qp_params.tx_mq = mq_open((q_name + "_TX").c_str(), O_CREAT | O_WRONLY, 0644, &attr);

      if (qp_params.rx_mq == (mqd_t)-1 || 
          qp_params.tx_mq == (mqd_t)-1) {
        HOLOSCAN_LOG_FATAL("Failed to create message queues for {}", q_name);
        return;
      }

      sparams->qp_params.emplace_back(qp_params);        
    }
  }

  inline int set_affinity(int cpu_core) {
    // Set the CPU affinity of our thread
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
      HOLOSCAN_LOG_FATAL("Failed to set thread affinity to core {}", cpu_core);
      return -1;
    }

    return 0;    
  }


  /**
   * Worker thread for a server SQ
  */
  void RDMAMgr::server_tx(int if_idx, int q) {
    struct ibv_wc wc;
    int num_comp;
    const auto &qref = cfg_.tx_[if_idx].queues_[q];
    const auto &rdma_qref = sparams_[if_idx].qp_params[q];
    const long cpu_core = strtol(qref.common_.cpu_cores_.c_str(), NULL, 10);

    if (set_affinity(cpu_core) != 0) {
      HOLOSCAN_LOG_FATAL("Failed to set TX core affinity");
      return;
    }

    HOLOSCAN_LOG_INFO("Affined TX thread to core {}", cpu_core);

    // Main TX loop. Wait for send requests from the transmitters to arrive for sending. Also
    // periodically poll the CQ.
    while (!force_quit.load()) {
      while ((num_comp = ibv_poll_cq(cq, 1, &wc)) != 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          HOLOSCAN_LOG_ERROR("CQ error {} for WRID {} and opcode {}", wc.status, wc.wr_id, wc.opcode);
          continue;
        }

        if (wc.opcode == IBV_WR_RDMA_READ) {
          AdvNetBurstParams read_burst;
          read_burst.hdr.hdr.

          out_wr_[cur_wc_id_].wr_id = cur_wc_id_;
          out_wr_[cur_wc_id_].done  = false;
          out_wr_[cur_wc_id_].mr = *(endpt->second);
          out_wr_[cur_wc_id_].mr.ptr = burst.cpu_pkts[p];
          cur_wc_id_++;

          auto ret = mq_send(rdma_qref.rx_mq, &read_burst, sizeof(read_burst), 10);
          if (ret != 0) {
            HOLOSCAN_LOG_FATAL("Failed to send to message queue: {}", errno);
            continue;
          }
        }
      }

      // Now handle any incoming sends
      AdvNetBurstParams burst;
      ssize_t bytes = mq_receive(rdma_qref.tx_mq, &burst, sizeof(burst), nullptr);
      if (bytes > 0) {
        const auto endpt = endpoints_.find(static_cast<struct rdma_cm_id*>(burst.hdr.hdr.rdma.dst_key));
        if (endpt == endpoints_.end()) {
          HOLOSCAN_LOG_ERROR("Trying to send to client {}, but that client is not connected", burst.hdr.hdr.rdma.dst_key);
          continue;
        }

        const auto local_mr = mrs_.find(burst.hdr.hdr.local_mr_name);
        if (local_mr == mrs_.end()) {
          HOLOSCAN_LOG_FATAL("Couldn't find MR with name {} in registry", burst.hdr.hdr.local_mr_name);
          free_tx_burst(&burst);
          continue;
        }

        switch (burst.hdr.hdr.opcode) {
          case AdvNetOpCode::SEND:
          { // perform send operation
            // Currently we expect SEND operations to be rare, and certainly not with thousands of
            // packets. For that reason we post one at a time and do not try to batch.
            for (int p = 0; p < burst.hdr.hdr.num_pkts; p++) {
              ibv_send_wr wr;
              ibv_send_wr *bad_wr;
              ibv_sge sge;

              memset(&wr, 0, sizeof(wr));
              sge.addr      = (uint64_t)burst.cpu_pkts[p];
              sge.length    = (uint32_t)burst.cpu_pkt_lens[p];
              sge.lkey      = local_mr->lkey;
              wr.sg_list    = &sge;
              wr.num_sge    = 1;
              wr.opcode     = IBV_WR_SEND;
              wr.send_flags = IBV_SEND_SIGNALED;

              ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
              if (ret != 0) {
                HOLOSCAN_LOG_FATAL("Failed to post SEND request, errno: {}", errno);
                free_tx_burst(&burst);
                continue;
              }   
            }

            break; 
          }
          case AdvNetOpCode::RDMA_WRITE: [[fall_through]]
          case AdvNetOpCode::RDMA_WRITE_IMM:
          {
            for (int p = 0; p < burst.hdr.hdr.num_pkts; p++) {
              ibv_send_wr wr;
              ibv_send_wr *bad_wr;
              ibv_sge sge;

              memset(&wr, 0, sizeof(wr));
              sge.addr      = (uint64_t)burst.cpu_pkts[p];
              sge.length    = (uint32_t)burst.cpu_pkt_lens[p];
              sge.lkey      = local_mr->lkey;
              wr.sg_list    = &sge;
              wr.num_sge    = 1;
              if (burst.hdr.hdr.opcode==AdvNetOpCode::RDMA_WRITE) {
                wr.opcode   = IBV_WR_RDMA_WRITE;
              }
              else {
                wr.opcode   = IBV_WR_RDMA_WRITE_WITH_IMM;
                wr.imm_data = htonl(burst.hdr.hdr.rdma.imm);
              }
 
              wr.send_flags = IBV_SEND_SIGNALED;

              // Look up remote key
              const auto remote_mr = endpt->second.find(burst.hdr.hdr.rdma.remote_mr_name);
              if (remote_mr == endpt->second.end()) {
                HOLOSCAN_LOG_FATAL("Couldn't find MR with name {} in registry for client {}", 
                      burst.hdr.hdr.remote_mr_name, burst.hdr.hdr.rdma.dst_key);
                free_tx_burst(&burst);
                continue;
              }              
              wr.wr.rdma_key = remote_mr->key;
              wr.wr.rdma_remote_addr = burst.hdr.hdr.rdma.raddr;

              ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
              if (ret != 0) {
                HOLOSCAN_LOG_FATAL("Failed to post SEND request, errno: {}", errno);
                free_tx_burst(&burst);
                continue;
              }
            }
            break;
          }
          case AdvNetOpCode::RDMA_READ:
          {
            for (int p = 0; p < burst.hdr.hdr.num_pkts; p++) {
              ibv_send_wr wr;
              ibv_send_wr *bad_wr;
              ibv_sge sge;

              memset(&wr, 0, sizeof(wr));
              sge.addr      = (uint64_t)burst.cpu_pkts[p];
              sge.length    = (uint32_t)burst.cpu_pkt_lens[p];
              sge.lkey      = local_mr->lkey;
              wr.sg_list    = &sge;
              wr.num_sge    = 1;
              wr.opcode     = IBV_WR_RDMA_READ;
              wr.send_flags = IBV_SEND_SIGNALED;

              // Look up remote key
              const auto remote_mr = endpt->second.find(burst.hdr.hdr.rdma.remote_mr_name);
              if (remote_mr == endpt->second.end()) {
                HOLOSCAN_LOG_FATAL("Couldn't find MR with name {} in registry for client {}", 
                      burst.hdr.hdr.remote_mr_name, burst.hdr.hdr.rdma.dst_key);
                free_tx_burst(&burst);
                continue;
              }              
              wr.wr.rdma_key = remote_mr->key;
              wr.wr.rdma_remote_addr = burst.hdr.hdr.rdma.raddr;

              ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
              if (ret != 0) {
                HOLOSCAN_LOG_FATAL("Failed to post SEND request, errno: {}", errno);
                free_tx_burst(&burst);
                continue;
              }

              out_wr_[cur_wc_id_].wr_id = cur_wc_id_;
              out_wr_[cur_wc_id_].done  = false;
              out_wr_[cur_wc_id_].mr = *(endpt->second);
              out_wr_[cur_wc_id_].mr.ptr = burst.cpu_pkts[p];
              cur_wc_id_++;
            }
            break;
          }          
        }
      }
    }
  }

  /**
   * Worker thread for a server RQ
  */
  void RDMAMgr::server_rx() {
    struct ibv_wc wc;
    int num_comp;
    const auto &qref = cfg_.rx_[if_idx].queues_[q];
    const auto &rdma_qref = sparams_[if_idx].qp_params[q];
    const long cpu_core = strtol(qref.common_.cpu_cores_.c_str(), NULL, 10);


    if (set_affinity(cpu_core) != 0) {
      HOLOSCAN_LOG_FATAL("Failed to set RX core affinity");
      return;
    }

    HOLOSCAN_LOG_INFO("Affined RX thread to core {}", cpu_core);

    // Main TX loop. Wait for send requests from the transmitters to arrive for sending. Also
    // periodically poll the CQ.
    while (!force_quit.load()) {
      while ((num_comp = ibv_poll_cq(cq, 1, &wc)) != 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          HOLOSCAN_LOG_ERROR("CQ error {} for WRID {} and opcode {}", wc.status, wc.wr_id, wc.opcode);
        }
      }    
  }

  void RDMAMgr::run_server() {
    int ret;

    if (set_affinity(cfg_.common_.master_core_) != 0) {
      HOLOSCAN_LOG_FATAL("Failed to set master core affinity");
      return;
    }

    int num_ib_devices;    
    ibv_context **ib_devices = rdma_get_devices(&num_ib_devices);
    if (num_ib_devices == 0) {
      HOLOSCAN_LOG_FATAL("No RDMA-capable devices found!");
      return;
    }

    HOLOSCAN_LOG_INFO("Found {} RDMA-capable devices", num_ib_devices);
      
    // Create a channel for events. Only the master thread reads from this channel
    cm_event_channel_ = rdma_create_event_channel();    
    if (cm_event_channel_ == nullptr) {
      HOLOSCAN_LOG_FATAL("Failed to create a CM channel: {}", errno);
      return;
    }

    // Start an RDMA server on each interface specified
    for (const auto &intf: cfg_.ifs_) {
      // Initialize all the setup before the main loop
      sockaddr_in server_addr;
      int ret;

      // Translate name of interface into a sockaddr
      if (!get_ip_from_interface(intf.if_name_, server_addr)) {
        HOLOSCAN_LOG_ERROR("Failed to look up interface {}", intf.if_name_);
        return;
      }

      HOLOSCAN_LOG_INFO("Successfully created CM event channel");

      struct rdma_cm_id *s_id;
      ret = rdma_create_id(cm_event_channel_, &s_id, nullptr, RDMA_PS_TCP);
      if (ret != 0) {
        HOLOSCAN_LOG_FATAL("Failed to create RDMA server ID {}", errno);
        return;
      }
      cm_server_id_.push_back(s_id);

      HOLOSCAN_LOG_INFO("Created RDMA server ID successfully");

      ret = rdma_bind_addr(s_id, reinterpret_cast<struct sockaddr*>(server_addr));
      if (ret != 0) {
        HOLOSCAN_LOG_FATAL("Failed to bind for RDMA server: {}", errno);
        return;
      }

      // Create a protection domain
      auto pd = ibv_alloc_pd(s_id->verbs);
      if (pd == nullptr) {
        HOLOSCAN_LOG_FATAL("Failed to allocate PD for device! {}", (void*)pd);
        return;
      }

      pd_map_[intf.name_] = pd;

      ret = rdma_listen(s_id, MAX_RDMA_CONNECTIONS);
      if (ret != 0) {
        HOLOSCAN_LOG_FATAL("Failed to listen for RDMA server: {}", errno);
        return;
      }

      HOLOSCAN_LOG_INFO("RDMA server successfully started");
    }


    if (rdma_register_cfg_mrs() < 0) {
      HOLOSCAN_LOG_ERROR("Failed to register MRs");
      return;
    }    

    // Create enough server parameters for each interface
    sparams_.resize(cfg_.rx.size());

    // Our master thread's job is to wait on connections from clients, set up all the needed
    // information for them (QPs, MRs, etc), and spawn client threads to monitor both TX and RX
    struct rdma_cm_event *cm_event = nullptr;    
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
          rdma_server_params sparams{};
          int listen_idx;
          sparams.client_id = cm_event->id;
          const auto listen_id = cm_event->listen_id;          
          ack_event(cm_event);

          const auto listen_iter = std::find(cm_server_id_.begin(), cm_server_id_.end(), listen_id);
          if (listen_iter == cm_server_id_.end()) {
            HOLOSCAN_LOG_ERROR("Failed to find listener ID for {}", listen_id);
            break;
          }
          else {
            // resize sparams_ for number of interfaces and add to array. finish posix mq work
            listen_idx = listen_iter - cm_server_id_.begin();
            setup_client_params_for_server(&sparams, listen_idx);
            HOLOSCAN_LOG_INFO("Configured queues for client. Launching threads");

            sparams_[listen_idx] = sparams;

            // Spawn a new TX and RX thread for each QP
            const auto num_queues = cfg_.rx[listen_idx].queues_.size();
            for (int q = 0; q < num_queues; q++) {
              txrx_workers.emplace_back(std::thread(&RDMAMgr::server_tx, this, listen_idx, q));
              txrx_workers.emplace_back(std::thread(&RDMAMgr::server_rx, this, listen_idx, q));
            }
          }
          break;
        }
        default:
          HOLOSCAN_LOG_INFO("Cannot handle event type {}", cm_event->event);
      }     
    }

    // Join any client threads we had spawned
    HOLOSCAN_LOG_INFO("Waiting for server TX/RX threads to complete");
    for (auto &w: tx_rx_workers) {
      w.join();
    }

    HOLOSCAN_LOG_INFO("Finished cleaning up TX/RX workers");
  }


  void RDMAMgr::initialize() {
    if (cfg_.common_.rdma_.mode_ == RDMAMode::SERVER) {
      std::thread t(&RDMAMgr::run_server, this);
      t.join();      
    }
    else {
      init_client();
      std::thread t(&RDMAMgr::run_client, this);
      t.join();      
    }
  }

}