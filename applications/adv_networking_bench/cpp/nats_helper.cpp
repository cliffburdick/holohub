#include "nats_helper.h"
#include <sstream>
#include <iostream>
#include <string.h>

void NATS::MsgCallback([[maybe_unused]] natsConnection *nc, [[maybe_unused]] natsSubscription *sub, natsMsg *msg, void *closure)
{
    auto config = reinterpret_cast<CBConfig *>(closure);
    auto q = config->q;
    auto buf_pool = config->buf_pool;

    printf("Incoming message from GUI\n");

    void *buf;
    {
        std::lock_guard<std::mutex> guard(*config->buf_mutex);
        buf = buf_pool->front();
        buf_pool->pop();
    }

    memcpy(buf, natsMsg_GetData(msg), natsMsg_GetDataLength(msg));

    {
        std::lock_guard<std::mutex> guard(*config->q_mutex);
        q->push(buf);
    }

    natsMsg_Destroy(msg);
}

void NATS::Init(const std::string &_host) 
{
    host = _host;
    printf("Initializing NATS\n");

    natsOptions_Create(&opts);
    natsOptions_SetIOBufSize(opts, (1 << 25));
    natsOptions_SetURL(opts, host.c_str());

    tx_buf = malloc(1<<25);

    for (int b = 0; b < NUM_RX_BUFS; b++) {
        buf_pool.push(malloc(65536));
    }

    config = CBConfig{&q, &buf_pool, &q_mutex, &buf_mutex};

    printf("Connecting to NATS at %s\n", host.c_str());
    Connect();
    Subscribe(RECEIVER_SUBJECT);
    printf("Subscribed\n");
}

void NATS::Connect() {
    if (natsConnection_Connect(&nc, opts) != NATS_OK) {
        printf("Failed to connect to NATS\n");
    }
    else {
        printf("Connected successfully to NATS\n");
    }
}

bool NATS::GetItem(void **data) {
    if (q.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> guard(q_mutex);
    auto val = q.front();
    q.pop();

    *data = val;
    return true;
}

void NATS::FreeBuf(void *buf) {
    std::lock_guard<std::mutex> guard(buf_mutex);
    buf_pool.push(buf);
}

void NATS::SendToGUI(const void *buf, const char *subject, size_t len)
{
    //uint32_t *htype = reinterpret_cast<uint32_t*>(tx_buf);
    //uint8_t  *hbuf  = reinterpret_cast<uint8_t* >(tx_buf) + sizeof(*htype);
    //*htype = type;
    //memcpy(hbuf, buf, len);
    //natsConnection_Publish(nc, subject, tx_buf, len + sizeof(*htype));
    natsConnection_Publish(nc, subject, buf, len);
}


void NATS::Subscribe(const std::string &stream) {

    natsConnection_Subscribe(&sub, nc, stream.c_str(), NATS::MsgCallback, &config);
}
