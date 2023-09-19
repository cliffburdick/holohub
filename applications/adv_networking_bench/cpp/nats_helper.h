#ifndef C921E9FD_2AA6_4134_ACDA_9CF40EC02B30
#define C921E9FD_2AA6_4134_ACDA_9CF40EC02B30


#include <string>
#include "nats.h"
#include <mutex>
#include <queue>


enum ADIMsgType {
    MSG_TYPE_PSD = 1,
    MSG_TYPE_CONSTELLATION = 2,
};

struct ADIRxMsg {
    uint32_t type;
    uint8_t payload[];
};

struct CBConfig {
    std::queue<void *> *q;
    std::queue<void *> *buf_pool; 
    std::mutex *q_mutex;  
    std::mutex *buf_mutex; 
};

#define NUM_RX_BUFS 256


class NATS {

    public:
        static inline const char *GUI_SUBJECT_PSD = "adi.gui.psd";
        static inline const char *GUI_SUBJECT_CONST_REF = "adi.gui.constellation_ref";
        static inline const char *GUI_SUBJECT_CONST_RX = "adi.gui.constellation_rx";
        static inline const char *GUI_SUBJECT_EVM = "adi.gui.evm";
        static inline const char *RECEIVER_SUBJECT = "adi.receiver";


        void Init(const std::string &host);
        void SendToGUI(const void *buf, const char *subject, size_t len);
        void Connect();
        void FreeBuf(void *buf);
        bool GetItem(void **data);
        void Subscribe(const std::string &stream);
        static void MsgCallback(natsConnection *nc, natsSubscription *sub, natsMsg *msg, void *closure);

    private:
        std::string host;
        natsOptions         *opts = nullptr;
        natsConnection      *nc  = nullptr;
        natsSubscription    *sub = nullptr;
        natsMsg             *msg = nullptr;
        void                *tx_buf = nullptr;
        std::queue<void *> q;
        std::queue<void *> buf_pool;
        std::mutex q_mutex;
        std::mutex buf_mutex;
        CBConfig config;
};


#endif /* C921E9FD_2AA6_4134_ACDA_9CF40EC02B30 */
