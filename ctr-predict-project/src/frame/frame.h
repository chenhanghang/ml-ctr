#ifndef _CTR_PREDICT_FRAME_H
#define _CTR_PREDICT_FRAME_H

#include <string>
#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <iostream>
#include "src/frame/task.h"
#include "src/sample/sample.h"
#include "src/utils/trainer_option.h"

namespace ml {
namespace ctr {

class Frame {
public:
    Frame(){}
    virtual bool init(Task * task, TrainerOption *opt);
    void run();
    virtual void pro_thread();
    virtual void con_thread();

protected:
    Task* task;
    std::mutex buf_mtx;
    std::condition_variable _cv_queue_not_full;
    std::condition_variable _cv_queue_not_empty;

    std::queue<std::vector<Sample> > buffer;
    std::vector<std::thread> thread_vec;
    int _thread_num;
    int _buf_size;
    int _log_num;
    int finished_flag;
};

}
}
#endif //_CTR_PREDICT_FRAME_H

