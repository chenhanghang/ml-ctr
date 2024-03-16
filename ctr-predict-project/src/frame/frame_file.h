#ifndef _CTR_PREDICT_FRAME_FILE_H
#define _CTR_PREDICT_FRAME_FILE_H

#include <string>
#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <semaphore.h>
#include <iostream>
#include "src/frame/task.h"
#include "src/frame/frame.h"
#include "src/sample/sample.h"

namespace ml {
namespace ctr {

class FrameFile: public Frame{
public:
    bool init(Task * task, TrainerOption *opt);
    void pro_thread();

private:
    std::string _train_path;

};

}
}
#endif //_CTR_PREDICT_FRAME_FILE_H

