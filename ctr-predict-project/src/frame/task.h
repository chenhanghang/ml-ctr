#ifndef CTR_PREDICT_TASK_H
#define CTR_PREDICT_TASK_H

#include <vector>
#include "src/sample/sample.h"

namespace ml {
namespace ctr {

class Task {
public:
    Task(){}
    virtual void run_task(std::vector<Sample>& data_buffer) {};
    //list wise 需要
    virtual void run_task(std::vector<std::vector<Sample>>& data_buffer) {};
};

}
}


#endif //CTR_PREDICT_TASK_H
