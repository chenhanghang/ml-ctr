#ifndef _CTR_PREDICT_BASE_TRAINER_H_
#define _CTR_PREDICT_BASE_TRAINER_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/frame/task.h"
#include "src/utils/trainer_option.h"

namespace ml {
namespace ctr {

class BaseTrainer : public Task{
    public:
        virtual void train(Sample & sample)=0;
        virtual void test()=0;//测试模型效果
        virtual void input_model(std::string model_path)=0;
        virtual void output_model(std::string model_path)=0;
        virtual void run_task(std::vector<Sample>& data_buffer)=0;
        virtual void run_task(std::vector<std::vector<Sample>>& data_buffer)=0;
        virtual ~BaseTrainer() {
        };
    protected:
        TrainerOption * opt;
        double _intersect(double x);
};

}
}

#endif


