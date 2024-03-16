#ifndef _CTR_PREDICT_FFM_TRAINER_H_
#define _CTR_PREDICT_FFM_TRAINER_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/frame/task.h"
#include "src/utils/trainer_option.h"
#include "src/models/base_trainer.h"
#include "src/models/ffm/ffm_model.h"

namespace ml {
namespace ctr {

class FfmTrainer : public BaseTrainer{
    public:
        FfmTrainer(TrainerOption * opt);
        virtual void train(Sample & sample);
        virtual void test();//测试模型效果
        virtual void input_model(std::string model_path);
        virtual void output_model(std::string model_path);
        virtual void run_task(std::vector<Sample>& data_buffer);
        virtual void run_task(std::vector<std::vector<Sample>>& data_buffer);
    private:
        std::shared_ptr<FfmModel> _p_model;
};

}
}

#endif


