#ifndef _CTR_PREDICT_FM_FTRL_TRAINER_H_
#define _CTR_PREDICT_FM_FTRL_TRAINER_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/frame/task.h"
#include "src/utils/trainer_option.h"
#include "src/models/fm_ftrl/fm_ftrl_model.h"
#include "src/models/base_trainer.h"

namespace ml {
namespace ctr {

class FmFtrlTrainer : public BaseTrainer{
    public:
        FmFtrlTrainer(TrainerOption * opt);
        virtual void train(Sample & sample);
        virtual void test();//测试模型效果
        virtual void input_model(std::string model_path);
        virtual void output_model(std::string model_path);
        virtual void run_task(std::vector<Sample>& data_buffer);
        virtual void run_task(std::vector<std::vector<Sample>>& data_buffer);
    private:
        std::shared_ptr<FmFtrlModel> _p_model;
};

}
}

#endif


