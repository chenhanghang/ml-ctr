#ifndef _CTR_PREDICT_FM_TRAINER_H_
#define _CTR_PREDICT_FM_TRAINER_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/sample/pairwise_sample.h"
#include "src/frame/task.h"
#include "src/utils/trainer_option.h"
#include "src/models/fm_pairwise/fm_pairwise_model.h"
#include "src/models/base_trainer.h"

namespace ml {
namespace ctr {

class FmPairwiseTrainer{
    public:
        FmPairwiseTrainer(TrainerOption * opt);
        void train(PairwiseSample & sample);
        void test();//测试模型效果
        void input_model(std::string model_path);
        void output_model(std::string model_path);
        void run_task(std::vector<PairwiseSample>& data_buffer);
        void run_task(std::vector<std::vector<PairwiseSample>>& data_buffer);
    private:
        std::shared_ptr<FmPairwiseModel> _p_model;
        void update_per_sample(Sample &sample, double lambda);
        TrainerOption * opt;
};

}
}

#endif


