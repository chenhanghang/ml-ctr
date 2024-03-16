#ifndef _LR_PAIRWISE_LRMODEL_H_
#define _LR_PAIRWISE_LRMODEL_H_

#include<map>
#include<vector>
#include<string>

#include "sample.h"
#include "pairwise_sample.h"
#include "ctr_model.h"

namespace ml {
namespace ctr {
class LrPairwiseModel : public CtrModel{
    public:
        LrPairwiseModel();
        void train(const std::string & train_path, const std::string & test_path) override;
        void train_SGD(const std::string & train_path, const std::string & test_path);
        void train_BGD(const std::string & train_path, const std::string & test_path);
        void test(const std::string & test_path) override;
        double predict(const Sample & sample) override;
        void save() override;
        void init() override;
        bool load_bin_model(const std::string& model_path) override;
        bool load_txt_model(const std::string& model_path) override;
        bool save_bin_model(const std::string& model_path) override;
        bool save_txt_model(const std::string& model_path) override;
    private:
        std::map<std::string, double> _model;
        static const std::string model_spliter;
        void update_per_sample(Sample & x, double & lambda, int batch_size);//更新一条样本
        double computePhiHat(const Sample & sample);
};

}
}

#endif


