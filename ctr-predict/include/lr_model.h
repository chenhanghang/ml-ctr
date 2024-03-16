#ifndef _CTR_PREDICT_LRMODEL_H_
#define _CTR_PREDICT_LRMODEL_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "sample.h"
#include "ctr_model.h"

namespace ml {
namespace ctr {
class LrModel : public CtrModel{
    public:
        LrModel();
        void train(const std::string & train_path, const std::string & test_path) override;
        void train_SGD(const std::string & train_path, const std::string & test_path);
        void train_BGD(const std::string & train_path, const std::string & test_path);
        void test(const std::string & test_path) override;
        double predict(const Sample & sample) override;
        void save() override;
        void init() override;
    private:
        std::unordered_map<std::string, double> _model;
        static const std::string model_spliter;
};

}
}

#endif


