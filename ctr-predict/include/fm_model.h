#ifndef _CTR_PREDICT_FMMODEL_H_
#define _CTR_PREDICT_FMMODEL_H_

#include<map>
#include<vector>
#include<string>

#include "sample.h"
#include "ctr_model.h"

namespace ml {
namespace ctr {
#define FM_K 12
struct FmItem {
    double w=0.0;
    double *v = new double[FM_K]();
};

class FmModel : public CtrModel{
    public:
        FmModel();
        void train(const std::string & train_path, const std::string & test_path) override;
        void test(const std::string & test_path) override;
        double predict(const Sample & sample) override;
        void save() override;
        void init() override;
        void train_SGD(const std::string & train_path, const std::string & test_path);
        void train_BGD(const std::string & train_path, const std::string & test_path);
        void train_FTRL(const std::string & train_path, const std::string & test_path);
    private:
        std::map<std::string, FmItem> _model;
        const int K=FM_K; //隐向量长度
        static const std::string _model_spliter;
        double b=0.0;
};

}
}

#endif


