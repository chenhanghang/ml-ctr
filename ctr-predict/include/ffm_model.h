#ifndef _CTR_PREDICT_FFMMODEL_H_
#define _CTR_PREDICT_FFMMODEL_H_

#include<map>
#include<vector>
#include<string>

#include "sample.h"
#include "ctr_model.h"

namespace ml {
namespace ctr {
#define FFM_K 12
#define FFM_F 1

struct FfmItem {
    double w=0.0;
    double (*v)[FFM_K] = new double[FFM_F][FFM_K]();
    /*double **v;
    FfmItem() {
        w = 0.0;
        //double (*v)[FFM_K] = new double[FFM_F][FFM_K]();
        v = new double *[FFM_F];
        for(int r=0; r<FFM_F; r++) {
          v[r] = new double[FFM_K]();
        }
    }
    ~FfmItem() {
        for(int r=0; r<FFM_F; r++)
            delete[] v[r];
        delete[] v;
    }*/
};

class FfmModel : public CtrModel{
    public:
        FfmModel();
        void train(const std::string & train_path, const std::string & test_path) override;
        void test(const std::string & test_path) override;
        double predict(const Sample & sample) override;
        void save() override;
        void init() override;
        void train_SGD(const std::string & train_path, const std::string & test_path);
        void train_BGD(const std::string & train_path, const std::string & test_path);
        void train_FTRL(const std::string & train_path, const std::string & test_path);
    private:
        std::map<std::string, FfmItem> _model;
        const int K=FFM_K; //隐向量长度
        const int F=FFM_F;
        static const std::string _model_spliter;
        double b=0.0;
};

}
}

#endif

