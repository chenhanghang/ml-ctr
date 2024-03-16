#ifndef _CTR_PREDICT_LRMODEL_H_
#define _CTR_PREDICT_LRMODEL_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"

namespace ml {
namespace ctr {


class lr_model_unit {
public:
    double wi;
    std::mutex mtx;
public:
    lr_model_unit() {
        wi = 0.0;
    }
    lr_model_unit(double wi) {
        this->wi = wi;
    }
};



class LrModel {
    public:
        LrModel();
        double predict(const Sample & sample);
        void save(const std::string & model_path);
        void init(const std::string & model_path);
        lr_model_unit * get_or_init_model_unit_bias();
        lr_model_unit * get_or_init_model_unit(std::string key);
    private:
        std::unordered_map<std::string, lr_model_unit*> _model;
        static const std::string model_spliter;
        std::mutex mtx;//模型锁
};

}
}

#endif


