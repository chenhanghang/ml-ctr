#ifndef _CTR_PREDICT_CTRMODEL_H_
#define _CTR_PREDICT_CTRMODEL_H_

#include<map>
#include<vector>
#include<string>

#include "sample.h"

namespace ml {
namespace ctr {
class CtrModel {
    public:
        CtrModel(const std::string & model_path, const std::string & mode);
        CtrModel();
        virtual void train(const std::string & train_path, const std::string & test_path)=0;
        virtual void test(const std::string & test_path)=0;
        virtual double predict(const Sample & sample)=0;
        virtual void save()=0;
        virtual void init()=0;
        virtual bool load_txt_model(const std::string& model_path);
        virtual bool load_bin_model(const std::string& model_path);
        virtual bool save_txt_model(const std::string& model_path);
        virtual bool save_bin_model(const std::string& model_path);
        unsigned long get_size();
        CtrModel & set_alpha(double alpha);
        CtrModel & set_epoch(int epoch);
        CtrModel & set_batch(int batch);
        CtrModel & set_reg(double reg);
        CtrModel & set_mode(std::string mode);
        CtrModel & set_threshold(double threshold);
        CtrModel & set_optimizer(std::string optimizer);
        CtrModel & set_model_path(std::string model_path);
        virtual ~CtrModel(){};
    protected:
        double _alpha;//迭代步长
        int _epoch;   //迭代轮数
        int _batch; //batchsize
        double _reg; //l2 regularization
        std::string _mode; //test or train
        unsigned long _size;//模型大小
        double _threshold=0.8; //计算准确率时候使用
        std::string _optimizer;
        std::string _model_path;
        double _intersect(double x);
};

}
}
#endif


