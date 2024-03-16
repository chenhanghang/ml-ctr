#include<iostream>
#include<gflags/gflags.h>

#include "lr_model.h"
#include "fm_model.h"
#include "ffm_model.h"

//http://dreamrunner.org/blog/2014/03/09/gflags-jian-ming-shi-yong/
DEFINE_string(algorithm, "lr", "algorithm for predict. lr fm ffm ..");
DEFINE_string(model_path, "./data/lr.model", "model save path");
DEFINE_string(mode, "test", "mode train or test");
DEFINE_string(train_path, "./data/train.dat", "mode train or test");
DEFINE_string(test_path, "./data/test.dat", "mode train or test");

DEFINE_int32(batch, 100, "batch size to train.");
DEFINE_int32(epoch, 30, "train epoch.");
DEFINE_double(alpha, 0.03, "train rate.");
DEFINE_double(reg, 0.001, "L2 regularization rate.");
DEFINE_string(optimizer, "bgd", "optimizer method sgd or bgd.");
DEFINE_double(threshold, 0.8, "threshold for calculate accuracy.");


void process(std::unique_ptr<::ml::ctr::CtrModel> &ctr_model) {
    if(ctr_model == nullptr) {
        std::cout<<"ctr_model ptr is null!!!"<<std::endl;
        return ;
    }
    ctr_model->set_epoch(FLAGS_epoch) //训练轮数
        .set_alpha(FLAGS_alpha)    //训练速度
        .set_batch(FLAGS_batch)    //batch大小
        .set_reg(FLAGS_reg)        //正则参数
        .set_optimizer(FLAGS_optimizer) //优化方法
        .set_threshold(FLAGS_threshold)
        .set_mode(FLAGS_mode)
        .set_model_path(FLAGS_model_path);
    ctr_model->init();
    //训练
    if(FLAGS_mode == "train") {
        std::cout<<"Begin train progress....."<<" mode:"<<FLAGS_mode<<" train_path:"<<FLAGS_train_path<<std::endl;
        //lrModel.trainSGD(FLAGS_train_path, FLAGS_test_path, FLAGS_epoch, FLAGS_alpha, FLAGS_batch, FLAGS_beta);
        ctr_model->train(FLAGS_train_path, FLAGS_test_path);
        ctr_model->save();
    } else if(FLAGS_mode == "test") {
        std::cout<<"Begin test progress....."<<" mode:"<<FLAGS_mode<<" test_path:"<<FLAGS_test_path<<std::endl;
        ctr_model->test(FLAGS_test_path);
    } else {
        std::cout<<"please give a right mode train or test!!!"<<std::endl;
    }
} 


int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::unique_ptr<::ml::ctr::CtrModel> ctr_model;
    if(FLAGS_algorithm == "lr") {
        ctr_model.reset(new ::ml::ctr::LrModel());
    } else if(FLAGS_algorithm == "fm"){
        ctr_model.reset(new ::ml::ctr::FmModel());
    } else if(FLAGS_algorithm == "ffm") {
        ctr_model.reset(new ::ml::ctr::FfmModel());
    }
    process(ctr_model);
    return 0;
}


