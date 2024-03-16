#include "src/models/lr/lr_trainer.h"
#include "src/models/lr/lr_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

LrTrainer::LrTrainer(TrainerOption * opt) {
    this->opt = opt;

}


void LrTrainer::input_model(std::string model_path) {
    this->_p_model = std::make_shared<LrModel>();
    this->_p_model->init(model_path);
}

//模型保存
void LrTrainer::output_model(std::string model_path) {
    this->_p_model->save(model_path);
}

void LrTrainer::run_task(std::vector<std::vector<Sample>>& data_buffer) {
    for(int i=0; i<data_buffer.size(); i++) {
        //std::cout<<"data_buffer[i]"<<data_buffer[i].size();
        this->run_task(data_buffer[i]);
    }
}


void LrTrainer::run_task(std::vector<Sample>& data_buffer) {
    for (auto sample : data_buffer) {
        this->train(sample);        
    }
}

void LrTrainer::train(Sample &sample) {
    double prob = sample.y==1 ? this->_p_model->predict(sample): 1-this->_p_model->predict(sample);
    int y = sample.y;
    for (auto iter = sample.xs.begin(); iter != sample.xs.end(); iter++) {
        double delta = y*(prob - 1)* (*iter)->v;
        lr_model_unit *model_unit = this->_p_model->get_or_init_model_unit((*iter)->i);
        model_unit->mtx.lock();
        model_unit->wi -= (this->opt->alpha)*delta - (model_unit->wi)*(this->opt->alpha)*(this->opt->reg);
        model_unit->wi = this->_intersect(model_unit->wi);
        model_unit->mtx.unlock();
    }
    lr_model_unit *model_unit = this->_p_model->get_or_init_model_unit_bias();
    model_unit->mtx.lock();
    model_unit->wi -= (this->opt->alpha)*y*(prob - 1);
    model_unit->wi = this->_intersect(model_unit->wi);
    model_unit->mtx.unlock();
}

void LrTrainer::test() {
    std::ifstream infile;
    infile.open(opt->test_path.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    std::string s;
    std::vector<Sample> samples;
    while (getline(infile,s))  {
        samples.push_back(Sample(s));
    }
    infile.close(); //关闭文件输入流
    //std::cout<<"samples size" << samples.size()<<std::endl;
    //遍历sample并且进行predict预测
    std::vector<double> probs;
    std::vector<int> labels;
    for (std::vector<Sample>::iterator it = samples.begin(); it != samples.end(); ++it) {
        double prob = this->_p_model->predict(*it);
        probs.push_back(prob);
        labels.push_back(it->y);
        //std::cout<<prob<<std::endl;
    }
    double auc_value = Metric::auc(labels, probs);
    double log_loss_value = Metric::log_loss(labels, probs);
    double acuraccy_value = Metric::acuraccy(labels, probs, this->opt->threshold);
    std::cout<<"auc:"<<auc_value<<" log_loss:"<<log_loss_value<<" acuraccy:"<<acuraccy_value<<std::endl;
}


}
}


