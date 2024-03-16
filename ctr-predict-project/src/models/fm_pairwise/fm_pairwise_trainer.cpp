#include "src/models/fm_pairwise/fm_pairwise_trainer.h"
#include "src/models/fm_pairwise/fm_pairwise_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

FmPairwiseTrainer::FmPairwiseTrainer(TrainerOption * opt) {
    this->opt = opt;

}


void FmPairwiseTrainer::input_model(std::string model_path) {
    this->_p_model = std::make_shared<FmPairwiseModel>(this->opt->factor);
    this->_p_model->init(model_path);
}

//模型保存
void FmPairwiseTrainer::output_model(std::string model_path) {
    this->_p_model->save(model_path);
}

void FmPairwiseTrainer::run_task(std::vector<std::vector<PairwiseSample>>& data_buffer) {
    for(int i=0; i<data_buffer.size(); i++) {
        this->run_task(data_buffer[i]);
    }
}


void FmPairwiseTrainer::run_task(std::vector<PairwiseSample>& data_buffer) {
    for (auto sample : data_buffer) {
        this->train(sample);        
    }
}

void FmPairwiseTrainer::train(PairwiseSample &pair_sample) {
    if(!pair_sample.flag) { return; } //解析错误
    //std::cout<<s<<std::endl;
    Sample & sample_left = (*pair_sample.left);
    Sample & sample_right = (*pair_sample.right);
    const int x_len_left = sample_left.xs.size();
    vector<fm_pairwise_model_unit*> model_units_left(x_len_left, NULL);
    for(int i = 0; i < x_len_left; ++i) {
        std::string index = sample_left.xs[i]->i;
        model_units_left[i] = this->_p_model->get_or_init_model_unit(index);
    }
    const int x_len_right = sample_right.xs.size();
    vector<fm_pairwise_model_unit*> model_units_right(x_len_right, NULL);
    for(int i = 0; i < x_len_right; ++i) {
        std::string index = sample_right.xs[i]->i;
        model_units_right[i] = this->_p_model->get_or_init_model_unit(index);
    }
    fm_pairwise_model_unit *model_unit_bias = this->_p_model->get_or_init_model_unit_bias();
    double pos_phi_hat = this->_p_model->predict(sample_left, model_units_left, model_unit_bias);
    double neg_phi_hat = this->_p_model->predict(sample_right, model_units_right, model_unit_bias);
    double phi_hat = pos_phi_hat - neg_phi_hat;
    double lambda =  -1.0 / (1.0 + exp(phi_hat));
    this->update_per_sample(sample_left, lambda);
    this->update_per_sample(sample_right, lambda);

}

void FmPairwiseTrainer::update_per_sample(Sample &sample, double lambda) {
    //训练模型
    const int x_len = sample.xs.size();
    vector<fm_pairwise_model_unit*> model_units(x_len, NULL);
    int y = sample.y;

    for(int i = 0; i < x_len; ++i) {
        std::string index = sample.xs[i]->i;
        model_units[i] = this->_p_model->get_or_init_model_unit(index);
    }

    std::vector<double> sum_vf(this->opt->factor,0.0);
    //计算交叉项梯度求和
    for (std::vector<int>::size_type i=0; i< sample.xs.size(); i++) {
        fm_pairwise_model_unit *model_unit = model_units[i];
        for (int f=0; f<this->opt->factor; f++) {
            sum_vf[f] += model_unit->vi[f]*sample.xs[i]->v;
        }
    }
    //更新w和v[f]
    for (std::vector<int>::size_type i=0; i< sample.xs.size(); i++) {
        double v = sample.xs[i]->v;
        fm_pairwise_model_unit *model_unit = model_units[i];
        model_unit->mtx.lock();
        for (int f=0; f < this->opt->factor; f++) {
            //更新vif
            model_unit->vi[f] -= lambda*opt->alpha*y*(v*sum_vf[f] - model_unit->vi[f]*std::pow(v, 2));
            //model_unit->vi[f] = this->_intersect(model_unit->vi[f]);
        }
        //更新wi
        model_unit->wi -= lambda*opt->alpha*y*v;
        //model_unit->wi = this->_intersect(model_unit->wi);
        model_unit->mtx.unlock();

    }

}


void FmPairwiseTrainer::test() {
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


