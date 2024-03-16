#include "src/models/ffm/ffm_trainer.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

FfmTrainer::FfmTrainer(TrainerOption * opt) {
    this->opt = opt;

}


void FfmTrainer::input_model(std::string model_path) {
    this->_p_model = std::make_shared<FfmModel>(this->opt->field, this->opt->factor);
    this->_p_model->init(model_path);
}

//模型保存
void FfmTrainer::output_model(std::string model_path) {
    this->_p_model->save(model_path);
}

void FfmTrainer::run_task(std::vector<std::vector<Sample>>& data_buffer) {
    for(int i=0; i<data_buffer.size(); i++) {
        this->run_task(data_buffer[i]);
    }
}


void FfmTrainer::run_task(std::vector<Sample>& data_buffer) {
    for (auto sample : data_buffer) {
        this->train(sample);        
    }
}

void FfmTrainer::train(Sample &sample) {
    //训练模型
    const int x_len = sample.xs.size();
    vector<ffm_model_unit*> model_units(x_len, NULL);
    int y = sample.y;

    for(int i = 0; i < x_len; ++i) {
        std::string index = sample.xs[i]->i;
        model_units[i] = this->_p_model->get_or_init_model_unit(index);
    }
    ffm_model_unit *model_unit_bias = this->_p_model->get_or_init_model_unit_bias();

    //std::vector<double> sum_vf(this->opt->factor,0.0);
    //计算交叉项梯度求和
    //for (std::vector<int>::size_type i=0; i< sample.x.size(); i++) {
    //    ffm_model_unit *model_unit = model_units[i];
    //    for (int f=0; f<this->opt->factor; f++) {
    //        sum_vf[f] += model_unit->vif[0][f]*sample.x[i].second;
    //    }
    //}
    double prob = sample.y==1 ? this->_p_model->predict(sample, model_units, model_unit_bias): 1-this->_p_model->predict(sample, model_units, model_unit_bias);
    double sig_g = opt->alpha*y*(prob-1);
    //更新w梯度
    for (std::vector<int>::size_type i=0; i< sample.xs.size(); i++) {
        ffm_model_unit *model_unit = model_units[i];
        model_unit->mtx.lock();
        model_unit->wi -= sig_g*sample.xs[i]->v;
        model_unit->wi = this->_intersect(model_unit->wi);
        model_unit->mtx.unlock();

    }
    //更新w和v[f]
    for (std::vector<int>::size_type i=0; i< sample.xs.size(); i++) {
        for (std::vector<int>::size_type j=i+1; j< sample.xs.size(); j++) {
            ffm_model_unit *model_unit1 = model_units[i];
            ffm_model_unit *model_unit2 = model_units[j];
            int fd1 = sample.xs[i]->f;
            int fd2 = sample.xs[j]->f;
            for (int k=0; k < this->opt->factor; k++) {
                //更新vif
                model_unit1->mtx.lock();
                model_unit1->vif[fd2][k] -= sig_g*model_unit2->vif[fd1][k]*std::pow(sample.xs[i]->v, 2);
                model_unit1->vif[fd2][k] = this->_intersect(model_unit1->vif[fd1][k]);
                model_unit1->mtx.unlock();
                model_unit2->mtx.lock();
                model_unit2->vif[fd1][k] -= sig_g*model_unit1->vif[fd2][k]*std::pow(sample.xs[i]->v, 2);
                model_unit2->vif[fd1][k] = this->_intersect(model_unit2->vif[fd2][k]);
                model_unit2->mtx.unlock();
            }
        }

    }
    //更新bias
    model_unit_bias->mtx.lock();
    model_unit_bias->wi -= sig_g;
    model_unit_bias->wi = this->_intersect(model_unit_bias->wi);
    model_unit_bias->mtx.unlock();

}


void FfmTrainer::test() {
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


