#include "src/models/fm_ftrl/fm_ftrl_trainer.h"
#include "src/models/fm_ftrl/fm_ftrl_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cmath>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

FmFtrlTrainer::FmFtrlTrainer(TrainerOption * opt) {
    this->opt = opt;

}


void FmFtrlTrainer::input_model(std::string model_path) {
    this->_p_model = std::make_shared<FmFtrlModel>(this->opt->factor);
    this->_p_model->init(model_path);
}

//模型保存
void FmFtrlTrainer::output_model(std::string model_path) {
    this->_p_model->save(model_path);
}

void FmFtrlTrainer::run_task(std::vector<std::vector<Sample>>& data_buffer) {
    for(int i=0; i<data_buffer.size(); i++) {
        this->run_task(data_buffer[i]);
    }
}


void FmFtrlTrainer::run_task(std::vector<Sample>& data_buffer) {
    for (auto sample : data_buffer) {
        this->train(sample);        
    }
}

void FmFtrlTrainer::train(Sample &sample) {
    const int x_len = sample.xs.size();
    vector<fm_ftrl_model_unit*> model_units(x_len, NULL);

    for(int i = 0; i < x_len; ++i) {
        std::string index = sample.xs[i]->i;
        model_units[i] = this->_p_model->get_or_init_model_unit(index); 
    }
    fm_ftrl_model_unit *model_unit_bias = this->_p_model->get_or_init_model_unit_bias();

    //update w,v
    for(int i = 0; i <= x_len; ++i) {
        fm_ftrl_model_unit& model_unit = i<x_len ? *(model_units[i]) : *(model_unit_bias);
        model_unit.mtx.lock();
        if(fabs(model_unit.w_zi) <= this->opt->w_lambda1) {
            model_unit.wi = 0.0; 
        } else {
            model_unit.wi = (-1) * (1 / (this->opt->w_lambda2 + (this->opt->w_beta + sqrt(model_unit.w_ni)) / this->opt->w_alpha)) *
                    (model_unit.w_zi - Util::sgn(model_unit.w_zi) * this->opt->w_lambda1);
        }
        if(i >= x_len) {
            model_unit.mtx.unlock();
            break;
        }
        for(int f = 0; f < this->opt->factor; ++f) {
            if(model_unit.vi[f] > 0) {
                if(fabs(model_unit.v_zi[f]) <= this->opt->v_lambda1) {
                    model_unit.vi[f] = 0.0; 
                } else {
                    model_unit.vi[f]  = (-1) * 
                        (1 / (this->opt->v_lambda2 + (this->opt->v_beta + sqrt(model_unit.v_ni[f])) / this->opt->v_alpha)) *
                        (model_unit.v_zi[f] - Util::sgn(model_unit.v_zi[f]) * this->opt->v_lambda1);
                }
            }
        
        }
        model_unit.mtx.unlock();
    
    }

    double prob = sample.y==1 ? this->_p_model->predict(sample,  model_units, model_unit_bias): 1-this->_p_model->predict(sample, model_units, model_unit_bias);
    //std::cout<<this->_p_model->predict(sample)<<"==="<<sample.xs.size()<<"=="<<this->opt->v_alpha<<std::endl;
    int y = sample.y;
    std::vector<double> sum_vf(this->opt->factor,0.0);
    //计算交叉项梯度求和
    for(int i = 0; i < x_len; ++i) {
        fm_ftrl_model_unit& model_unit = *(model_units[i]);
        for (int f=0; f<this->opt->factor; f++) {
            sum_vf[f] += model_unit.vi[f]*sample.xs[i]->v;
        }
    }

    //update w_n, w_z
    for(int i = 0; i <= x_len; ++i) {
        double xi = i< x_len ? sample.xs[i]->v : 1.0;
        fm_ftrl_model_unit& model_unit = i<x_len ? *(model_units[i]) : *(model_unit_bias);
        model_unit.mtx.lock();
        double w_gi = y*(prob-1) * xi;
        double w_thetai = 1 / this->opt->w_alpha * (sqrt(model_unit.w_ni + w_gi * w_gi) - sqrt(model_unit.w_ni));
        model_unit.w_zi += w_gi - w_thetai * model_unit.wi;
        model_unit.w_ni += w_gi * w_gi;

        if(i == x_len) {
            model_unit.mtx.unlock();
            break;//bias no factor 
        }
        
        for (int f=0; f<this->opt->factor; f++) {
            double v_gif = y*(prob-1) * (sum_vf[f] * xi - model_unit.vi[f] * xi * xi);
            double v_thetaif = 1 / this->opt->v_alpha * (sqrt(model_unit.v_ni[f] + v_gif * v_gif) - sqrt(model_unit.v_ni[f]));
            model_unit.v_zi[f] += v_gif - v_thetaif * model_unit.vi[f];
            model_unit.v_ni[f] += v_gif * v_gif;
        }

        model_unit.mtx.unlock();
    }

}


void FmFtrlTrainer::test() {
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


