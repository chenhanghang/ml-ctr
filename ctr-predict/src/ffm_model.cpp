#include "ffm_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cmath>

#include "util.h"
#include "metric.h"

namespace ml {
namespace ctr {

const std::string FfmModel::_model_spliter = " ";

FfmModel::FfmModel() {
    if (this->_mode == "train") {
        return ;    
    }

}

void FfmModel::init() {
    if (this->_mode == "train") {
        return ;    
    }

}

void FfmModel::train(const std::string & train_path, const std::string & test_path) {
    if (this->_optimizer == "sgd") {
        this->train_SGD(train_path, test_path);
    } else if (this->_optimizer == "bgd"){
        this->train_BGD(train_path, test_path);
    } else if (this->_optimizer == "ftrl") {
        this->train_FTRL(train_path, test_path);
    }
}

//测试
void FfmModel::test(const std::string & test_path) {
    std::ifstream infile;
    infile.open(test_path.data());   //将文件流对象与文件连接起来
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
        double prob = this->predict(*it);
        probs.push_back(prob);
        labels.push_back(it->y);
        //std::cout<<prob<<std::endl;
    }
    double auc_value = Metric::auc(labels, probs);
    double log_loss_value = Metric::log_loss(labels, probs);
    double acuraccy_value = Metric::acuraccy(labels, probs, this->_threshold);
    std::cout<<"auc:"<<auc_value<<" log_loss:"<<log_loss_value<<" acuraccy:"<<acuraccy_value<<std::endl;
}

//predict 预测
double FfmModel::predict(const Sample & sample) {
    double z = 0.0;
    const std::vector<std::pair<std::string, double> > &x = sample.x;
    //计算线性项目
    double sum_w = 0.0;
    for (std::vector<int>::size_type i=0; i<x.size(); i++) {
        auto mIt = this->_model.find(x[i].first);
        if (mIt != this->_model.end()) {
            sum_w += mIt->second.w*x[i].second;
        }
    }
    double sum_v=0.0;
    for (std::vector<int>::size_type i=0; i<x.size(); i++) {
        auto mIt_vif = this->_model.find(x[i].first);
        if(mIt_vif == this->_model.end()) {
            continue;
        }
        for (std::vector<int>::size_type j=i+1; j<x.size(); j++) {
            auto mIt_vjf = this->_model.find(x[j].first);
            if(mIt_vjf == this->_model.end()) {
                continue;
            }
            FfmItem & ffm_item1 = this->_model.find(x[i].first)->second;
            FfmItem & ffm_item2 = this->_model.find(x[j].first)->second;
            //todo 此处应该获取对应的field， 因为样本暂时都为1
            double * vif = ffm_item1.v[0];
            double * vjf = ffm_item2.v[0];
            double dot_v=0;
            for (int k=0; k < this->K; k++) {
                dot_v += vif[k]*vjf[k]; 
            }
           sum_v += dot_v*x[i].second*x[j].second;
        }
    }
    z = this->b + sum_w + sum_v;
    return Util::sigmoid(z);
}

void FfmModel::train_SGD(const std::string & train_path, const std::string & test_path) {
    //开始迭代
    for (int i=0; i<this->_epoch; i++) {
        std::cout<<"epoch "<<i<<std::endl;
        std::ifstream infile;
        infile.open(train_path.data());
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        std::string s;
        while (getline(infile,s))  {
            Sample sample(s);//读取一个样本
            //std::cout<<s<<std::endl;
            double prob = sample.y==1 ? this->predict(sample):1-this->predict(sample);
            int y = sample.y;
            const std::vector<std::pair<std::string, double> > &x = sample.x;
           
            double sig_g = this->_alpha*y*(prob-1);
            //计算w梯度
            for (std::vector<int>::size_type i=0; i<x.size(); i++) {
                auto mIt = this->_model.find(x[i].first);
                if(mIt == this->_model.end()) {
                    this->_model[x[i].first] = FfmItem();
                    //this->_model.insert(std::pair<std::string, FfmItem>(x[i].first, FfmItem()));
                }
                this->_model[x[i].first].w -= sig_g*x[i].second;
                this->_model[x[i].first].w = this->_intersect(this->_model[x[i].first].w);
            }
            
            //计算vif 
            for (std::vector<int>::size_type i=0; i<x.size(); i++) {
                for (std::vector<int>::size_type j=i+1; j<x.size(); j++) {
                    auto mIt_vif = this->_model.find(x[i].first);
                    auto mIt_vjf = this->_model.find(x[j].first);
                    if(mIt_vif == this->_model.end() || mIt_vjf == this->_model.end()) {
                        continue;
                    }
                    FfmItem & ffm_item1 = this->_model.find(x[i].first)->second;
                    FfmItem & ffm_item2 = this->_model.find(x[j].first)->second;
                    //todo 此处应该根据样本x[1].first 获取field信息
                    double * vif = ffm_item1.v[0];
                    double * vjf = ffm_item2.v[0];
                    for (int k=0; k < this->K; k++) {
                        double temp_v1 = vif[k];
                        vif[k] -= sig_g*vjf[k]*x[i].second*x[j].second;
                        vif[k] = this->_intersect(vif[k]);

                        vjf[k] -= sig_g*vif[k]*x[i].second*x[j].second;
                        vjf[k] = this->_intersect(vjf[k]);
                    }
                }
            
            }
            //更新b即w0
            this->b -= this->_alpha*y*(prob-1);
        }
        infile.close(); //关闭文件输入流
        //每次训练计算测试集表现
        this->test(test_path);
    }

}

void FfmModel::train_BGD(const std::string & train_path, const std::string & test_path) {
    //开始迭代
    for (int i=0; i< this->_epoch; i++) {
        std::cout<<"epoch "<<i<<std::endl;
        std::ifstream infile;
        infile.open(train_path.data());
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        std::string s;
        std::vector<Sample> batch_samples;
        int batch_count;
        while (!infile.eof()) {
            batch_count = 0;
            batch_samples.clear();
            while (getline(infile,s))  {//读取batch个样本
                batch_samples.push_back(Sample(s));//读取一个样本
                if (++batch_count >= this->_batch)
                  break;
            }
            for (auto sample : batch_samples) {
                double prob = sample.y==1 ? this->predict(sample):1-this->predict(sample);
                int y = sample.y;
                const std::vector<std::pair<std::string, double> > &x = sample.x;
           
                double sig_g = this->_alpha*y*(prob-1);
                //计算w梯度
                for (std::vector<int>::size_type i=0; i<x.size(); i++) {
                    auto mIt = this->_model.find(x[i].first);
                    if(mIt == this->_model.end()) {
                        this->_model[x[i].first] = FfmItem();
                        //this->_model.insert(std::pair<std::string, FfmItem>(x[i].first, FfmItem()));
                    }
                    this->_model[x[i].first].w -= sig_g*x[i].second/this->_batch;
                    this->_model[x[i].first].w = this->_intersect(this->_model[x[i].first].w);
                }
            
                 //计算vif 
                for (std::vector<int>::size_type i=0; i<x.size(); i++) {
                    for (std::vector<int>::size_type j=j+1; j<x.size(); j++) {
                        auto mIt_vif = this->_model.find(x[i].first);
                        auto mIt_vjf = this->_model.find(x[j].first);
                        if(mIt_vif == this->_model.end() || mIt_vjf == this->_model.end()) {
                            continue;
                        }
                        FfmItem & ffm_item1 = this->_model.find(x[i].first)->second;
                        FfmItem & ffm_item2 = this->_model.find(x[j].first)->second;
                        //todo 此处应该根据样本x[1].first 获取field信息
                        double * vif = ffm_item1.v[0];
                        double * vjf = ffm_item2.v[0];
                        for (int k=0; k < this->K; k++) {
                            double temp_v1 = vif[k];
                            vif[k] -= sig_g*vjf[k]*x[i].second*x[j].second/this->_batch;
                            vif[k] = this->_intersect(vif[k]);

                            vjf[k] -= sig_g*vif[k]*x[i].second*x[j].second/this->_batch;
                            vjf[k] = this->_intersect(vjf[k]);
                        }
                    }
                
                }
                //更新b即w0
                this->b -= this->_alpha*y*(prob-1)/this->_batch;
            }
        }
        infile.close(); //关闭文件输入流
        //每次训练计算测试集表现
        this->test(test_path);
    }
}

void FfmModel::train_FTRL(const std::string & train_path, const std::string & test_path) {

}

//保存模型
void FfmModel::save() {

}

}
}


