#include "fm_pairwise_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cmath>

#include "util.h"
#include "metric.h"

namespace ml {
namespace ctr {

const std::string FmPairwiseModel::_model_spliter = " ";

FmPairwiseModel::FmPairwiseModel(const std::string & model_path, const std::string & mode):CtrModel(model_path, mode) {
    if (mode == "train") {
        return ;    
    }

}

void FmPairwiseModel::train(const std::string & train_path, const std::string & test_path) {
    if (this->_optimizer == "sgd") {
        this->train_SGD(train_path, test_path);
    } else if (this->_optimizer == "bgd"){
        this->train_BGD(train_path, test_path);
    } else if (this->_optimizer == "ftrl") {
        this->train_FTRL(train_path, test_path);
    }
}

//测试
void FmPairwiseModel::test(const std::string & test_path) {
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
double FmPairwiseModel::predict(const Sample & sample) {
    return Util::sigmoid(computePhiHat(sample));
}

//predict 预测
double FmPairwiseModel::computePhiHat(const Sample & sample) {
    double z = 0.0;
    const std::vector<std::pair<std::string, double> > &x = sample.x;
    //计算线性项目
    double sumW = 0.0;
    for (std::vector<int>::size_type i=0; i<x.size(); i++) {
        auto mIt = this->_model.find(x[i].first);
        if (mIt != this->_model.end()) {
            sumW += mIt->second.w*x[i].second;
        }
    }
    //交叉项
    double sum_sq=0.0, sq_sum=0.0;
    for (int f=0; f<this->K; f++) {
       double temp_sum_sq = 0.0;
       for (std::vector<int>::size_type  i=0; i<x.size(); i++) {
            auto mIt = this->_model.find(x[i].first);
            if (mIt != this->_model.end()) {
                temp_sum_sq += mIt->second.v[f]*x[i].second;
                sq_sum += std::pow(mIt->second.v[f], 2)*std::pow(x[i].second, 2);
           }
       }
       sum_sq += std::pow(temp_sum_sq, 2);
    }
    //求和并截断
    z = this->b + sumW + 0.5*(sum_sq - sq_sum);
    return z;
}

void FmPairwiseModel::update_per_sample(Sample & sample, double & lambda, int batch_size=1) {
    const std::vector<std::pair<std::string, double> > &x = sample.x;
    int y = sample.y;
    std::unique_ptr<double[]> sum_vf(new double[this->K]{0.0});
    //计算交叉项梯度求和
    for (std::vector<int>::size_type i=0; i<x.size(); i++) {
        for (int f=0; f<this->K; f++) {
            auto mIt = this->_model.find(x[i].first);
            if (mIt != this->_model.end()) {
                sum_vf[f] += mIt->second.v[f]*x[i].second; 
            }
        }
    }

    //更新w和v[f]
    for (std::vector<int>::size_type i=0; i<x.size(); i++) {
        auto mIt = this->_model.find(x[i].first);
        if (mIt == this->_model.end()) {
            this->_model[x[i].first] = FmItem();
        }
        double *vi = this->_model[x[i].first].v;
        for (int f=0; f < this->K; f++) {
            //更新vif
            vi[f] -= lambda/batch_size*this->_alpha*(y*(x[i].second*sum_vf[f] - vi[f]*std::pow(x[i].second, 2)) );
            vi[f] = this->_intersect(vi[f]);
        }
        //std::cout<<"&&&"<<x[i].first<<std::endl;
        //更新wi
        this->_model[x[i].first].w -= lambda/batch_size*this->_alpha*y*x[i].second;
        this->_model[x[i].first].w = this->_intersect(this->_model[x[i].first].w);
    } 
    //更新b即w0
    //this->b -= lambda*this->_alpha*y;
}

void FmPairwiseModel::train_SGD(const std::string & train_path, const std::string & test_path) {
    //开始迭代
    for (int i=0; i<this->_epoch; i++) {
        std::cout<<"epoch "<<i<<std::endl;
        std::ifstream infile;
        infile.open(train_path.data());
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        std::string s;
        std::unique_ptr<double[]> sum_vf(new double[this->K]{0.0});
        while (getline(infile,s))  {
            PairwiseSample pairwiseSample(s);//读取一个样本
            if(!pairwiseSample.flag) { continue; } //解析错误
            //std::cout<<s<<std::endl;
            double posPhiHat = this->computePhiHat(*(pairwiseSample.left));
            double negPhiHat = this->computePhiHat(*(pairwiseSample.right));
            double phiHat = posPhiHat - negPhiHat;
            double lambda =  -1.0 / (1.0 + exp(phiHat));
            this->update_per_sample(*(pairwiseSample.left), lambda);         
            this->update_per_sample(*(pairwiseSample.right), lambda);         
        }
        infile.close(); //关闭文件输入流
        //每次训练计算测试集表现
        this->test(test_path);
    }
}

void FmPairwiseModel::train_BGD(const std::string & train_path, const std::string & test_path) {
    //开始迭代
    for (int i=0; i< this->_epoch; i++) {
        std::cout<<"epoch "<<i<<std::endl;
        std::ifstream infile;
        infile.open(train_path.data());
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        std::string s;
        std::vector<PairwiseSample> batch_samples;
        int batch_count;
        while (!infile.eof()) {
            batch_count = 0;
            batch_samples.clear();
            while (getline(infile,s))  {//读取batch个样本
                batch_samples.push_back(PairwiseSample(s));//读取一个样本
                if (++batch_count >= this->_batch)
                  break;
            }
            for (auto pairwiseSample : batch_samples) {
                if(!pairwiseSample.flag) { continue; } //解析错误
                double posPhiHat = this->computePhiHat(*(pairwiseSample.left));
                double negPhiHat = this->computePhiHat(*(pairwiseSample.right));
                double phiHat = posPhiHat - negPhiHat;
                double lambda =  -1.0 / (1.0 + exp(phiHat));
                this->update_per_sample(*(pairwiseSample.left), lambda, batch_count);         
                this->update_per_sample(*(pairwiseSample.right), lambda, batch_count);         
            }
        }
        infile.close(); //关闭文件输入流
        //每次训练计算测试集表现
        this->test(test_path);
    }
}

void FmPairwiseModel::train_FTRL(const std::string & train_path, const std::string & test_path) {

}

//保存模型
void FmPairwiseModel::save() {

}

}
}

