#include "lr_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "util.h"
#include "metric.h"

namespace ml {
namespace ctr {

const std::string LrModel::model_spliter = " ";


LrModel::LrModel() {

}

void LrModel::init() {
    if (this->_mode == "train") {
        return ;    
    }
    std::ifstream infile; 
    infile.open(this->_model_path.data());   //将文件流对象与文件连接起来 
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
    std::string s;
    while (getline(infile,s))  {
        std::vector<std::string> items;
        Util::split(items, s, LrModel::model_spliter);
        if (items.size() == 2 ) {
           this->_model.insert(std::pair<std::string, double>(items[0], std::stod(items[1]))); 
        }
    }
    infile.close(); //关闭文件输入流 
    this->_size = this->_model.size();

}


void LrModel::train(const std::string & train_path, const std::string & test_path) {
    if (this->_optimizer == "sgd") {
        this->train_SGD(train_path, test_path);
    } else if (this->_optimizer == "bgd"){
        this->train_BGD(train_path, test_path);
    }
}

//训练
void LrModel::train_SGD(const std::string & train_path, const std::string & test_path) {
    //开始迭代
    for (int i=0; i<this->_epoch; i++) {
        std::cout<<"epoch "<<i<<std::endl;
        std::ifstream infile;
        infile.open(train_path.data());
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        std::string s;
        while (getline(infile,s))  {
            Sample sample(s);//读取一个样本
            double prob = sample.y==1?this->predict(sample): 1-this->predict(sample);
            int y = sample.y;
            for (auto begin = sample.x.begin(); begin != sample.x.end(); begin++) {
                std::pair<std::string, double> xi = *begin; 
                double delta = y*(prob - 1)*xi.second;
                auto it = this->_model.find(xi.first);
                if (it ==  this->_model.end()) {
                    this->_model[xi.first] =  -this->_alpha*delta;
                } else {
                    this->_model[xi.first] -= this->_alpha*delta;
                }
                this->_model[xi.first] = this->_intersect(this->_model[xi.first]);
            }
            this->_model["0"] -= this->_alpha*y*(prob - 1);
        }
        infile.close(); //关闭文件输入流 
        //每次训练计算测试集表现
        this->test(test_path);
        
    }
    //auto iter = this->_model.begin();
    //while (iter != this->_model.end()) {
        //std::cout<<iter->first<<" "<<iter->second<<std::endl;
   //     iter++;
   // }
}

void LrModel::train_BGD(const std::string & train_path, const std::string & test_path) {
    //开始迭代
    for (int i=0; i<this->_epoch; i++) {
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
                double prob = sample.y==1 ? this->predict(sample): 1-this->predict(sample);
                int y = sample.y;
                for (auto begin = sample.x.begin(); begin != sample.x.end(); begin++) {
                    std::pair<std::string, double> xi = *begin; 
                    double delta = y*(prob - 1)*xi.second;
                    auto it = this->_model.find(xi.first);
                    if (it ==  this->_model.end()) {
                        this->_model[xi.first] =  -this->_alpha*delta/this->_batch - this->_model[xi.first]*this->_alpha*this->_reg/this->_batch;
                    } else {
                        this->_model[xi.first] -= this->_alpha*delta/this->_batch - this->_model[xi.first]*this->_alpha*this->_reg/this->_batch;
                    }
                    this->_model[xi.first] = this->_intersect(this->_model[xi.first]);
                 }
                this->_model["0"] -= this->_alpha*y*(prob - 1)/this->_batch;
            }
        }
        infile.close(); //关闭文件输入流 
        //每次训练计算测试集表现
        this->test(test_path);
        
    }
    auto iter = this->_model.begin();
    while (iter != this->_model.end()) {
        //std::cout<<iter->first<<" "<<iter->second<<std::endl;
        iter++;
    }
}

//测试
void LrModel::test(const std::string & test_path) {
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
double LrModel::predict(const Sample & sample) {
    const std::vector<std::pair<std::string, double> > &x = sample.x;
    double z = 0.0;
    for (std::vector<std::pair<std::string, double> >::const_iterator it = x.cbegin(); it != x.cend(); ++it) {
        auto mIt = this->_model.find(it->first);
        if (mIt != this->_model.end())
             z+= it->second*mIt->second;
    }
    z+=this->_model["0"];//b
    return Util::sigmoid(z);
}

//模型保存
void LrModel::save() {
    std::ofstream out(this->_model_path.c_str(), std::ofstream::out);
    //todo impl
    std::cout<<"save model to:"<<this->_model_path<<std::endl;
    for(std::unordered_map<std::string, double>::iterator iter = this->_model.begin(); iter != this->_model.end(); iter ++) {
         out << iter->first << LrModel::model_spliter << iter->second << std::endl;
    }
    out.close();
}

}
}


