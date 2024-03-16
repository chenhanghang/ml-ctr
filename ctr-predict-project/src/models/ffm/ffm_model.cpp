#include "src/models/ffm/ffm_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

const std::string FfmModel::model_spliter = " ";


FfmModel::FfmModel(int field_num, int factor_num) {
    this->field_num = field_num;
    this->factor_num = factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;

}

FfmModel::FfmModel(int field_num, int factor_num, double mean, double stdev) {
    this->field_num = field_num;
    this->factor_num = factor_num;
    this->init_mean = mean;
    this->init_stdev = stdev;

}

double FfmModel::get_wi(const std::string& index) {
    std::unordered_map<std::string, ffm_model_unit*>::iterator iter = this->_model.find(index);
    if(iter == this->_model.end()) {
        return 0.0;
    }
    else {
        return iter->second->wi;
    }
}

double FfmModel::get_vif(const std::string& index, int field, int factor) {
    std::unordered_map<std::string, ffm_model_unit*>::iterator iter = this->_model.find(index);
    if(iter == this->_model.end()) {
        return 0.0;
    } else {
        return iter->second->vif[field][factor];
    }

}

void FfmModel::init(const std::string & model_path) {
    std::ifstream infile; 
    infile.open(model_path.data());   //将文件流对象与文件连接起来 
    if(!infile.is_open()) {
        return;
    }
    std::string s;
    while (getline(infile,s))  {
        std::vector<std::string> items;
        Util::split(items, s, FfmModel::model_spliter);
        if (items.size() == factor_num+2 ) {
            ffm_model_unit* p_mu = new ffm_model_unit(this->field_num, this->factor_num, items);
            this->_model.insert(std::pair<std::string, ffm_model_unit*>(items[0], p_mu)); 
        }
    }
    infile.close(); //关闭文件输入流 

}


//predict 预测
double FfmModel::predict(const Sample & sample) {
    const std::vector<SampleItem *> &x = sample.xs;
    double result = 0.0;
    for (std::vector<SampleItem *>::const_iterator iter = x.cbegin(); iter != x.cend(); ++iter) {
        const std::string& index = (*iter)->i;
        result += this->get_wi(index) * (*iter)->v;
    }
    ffm_model_unit * bias = this->get_or_init_model_unit_bias();
    result += bias->wi;//b
    for(int k = 0; k < this->factor_num; ++k) {
        double sum=0.0, sum_sqr=0.0;
        for(std::vector<SampleItem *>::const_iterator iter = x.cbegin(); iter != x.cend(); ++iter) {
            const std::string& index = (*iter)->i;
            //todo 此处应该获取对应的field， 因为样本暂时都为0
            double d = this->get_vif(index, (*iter)->f, k) * (*iter)->v;
            sum += d;
            sum_sqr += d * d;
        }
        result += 0.5 * (sum * sum - sum_sqr);
    }
    return Util::sigmoid(result);
}

double FfmModel::predict(const Sample & sample, std::vector<ffm_model_unit*> theta, ffm_model_unit* model_unit_bias) {
    const std::vector<SampleItem *> &x = sample.xs;
    double result = 0.0;
    result += model_unit_bias->wi;
    for(int i=0; i< x.size(); i++){
        result += theta[i]->wi * x[i]->v;
    }
    for(int f = 0; f < this->factor_num; ++f) {
        double sum=0.0, sum_sqr=0.0;
        for(int i = 0; i < x.size(); ++i) {
            double d = theta[i]->vif[0][f] * x[i]->v;
            sum += d;
            sum_sqr += d * d;
        }
        result += 0.5 * (sum * sum - sum_sqr);
    }
    return Util::sigmoid(result);

}

//模型保存
void FfmModel::save(const std::string & model_path) {
    std::ofstream out(model_path.c_str(), std::ofstream::out);
    //todo impl
    std::cout<<"save model to:"<<model_path<<" "<<this->_model.size()<<std::endl;
    for(std::unordered_map<std::string, ffm_model_unit*>::iterator iter = this->_model.begin(); iter != this->_model.end(); iter ++) {
         out << iter->first << FfmModel::model_spliter << *(iter->second) << std::endl;
    }
    out.close();
}

ffm_model_unit * FfmModel::get_or_init_model_unit_bias() {
    return get_or_init_model_unit("0");
}

ffm_model_unit * FfmModel::get_or_init_model_unit(std::string index) {
    std::unordered_map<std::string, ffm_model_unit*>::iterator iter = this->_model.find(index);
    if(iter == this->_model.end())
    {
        mtx.lock();
        ffm_model_unit* p_mu = new ffm_model_unit(field_num, factor_num, 0.0, 1.0);
        this->_model.insert(make_pair(index, p_mu));
        mtx.unlock();
        return p_mu;
    }
    else
    {
        return iter->second;
    }
}


}
}


