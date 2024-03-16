#include "src/models/fm_ftrl/fm_ftrl_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

const std::string FmFtrlModel::model_spliter = " ";


FmFtrlModel::FmFtrlModel(double _factor_num) {
    factor_num = _factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;

}

FmFtrlModel::FmFtrlModel(double _factor_num, double _mean, double _stdev) {
    factor_num = _factor_num;
    init_mean = _mean;
    init_stdev = _stdev;

}

double FmFtrlModel::get_wi(const std::string& index) {
    std::unordered_map<std::string, fm_ftrl_model_unit*>::iterator iter = this->_model.find(index);
    if(iter == this->_model.end()) {
        return 0.0;
    }
    else {
        return iter->second->wi;
    }
}

double FmFtrlModel::get_vif(const std::string& index, int f) {
    //????to do check why
    mtx.lock();
    std::unordered_map<std::string, fm_ftrl_model_unit*>::iterator iter = this->_model.find(index);
    mtx.unlock();
    if(iter == this->_model.end()) {
        return 0.0;
    } else {
        return iter->second->vi[f];
    }

}

void FmFtrlModel::init(const std::string & model_path) {
    std::ifstream infile; 
    infile.open(model_path.data());   //将文件流对象与文件连接起来 
    if(!infile.is_open()) {
        return;
    }
    std::string s;
    while (getline(infile,s))  {
        std::vector<std::string> items;
        Util::split(items, s, FmFtrlModel::model_spliter);
        if (items.size() == factor_num+2 ) {
            fm_ftrl_model_unit* p_mu = new fm_ftrl_model_unit(this->factor_num, items);
            this->_model.insert(std::pair<std::string, fm_ftrl_model_unit*>(items[0], p_mu)); 
        }
    }
    infile.close(); //关闭文件输入流 

}


//predict 预测
double FmFtrlModel::predict(const Sample & sample) {
    const std::vector<SampleItem *> &x = sample.xs;
    double result = 0.0;
    for (std::vector<SampleItem * >::const_iterator iter = x.cbegin(); iter != x.cend(); ++iter) {
        result += this->get_wi((*iter)->i) * (*iter)->v;
    }
    fm_ftrl_model_unit * bias = this->get_or_init_model_unit_bias();
    result += bias->wi;//b
    for(int f = 0; f < this->factor_num; ++f) {
        double sum=0.0, sum_sqr=0.0;
        for(int i = 0; i < x.size(); ++i) {
            const std::string& index = x[i]->i;
            double d = this->get_vif(index, f) * x[i]->v;
            sum += d;
            sum_sqr += d * d;
        }
        result += 0.5 * (sum * sum - sum_sqr);
    }
    return Util::sigmoid(result);
}

double FmFtrlModel::predict(const Sample & sample, std::vector<fm_ftrl_model_unit*> theta, fm_ftrl_model_unit* model_unit_bias) {
    const std::vector<SampleItem *> &x = sample.xs;
    double result = 0.0;
    result += model_unit_bias->wi;
    for(int i=0; i< x.size(); i++){
        result += theta[i]->wi * x[i]->v;
    }
    for(int f = 0; f < this->factor_num; ++f) {
        double sum=0.0, sum_sqr=0.0;
        for(int i = 0; i < x.size(); ++i) {
            double d = theta[i]->vi[f] * x[i]->v;
            sum += d;
            sum_sqr += d * d;
        }
        result += 0.5 * (sum * sum - sum_sqr);
    }
    return Util::sigmoid(result);

}

//模型保存
void FmFtrlModel::save(const std::string & model_path) {
    std::ofstream out(model_path.c_str(), std::ofstream::out);
    //todo impl
    std::cout<<"save model to:"<<model_path<<" "<<this->_model.size()<<std::endl;
    for(std::unordered_map<std::string, fm_ftrl_model_unit*>::iterator iter = this->_model.begin(); iter != this->_model.end(); iter ++) {
         out << iter->first << FmFtrlModel::model_spliter << *(iter->second) << std::endl;
    }
    out.close();
}

fm_ftrl_model_unit * FmFtrlModel::get_or_init_model_unit_bias() {
    return get_or_init_model_unit("0");
}


int FmFtrlModel::get_model_size() {
    return this->_model.size();
}
fm_ftrl_model_unit * FmFtrlModel::get_or_init_model_unit(std::string index) {
    std::unordered_map<std::string, fm_ftrl_model_unit*>::iterator iter = this->_model.find(index);
    if(iter == this->_model.end())
    {
        mtx.lock();
        fm_ftrl_model_unit* p_mu = nullptr;
        iter = this->_model.find(index);
        if(iter == this->_model.end()) {
            p_mu = new fm_ftrl_model_unit(factor_num, 0.0, 0.1);
            this->_model.insert(make_pair(index, p_mu));
        } else {
            p_mu = iter->second;
        }
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


