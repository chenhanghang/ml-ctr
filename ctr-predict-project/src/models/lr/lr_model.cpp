#include "src/models/lr/lr_model.h"

#include <fstream>
#include <cassert>
#include <sstream>
#include <iostream>

#include "src/utils/util.h"
#include "src/utils/metric.h"

namespace ml {
namespace ctr {

const std::string LrModel::model_spliter = " ";


LrModel::LrModel() {

}

void LrModel::init(const std::string & model_path) {
    std::ifstream infile; 
    infile.open(model_path.data());   //将文件流对象与文件连接起来 
    if(!infile.is_open()) {
        return;
    }
    std::string s;
    while (getline(infile,s))  {
        std::vector<std::string> items;
        Util::split(items, s, LrModel::model_spliter);
        if (items.size() == 2 ) {
           this->_model.insert(std::pair<std::string, lr_model_unit*>(items[0], new lr_model_unit(std::stod(items[1])))); 
        }
    }
    infile.close(); //关闭文件输入流 

}


//predict 预测
double LrModel::predict(const Sample & sample) {
    const std::vector<SampleItem *> &xs = sample.xs;
    double z = 0.0;
    for (std::vector<SampleItem *>::const_iterator it = xs.cbegin(); it != xs.cend(); ++it) {
        auto mIt = this->_model.find((*it)->i);
        if (mIt != this->_model.end())
             z+= (*it)->v*(mIt->second->wi);
    }
    lr_model_unit * bias = this->get_or_init_model_unit_bias();
    z+=bias->wi;//b
    return Util::sigmoid(z);
}

//模型保存
void LrModel::save(const std::string & model_path) {
    std::ofstream out(model_path.c_str(), std::ofstream::out);
    //todo impl
    std::cout<<"save model to:"<<model_path<<std::endl;
    for(std::unordered_map<std::string, lr_model_unit*>::iterator iter = this->_model.begin(); iter != this->_model.end(); iter ++) {
         out << iter->first << LrModel::model_spliter << iter->second->wi << std::endl;
    }
    out.close();
}

lr_model_unit * LrModel::get_or_init_model_unit_bias() {
    return get_or_init_model_unit("0");
}

lr_model_unit * LrModel::get_or_init_model_unit(std::string index) {
    std::unordered_map<std::string, lr_model_unit*>::iterator iter = this->_model.find(index);
    if(iter == this->_model.end())
    {
        mtx.lock();
        lr_model_unit* p_mu = new lr_model_unit();
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


