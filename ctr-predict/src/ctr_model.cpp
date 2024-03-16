#include "ctr_model.h"
#include "metric.h"
#include <iostream>

namespace ml {
namespace ctr {

CtrModel::CtrModel(const std::string & model_path, const std::string & mode) {
    this->_mode = mode;
    this->_model_path = model_path;
}
CtrModel::CtrModel() {
}

double CtrModel::_intersect(double x) {
    if (x > 10) return 10;
    else if (x < -10) return -10;
    return x;

}

CtrModel & CtrModel::set_alpha(double alpha) { 
    this->_alpha = alpha;
    return *this;
}

CtrModel & CtrModel::set_epoch(int epoch) { 
    this->_epoch=epoch;
    return *this;
}

CtrModel & CtrModel::set_batch(int batch) { 
    this->_batch=batch;
    return *this;
}

CtrModel & CtrModel::set_reg(double reg) { 
    this->_reg=reg;
    return *this;
}

CtrModel & CtrModel::set_mode(std::string mode) { 
    this->_mode=mode;
    return *this;
}

CtrModel & CtrModel::set_threshold(double threshold) {
    this->_threshold = threshold;
    return *this;
}

CtrModel & CtrModel::set_optimizer(std::string optimizer) {
    this->_optimizer = optimizer;
    return *this;
}


CtrModel & CtrModel::set_model_path(std::string model_path) {
    this->_model_path = model_path;
    return *this;
}

bool CtrModel::load_txt_model(const std::string& model_path) {
    return false;
}

bool CtrModel::load_bin_model(const std::string& model_path) {
    return false;
}

bool CtrModel::save_txt_model(const std::string& model_path) {
    return false;
}

bool CtrModel::save_bin_model(const std::string& model_path) {
    return false;
}


}
}

