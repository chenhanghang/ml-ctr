#include "sample.h"

namespace es {
namespace model {

//分割符
const std::string Sample::_spliter = "\t";
const std::string Sample::_feature_spliter = " ";
const std::string Sample::_feature_inner_spliter = " ";


Sample::Sample(const std::string& line) {
    if(line.size() == 0) {
        this->flag = false;
        return;
    }
    std::vector<std::string> tokens;
    //0:exp_name 1: reward 2:sigma 3:features
    Util::split(tokens, line, PairwiseSample::_spliter);
    if(tokens.size() != 4) {
        this->flag = false;
        return;
    }
    this->exp_name = tokens[0]
    this->reward = std::atof(tokens[1].c_str());
    this->sigma = std::atof(tokens[2].c_str());
    std::vector<std::string> feature_items;
    Util::split(feature_items, tokens[3], PairwiseSample::_feature_spliter);
    if(feature_items.size() == 0) {
        this->flag = false;
        return;
    }
    std::vector<std::string> one_feature;
    for(auto feature : feature_items) {
        Util::split(one_feature, feature, PairwiseSample::_feature_inner_spliter);
        if(one_feature.size() != 3){
            this->flag = false;
            return;
        }
        Feature feature;
        feature.index = std::atoi(one_feature[0].c_str());
        feature.value = std::atof(one_feature[1].c_str());
        feature.noise = std::atof(one_feature[2].c_str());
        this->features.push_back(feature);
    }
    if(this->features.size() == 0) {
        this->flag = false;
        return;
    }
    this->flag = true;
}

}
}
