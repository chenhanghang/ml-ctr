#include "pairwise_sample.h"

namespace ml {
namespace ctr {

const std::string PairwiseSample::_spliter = "\t";

PairwiseSample::PairwiseSample(const std::string& line) {
    if(line.size() == 0) {
        this->flag = false;
        return;
    }
    std::vector<std::string> tokens;
    Util::split(tokens, line, PairwiseSample::_spliter); 
    if(tokens.size() != 2) {
        this->flag = false;
        return;
    }

    this->left = new Sample(tokens[0]);
    this->right = new Sample(tokens[1]);
    if(this->left->flag == false || this->right->flag == false) {
        this->flag = false;
        return;
    }
    this->flag = true;
}


PairwiseSample::~PairwiseSample() {
    if(this->left != nullptr) {
        delete this->left;
    }
    if(this->right != nullptr) {
        delete this->right;
    }
}

}
}

