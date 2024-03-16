#ifndef _CTR_PREDICT_PAIR_SAMPLE_H_
#define _CTR_PREDICT_PAIR_SAMPLE_H_

#include<iostream>
#include<vector>
#include<string>
#include<utility>
#include "sample.h"
#include "util.h"

namespace ml {
namespace ctr {

/**
 * 保证left 永远是正样本，right是负样本
 */
class PairwiseSample {
    public:
        Sample *left;
        Sample *right;
        bool flag = false;
        PairwiseSample(const std::string& line);
        ~PairwiseSample();
    private:
        static const std::string _spliter;
};

}
}

#endif


