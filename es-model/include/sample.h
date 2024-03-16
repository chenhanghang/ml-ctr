#ifndef _SAMPLE_H_
#define _SAMPLE_H_

#include<vector>
#include<iostream>
#include<string>

#include "util.h"

namespace es {
namespace model {

/**
 * exp_name reward sigma feature_index:feature_value:noise
 *
 */

struct Feature {
    int index;
    double value;
    double noise;
};

class Sample {
    public:
        std::vector<Feature> features;
        double reward;//收益
        double sigma;//撒点使用的sigma
        bool flag = false;
        std::string exp_name;
        Sample(const std::string& line);
    private:
        static const std::string _spliter;
        static const std::string _feature_spliter;
};


}
}

#endif

