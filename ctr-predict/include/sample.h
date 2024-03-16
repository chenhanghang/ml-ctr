#ifndef _CTR_PREDICT_SAMPLE_H_
#define _CTR_PREDICT_SAMPLE_H_

#include<iostream>
#include<vector>
#include<string>
#include<utility>

namespace ml {
namespace ctr {
class Sample {
    public:
        std::vector<std::pair<std::string, double> > x;
        int y;//y=-1,1
        bool flag = false;
        Sample(const std::string& line);
    private:
        static const std::string _spliter;
        static const std::string _inner_spliter;
};

}
}

#endif


