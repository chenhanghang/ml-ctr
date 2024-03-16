#ifndef _CTR_PREDICT_SAMPLE_H_
#define _CTR_PREDICT_SAMPLE_H_

#include<iostream>
#include<vector>
#include<string>
#include<utility>

namespace ml {
namespace ctr {

class SampleItem {
    public:
        std::string i;//index
        double v; //value
        double f; //filed
        SampleItem(std::string i, double f, double v){
            this->i = i;
            this->f = f;
            this->v = v;
        };

};

class Sample {
    public:
        //std::vector<std::pair<std::string, double> > x;
        std::vector<SampleItem*> xs;
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


