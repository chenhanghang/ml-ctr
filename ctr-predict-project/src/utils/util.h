#ifndef _CTR_PREDICT_UTIL_H_
#define _CTR_PREDICT_UTIL_H_

#include<vector>
#include<string>

namespace ml {
namespace ctr {

class Util {
    public:
        Util() {}
    
        static void split(std::vector<std::string> &result, std::string str,std::string pattern);//字符串分割函数
        static double sigmoid(double z);
        double static uniform();
        double static gaussian();
        double static gaussian(double mean, double stdev);
        static int sgn(double x);

};

}
}

#endif


