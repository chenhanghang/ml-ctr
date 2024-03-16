#include <stdlib.h>
#include <cmath>

#include "src/utils/util.h"

namespace ml {
namespace ctr {

void Util::split(std::vector<std::string> &result, std::string str,std::string pattern) {
    std::string::size_type pos;
    str+=pattern;//扩展字符串以方便操作
    int size=str.size();

    for(int i=0; i<size; i++) {
        pos=str.find(pattern,i);
        if(pos<size) {
            std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
}

double Util::sigmoid(double z) {
    return 1.0/(1.0+exp(-z));
}


double Util::uniform()
{
    return rand()/((double)RAND_MAX + 1.0);
}


double Util::gaussian()
{
    double u,v, x, y, Q;
    do
    {
        do
        {
            u = uniform();
        } while (u == 0.0);

        v = 1.7156 * (uniform() - 0.5);
        x = u - 0.449871;
        y = fabs(v) + 0.386595;
        Q = x * x + y * (0.19600 * y - 0.25472 * x);
    } while (Q >= 0.27597 && (Q > 0.27846 || v * v > -4.0 * u * u * log(u)));
    return v / u;
}

double Util::gaussian(double mean, double stdev) {
    if(0.0 == stdev)
    {
        return mean;
    }
    else
    {
        return mean + stdev * gaussian();
    }
}

int Util::sgn(double x) {
    if(x > 0.0000001) return 1;
    else return -1;
}

}
}
