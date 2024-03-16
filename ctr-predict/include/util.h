#ifndef _CTR_PREDICT_UTIL_H_
#define _CTR_PREDICT_UTIL_H_

#include<vector>
#include<string>
#include<math.h>

namespace ml {
namespace ctr {

class Util {
    public:
        Util() {}
        ~Util() {}
    
        //字符串分割函数
        static void split(std::vector<std::string> &result, std::string str,std::string pattern) {
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

        static double sigmoid(double z) {
            return 1.0/(1.0+exp(-z));
        }
};

}
}

#endif


