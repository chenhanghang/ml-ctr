#ifndef _CTR_PREDICT_UTIL_H_
#define _CTR_PREDICT_UTIL_H_

#include<vector>
#include<string>
#include<math.h>

namespace es {
namespace model {

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

        static uint32_t hash(std::string& seed, bool method=false){
            uint32_t result = 0;
            for (auto ch : seed){
                if (!method) {
                    result += (ch - '0');
                } else {
                    result = (result * 131 + (ch - '0' + 1)) % 10007;
                }
             }
            return result;
        }
};

}
}

#endif


