#ifndef _DEFAULT_NOISE_COMPONENT_H_
#define _DEFAULT_NOISE_COMPONENT_H_

#include <time.h>
#include<vector>
#include<string>

namespace es {
namespace model {

/**
 *撒点
 */
class DefaultNoiseComponent {
    public:
        DefaultNoiseComponent(float sigma, std::string seed_str);
        bool generateNoise(std::vector<float> noises, int feature_size);
    protected:
        float sigma;
        std::default_random_engine generator;
        std::normal_distribution<double> norm(0, 1);


};

}
}



#endif
