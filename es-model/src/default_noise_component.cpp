#include "default_noise_component.h"

namespace es {
namespace model {

DefaultNoiseComponent::DefaultNoiseComponent(float sigma, std::string seed_str) {
    this->sigma = sigma;
    time_t t = time(NULL);
    struct tm *temp_tm = localtime(&t);
    int year = temp_tm->tm_year + 1900;
    int month = temp_tm->tm_mon + 1;
    int day = temp_tm->tm_mday;
    int hour = temp_tm->tm_hour;
    uint64_t curr_time = year * 1000000 + month * 10000 + day * 100 + hour;

    uint64_t seed = curr_time + hash_str;
    this->generator.seed(this->seed);
}

bool DefaultNoiseComponent::generateNoise(std::vector<float> noises, int feature_size, string hash_str) {
    float number = 0.0;
    float tmp_max = 1000000.0f;
    for (int i = 0; i < feature_size; ++i) {
        number = this->norm(this->generator);
        number = int(number * 1000000) / tmp_max;
        noises.push_back(number*sigma);
    }
    return true;
}


}
}
