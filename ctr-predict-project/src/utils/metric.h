#ifndef _CTR_PREDICT_METRIC_H_
#define _CTR_PREDICT_METRIC_H_

#include<utility>
#include<vector>

namespace ml {
namespace ctr {
class Metric {
    private:
        static bool less(const std::pair<int, double>& s1,const std::pair<int, double> & s2);
    public:
        static double auc(const std::vector<int> & labels, const std::vector<double> & probs);
        static double log_loss(const std::vector<int> & labels, const std::vector<double> & probs);
        static double acuraccy(const std::vector<int> & labels, const std::vector<double> & probs, double threshold);
};

}
}

#endif


