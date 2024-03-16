#ifndef _CTR_PREDICT_FMMODEL_H_
#define _CTR_PREDICT_FMMODEL_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/utils/util.h"
using namespace std;

namespace ml {
namespace ctr {


class fm_pairwise_model_unit {
public:
    double wi;
    std::vector<double> vi;
    std::mutex mtx;
public:
    fm_pairwise_model_unit(int factor_num, double v_mean, double v_stdev) {
        wi = 0.0;
        this->vi.resize(factor_num);
        for(int f = 0; f < factor_num; ++f) {
            vi[f] = Util::gaussian(v_mean, v_stdev);
        }
    }
    fm_pairwise_model_unit(int factor_num, const std::vector<std::string>& model_line_seg) {
        this->vi.resize(factor_num);
        this->wi = stod(model_line_seg[1]);
        for(int f = 0; f < factor_num; ++f)
        {
            this->vi[f] = stod(model_line_seg[2 + f]);
        }
    }
    void reinit_vi(double v_mean, double v_stdev) {
        int size = vi.size();
        for(int f = 0; f < size; ++f)
        {
            vi[f] = Util::gaussian(v_mean, v_stdev);
        }
    }
    //friend inline std::ostream& operator <<(std::ostream& os, fm_pairwise_model_unit& mu) {
    friend std::ostream& operator<<(std::ostream& os, fm_pairwise_model_unit& mu) {
        os << mu.wi;
        for(int f = 0; f < mu.vi.size(); ++f) {
            os << " " << mu.vi[f];
        }
        return os;
    }
};




class FmPairwiseModel {
    public:
        FmPairwiseModel(double _factor_num);
        FmPairwiseModel(double _factor_num, double _mean, double _stdev);


        double predict(const Sample & sample);
        double predict(const Sample & sample, std::vector<fm_pairwise_model_unit*> theta, fm_pairwise_model_unit* model_unit_bias);
        void save(const std::string & model_path);
        void init(const std::string & model_path);
        fm_pairwise_model_unit * get_or_init_model_unit_bias();
        fm_pairwise_model_unit * get_or_init_model_unit(std::string key);
    private:
        double get_wi(const std::string& index);
        double get_vif(const std::string& index, int f);

    public:
        int factor_num;
        double init_stdev;
        double init_mean;

    private:
        std::unordered_map<std::string, fm_pairwise_model_unit*> _model;
        static const std::string model_spliter;
        std::mutex mtx;//模型锁

};

}
}

#endif


