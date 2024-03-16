#ifndef _CTR_PREDICT_FMFMODEL_H_
#define _CTR_PREDICT_FMFMODEL_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/utils/util.h"
using namespace std;

namespace ml {
namespace ctr {


class ffm_model_unit {
public:
    double wi;
    std::vector<std::vector<double>> vif;
    std::mutex mtx;
public:
    ffm_model_unit(int field_num, int factor_num, double v_mean, double v_stdev) {
        this->wi = 0.0;
        this->vif.resize(field_num);
        for(int fd = 0; fd < field_num; fd++) {
            std::vector<double> vi;
            vi.resize(factor_num);
            for(int f = 0; f < factor_num; ++f) {
                vi[f] = Util::gaussian(v_mean, v_stdev);
            }
            vif[fd] = vi;
        }
    }
    ffm_model_unit(int field_num, int factor_num, const std::vector<std::string>& model_line_seg) {
        this->wi = stod(model_line_seg[1]);
        this->vif.resize(field_num);
        for(int fd = 0; fd < field_num; fd++) {
            std::vector<double> vi;
            vi.resize(factor_num);
            for(int f = 0; f < factor_num; ++f) {
                vi[f] = stod(model_line_seg[2 + f]);
            }
            vif[fd] = vi;
        }
    }
    void reinit_vi(double v_mean, double v_stdev) {
        int field_num = vif.size();
        for(int fd = 0; fd < field_num; fd++) {
            int factor_num = vif[fd].size();
            for(int f = 0; f < factor_num; ++f) {
                vif[fd][f] = Util::gaussian(v_mean, v_stdev);
            }
        }
    }
    //friend inline std::ostream& operator <<(std::ostream& os, ffm_model_unit& mu) {
    friend std::ostream& operator<<(std::ostream& os, ffm_model_unit& mu) {
        os << mu.wi;
        for(int fd = 0; fd < mu.vif.size(); fd++) {
            for(int f = 0; f < mu.vif[fd].size(); ++f) {
                os << " " << mu.vif[fd][f];
            }
        }
        return os;
    }
};




class FfmModel {
    public:
        FfmModel(int field_num, int factor_num);
        FfmModel(int field_num, int factor_num, double mean, double stdev);


        double predict(const Sample & sample);
        double predict(const Sample & sample, std::vector<ffm_model_unit*> theta, ffm_model_unit* model_unit_bias);
        void save(const std::string & model_path);
        void init(const std::string & model_path);
        ffm_model_unit * get_or_init_model_unit_bias();
        ffm_model_unit * get_or_init_model_unit(std::string key);
    private:
        double get_wi(const std::string& index);
        double get_vif(const std::string& index, int field, int factor);

    public:
        int field_num;
        int factor_num;
        double init_stdev;
        double init_mean;

    private:
        std::unordered_map<std::string, ffm_model_unit*> _model;
        static const std::string model_spliter;
        std::mutex mtx;//模型锁

};

}
}

#endif


