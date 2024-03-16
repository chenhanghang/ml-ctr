#ifndef _CTR_PREDICT_FMFTRLMODEL_H_
#define _CTR_PREDICT_FMFTRLMODEL_H_

#include<map>
#include<vector>
#include<string>
#include<unordered_map>

#include "src/sample/sample.h"
#include "src/utils/util.h"
using namespace std;

namespace ml {
namespace ctr {


class fm_ftrl_model_unit {
public:
    double wi;
    double w_ni;
    double w_zi;
    std::vector<double> vi;
    vector<double> v_ni;
    vector<double> v_zi;
    std::mutex mtx;
public:
    fm_ftrl_model_unit(int factor_num, double v_mean, double v_stdev) {
        wi = 0.0;
        w_ni = 0.0;
        w_zi = 0.0;
        this->vi.resize(factor_num);
        this->v_ni.resize(factor_num);
        this->v_zi.resize(factor_num);
        for(int f = 0; f < factor_num; ++f) {
            vi[f] = Util::gaussian(v_mean, v_stdev);
            v_ni[f] = 0.0;
            v_zi[f] = 0.0;
        }
    }
    fm_ftrl_model_unit(int factor_num, const std::vector<std::string>& model_line_seg) {
        this->vi.resize(factor_num);
        this->v_ni.resize(factor_num);
        this->v_zi.resize(factor_num);

        this->wi = stod(model_line_seg[1]);
        this->w_ni = stod(model_line_seg[2 + factor_num]);
        this->w_zi = stod(model_line_seg[3 + factor_num]);
        for(int f = 0; f < factor_num; ++f)
        {
            this->vi[f] = stod(model_line_seg[2 + f]);
            this->v_ni[f] = stod(model_line_seg[4 + factor_num + f]);
            this->v_zi[f] = stod(model_line_seg[4 + 2*factor_num + f]);
        }
    }
    void reinit_vi(double v_mean, double v_stdev) {
        int size = vi.size();
        for(int f = 0; f < size; ++f)
        {
            vi[f] = Util::gaussian(v_mean, v_stdev);
            v_ni[f] = 0.0;
            v_zi[f] = 0.0;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, fm_ftrl_model_unit& mu) {
        os << mu.wi;
        for(int f = 0; f < mu.vi.size(); ++f) {
            os << " " << mu.vi[f];
        }
        os << " " << mu.w_ni << " " << mu.w_zi;
        for(int f = 0; f < mu.v_ni.size(); ++f)
        {
            os << " " << mu.v_ni[f];
        }
        for(int f = 0; f < mu.v_zi.size(); ++f)
        {
            os << " " << mu.v_zi[f];
        }
        return os;
    }
};




class FmFtrlModel {
    public:
        FmFtrlModel(double _factor_num);
        FmFtrlModel(double _factor_num, double _mean, double _stdev);


        double predict(const Sample & sample);
        double predict(const Sample & sample, std::vector<fm_ftrl_model_unit*> theta, fm_ftrl_model_unit* model_unit_bias);
        void save(const std::string & model_path);
        void init(const std::string & model_path);
        fm_ftrl_model_unit * get_or_init_model_unit_bias();
        fm_ftrl_model_unit * get_or_init_model_unit(std::string key);
        int get_model_size();
    private:
        double get_wi(const std::string& index);
        double get_vif(const std::string& index, int f);

    public:
        int factor_num;
        double init_stdev;
        double init_mean;

    private:
        std::unordered_map<std::string, fm_ftrl_model_unit*> _model;
        static const std::string model_spliter;
        std::mutex mtx;//模型锁

};

}
}

#endif


