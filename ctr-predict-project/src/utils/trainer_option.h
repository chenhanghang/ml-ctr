#ifndef _CTR_PREDICT_TRAINER_OPTION_H_
#define _CTR_PREDICT_TRAINER_OPTION_H_
namespace ml {
namespace ctr {

struct TrainerOption {
    TrainerOption():init_mean(0.0), init_stdev(0.1),algorithm("lr"),
        model_path("./data/lr.model"),mode("test"),train_path("./data/train.dat"),test_path("./data/test.dat"),
        batch(100),epoch(30),alpha(0.03),reg(0.001),optimizer("bgd"),threshold(0.8),threads_num(1){}
    double init_mean, init_stdev;
    std::string algorithm;
    std::string model_path;
    std::string mode;
    std::string train_path;
    std::string test_path;
    int batch;
    int epoch;
    double alpha;
    double reg;
    std::string optimizer;
    double threshold;
    int threads_num;
    int log_num;
    int buffer_size;

    //fm
    int factor;
    int field;
    //ftrl
    double w_alpha=0.05;
    double w_beta=1.0;
    double w_lambda1=0.1;
    double w_lambda2=5.0;
    double v_alpha=0.05;
    double v_beta=1.0;
    double v_lambda1=0.1;
    double v_lambda2=5.0;

};

}
}

#endif
