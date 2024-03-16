#include<iostream>
#include<gflags/gflags.h>

#include "src/frame/frame.h"
#include "src/frame/frame_file.h"
#include "src/models/lr/lr_trainer.h"
#include "src/models/fm/fm_trainer.h"
#include "src/models/ffm/ffm_trainer.h"
#include "src/models/fm_ftrl/fm_ftrl_trainer.h"
#include "src/utils/trainer_option.h"

using namespace ml::ctr;
//http://dreamrunner.org/blog/2014/03/09/gflags-jian-ming-shi-yong/
DEFINE_string(algorithm, "lr", "algorithm for predict. lr fm ffm ..");
DEFINE_string(model_path, "./model/lr.model", "model save path");
DEFINE_string(mode, "test", "mode train or test");
DEFINE_string(train_path, "./data/agaricus.txt.train", "mode train or test");
DEFINE_string(test_path, "./data/agaricus.txt.test", "mode train or test");
DEFINE_int32(threads_num, 10, "thread to train");

DEFINE_int32(batch, 100, "batch size to train.");
DEFINE_int32(epoch, 30, "train epoch.");
DEFINE_double(alpha, 0.03, "train rate.");
DEFINE_double(reg, 0.001, "L2 regularization rate.");
DEFINE_string(optimizer, "bgd", "optimizer method sgd or bgd.");
DEFINE_double(threshold, 0.8, "threshold for calculate accuracy.");
DEFINE_int32(log_num, 2000, "log size to train.");
DEFINE_int32(buffer_size, 2000, "size per thread to buffer.");

//fm related
DEFINE_int32(factor, 6, "fm factor num");
DEFINE_int32(field, 1, "ffm field num");


TrainerOption get_opt() {
    TrainerOption  opt;
    opt.algorithm = FLAGS_algorithm;
    opt.mode = FLAGS_mode;

    opt.model_path = FLAGS_model_path;
    opt.train_path = FLAGS_train_path;
    opt.test_path = FLAGS_test_path;

    opt.threads_num = FLAGS_threads_num;
    opt.log_num = FLAGS_log_num;
    opt.buffer_size = FLAGS_buffer_size;
    opt.batch = FLAGS_batch;
    opt.epoch = FLAGS_epoch;
    opt.alpha = FLAGS_alpha;
    opt.reg = FLAGS_reg;
    opt.optimizer = FLAGS_optimizer;
    opt.threshold = FLAGS_threshold;

    opt.factor = FLAGS_factor;
    opt.field = FLAGS_field;
    return opt;
} 


int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    TrainerOption  opt = get_opt();
    BaseTrainer *trainer = nullptr;
    if(opt.algorithm == "lr") {
        trainer = new LrTrainer(&opt);
    } else if(opt.algorithm == "fm") {
        trainer = new FmTrainer(&opt); 
    } else if(opt.algorithm == "fm_ftrl") {
        trainer = new FmFtrlTrainer(&opt); 
    } else if(opt.algorithm == "ffm") { 
        trainer = new FfmTrainer(&opt); 
    } else {
        std::cout<<"choose train algorithm....";
        return 0;
    }
    //LrTrainer trainer(&opt);
    //FmTrainer trainer(&opt);
    //FmFtrlTrainer trainer(&opt);
    trainer->input_model(FLAGS_model_path);

    FrameFile frame;
    frame.init(&(*trainer), &opt);
    frame.run();

    //训练完后测试
    std::cout<<"test model...";
    trainer->test();
    trainer->output_model(FLAGS_model_path);

    return 0;
}


