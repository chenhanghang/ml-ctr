#include "src/frame/frame_file.h"
#include <fstream>
#include <unistd.h>

namespace ml {
namespace ctr {

bool FrameFile::init(Task* task, TrainerOption * opt) {
    
    this->_train_path = opt->train_path;
    return this->Frame::init(task, opt);
}


void FrameFile::pro_thread() {
    std::string line;
    int line_num = 0;
    int i = 0;
    finished_flag = false;
    std::vector<Sample> sample_grp;
    std::ifstream infile;
    infile.open(this->_train_path.data());
    assert(infile.is_open());
    while(true) {
        std::unique_lock<std::mutex> lock(buf_mtx);
        _cv_queue_not_full.wait(lock, [&] { return buffer.size() < 1; });
        //_cv_queue_not_full.wait(lock);
        std::cout<<"+++++++++++production"<<std::endl;
        for(i = 0; i < this->_buf_size;i++) {
            if(!std::getline(infile, line)) {
                finished_flag = true;
                break;
            }
            line_num++;
            Sample sample(line);
            if(sample_grp.size() > 64) {
                buffer.push(sample_grp);
                sample_grp.clear();
            }
            sample_grp.push_back(sample);
            if(line_num%this->_log_num == 0) { //日志记录
                std::cout << line_num << " lines have finished" << std::endl;
            }
        }

        if(!sample_grp.empty()) {
            buffer.push(sample_grp);
            sample_grp.clear();
        }
        _cv_queue_not_empty.notify_one();
        lock.unlock();
        std::cout<<"_buf_size:"<<this->_buf_size<<"  "<<buffer.size()<<" finished_flag:"<<finished_flag<<std::endl;
        if(finished_flag) {
            break;
        }
    }
    _cv_queue_not_empty.notify_all();
    std::cout<<"prodcut  end...."<<std::endl;
    infile.close(); //关闭文件输入流
}

}
}
