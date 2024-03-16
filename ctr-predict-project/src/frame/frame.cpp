#include "src/frame/frame.h"
#include <unistd.h>

namespace ml {
namespace ctr {

bool Frame::init(Task* task, TrainerOption * opt) {
    // t_num 线程数量
    // buf_size 单个线程处理数量
    this->task = task;
    this->_thread_num = opt->threads_num;
    this->_buf_size = opt->buffer_size;
    this->_log_num = opt->log_num;
    thread_vec.clear();
    //生产者，样本读取
    thread_vec.push_back(std::thread(&Frame::pro_thread, this));
    for(int i = 0; i < _thread_num; ++i) {
        //消费者，训练模型
        thread_vec.push_back(std::thread(&Frame::con_thread, this));
    }
    return true;
}


void Frame::run() {
    for(int i = 0; i < thread_vec.size(); ++i) {
        thread_vec[i].join();
    }
}


void Frame::pro_thread() {
    std::string line;
    int line_num = 0;
    int i = 0;
    finished_flag = false;
    std::vector<Sample> sample_grp;
    while(true) {
        std::unique_lock<std::mutex> lock(buf_mtx);
        _cv_queue_not_full.wait(lock, [&] { return buffer.size() < 1; });
        std::cout<<">>>>>> product sample"<<std::endl;
        for(i = 0; i < this->_buf_size;i++) {
            if(!std::getline(std::cin, line)) {
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
            if(line_num%_log_num == 0) { //日志记录
                std::cout << line_num << " lines have finished" << std::endl;
            }
        }

        if(!sample_grp.empty()) {
            buffer.push(sample_grp);
            sample_grp.clear();
        }
        _cv_queue_not_empty.notify_one();
        std::cout<<">>>>>>>>_buf_size:"<<this->_buf_size<<"  iiii"<<buffer.size()<<std::endl;
        if(finished_flag) {
            break;
        }
    }
}


void Frame::con_thread() {
    std::vector<std::vector<Sample> > input_vec;
    while(true) {
        input_vec.clear();
        std::cout<<">>>>>> consumer sample 0"<<std::endl;
        std::unique_lock<std::mutex> lock(buf_mtx);
        _cv_queue_not_empty.wait(lock, [&] {return !buffer.empty() || finished_flag; });
        //_cv_queue_not_empty.wait(lock);
        std::cout<<">>>>>> consumer sample 1"<<std::endl;
        for(int i = 0; i < this->_buf_size;) {
            if(buffer.empty()) {
                finished_flag = true;
                break;
            }
            input_vec.push_back(buffer.front());
            i += buffer.front().size();
            buffer.pop();
        }
        _cv_queue_not_full.notify_one();
        lock.unlock();
        std::cout<<"consumer..."<<input_vec.size()<<std::endl;
        if(input_vec.size() > 0) {
            task->run_task(input_vec);
        } else {
            break;
        }
        //if(finished_flag) {
        //    break;
        //}
    }
    std::cout<<"consumer... end"<<std::endl;
}



}
}
