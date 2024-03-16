#include "sample.h"

namespace ml {
namespace ctr {

//分割符
const std::string Sample::_spliter = " ";
const std::string Sample::_inner_spliter = ":";

Sample::Sample(const std::string& line) {
    this->xs.clear();
    if(line.size() == 0) {
        this->flag = false;
        return;
    }
    std::size_t posb = line.find_first_not_of(Sample::_spliter, 0);
    std::size_t pose = line.find_first_of(Sample::_spliter, posb);
    int label = std::atoi(line.substr(posb, pose-posb).c_str());
    this->y = label > 0 ? 1 : -1;
    std::string key;
    int field = 0;
    double value; 
    while(pose < line.size()) {
        posb = line.find_first_not_of(Sample::_spliter, pose);
        if(posb == std::string::npos) {
            break;
        }
        pose = line.find_first_of(Sample::_inner_spliter, posb);
        if(pose == std::string::npos) {
            std::cerr << "wrong line of sample input\n" << line << std::endl;
            exit(1);
        }
        key = line.substr(posb, pose-posb);
        posb = pose + 1;
        if(posb >= line.size()) {
            std::cerr << "wrong line of sample input\n" << line << std::endl;
            exit(1);
        }
        pose = line.find_first_of(Sample::_spliter, posb);
        std::size_t pose1 = line.find_first_of(Sample::_inner_spliter, posb);
        if(pose1 < pose) {
            value = std::stod(line.substr(posb, pose1-posb));
            field = std::stoi(line.substr(pose1, pose-pose1));
        } else {
            value = std::stod(line.substr(posb, pose-posb));
            field = 0;
        }

        if(value != 0) {
            //this->x.push_back(std::make_pair(key, value));
            this->xs.push_back(new SampleItem(key, field, value));
        }
    }
    this->flag = true;
}

}
}

