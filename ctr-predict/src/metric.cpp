#include "metric.h"

#include<vector>
#include<cmath>
#include<algorithm>
#include<iostream>

namespace ml {
namespace ctr {

bool Metric::less(const std::pair<int, double>& s1,const std::pair<int, double> & s2) {
    return s1.second > s2.second;
}

double Metric::auc(const std::vector<int> & labels, const std::vector<double> & probs) {
    double aucValue = 0.0;
    if(labels.size() != probs.size()) {
        return aucValue;
    }
    std::vector<std::pair<int, double> > labelProbs;
    long long posCount=0, negCount=0, rankCount=0;
    for(int i=0; i<labels.size(); i++) {
        labelProbs.push_back(std::pair<int, double>(labels[i], probs[i]));
        if(labels[i] == 1) posCount++;
        else negCount++;
    }
    //sort 排序
    std::sort(labelProbs.begin(), labelProbs.end(), Metric::less);
    for(int i=0; i<labels.size(); i++) {
        if(labelProbs[i].first == 1) {
            rankCount += (labels.size()-i-1);
        }
    }
    //std::cout<<"pos:"<<posCount<<"neg:"<<negCount<<"rankCount"<<rankCount<<std::endl;
    aucValue = (rankCount - posCount*(posCount+1)/2.0)/(posCount*negCount);
    return aucValue;
}


double Metric::log_loss(const std::vector<int> & labels, const std::vector<double> & probs) {
    double logLossValue = 0.0;
    if(labels.size() != probs.size()) {
        return logLossValue;
    }
    double epsilon=1e-7;
    for(int i=0; i<labels.size(); i++) {
        if(labels[i] == 1) {
            logLossValue += log(std::max(probs[i], epsilon));
            //std::cout<<"logLoss:"<<probs[i]<<std::endl;
        } else {
            logLossValue += log(std::max(1-probs[i], epsilon));
            //std::cout<<"logLoss:"<<-1*log(std::max(1-probs[i], epsilon))<<std::endl;
        }
    }
    //std::cout<<"logLoss:"<<logLossValue<<std::endl;
    logLossValue /= labels.size();
    return -logLossValue;
}

double Metric::acuraccy(const std::vector<int> & labels, const std::vector<double> & probs, double threshold) {
    double acurracyValue = 0.0;
    if(labels.size() != probs.size()) {
        return acurracyValue;
    }
    long long count = 0l;
    for(int i=0; i<labels.size(); i++) {
        //std::cout<<labels[i]<<" "<<probs[i]<<std::endl;
        if((labels[i] == 1 && probs[i]>threshold) || (labels[i] == -1 && probs[i]<=threshold)) {
            count++;
        }
    }
    return (count*1.0)/labels.size();

}

}
}

