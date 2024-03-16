#include "src/models/base_trainer.h"

#include <sstream>

namespace ml {
namespace ctr {

double BaseTrainer::_intersect(double x) {
    if (x > 10) return 10;
    else if (x < -10) return -10;
    return x;

}

}
}


