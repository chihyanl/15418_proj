#include "layer.h"
#include <cstdint>
#include <vector>

using namespace std;

class Model {
public:
  Model() {}
  Tensor Forward(const Tensor &input);

  std::vector<Layer> layers;
};

class BaseModel : public Model {
public:
  BaseModel() {
    layers.reserve(5);
    layers[0] = std::move(Conv2d(1, 10, 5, nullptr));
    layers[1] = std::move(Conv2d(10, 20, 5, nullptr));
    layers[2] = std::move(Dropout2d());
    layers[3] = std::move(Linear(320, 50));
    layers[4] = std::move(Linear(50, 10));
  }
};

class FusionModel : public Model {
public:
  FusionModel() {
    layers.reserve(4);
    layers[0] = std::move(Conv2dFusion(1, 10, 20, 5, nullptr));
    layers[1] = std::move(Dropout2d());
    layers[2] = std::move(Linear(320, 50));
    layers[3] = std::move(Linear(50, 10));
  }
};
