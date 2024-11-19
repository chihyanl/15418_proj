#include "model.h"

Tensor Model::Forward(const Tensor &input) {
  Tensor out_tensor = std::move(input);
  for (Layer &l : layers) {
    out_tensor = l.Apply(out_tensor);
  }

  return std::move(out_tensor);
}