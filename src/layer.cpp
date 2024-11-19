#include "layer.h"

Tensor Conv2d::Apply(const Tensor &input) {
  // This is going under the assumption that the input has a shape of 32*32*1
  int out_dim = input.shape[0] - _kernel_size + 1;
  std::vector<int> shape = {out_dim, out_dim, _out_channels};
  Tensor out_tensor(shape);

  // TODO: actually implement this

  return std::move(out_tensor);
}
