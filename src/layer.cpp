#include "layer.h"
#include <random>

Tensor Conv2d::Apply(const Tensor &input) {
  // This is going under the assumption that the input has a shape of 32*32*1
  int out_dim = input.shape[0] - _kernel_size + 1;
  std::vector<int> shape = {out_dim, out_dim, _out_channels};
  Tensor output(shape);

  // TODO: actually implement this

  return output;
}

Tensor Dropout2d::Apply(const Tensor &input) {
  if (_p == 0 || !_is_training) {
    return input;
  }

  Tensor output(input.shape);
  std::bernoulli_distribution bdist(1.0 - _p);

  std::mt19937 rng(42);

  for (int i = 0; i < input.vals.size(); i++) {
    bool keep = bdist(rng);
    output.vals.push_back(keep ? input.vals[i] / (1.0 - _p) : 0.0);
  }

  return output;
}

Tensor Linear::Apply(const Tensor &input) {
  Tensor output({_out_channels});

  for (int i = 0; i < _out_channels; i++) {
    float val = _biases[i];
    for (int j = 0; j < _in_channels; j++) {
      val += _weights[i][j] * input.vals[j];
    }

    output.vals[i] = val;
  }

  return output;
}

void Linear::InitWeights() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.1, 0.1);

  for (int r = 0; r < _weights.size(); r++) {
    for (int c = 0; c < _weights[0].size(); c++) {
      _weights[r][c] = dist(rng);
    }
  }

  for (int i = 0; i < _biases.size(); i++) {
    _biases[i] = dist(rng);
  }
}
