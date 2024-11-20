#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace std;

class Tensor {
public:
  Tensor(std::vector<int> shape_in) : shape(std::move(shape_in)) {
    int size = 1;
    for (uint8_t dim : shape) {
      size *= dim;
    }

    vals.reserve(size);
  }

  std::vector<std::uint8_t> vals;
  std::vector<int> shape;
};

typedef Tensor (*ActivFunc)(Tensor &);

class Layer {
public:
  Tensor Apply(const Tensor &input) {
    throw std::runtime_error("Base Class apply called");
  }

  void SetIsTraining(bool is_training) { _is_training = is_training; }

protected:
  bool _is_training;
};

class Conv2d : public Layer {
public:
  Conv2d(int in_channels, int out_channels, int kernel_size,
         ActivFunc *activ_func)
      : _in_channels(in_channels), _out_channels(out_channels),
        _kernel_size(kernel_size), _activ_func(activ_func) {
    _weights.reserve(_in_channels * _out_channels * _kernel_size *
                     _kernel_size);
  }

  Tensor Apply(const Tensor &input);

private:
  int _in_channels;
  int _out_channels;
  int _kernel_size;
  vector<float> _weights;
  ActivFunc *_activ_func;
};

class Conv2dFusion : public Layer {
public:
  Conv2dFusion(int in_channels, int mid_channels, int out_channels,
               int kernel_size, ActivFunc *activ_func)
      : _in_channels(in_channels), _mid_channels(mid_channels),
        _out_channels(out_channels), _kernel_size(kernel_size),
        _activ_func(activ_func) {}

  Tensor Apply(const Tensor &input);

private:
  int _in_channels;
  int _mid_channels;
  int _out_channels;
  int _kernel_size;
  ActivFunc *_activ_func;
};

class Dropout2d : public Layer {
public:
  Dropout2d(float p) : _p(p) { _is_training = false; }

  Tensor Apply(const Tensor &input);

private:
  float _p;
};

class Linear : public Layer {
public:
  Linear(int in_channels, int out_channels)
      : _in_channels(in_channels), _out_channels(out_channels) {
    _weights.resize(_out_channels, std::vector<float>(_in_channels));
    _biases.resize(_out_channels);
    InitWeights();
  }

  Tensor Apply(const Tensor &input);

  void InitWeights();

private:
  int _in_channels;
  int _out_channels;
  std::vector<std::vector<float>> _weights;
  std::vector<float> _biases;
};
