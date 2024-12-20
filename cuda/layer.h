
void getCorrect(int label, float* output, float* error, int size, int* correct);

class Layer {
  public:
    float *output;
    float *bias;
    float *weight;

    float *u_weight;
    float *u_bias;
    float *error;
};

class Conv : public Layer {
  public:
    int height, width, in_channels, out_channels, kernel_h, kernel_w, stride, pad, height_out, width_out;

    explicit Conv(int in_channels, int out_channels, int height, int width, int kernel_h, int kernel_w, int stride, int pad);
    ~Conv();
    void forward(float* input);
    void backward(float* input, float* src_error);
    void update(float rate);
};

class Linear : public Layer {
  public:
    int in_channels, out_channels;

    explicit Linear(int in_channels, int out_channels);
    ~Linear();
    void forward(float* input);
    void backward(float* input, float* src_error);
    void update(float rate);
};
