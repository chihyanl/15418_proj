
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

struct ConvLayerConfig {
  int height, width; // input width and height on the layer
  int kernel_h, kernel_w, stride, pad;
};

class ConvFuse : public Layer {
  public:
    int in_height, in_width, in_channels, mid_channels, out_channels, l1_h_out, l1_w_out, l2_h_out, l2_w_out;
    Layer mid_layer;

    ConvLayerConfig l1_config, l2_config;

    explicit ConvFuse(int in_channels, int mid_channels, int out_channels, ConvLayerConfig &l1_config, ConvLayerConfig &l2_config);
    ~ConvFuse();
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
