
#include <cuda.h>

#include "dataset.h"
#include "layer.h"
#include "CycleTimer.h"

#define EPOCH 2
#define LR 0.005
#define BATCH 1

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < 50; i++) printf("-");
    printf("\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
}

int main(int argc, char** argv) {
  int train_count, test_count, epoch_count;
  bool fuse = false;

  if (argc == 1) {
    train_count = TRAIN_SIZE;
    test_count = TEST_SIZE;
    epoch_count = EPOCH;
  } else if (argc == 2) {
    train_count = TRAIN_SIZE;
    test_count = TEST_SIZE;
    epoch_count = EPOCH;
    fuse = argv[1][0] == 'f';
  }
  else if (argc == 4) {
    train_count = stoi(argv[1]);
    test_count = stoi(argv[2]);
    epoch_count = EPOCH;
  } else if (argc == 4) {
    train_count = stoi(argv[1]);
    test_count = stoi(argv[2]);
    epoch_count = stoi(argv[3]);
  } else {
    printf("Error, incorrect argument count\n");
    return 1;
  }

  printCudaInfo();

  for (int i = 0; i < 50; i++) printf("-");
  printf("\nTraining Samples: %d\nTesting Samples: %d\nEpochs: %d\nLearning Rate: %g\nBatch Size: %d\n", train_count, test_count, epoch_count, LR, BATCH);
  for (int i = 0; i < 50; i++) printf("-");
  printf("\nInput=%dx%dx1\n", IMAGE_WIDTH, IMAGE_HEIGHT);


  // Create model
  // Conv(int in_channels, int out_channels, int height, int width, int kernel_h, int kernel_w, int stride, int pad)
  // Linear(int in_channels, int out_channels)
  Conv* l1 = new Conv(1, 6, 28, 28, 5, 5, 1, 2);  // 28x28x1 -> 28x28x6
  Conv* l2 = new Conv(6, 16, 28, 28, 4, 4, 2, 0);  // 28x28x6 -> 13x13x16
  ConvFuse* lfuse = new ConvFuse(l1, l2);
  Conv* l3 = new Conv(16, 8, 13, 13, 3, 3, 1, 1);  // 13x13x16 -> 13x13x8
  Conv* l4 = new Conv(8, 4, 13, 13, 3, 3, 1, 0);  // 13x13x8 -> 11x11x4
  Linear* l5 = new Linear(484, 10);

  int correct;
  int* device_correct;
  cudaMalloc(&device_correct, sizeof(int));

  // train
  for (int i = 0; i < 50; i++) printf("-");
  printf("\nTRAINING\n");
  {
    int err = 0;
    float* train_data = (float*)malloc(sizeof(float) * TRAIN_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH);
    int* train_label = (int*)malloc(sizeof(int) * TRAIN_SIZE);

    err += get_MNIST_data("../dataset/train-images-idx3-ubyte", train_data, true);
    err += get_MNIST_label("../dataset/train-labels-idx1-ubyte", train_label, true);

    if (err) {
      return 1;
    }

    float* device_train_data;
    cudaMalloc(&device_train_data, sizeof(float) * TRAIN_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH);
    cudaMemcpy(device_train_data, train_data, sizeof(float) * TRAIN_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH, cudaMemcpyHostToDevice);

    for (int i = 1; i <= epoch_count; i++) {
      cudaMemset(device_correct, 0, sizeof(int));
      double convAccTime = 0;
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < train_count; j++) {
        // forward
        double convTimeStart = CycleTimer::currentSeconds();
        if (fuse) {
            // lfuse->forward(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
            lfuse->forward2(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
        } else {
            l1->forward(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
            l2->forward(l1->output);
        }
        double convTimeEnd = CycleTimer::currentSeconds();
        convAccTime += convTimeEnd - convTimeStart;

        float *l2_out = fuse ? lfuse->output() : l2->output;
        l3->forward(l2_out);
        l4->forward(l3->output);
        l5->forward(l4->output);

        getCorrect(train_label[j], l5->output, l5->error, l5->out_channels, device_correct);

        // backward
        l5->backward(l4->output, l4->error);
        l4->backward(l3->output, l3->error);
        l3->backward(l2->output, l2->error);

        if (fuse) {
            lfuse->backward(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
        } else {
            l2->backward(l1->output, l1->error);
            l1->backward(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH], nullptr);
        }

        // update
        if (j % BATCH == 0) {
          if (fuse) {
            lfuse->update(LR / BATCH);
          } else {
            l1->update(LR / BATCH);
            l2->update(LR / BATCH);
          }
          l3->update(LR / BATCH);
          l4->update(LR / BATCH);
          l5->update(LR / BATCH);
        }
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("%d/%d: Accuracy %.2f%%, Time %.5f s\n", i, epoch_count, (float)correct/train_count*100, endTime-startTime);
      printf("L1 & L2 Acc Forward time: %.5f s\n", convAccTime);
    }

    free(train_data);
    free(train_label);
    cudaFree(device_train_data);
  }

  // test
  for (int i = 0; i < 50; i++) printf("-");
  printf("\nTESTING\n");
  {
    int err = 0;
    float* test_data = (float*)malloc(sizeof(float) * TEST_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH);
    int* test_label = (int*)malloc(sizeof(int) * TEST_SIZE);

    err += get_MNIST_data("../dataset/t10k-images-idx3-ubyte", test_data, false);
    err += get_MNIST_label("../dataset/t10k-labels-idx1-ubyte", test_label, false);

    if (err) {
      return 1;
    }

    float* device_test_data;
    cudaMalloc((void**)&device_test_data, sizeof(float) * TEST_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH);
    cudaMemcpy(device_test_data, test_data, sizeof(float) * TEST_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH, cudaMemcpyHostToDevice);

    cudaMemset(device_correct, 0, sizeof(int));
    double convAccTime = 0;
    double startTime = CycleTimer::currentSeconds();
    for (int j = 0; j < test_count; j++) {
      // forward
      double convTimeStart = CycleTimer::currentSeconds();
      if (fuse) {
        lfuse->forward(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
      } else {
        l1->forward(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
        l2->forward(l1->output);
      }
      double convTimeEnd = CycleTimer::currentSeconds();
      convAccTime += convTimeEnd - convTimeStart;
      float *l2_out = fuse ? lfuse->output() : l2->output;
      l3->forward(l2_out);
      l4->forward(l3->output);
      l5->forward(l4->output);

      getCorrect(test_label[j], l5->output, l5->error, l5->out_channels, device_correct);
    }
    double endTime = CycleTimer::currentSeconds();
    cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Accuracy %.2f%%, Time %.5f s\n", (float)correct/test_count*100, endTime-startTime);
    printf("L1 & L2 Acc Forward time: %.5f s\n", convAccTime);

    free(test_data);
    free(test_label);
    cudaFree(device_test_data);
  }

  cudaFree(device_correct);
  return 0;
}
