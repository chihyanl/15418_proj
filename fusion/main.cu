
#include <cuda.h>

#include "dataset.h"
#include "layer.h"
#include "fusion.h"
#include "CycleTimer.h"
#include <unistd.h>

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

  if (argc == 1) {
    train_count = TRAIN_SIZE;
    test_count = TEST_SIZE;
    epoch_count = EPOCH;
  } else if (argc == 3) {
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
  Conv* l1 = new Conv(1, 6, 28, 28, 5, 5, 1, 2);  // 28x28x1 -> 28x28x6
  Conv* l2 = new Conv(6, 16, 28, 28, 4, 4, 2, 0);  // 28x28x6 -> 13x13x16
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
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < train_count; j++) {
        // forward
        l1->forward(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
        l2->forward(l1->output);
        l3->forward(l2->output);
        l4->forward(l3->output);
        l5->forward(l4->output);

        getCorrect(train_label[j], l5->output, l5->error, l5->out_channels, device_correct);

        // backward
        l5->backward(l4->output, l4->error);
        l4->backward(l3->output, l3->error);
        l3->backward(l2->output, l2->error);
        l2->backward(l1->output, l1->error);
        l1->backward(&device_train_data[j*IMAGE_HEIGHT*IMAGE_WIDTH], nullptr);

        // update
        if (j % BATCH == 0) {
          l1->update(LR / BATCH);
          l2->update(LR / BATCH);
          l3->update(LR / BATCH);
          l4->update(LR / BATCH);
          l5->update(LR / BATCH);
        }
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("%d/%d: Accuracy %.2f%%, Total Time %.5f s\n", i, epoch_count, (float)correct/train_count*100, endTime-startTime);
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
 
    { // Conventional
      correct = 0;
      cudaMemset(device_correct, 0, sizeof(int)); 
      double l1Time = 0;
      double l2Time = 0;
      double l3Time = 0;
      double l4Time = 0;
      double l5Time = 0;
      double resultTime = 0;
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < test_count; j++) {
        // forward
        double l1StartTime = CycleTimer::currentSeconds();
        l1->forward(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH]);
        double l1EndTime = CycleTimer::currentSeconds();
        double l2StartTime = CycleTimer::currentSeconds();
        l2->forward(l1->output);
        double l2EndTime = CycleTimer::currentSeconds();
        double l3StartTime = CycleTimer::currentSeconds();
        l3->forward(l2->output);
        double l3EndTime = CycleTimer::currentSeconds();
        double l4StartTime = CycleTimer::currentSeconds();
        l4->forward(l3->output);
        double l4EndTime = CycleTimer::currentSeconds();
        double l5StartTime = CycleTimer::currentSeconds();
        l5->forward(l4->output);
        double l5EndTime = CycleTimer::currentSeconds();

        double resultStartTime = CycleTimer::currentSeconds();
        getCorrect(test_label[j], l5->output, l5->error, l5->out_channels, device_correct);
        double resultEndTime = CycleTimer::currentSeconds();

        l1Time += l1EndTime - l1StartTime;
        l2Time += l2EndTime - l2StartTime;
        l3Time += l3EndTime - l3StartTime;
        l4Time += l4EndTime - l4StartTime;
        l5Time += l5EndTime - l5StartTime;
        resultTime += resultEndTime - resultStartTime;
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("Coventional:        Accuracy %.2f%%, Total Time %.5f s, ", (float)correct/test_count*100, endTime-startTime);
      printf("L1 Time %.5f s, L2 Time %.5f s, L3 Time %.5f s, L4 Time %.5f s, L5 Time %.5f s, Result Time %.5f s\n", l1Time, l2Time, l3Time, l4Time, l5Time, resultTime);
    }
    { // 2 Layers Fused
      correct = 0;
      cudaMemset(device_correct, 0, sizeof(int));
      double l1l2Time = 0;
      double l3Time = 0;
      double l4Time = 0;
      double l5Time = 0;
      double resultTime = 0;
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < test_count; j++) {
        // forward
        double l1l2StartTime = CycleTimer::currentSeconds();
        convFuseForward2(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH], l1, l2, 3, 3);
        double l1l2EndTime = CycleTimer::currentSeconds();
        double l3StartTime = CycleTimer::currentSeconds();
        l3->forward(l2->output);
        double l3EndTime = CycleTimer::currentSeconds();
        double l4StartTime = CycleTimer::currentSeconds();
        l4->forward(l3->output);
        double l4EndTime = CycleTimer::currentSeconds();
        double l5StartTime = CycleTimer::currentSeconds();
        l5->forward(l4->output);
        double l5EndTime = CycleTimer::currentSeconds();

        double resultStartTime = CycleTimer::currentSeconds();
        getCorrect(test_label[j], l5->output, l5->error, l5->out_channels, device_correct);
        double resultEndTime = CycleTimer::currentSeconds();

        l1l2Time += l1l2EndTime - l1l2StartTime;
        l3Time += l3EndTime - l3StartTime;
        l4Time += l4EndTime - l4StartTime;
        l5Time += l5EndTime - l5StartTime;
        resultTime += resultEndTime - resultStartTime;
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("2 Layers Fused:     Accuracy %.2f%%, Total Time %.5f s, ", (float)correct/test_count*100, endTime-startTime);
      printf("L1+L2 Time %.5f s, L3 Time %.5f s, L4 Time %.5f s, L5 Time %.5f s, Result Time %.5f s\n", l1l2Time, l3Time, l4Time, l5Time, resultTime);
    }
    { // 3 Layers Fused
      correct = 0;
      cudaMemset(device_correct, 0, sizeof(int));
      double l1l2l3Time = 0;
      double l4Time = 0;
      double l5Time = 0;
      double resultTime = 0;
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < test_count; j++) {
        // forward
        double l1l2l3StartTime = CycleTimer::currentSeconds();
        convFuseForward3(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH], l1, l2, l3, 2, 2);
        double l1l2l3EndTime = CycleTimer::currentSeconds();
        double l4StartTime = CycleTimer::currentSeconds();
        l4->forward(l3->output);
        double l4EndTime = CycleTimer::currentSeconds();
        double l5StartTime = CycleTimer::currentSeconds();
        l5->forward(l4->output);
        double l5EndTime = CycleTimer::currentSeconds();

        double resultStartTime = CycleTimer::currentSeconds();
        getCorrect(test_label[j], l5->output, l5->error, l5->out_channels, device_correct);
        double resultEndTime = CycleTimer::currentSeconds();

        l1l2l3Time += l1l2l3EndTime - l1l2l3StartTime;
        l4Time += l4EndTime - l4StartTime;
        l5Time += l5EndTime - l5StartTime;
        resultTime += resultEndTime - resultStartTime;
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("3 Layers Fused:     Accuracy %.2f%%, Total Time %.5f s, ", (float)correct/test_count*100, endTime-startTime);
      printf("L1+L2+L3 Time %.5f s, L4 Time %.5f s, L5 Time %.5f s, Result Time %.5f s\n", l1l2l3Time, l4Time, l5Time, resultTime);
    }
    { // 4 Layers Fused
      correct = 0;
      cudaMemset(device_correct, 0, sizeof(int));
      double l1l2l3l4Time = 0;
      double l5Time = 0;
      double resultTime = 0;
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < test_count; j++) {
        // forward
        double l1l2l3l4StartTime = CycleTimer::currentSeconds();
        convFuseForward4(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH], l1, l2, l3, l4, 1, 1);
        double l1l2l3l4EndTime = CycleTimer::currentSeconds();
        double l5StartTime = CycleTimer::currentSeconds();
        l5->forward(l4->output);
        double l5EndTime = CycleTimer::currentSeconds();

        double resultStartTime = CycleTimer::currentSeconds();
        getCorrect(test_label[j], l5->output, l5->error, l5->out_channels, device_correct);
        double resultEndTime = CycleTimer::currentSeconds();

        l1l2l3l4Time += l1l2l3l4EndTime - l1l2l3l4StartTime;
        l5Time += l5EndTime - l5StartTime;
        resultTime += resultEndTime - resultStartTime;
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("4 Layers Fused:     Accuracy %.2f%%, Total Time %.5f s, ", (float)correct/test_count*100, endTime-startTime);
      printf("L1+L2+L3+L4 Time %.5f s, L5 Time %.5f s, Result Time %.5f s\n", l1l2l3l4Time, l5Time, resultTime);
    }
    { // 2x2 Layers Fused
      correct = 0;
      cudaMemset(device_correct, 0, sizeof(int));
      double l1l2Time = 0;
      double l3l4Time = 0;
      double l5Time = 0;
      double resultTime = 0;
      double startTime = CycleTimer::currentSeconds();
      for (int j = 0; j < test_count; j++) {
        // forward
        double l1l2StartTime = CycleTimer::currentSeconds();
        convFuseForward2(&device_test_data[j*IMAGE_HEIGHT*IMAGE_WIDTH], l1, l2, 5, 5);
        double l1l2EndTime = CycleTimer::currentSeconds();
        double l3l4StartTime = CycleTimer::currentSeconds();
        convFuseForward2(l2->output, l3, l4, 3, 3);
        double l3l4EndTime = CycleTimer::currentSeconds();
        double l5StartTime = CycleTimer::currentSeconds();
        l5->forward(l4->output);
        double l5EndTime = CycleTimer::currentSeconds();

        double resultStartTime = CycleTimer::currentSeconds();
        getCorrect(test_label[j], l5->output, l5->error, l5->out_channels, device_correct);
        double resultEndTime = CycleTimer::currentSeconds();

        l1l2Time += l1l2EndTime - l1l2StartTime;
        l3l4Time += l3l4EndTime - l3l4StartTime;
        l5Time += l5EndTime - l5StartTime;
        resultTime += resultEndTime - resultStartTime;
      }
      double endTime = CycleTimer::currentSeconds();
      cudaMemcpy(&correct, device_correct, sizeof(int), cudaMemcpyDeviceToHost);
      printf("2x2 Layers Fused:   Accuracy %.2f%%, Total Time %.5f s, ", (float)correct/test_count*100, endTime-startTime);
      printf("L1+L2 Time %.5f s, L3+L4 Time %.5f s, L5 Time %.5f s, Result Time %.5f s\n", l1l2Time, l3l4Time, l5Time, resultTime);
    }

    free(test_data);
    free(test_label);
    cudaFree(device_test_data);
  }

  cudaFree(device_correct);
  return 0;
}
