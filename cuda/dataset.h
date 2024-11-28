#include <string>
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049

using namespace std;

int get_MNIST_data(string path, float* ptr, bool train);
int get_MNIST_label(string path, int* ptr, bool train);
