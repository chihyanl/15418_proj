#include <iostream>
#include <fstream>

#include "dataset.h"

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

int get_MNIST_data(string path, float* ptr, bool train) {
  ifstream file(path, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int numImg = 0;
    int numRow = 0;
    int numCol = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    if (ReverseInt(magic_number) != IMAGE_MAGIC_NUMBER) {
      cout << "Magic Number Error when opening " << path << "\n";
      return 1;
    }
    file.read((char*)&numImg, sizeof(numImg));
    numImg = ReverseInt(numImg);
    if (train && numImg != TRAIN_SIZE) {
      printf("Train Data Count Incorrect\n");
      return 1;
    } else if (!train && numImg != TEST_SIZE) {
      printf("Test Data Count Incorrect\n");
      return 1;
    }

    file.read((char*)&numRow, sizeof(numRow));
    if (ReverseInt(numRow) != IMAGE_WIDTH) {
      cout << "Row Number Error when opening " << path << "\n";
      return 1;
    }
    file.read((char*)&numCol, sizeof(numCol));
    if (ReverseInt(numCol) != IMAGE_HEIGHT) {
      cout << "Column Number Error when opening " << path << "\n";
      return 1;
    }

    int index;
    for (int i = 0; i < numImg; i++) {
      for (int row = 0; row < IMAGE_WIDTH; row++) {
        for (int col = 0; col < IMAGE_HEIGHT; col++) {
          unsigned char data = 0;
          file.read((char*)&data, sizeof(data));
          index = i * IMAGE_WIDTH * IMAGE_HEIGHT + row * IMAGE_WIDTH + col;
          ptr[index] = (float)data;
        }
      }
      for (int row = 0; row < IMAGE_WIDTH; row++) {
        for (int col = 0; col < IMAGE_HEIGHT; col++) {
          index = i * IMAGE_WIDTH * IMAGE_HEIGHT + row * IMAGE_WIDTH + col;
          ptr[index] = ptr[index] / 255;
        }
      }
    }
  } else {
    cout << "Error opening " << path << "\n";
    return 1;
  }

  return 0;
}

int get_MNIST_label(string path, int* ptr, bool train) {
  ifstream file(path, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int numImg = 0;

    file.read((char*)&magic_number, sizeof(magic_number)); 
    if (ReverseInt(magic_number) != LABEL_MAGIC_NUMBER) {
      cout << "Magic Number Error when opening " << path << "\n";
      return 1;
    }
    file.read((char*)&numImg, sizeof(numImg));
    numImg = ReverseInt(numImg);
    if (train && numImg != TRAIN_SIZE) {
      printf("Train Data Count Incorrect\n");
      return 1;
    } else if (!train && numImg != TEST_SIZE) {
      printf("Test Data Count Incorrect\n");
      return 1;
    }

    for (int i = 0; i < numImg; i++) {
      unsigned char data = 0;
      file.read((char*)&data, sizeof(data));
      ptr[i] = (int)data;
    }
  } else {
    cout << "Error opening " << path << "\n";
    return 1;
  }

  return 0;
}

