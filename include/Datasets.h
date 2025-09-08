#ifndef DATASETS_H
#define DATASETS_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#define MNIST_TRAIN_LEN 60000
#define MNIST_TEST_LEN 10000
#define MNIST_IMAGE_SIZE 784 

class MNIST {
public:
    void get_mnist(
        std::vector<std::vector<float>>& Train_DS, std::vector<int>& Train_L,
        std::vector<std::vector<float>>& Test_DS, std::vector<int>& Test_L);

private:
    int ReverseInt(int i);
    void read_images(const std::string& path, int num_images, int image_size, std::vector<std::vector<float>>& dataset);
    void read_labels(const std::string& path, int num_labels, std::vector<int>& labels);
};

#endif
