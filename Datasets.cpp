#include "Datasets.h"
#include <fstream>
#include <omp.h>

using namespace std;

int MNIST::ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNIST::read_images(const std::string& path, int num_images, int image_size, std::vector<std::vector<float>>& dataset) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open image file " + path);
    }

    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    n_cols = ReverseInt(n_cols);

    if (number_of_images < num_images) {
        throw std::runtime_error("Error: Not enough images in " + path);
    }

    dataset.resize(num_images, std::vector<float>(image_size));

    std::vector<unsigned char> buffer(num_images * image_size);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    #pragma omp parallel for
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            dataset[i][j] = static_cast<float>(buffer[i * image_size + j]) / 255.0f;
        }
    }
}

void MNIST::read_labels(const std::string& path, int num_labels, std::vector<int>& labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open label file " + path);
    }

    int magic_number = 0, number_of_labels = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = ReverseInt(number_of_labels);

    if (number_of_labels < num_labels) {
        throw std::runtime_error("Error: Not enough labels in " + path);
    }

    labels.resize(num_labels);
    std::vector<unsigned char> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    #pragma omp parallel for
    for (int i = 0; i < num_labels; ++i) {
        labels[i] = static_cast<int>(buffer[i]);
    }
}

void MNIST::get_mnist(
    std::vector<std::vector<float>>& Train_DS, std::vector<int>& Train_L,
    std::vector<std::vector<float>>& Test_DS, std::vector<int>& Test_L) {

    std::cout << "\nLoading MNIST dataset...\n" << std::endl;
    auto load_start = chrono::high_resolution_clock::now();

    read_images("MNIST_data/train-images.idx3-ubyte", MNIST_TRAIN_LEN, MNIST_IMAGE_SIZE, Train_DS);
    read_labels("MNIST_data/train-labels.idx1-ubyte", MNIST_TRAIN_LEN, Train_L);

    read_images("MNIST_data/t10k-images.idx3-ubyte", MNIST_TEST_LEN, MNIST_IMAGE_SIZE, Test_DS);
    read_labels("MNIST_data/t10k-labels.idx1-ubyte", MNIST_TEST_LEN, Test_L);

    auto load_end = chrono::high_resolution_clock::now();
    chrono::duration<double> load_time = load_end - load_start;

    std::cout << "Done loading MNIST!\n" << "Loading time: " << load_time.count() << " seconds\n" << std::endl;
}
