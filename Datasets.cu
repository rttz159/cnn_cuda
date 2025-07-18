#include "Datasets.cuh"
#include "image_loader.h"

void OpenCVLoader::load_from_folders()
{
    std::cout << "Loading Images\n";
    load_images_into_batches("data/training_data",training_batches,training_label_batches,local_class_to_label,batch_size,target_channels,target_height,target_width);
    load_images_into_batches("data/testing_data",testing_batches,testing_label_batches,local_class_to_label,batch_size,target_channels,target_height,target_width);
    load_images_into_batches("data/validation_data",validation_batches,validation_label_batches,local_class_to_label,batch_size,target_channels,target_height,target_width);
    
    if(cudaEnabled){
        for(int i = 0; i < training_batches.size(); i++){
            CudaTensor<4> tempTensor(training_batches.at(i).get_shape());
            tempTensor.copy_from_host(training_batches.at(i).raw_data_arr());
            training_batches_cuda.push_back(tempTensor);
        }
        for(int i = 0; i < training_label_batches.size(); i++){
            CudaTensor<2> tempTensor(training_label_batches.at(i).get_shape());
            tempTensor.copy_from_host(training_label_batches.at(i).raw_data_arr());
            training_label_batches_cuda.push_back(tempTensor);
        }
        for(int i = 0; i < testing_batches.size(); i++){
            CudaTensor<4> tempTensor(testing_batches.at(i).get_shape());
            tempTensor.copy_from_host(testing_batches.at(i).raw_data_arr());
            testing_batches_cuda.push_back(tempTensor);
        }
        for(int i = 0; i < testing_label_batches.size(); i++){
            CudaTensor<2> tempTensor(testing_label_batches.at(i).get_shape());
            tempTensor.copy_from_host(testing_label_batches.at(i).raw_data_arr());
            testing_label_batches_cuda.push_back(tempTensor);
        }
        for(int i = 0; i < validation_batches.size(); i++){
            CudaTensor<4> tempTensor(validation_batches.at(i).get_shape());
            tempTensor.copy_from_host(validation_batches.at(i).raw_data_arr());
            validation_batches_cuda.push_back(tempTensor);
        }
        for(int i = 0; i < validation_label_batches.size(); i++){
            CudaTensor<2> tempTensor(validation_label_batches.at(i).get_shape());
            tempTensor.copy_from_host(validation_label_batches.at(i).raw_data_arr());
            validation_label_batches_cuda.push_back(tempTensor);
        }
    }

    std::cout << "Images Loading Done\n";
}

void OpenCVLoader::load_images_into_batches(
    const std::string &dataset_path,
    std::vector<Tensor<4>> &batches,
    std::vector<Tensor<2>> &label_batches,
    std::map<std::string, int> &class_to_index,
    int batch_size,
    int target_channels,
    int target_height,
    int target_width)
{
    auto loaded_data = load_images_with_labels(dataset_path, class_to_index, target_width, target_height);

    size_t total_samples = loaded_data.size();
    size_t num_batches = (total_samples + batch_size - 1) / batch_size;

    for (size_t b = 0; b < num_batches; ++b) {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, total_samples);
        size_t current_batch_size = end - start;

        Tensor<4> batch({(size_t)batch_size, (size_t)target_channels, (size_t)target_height, (size_t)target_width});
        Tensor<2> label_tensor({(size_t)batch_size, 1});

        for (size_t i = 0; i < current_batch_size; ++i) {
            const auto &[img, label] = loaded_data[start + i];
            for (int c = 0; c < target_channels; ++c)
                for (int h = 0; h < target_height; ++h)
                    for (int w = 0; w < target_width; ++w)
                        batch(i, c, h, w) = img.at<cv::Vec3f>(h, w)[c];
            label_tensor(i, 0) = static_cast<float>(label);
        }

        for (size_t i = current_batch_size; i < (size_t)batch_size; ++i) {
            for (int c = 0; c < target_channels; ++c)
                for (int h = 0; h < target_height; ++h)
                    for (int w = 0; w < target_width; ++w)
                        batch(i, c, h, w) = 0.0f;
            label_tensor(i, 0) = -1.0f;
        }

        Tensor<1> mask({current_batch_size});
        for (size_t i = 0; i < current_batch_size; ++i) {
            mask(i) = (label_tensor(i, 0) != -1) ? 1.0f : 0.0f;
        }

        mask_batches.push_back(mask);
        batches.push_back(batch);
        label_batches.push_back(label_tensor);
    }
}
