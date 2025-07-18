// image_loader.cpp
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::pair<cv::Mat, int>> load_images_with_labels(
    const std::string &dataset_path,
    std::map<std::string, int> &class_to_index,
    int target_width,
    int target_height)
{
    std::vector<std::pair<cv::Mat, int>> result;
    int current_label = class_to_index.size();

    for (const auto &entry : fs::directory_iterator(dataset_path))
    {
        if (!entry.is_directory())
            continue;

        std::string class_name = entry.path().filename().string();
        if (class_to_index.find(class_name) == class_to_index.end())
            class_to_index[class_name] = current_label++;

        int label = class_to_index[class_name];

        for (const auto &img_entry : fs::directory_iterator(entry.path()))
        {
            std::string img_path = img_entry.path().string();
            cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
            if (img.empty())
                continue;

            if (img.cols != target_width || img.rows != target_height)
                cv::resize(img, img, cv::Size(target_width, target_height));

            img.convertTo(img, CV_32F, 1.0 / 255.0);

            result.emplace_back(img, label);
        }
    }

    return result;
}
