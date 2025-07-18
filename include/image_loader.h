#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>

std::vector<std::pair<cv::Mat, int>> load_images_with_labels(
    const std::string &dataset_path,
    std::map<std::string, int> &class_to_index,
    int target_width,
    int target_height);
