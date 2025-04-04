/*
 * Licensed under the Apache License, Version 2.0.
 * See the LICENSE file for more information.
 */
#include "utils.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

// set ONEDNN_CACHE_CAPACITY
void ConfigureOneDNNCache() {
#ifdef _WIN32
    auto status = _putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=100");
#elif __linux__
    auto status = setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "100", true);
#else
    std::cout << "Running on an unknown OS" << std::endl;
#endif
    // TODO : Add try catch block here
    if (status == 0) {
        char* onednn_kernel_capacity = std::getenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY");
        int num = std::stoi(std::string(onednn_kernel_capacity));
        assert((num == 100) && "[ERROR] Set ONEDNN_PRIMITIVE_CACHE_CAPACITY fails!");
        std::cout << "[INFO] Set ONEDNN_PRIMITIVE_CACHE_CAPACITY: " << onednn_kernel_capacity << "\n";
    }
}
// set ONEDNN_MAX_CPU_ISA=AVX2_VNNI
// A workaround for int8 model's inference on Lunar Lake
// Should not affect on Meteor lake or processors and will be removed in the future.
// Ref https://oneapi-src.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html
void SetOneDNN_CPU_MAX_ISA() {
#ifdef _WIN32
    auto status = _putenv("ONEDNN_MAX_CPU_ISA=AVX2_VNNI");
#elif __linux__
    auto status = setenv("ONEDNN_MAX_CPU_ISA", "AVX2_VNNI", true);
#else
    std::cout << "Running on an unknown OS" << std::endl;
#endif
    if (status == 0) {
        char* onednn_max_cpu_isa = std::getenv("ONEDNN_MAX_CPU_ISA");
        assert((std::string(onednn_max_cpu_isa) == "AVX2_VNNI") && "[ERROR] Set ONEDNN_MAX_CPU_ISA fails!");
        std::cout << "[INFO] Set ONEDNN_MAX_CPU_ISA: " << onednn_max_cpu_isa << "\n";
    }
}

std::vector<std::string> read_file_lines(const std::filesystem::path& file_path) {
    std::vector<std::string> lines;
    std::ifstream file(file_path);

    if (!std::filesystem::exists(file_path) || !file.is_open()) {
        std::cerr << "Error: File either does not exist or could not be opened: " << file_path << std::endl;
        return lines;  // return empty vector if file cannot be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);  // add each line to the vector
    }

    file.close();
    std::cout << "Info: Read input path successfully!\n";
    return lines;
}

/**
 * helper function splits a UTF-8 encoded string containing only Chinese characters
 * e.g. 右值的生命周期 -> {右,值,的,生,命,周,期,}
 * (excluding punctuation and English letters) into individual Chinese characters.
 */
std::vector<std::string> split_utf8_chinese(const std::string& str) {
    std::vector<std::string> res;  // chinese characters
    int strSize = str.size();
    int i = 0;

    while (i < strSize) {
        int len = 1;
        for (int j = 0; j < 6 && (str[i] & (0x80 >> j)); j++) {
            len = j + 1;
        }
        res.push_back(str.substr(i, len));
        i += len;
    }
    return res;
}
// torch.mean(res, dim=1)
// This fuction computes the average of all columns in each row, resulting in a tensor that only has the row dimension
// remaining
std::vector<float> cal_row_mean(const ov::Tensor& tensor_2d, bool print_shape) {
    ov::Shape tensor_shape = tensor_2d.get_shape();
    if (print_shape)
        std::cout << "tensor_2d_shape" << tensor_shape << std::endl;
    int row = tensor_shape[0], col = tensor_shape[1];
    const float* data = tensor_2d.data<const float>();
    std::vector<float> res;
    for (int i = 0; i < row; ++i) {
        float sum = 0.f;
        for (int j = 0; j < col; ++j) {
            sum += data[col * i + j];
        }
        res.emplace_back(sum / col);
    }
    return res;
}
// torch.var(res, dim=1, unbiased=False)
std::vector<float> cal_row_variance(const ov::Tensor& tensor_2d, bool print_shape) {
    ov::Shape tensor_shape = tensor_2d.get_shape();
    if (print_shape)
        std::cout << " tensor_2d_shape" << tensor_shape << std::endl;
    int row = tensor_shape[0], col = tensor_shape[1];
    const float* data = tensor_2d.data<const float>();
    std::vector<float> res;
    for (int i = 0; i < row; ++i) {
        float sum = 0.f;
        for (int j = 0; j < col; ++j) {
            sum += data[col * i + j];
        }
        float mean = sum / col;
        float variance = 0.f;
        for (int j = 0; j < col; ++j) {
            variance += (data[col * i + j] - mean) * (data[col * i + j] - mean);
        }
        res.emplace_back(variance / col);
    }
    return res;
}