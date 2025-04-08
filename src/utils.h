/*
 * Licensed under the Apache License, Version 2.0.
 * See the LICENSE file for more information.
 */
#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <chrono>
#include <filesystem>
#include <format>
#include <iomanip>
#include <numeric>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

// This utils module defines a collection of utility functions that are
// used throughout the program

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::microseconds us;
inline long long get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ms>(Time::now() - startTime).count();
    ;
};
inline long long get_duration_us_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<us>(Time::now() - startTime).count();
    ;
};
void ConfigureOneDNNCache();
void SetOneDNN_CPU_MAX_ISA();

// Lambda for calculating mean
// const please refer to https://stackoverflow.com/questions/18113164/lambda-in-header-file-error
const auto calculate_mean = [](const std::vector<float>& v) -> float {
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
};

// Lambda for calculating variance
const auto calculate_variance = [](const std::vector<float>& v, float mean) -> float {
    float variance = std::accumulate(v.begin(), v.end(), 0.0f, [mean](float sum, float value) {
        return sum + (value - mean) * (value - mean);
    });
    return variance / v.size();
};

const auto print_mean_variance = [](const std::string& name, const std::vector<std::vector<float>>& v) -> void {
    std::cout << name << " 's size is" << v.size() << " " << v.front().size() << std::endl;
    for (const auto& row : v) {
        const auto mean = calculate_mean(row);
        std::cout << "mean is " << mean << ",";
        std::cout << "variance is " << calculate_variance(row, mean) << std::endl;
    }
};

const auto printVec = [](const auto& vec, const std::string& vecName) {
    std::cout << vecName << ":\n";
    for (const auto& row : vec) {
        for (const auto& x : row)
            std::cout << x << " ";
        std::cout << "|" << row.size() << std::endl;
    }
    std::cout << std::endl;
};

std::vector<float> cal_row_mean(const ov::Tensor& tensor_2d, bool print_shape = false);
std::vector<float> cal_row_variance(const ov::Tensor& tensor_2d, bool print_shape = false);

std::vector<std::string> read_file_lines(const std::filesystem::path& file_path);

std::vector<std::string> split_utf8_chinese(const std::string& str);

// This function mimics Python's len() function.
// It counts each character, treating both letters, Chinese characters and space as 1 unit of length.
// use utf-8 Chinese characters
// no punctuation here!
inline size_t str_len(const std::string& s) {
    int strSize = s.size();
    int i = 0;
    int cnt = 0;
    while (i < strSize) {
        // English letters
        if (s[i] <= 'z' && s[i] >= 'a' || s[i] <= 'Z' && s[i] >= 'A') {
            ++cnt;
            ++i;
        } else {  // Chinese characters
            int len = 1;
            for (int j = 0; j < 6 && (s[i] & (0x80 >> j)); j++) {
                len = j + 1;
            }
            ++cnt;
            i += len;
        }
    }
    return cnt;
}

// function to get profiling info, used after inference with config "device_config[ov::enable_profiling.name()] =
// false;" Refer to
// https://github.com/sammysun0711/ov_llm_bench/blob/6a03a1aacab550ec7e3b84948abf1c7fe186e652/inference_engine.py#L215-L220
[[maybe_unused]] inline void get_profiling_info(std::unique_ptr<ov::InferRequest>& _infer_request) {
    std::vector<ov::ProfilingInfo> perfs_count_list = _infer_request->get_profiling_info();
    perfs_count_list.erase(std::remove_if(perfs_count_list.begin(),
                                          perfs_count_list.end(),
                                          [](ov::ProfilingInfo info) {
                                              return info.status == ov::ProfilingInfo::Status::NOT_RUN;
                                          }),
                           perfs_count_list.end());
    std::sort(perfs_count_list.begin(), perfs_count_list.end(), [&](ov::ProfilingInfo x, ov::ProfilingInfo y) {
        return x.real_time > y.real_time;
    });
    std::cout << std::endl;
    for (const auto& x : perfs_count_list) {
        if (x.status == ov::ProfilingInfo::Status::NOT_RUN)
            continue;

        std::cout << std::left << std::setw(10) << "LayerName: " << std::left << std::setw(60) << x.node_name
                  << std::left << std::setw(10) << "LayerType: " << std::left << std::setw(20) << x.node_type
                  << std::left << std::setw(10) << "Status: " << std::left << std::setw(5) << (int)(x.status)
                  << std::left << std::setw(10) << "execType: " << std::left << std::setw(30) << x.exec_type
                  << std::left << std::setw(10) << "cpuTime: " << std::left << std::setw(10) << x.cpu_time << std::left
                  << std::setw(10) << "realTime: " << x.real_time << std::endl;
    }
    std::cout << std::endl;
}

#endif  //  UTILS_H
