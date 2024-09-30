#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <format>
#include <chrono>
#include <numeric>

// This utils module defines a collection of utility functions that are
// used throughout the program

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
inline long long get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ms>(Time::now() - startTime).count();;
};
void ConfigureOneDNNCache();

// Lambda for calculating mean
// const please refer to https://stackoverflow.com/questions/18113164/lambda-in-header-file-error
const auto calculate_mean = [](const std::vector<float>& v) -> float {
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };

// Lambda for calculating variance
const auto calculate_variance = [](const std::vector<float>& v, float mean) -> float {
    float variance = std::accumulate(v.begin(), v.end(), 0.0f,
        [mean](float sum, float value) {
            return sum + (value - mean) * (value - mean);
        });
    return variance / v.size();
    };

const auto print_mean_variance = [](const std::string& name, const std::vector<std::vector<float>>& v) ->void {
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

        for (const auto& x : row) std::cout << x << " ";
        std::cout << "|" << row.size() << std::endl;

    }
    std::cout << std::endl;
    };

#endif //  UTILS_H
