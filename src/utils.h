#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <format>
#include <chrono>

// This utils module defines a collection of utility functions that are
// used throughout the program

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
inline double get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ms>(Time::now() - startTime).count();
};
void ConfigureOneDNNCache();
#endif //  UTILS_H
