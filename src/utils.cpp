#include <cassert>
#include <iostream>
#include "utils.h"

// set ONEDNN_CACHE_CAPACITY 
void ConfigureOneDNNCache() {
#ifdef _WIN32
    auto status = _putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=100");
#elif __linux__ 
    auto status = setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "100", true);
#else
    std::cout << "Running on an unknown OS" << std::endl;
    return 0;
#endif
    if (status == 0) {
        char* onednn_kernel_capacity = std::getenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY");
        int num = std::stoi(std::string(onednn_kernel_capacity));
        assert((num == 100) && "[ERROR] Set ONEDNN_PRIMITIVE_CACHE_CAPACITY fails!");
        std::cout << "set ONEDNN_PRIMITIVE_CACHE_CAPACITY: " << onednn_kernel_capacity << "\n";
    }
}