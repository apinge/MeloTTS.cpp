#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include "utils.h"

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
        std::cout << "set ONEDNN_PRIMITIVE_CACHE_CAPACITY: " << onednn_kernel_capacity << "\n";
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