/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cmudict.h"

#include <fstream>
#include <iostream>
#include <sstream>
namespace melo {

// Constructor that loads data from a file.
// @param filename The name of the file from which to load the data.
CMUDict::CMUDict(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "CMUDict::CMUDict: Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::vector<std::vector<std::string>> inner_vec;  // store result of each line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (!std::getline(iss, key, ':')) {
            continue;  // Skip lines that cannot be parsed.
        }

        std::vector<std::string> value;
        std::string segment;

        while (std::getline(iss, segment, ',')) {
            std::istringstream segment_ss(segment);
            std::vector<std::string> subValues;
            std::string subValue;
            while (segment_ss >> subValue) {
                subValues.push_back(subValue);
            }
            if (!subValues.empty()) {
                value.insert(value.end(), subValues.begin(), subValues.end());
            }
        }

        if (!key.empty()) {
            dict_[key] = value;
        }
    }

    file.close();
    std::cout << "CMUDict::CMUDict: Construct CMUDict\n";
}

// Overloads the operator<< to print the contents of a CMUDict object.
// @param out The output stream to which the data is to be printed.
// @param cmudict The CMUDict object to be printed.
// @return Returns the output stream for chaining.
[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const CMUDict& dict) {
    for (const auto& pair : dict.dict_) {
        const std::string& key = pair.first;
        const std::vector<std::string>& value = pair.second;

        os << key << ":";

        for (auto& x : value)
            os << x;
        os << " ";
        os << std::endl;
    }
    return os;
}

}  // namespace melo