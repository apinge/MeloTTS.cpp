#include <fstream>
#include <iostream>
#include <sstream>
#include "cmudict.h"
namespace melo {

    // 从文件加载数据的构造函数
    CMUDict::CMUDict(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            if (!std::getline(iss, key, ':')) {
                continue; // 跳过无法解析的行
            }

            std::vector<std::vector<std::string>> value;
            std::string part;
            // 处理值部分
            while (std::getline(iss, part)) {
                std::istringstream partStream(part);
                std::vector<std::string> subValues;
                std::string subValue;
                while (partStream >> subValue) {
                    subValues.push_back(subValue);
                }
                if (!subValues.empty()) {
                    value.push_back(subValues);
                }
            }

            if (!key.empty()) {
                dict_[key] = value;
            }
        }

        file.close();
    }

    // 重载 operator<< 以打印 CMUDict
    [[maybe_unused]]  std::ostream& operator<<(std::ostream& os, const CMUDict& dict) {
        for (const auto& pair : dict.dict_) {
            const std::string& key = pair.first;
            const std::vector<std::vector<std::string>>& value = pair.second;

            os << key << ":";
            for (const auto& vec : value) {
                os << " ";
                for (size_t i = 0; i < vec.size(); ++i) {
                    os << vec[i];
                    if (i < vec.size() - 1) {
                        os << " ";
                    }
                }
            }
            os << std::endl;
        }
        return os;
    }

   
}