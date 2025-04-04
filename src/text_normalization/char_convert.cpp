/**
 * Copyright      2025    Alex G Chen (alex.g.chen@intel.com)
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
#include "char_convert.h"

#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#define NOGDI
#define NOCRYPT
#include <windows.h>
#endif

namespace text_normalization {
std::unordered_map<wchar_t, wchar_t> s2t_dict;
std::unordered_map<wchar_t, wchar_t> t2s_dict;
// 从文件中读取字符串
std::wstring readFile(const std::string& filename) {
    std::wifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERORR] text_normalization::readFile:: Cannot openfile:  " << filename << std::endl;
        return L"";
    }
    // 设置 locale 以处理 UTF-8 编码
    file.imbue(std::locale(std::locale(), new std::codecvt_utf8<wchar_t>));

    std::wstringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
// 保存映射到二进制文件
void save_map_to_binary_file(const std::unordered_map<wchar_t, wchar_t>& map, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERORR]save_map_to_binary_file:: Cannot openfile: " << filename << std::endl;
        return;
    }

    size_t size = map.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& pair : map) {
        file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
    file.close();
}
// 从二进制文件加载映射
std::unordered_map<wchar_t, wchar_t> load_map_from_binary_file(const std::string& filename) {
    std::unordered_map<wchar_t, wchar_t> map;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERORR] load_map_from_binary_file:: Cannot openfile: " << filename << std::endl;
        return map;
    }

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    wchar_t key, value;
    for (size_t i = 0; i < size; ++i) {
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        map[key] = value;
    }
    file.close();
    return map;
}
// 从文件中读取简体和繁体字符
// const std::wstring simplified_characters = readFile("simplified_characters.txt");
// const std::wstring traditional_characters = readFile("traditional_characters.txt");

// 创建映射
// std::unordered_map<wchar_t, wchar_t> s2t_dict;
// std::unordered_map<wchar_t, wchar_t> t2s_dict;

// 初始化映射
// void initialize_maps() {
//    // 确保简体和繁体字符的长度相同
//    if (simplified_characters.length() != traditional_characters.length()) {
//        throw std::invalid_argument("简体和繁体字符的长度不一致");
//    }
//    for (size_t i = 0; i < simplified_characters.length(); ++i) {
//        wchar_t s_char = static_cast<wchar_t>(simplified_characters[i]);
//        wchar_t t_char = static_cast<wchar_t>(traditional_characters[i]);
//        s2t_dict[s_char] = t_char;
//        t2s_dict[t_char] = s_char;
//    }
//    //save_map_to_binary_file(s2t_dict, L"s2t_map.bin");
//    //save_map_to_binary_file(t2s_dict, L"t2s_map.bin");
//}
std::string wstring_to_string(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}
std::wstring string_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}
// 将繁体转换为简体
std::wstring traditional_to_simplified(const std::wstring& text) {
    std::wstring result;

    for (wchar_t ch : text) {
        if (t2s_dict.find(ch) != t2s_dict.end()) {
            result += t2s_dict[ch];
        } else {
            result += ch;  // 保持原字符
        }
    }
    return result;
}

// 将简体转换为繁体
std::wstring simplified_to_traditional(const std::wstring& text) {
    std::wstring result;

    for (wchar_t ch : text) {
        if (s2t_dict.find(ch) != s2t_dict.end()) {
            result += s2t_dict[ch];
        } else {
            result += ch;  // 保持原字符
        }
    }

    return result;
}
void initialize_char_maps(const std::filesystem::path& char_map_folder) {
    std::filesystem::path s2t_path = char_map_folder / "s2t_map.bin";
    std::filesystem::path t2s_map = char_map_folder / "t2s_map.bin";
    // 从二进制文件加载映射
    s2t_dict = load_map_from_binary_file(s2t_path.string());
    t2s_dict = load_map_from_binary_file(t2s_map.string());
}
}  // namespace text_normalization

// int main() {
//
//#ifdef _WIN32
//    SetConsoleOutputCP(CP_UTF8);
//    // 使用系统默认区域设置
//    std::wcout.imbue(std::locale(""));
//#endif
//
//
//    // 初始化映射
//    //initialize_maps();
//
//    // 从二进制文件加载映射
//    s2t_dict = load_map_from_binary_file(L"s2t_map.bin");
//    t2s_dict = load_map_from_binary_file(L"t2s_map.bin");
//
//    // 示例文本（繁体）
//    std::string text =
//    "一般是指存取一個應用程式啟動時始終顯示在網站或網頁瀏覽器中的一個或多個初始網頁等畫面存在的站點"; std::cout <<
//    "Original text: " << text << std::endl;
//
//    // 转换为简体
//    std::string text_simple = traditional_to_simplified(text);
//    std::cout << "Simplified text: " << text_simple << std::endl;
//
//    // 再转换回繁体
//    std::string text_traditional = simplified_to_traditional(text_simple);
//    std::cout << "Traditional text: " << text_traditional << std::endl;
//
//    return 0;
//}
