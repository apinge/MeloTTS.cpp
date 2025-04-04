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
#pragma once
#ifndef CHAR_CONVERT_H
#define CHAR_CONVERT_H
#include <filesystem>
#include <unordered_map>
namespace text_normalization {
extern std::unordered_map<wchar_t, wchar_t> s2t_dict;
extern std::unordered_map<wchar_t, wchar_t> t2s_dict;

void initialize_char_maps(const std::filesystem::path& char_map_folder);
// 从文件中读取字符串
std::wstring readFile(const std::string& filename);
// 将繁体转换为简体
std::wstring traditional_to_simplified(const std::wstring& text);
// 将简体转换为繁体
std::wstring simplified_to_traditional(const std::wstring& text);
// 保存映射到二进制文件
void save_map_to_binary_file(const std::unordered_map<wchar_t, wchar_t>& map, const std::string& filename);
// 从二进制文件加载映射
std::unordered_map<wchar_t, wchar_t> load_map_from_binary_file(const std::string& filename);
std::string wstring_to_string(const std::wstring& wstr);
std::wstring string_to_wstring(const std::string& str);
}  // namespace text_normalization
#endif