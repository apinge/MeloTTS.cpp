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
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>

#include "number.h"

namespace text_normalization {
std::unordered_map<std::wstring, std::wstring> measure_dict = {
    {L"cm2", L"平方厘米"},
    {L"cm²", L"平方厘米"},
    {L"cm3", L"立方厘米"},
    {L"cm³", L"立方厘米"},
    //{L"cm", L"厘米"}, {L"db", L"分贝"}, {L"ds", L"毫秒"}, {L"kg", L"千克"}, {L"km", L"千米"},
    {L"m2", L"平方米"},
    {L"m²", L"平方米"},
    {L"m³", L"立方米"},
    {L"m3", L"立方米"},
    //{L"ml", L"毫升"}, {L"m", L"米"}, {L"mm", L"毫米"}, {L"s", L"秒"}
};

// 使用宽字符版本的正则表达式
std::wregex re_temperature(LR"((-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度))");

std::wstring replace_temperature(const std::wsmatch& match) {
    std::wstring sign = match.str(1);
    std::wstring temperature = match.str(2);
    std::wstring unit = match.str(4);
    sign = sign.empty() ? L"" : L"零下";
    temperature = num2str(temperature);  // 假设 num2str 返回宽字符串
    unit = (unit == L"摄氏度") ? L"摄氏度" : L"度";
    return match.prefix().str() + sign + temperature + unit + match.suffix().str();
}

std::wstring replace_measure(std::wstring sentence) {
    for (const auto& q_notation : measure_dict) {
        size_t pos = 0;
        while ((pos = sentence.find(q_notation.first, pos)) != std::wstring::npos) {
            sentence.replace(pos, q_notation.first.length(), q_notation.second);
            pos += q_notation.second.length();
        }
    }
    return sentence;
}
}  // namespace text_normalization

// int main() {
//     // Example usage
//     string temperature_example = "-3°C";
//
//     smatch match;
//     if (regex_search(temperature_example, match, re_temperature)) {
//         cout << replace_temperature(match) << endl;
//     }
//
//     string measure_example = "10cm2";
//     cout << replace_measure(measure_example) << endl;
//
//     return 0;
// }