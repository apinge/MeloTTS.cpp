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
#ifndef CHRONOLOGY_H
#define CHRONOLOGY_H
#include <regex>
#include <unordered_map>

namespace text_normalization {
//// 数字映射
// std::unordered_map<char, std::string> DIGITS = {
//     {'0', "零"}, {'1', "一"}, {'2', "二"}, {'3', "三"},
//     {'4', "四"}, {'5', "五"}, {'6', "六"}, {'7', "七"},
//     {'8', "八"}, {'9', "九"} };
//  时刻表达式
extern std::wregex RE_TIME;
// 时间范围，如8:30-12:30
extern std::wregex RE_TIME_RANGE;
// 日期表达式
extern std::wregex RE_DATE;
// 用 / 或者 - 分隔的 YY/MM/DD 或者
extern std::wregex RE_DATE2;

std::wstring _time_num2str(const std::wstring& num_string);
std::wstring replace_time(const std::wsmatch& match);
std::wstring replace_date(const std::wsmatch& match);
std::wstring replace_date2(const std::wsmatch& match);
// 假设 num2str 和 verbalize_digit, verbalize_cardinal 函数已经实现
// std::string num2str(const std::string& num);
// std::string verbalize_digit(const std::string& num);
// std::string verbalize_cardinal(const std::string& num);
}  // namespace text_normalization
#endif