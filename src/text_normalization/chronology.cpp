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
#include "chronology.h"

#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>

#include "number.h"

namespace text_normalization {
// 时刻表达式 (使用宽字符 wregex)
std::wregex RE_TIME(LR"(([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?)");
// 时间范围，如8:30-12:30
std::wregex RE_TIME_RANGE(
    LR"(([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?)");
// 日期表达式
std::wregex RE_DATE(LR"((\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?)");
// 用 / 或者 - 分隔的 YY/MM/DD 或者
std::wregex RE_DATE2(LR"((\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01]))");

// 特殊时间数字转换 (改为宽字符版本)
std::wstring _time_num2str(const std::wstring& num_string) {
    std::wstring result = num2str(num_string.substr(num_string.find_first_not_of(L'0')));
    if (num_string[0] == L'0') {
        result = DIGITS[L'0'] + result;
    }
    return result;
}

// 替换时间 (改为宽字符版本)
std::wstring replace_time(const std::wsmatch& match) {
    bool is_range = match.size() > 5;

    std::wstring hour = match.str(1);
    std::wstring minute = match.str(2);
    std::wstring second = match.str(4);

    std::wstring result = num2str(hour) + L"点";
    if (!minute.empty() && minute != L"00") {
        if (std::stoi(minute) == 30) {
            result += L"半";
        } else {
            result += _time_num2str(minute) + L"分";
        }
    }
    if (!second.empty() && second != L"00") {
        result += _time_num2str(second) + L"秒";
    }

    if (is_range) {
        std::wstring hour_2 = match.str(6);
        std::wstring minute_2 = match.str(7);
        std::wstring second_2 = match.str(9);

        result += L"至" + num2str(hour_2) + L"点";
        if (!minute_2.empty() && minute_2 != L"00") {
            if (std::stoi(minute_2) == 30) {
                result += L"半";
            } else {
                result += _time_num2str(minute_2) + L"分";
            }
        }
        if (!second_2.empty() && second_2 != L"00") {
            result += _time_num2str(second_2) + L"秒";
        }
    }

    return match.prefix().str() + result + match.suffix().str();
}

// 替换日期 (改为宽字符版本)
std::wstring replace_date(const std::wsmatch& match) {
    std::wstring year = match.str(1);
    std::wstring month = match.str(3);
    std::wstring day = match.str(5);
    std::wstring result;
    if (!year.empty()) {
        result += verbalize_digit(year) + L"年";
    }
    if (!month.empty()) {
        result += verbalize_cardinal(month) + L"月";
    }
    if (!day.empty()) {
        result += verbalize_cardinal(day) + L"日";
    }
    return match.prefix().str() + result + match.suffix().str();
}

// 替换日期2 (改为宽字符版本)
std::wstring replace_date2(const std::wsmatch& match) {
    std::wstring year = match.str(1);
    std::wstring month = match.str(3);
    std::wstring day = match.str(4);
    std::wstring result;
    if (!year.empty()) {
        result += verbalize_digit(year) + L"年";
    }
    if (!month.empty()) {
        result += verbalize_cardinal(month) + L"月";
    }
    if (!day.empty()) {
        result += verbalize_cardinal(day) + L"日";
    }
    return match.prefix().str() + result + match.suffix().str();
}

}  // namespace text_normalization

//// 假设 num2str 和 verbalize_digit, verbalize_cardinal 函数已经实现
// std::string num2str(const std::string& num) {
//     // 这里是一个简单的实现，实际实现可能更复杂
//     std::unordered_map<char, std::string> num_map = {
//         {'0', "零"}, {'1', "一"}, {'2', "二"}, {'3', "三"},
//         {'4', "四"}, {'5', "五"}, {'6', "六"}, {'7', "七"},
//         {'8', "八"}, {'9', "九"}
//     };
//     std::string result;
//     for (char ch : num) {
//         result += num_map[ch];
//     }
//     return result;
// }
//
// std::string verbalize_digit(const std::string& num) {
//     // 这里是一个简单的实现，实际实现可能更复杂
//     return num2str(num);
// }
//
// std::string verbalize_cardinal(const std::string& num) {
//     // 这里是一个简单的实现，实际实现可能更复杂
//     return num2str(num);
// }

// int main() {
//
//     // 测试时间替换
//     std::string time_text = "12:30";
//     std::smatch match;
//     if (std::regex_search(time_text, match, RE_TIME)) {
//         std::string replaced_time = replace_time(match);
//         std::cout << "Replaced time: " << replaced_time << std::endl;
//     }
//
//     // 测试时间范围替换
//     std::string time_range_text = "8:30-12:30";
//     if (std::regex_search(time_range_text, match, RE_TIME_RANGE)) {
//         std::string replaced_time_range = replace_time(match);
//         std::cout << "Replaced time range: " << replaced_time_range << std::endl;
//     }
//
//     // 测试日期替换
//     std::string date_text = "2021年2月14日";
//     if (std::regex_search(date_text, match, RE_DATE)) {
//         std::string replaced_date = replace_date(match);
//         std::cout << "Replaced date: " << replaced_date << std::endl;
//     }
//
//     // 测试日期2替换
//     std::string date2_text = "2021-02-01";
//     if (std::regex_search(date2_text, match, RE_DATE2)) {
//         std::string replaced_date2 = replace_date2(match);
//         std::cout << "Replaced date2: " << replaced_date2 << std::endl;
//     }
//
//     return 0;
// }