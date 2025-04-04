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
#include <cwctype>  // 用于 iswdigit 等函数
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "number.h"
#ifdef _WIN32
#include <iostream>
#include <locale>
#define NOGDI
#define NOCRYPT
#include <windows.h>
#endif

namespace text_normalization {
// 数字和单位的映射
std::unordered_map<wchar_t, std::wstring> DIGITS = {{L'0', L"零"},
                                                    {L'1', L"一"},
                                                    {L'2', L"二"},
                                                    {L'3', L"三"},
                                                    {L'4', L"四"},
                                                    {L'5', L"五"},
                                                    {L'6', L"六"},
                                                    {L'7', L"七"},
                                                    {L'8', L"八"},
                                                    {L'9', L"九"}};

std::map<int, std::wstring> UNITS = {{1, L"十"}, {2, L"百"}, {3, L"千"}, {4, L"万"}, {8, L"亿"}};

std::unordered_map<wchar_t, std::wstring> asmd_map = {
    {L'+', L"加"},
    {L'-', L"减"},
    {L'×', L"乘"},
    {L'÷', L"除"},
    {L'=', L"等于"},
    {L'>', L"大于"},
    {L'<', L"小于"},
    {L'≈', L"约等于"},
    {L'≥', L"大于等于"},
    {L'≤', L"小于等于"},
};

// 各种正则表达式
std::wregex re_frac(L"(-?)(\\d+)/(\\d+)");
std::wregex re_percentage(L"(-?)(\\d+(\\.\\d+)?)%");
std::wregex re_negative_num(L"(-)(\\d+)");
std::wregex re_default_num(L"\\d{3}\\d*");
std::wregex re_asmd(L"((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))(\\-)((-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+)))");
std::wregex re_math_symbol(L"[\\+\\×\\÷><=≈≤≥]");  // Note that there is no minus symbol '-' here. The hyphen could be
                                                   // used as a dash, so we need to handle it separately.
std::wregex re_positive_quantifier(
    L"(\\d+)([多余几\\+])?(封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|"
    L"担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|"
    L"贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|"
    L"尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|"
    L"瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|"
    L"颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|"
    L"百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)");
std::wregex re_number(L"(-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+))");
// wregex re_range(R"((?<![\d\+\-\×÷=])((-?)((\d+)(\.\d+)?))[-~]((-?)((\d+)(\.\d+)?))(?![\d\+\-\×÷=]))"); running error

std::wregex re_range(LR"((\b(-?\d+(\.\d+)?)\b[-~]\b(-?\d+(\.\d+)?)\b))");
// wregex
// re_to_range(R"(((-?)((\d+)(\.\d+)?)|(\.(\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)[~]((-?)((\d+)(\.\d+)?)|(\.(\d+)))(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s))");
// running error

std::wregex re_to_range(
    LR"((-?\d+(\.\d+)?)([~])(-?\d+(\.\d+)?)([%°C℃度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s]?))");

// 替换分数
std::wstring replace_frac(const std::wsmatch& match) {
    std::wstring sign = match.str(1);
    std::wstring nominator = match.str(2);
    std::wstring denominator = match.str(3);
    sign = sign.empty() ? L"" : L"负";
    nominator = num2str(nominator);
    denominator = num2str(denominator);
    return match.prefix().str() + sign + denominator + L"分之" + nominator + match.suffix().str();
}

// 替换百分比
std::wstring replace_percentage(const std::wsmatch& match) {
    std::wstring sign = match.str(1);
    std::wstring percent = match.str(2);
    sign = sign.empty() ? L"" : L"负";
    percent = num2str(percent);
    return match.prefix().str() + sign + L"百分之" + percent + match.suffix().str();
}

// 替换负数
std::wstring replace_negative_num(const std::wsmatch& match) {
    std::wstring sign = match.str(1);
    std::wstring number = match.str(2);
    sign = sign.empty() ? L"" : L"负";
    number = num2str(number);
    return match.prefix().str() + sign + number + match.suffix().str();
}

// 默认数字替换
std::wstring replace_default_num(const std::wsmatch& match) {
    std::wstring number = match.str(0);
    std::wstring result;
    for (wchar_t digit : number) {
        result += DIGITS[digit];
    }
    return match.prefix().str() + result + match.suffix().str();
}

// 四则运算替换
std::wstring replace_asmd(const std::wsmatch& match) {
    std::wstring result = match.str(1) + asmd_map[match.str(8)[0]] + match.str(9);
    return match.prefix().str() + result + match.suffix().str();
}
// 加、减、乘、除、大于、小于、等于
std::wstring replace_math_symbol(const std::wsmatch& match) {
    auto symbol = match.str(0)[0];
    if (asmd_map.contains(symbol))
        return match.prefix().str() + asmd_map.at(symbol) + match.suffix().str();
    return {};
}

// 量词替换
std::wstring replace_positive_quantifier(const std::wsmatch& match) {
    std::wstring number = match.str(1);
    std::wstring match_2 = match.str(2);
    if (match_2 == L"+") {
        match_2 = L"多";
    }
    std::wstring quantifiers = match.str(3);
    number = num2str(number);
    return match.prefix().str() + number + match_2 + quantifiers + match.suffix().str();
}

// 数字替换
std::wstring replace_number(const std::wsmatch& match) {
    std::wstring sign = match.str(1);
    std::wstring number = match.str(2);
    std::wstring pure_decimal = match.str(5);

    if (!pure_decimal.empty()) {
        return match.prefix().str() + num2str(pure_decimal) + match.suffix().str();
    } else {
        sign = sign.empty() ? L"" : L"负";
        number = num2str(number);
        return match.prefix().str() + sign + number + match.suffix().str();
        ;
    }
}

// 区间替换
std::wstring replace_range(const std::wsmatch& match) {
    std::wstring first = match.str(2);
    std::wstring second = match.str(4);

    std::wregex re(L"(-?)((\\d+)(\\.\\d+)?)|(\\.(\\d+))");

    // 使用回调替换 first 和 second
    first = replace_with_callback(first, re, replace_number);
    second = replace_with_callback(second, re, replace_number);

    return match.prefix().str() + first + L"到" + second + match.suffix().str();
}

// 使用"至"替换
std::wstring replace_to_range(const std::wsmatch& match) {
    std::wstring result = match.str(0);
    result.replace(result.find(L'~'), 1, L"至");
    return match.prefix().str() + result + match.suffix().str();
}

// 工具函数：带回调的替换
std::wstring replace_with_callback(const std::wstring& input,
                                   const std::wregex& re,
                                   const std::function<std::wstring(const std::wsmatch&)>& callback) {
    std::wstring output;
    auto begin = std::wsregex_iterator(input.begin(), input.end(), re);
    auto end = std::wsregex_iterator();

    size_t last_pos = 0;
    for (auto it = begin; it != end; ++it) {
        std::wsmatch match = *it;
        output += input.substr(last_pos, match.position() - last_pos);  // 复制未匹配部分
        output += callback(match);                                      // 使用回调函数替换匹配项
        last_pos = match.position() + match.length();                   // 移动最后的位置
    }
    output += input.substr(last_pos);  // 复制剩余的未匹配部分
    return output;
}
std::vector<std::wstring> _get_value(const std::wstring& value_string, bool use_zero) {
    std::wstring stripped = value_string;
    stripped.erase(0, std::min(stripped.find_first_not_of(L'0'), stripped.size() - 1));
    if (stripped.empty()) {
        return {};
    } else if (stripped.size() == 1) {
        if (use_zero && stripped.size() < value_string.size()) {
            return {DIGITS[L'0'], DIGITS[stripped[0]]};
        } else {
            return {DIGITS[stripped[0]]};
        }
    } else {
        int largest_unit = 0;
        for (auto it = UNITS.rbegin(); it != UNITS.rend(); ++it) {
            if (it->first < stripped.size()) {
                largest_unit = it->first;
                break;
            }
        }
        std::wstring first_part = value_string.substr(0, value_string.size() - largest_unit);
        std::wstring second_part = value_string.substr(value_string.size() - largest_unit);
        std::vector<std::wstring> result = _get_value(first_part);
        result.push_back(UNITS[largest_unit]);
        std::vector<std::wstring> second_result = _get_value(second_part);

        // 判断第二部分是否全为 '0'，如果是，则不再递归处理
        if (second_part.find_first_not_of(L'0') == std::wstring::npos) {
            // second_part 是 "0" 的情况下，不添加 "零"
            return result;  // 返回的结果仅包含前半部分和单位
        }
        result.insert(result.end(), second_result.begin(), second_result.end());
        return result;
    }
}

std::wstring verbalize_cardinal(const std::wstring& value_string) {
    if (value_string.empty()) {
        return L"";
    }

    std::wstring stripped = value_string;
    stripped.erase(0, std::min(stripped.find_first_not_of(L'0'), stripped.size() - 1));
    if (stripped.empty()) {
        return DIGITS[L'0'];
    }

    std::vector<std::wstring> result_symbols = _get_value(value_string);
    if (result_symbols.size() >= 2 && result_symbols[0] == DIGITS[L'1'] && result_symbols[1] == UNITS[1]) {
        result_symbols.erase(result_symbols.begin());
    }
    std::wstring result;
    for (const std::wstring& symbol : result_symbols) {
        result += symbol;
    }
    return result;
}

std::wstring verbalize_digit(const std::wstring& value_string, bool alt_one) {
    std::wstring result;
    for (wchar_t digit : value_string) {
        result += DIGITS[digit];
    }
    // 替换 "一" 为 "幺"
    if (alt_one) {
        size_t pos = result.find(L"一");
        while (pos != std::wstring::npos) {
            result.replace(pos, 1, L"幺");
            pos = result.find(L"一", pos + 1);  // 更新位置，跳过刚替换的字符
        }
    }
    return result;
}
// 数字转字符串
std::wstring num2str(const std::wstring& value_string) {
    size_t point_pos = value_string.find(L'.');
    std::wstring integer, decimal;
    if (point_pos == std::wstring::npos) {
        integer = value_string;
    } else {
        integer = value_string.substr(0, point_pos);
        decimal = value_string.substr(point_pos + 1);
    }

    std::wstring result = verbalize_cardinal(integer);

    decimal.erase(decimal.find_last_not_of(L'0') + 1);
    if (!decimal.empty()) {
        result = result.empty() ? L"零" : result;
        result += L"点" + verbalize_digit(decimal);
    }
    return result;
}
}  // namespace text_normalization

//
// int main() {
//
//#ifdef _WIN32
//        SetConsoleOutputCP(CP_UTF8);
//        // 使用系统默认区域设置
//        std::wcout.imbue(std::locale(""));
//#endif
//        // 测试示例
//        wstring test_frac = L"3/4";
//        wstring test_percentage = L"51%";
//        wstring test_negative_num = L"-10.2";
//        wstring test_default_num = L"100078";
//        wstring test_asmd = L"3+2";
//        wstring test_positive_quantifier = L"3个";
//        wstring test_number = L"-3.14";
//        wstring test_range = L"1-10";
//        wstring test_to_range = L"1~10%";
//
//        wsmatch match;
//
//        if (regex_search(test_frac, match, re_frac)) {
//            wcout << test_frac << L" Fraction: " << replace_frac(match) << endl;
//        }
//
//        if (regex_search(test_percentage, match, re_percentage)) {
//            wcout << test_percentage << L" Percentage: " << replace_percentage(match) << endl;
//        }
//
//        if (regex_search(test_negative_num, match, re_negative_num)) {
//            wcout << test_negative_num << L" Negative Number: " << replace_negative_num(match) << endl;
//        }
//
//        if (regex_search(test_default_num, match, re_default_num)) {
//            wcout << test_default_num << L" Default Number: " << replace_default_num(match) << endl;
//        }
//
//        if (regex_search(test_asmd, match, re_asmd)) {
//            wcout << test_asmd << L" ASMD: " << replace_asmd(match) << endl;
//        }
//
//        if (regex_search(test_positive_quantifier, match, re_positive_quantifier)) {
//            wcout << test_positive_quantifier << L" Positive Quantifier: " << replace_positive_quantifier(match) <<
//            endl;
//        }
//
//        if (regex_search(test_number, match, re_number)) {
//            wcout << test_number << L" Number: " << replace_number(match) << endl;
//        }
//
//        if (regex_search(test_range, match, re_range)) {
//                wcout << test_range << L" Range: " << replace_range(match) << endl;
//        }
//
//        if (regex_search(test_to_range, match, re_to_range)) {
//            wcout << test_to_range << L" To Range: " << replace_to_range(match) << endl;
//        }
//
//        return 0;
//}