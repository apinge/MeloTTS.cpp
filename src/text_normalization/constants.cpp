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
#include <cctype>
#include <cwchar>  // 用于宽字符处理
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>

#include "constant.h"

#ifdef _WIN32
#include <iostream>
#include <locale>
#define NOGDI
#define NOCRYPT
#include <windows.h>
#endif
namespace text_normalization {
std::unordered_map<wchar_t, wchar_t> F2H_ASCII_LETTERS;
std::unordered_map<wchar_t, wchar_t> H2F_ASCII_LETTERS;
std::unordered_map<wchar_t, wchar_t> F2H_DIGITS;
std::unordered_map<wchar_t, wchar_t> H2F_DIGITS;
std::unordered_map<wchar_t, wchar_t> F2H_PUNCTUATIONS;
std::unordered_map<wchar_t, wchar_t> H2F_PUNCTUATIONS;
std::unordered_map<wchar_t, wchar_t> F2H_SPACE;
std::unordered_map<wchar_t, wchar_t> H2F_SPACE;
// 初始化字符映射
void initialize_constant_maps() {
    // ASCII 字母 全角 -> 半角
    for (wchar_t ch = L'a'; ch <= L'z'; ++ch) {
        F2H_ASCII_LETTERS[ch + 65248] = ch;
        H2F_ASCII_LETTERS[ch] = ch + 65248;
    }
    for (wchar_t ch = L'A'; ch <= L'Z'; ++ch) {
        F2H_ASCII_LETTERS[ch + 65248] = ch;
        H2F_ASCII_LETTERS[ch] = ch + 65248;
    }

    // 数字字符 全角 -> 半角
    for (wchar_t ch = L'0'; ch <= L'9'; ++ch) {
        F2H_DIGITS[ch + 65248] = ch;
        H2F_DIGITS[ch] = ch + 65248;
    }

    // 标点符号 全角 -> 半角
    std::wstring punctuations = L"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    for (wchar_t ch : punctuations) {
        F2H_PUNCTUATIONS[ch + 65248] = ch;
        H2F_PUNCTUATIONS[ch] = ch + 65248;
    }

    // 空格 全角 -> 半角
    F2H_SPACE[L'\u3000'] = L' ';
    H2F_SPACE[L' '] = L'\u3000';
}

// 将全角字符转换为半角
std::wstring fullwidth_to_halfwidth(const std::wstring& input) {
    std::wstring result;
    for (wchar_t ch : input) {
        if (F2H_ASCII_LETTERS.count(ch)) {
            result += F2H_ASCII_LETTERS[ch];
        } else if (F2H_DIGITS.count(ch)) {
            result += F2H_DIGITS[ch];
        } else if (F2H_PUNCTUATIONS.count(ch)) {
            result += F2H_PUNCTUATIONS[ch];
        } else if (F2H_SPACE.count(ch)) {
            result += F2H_SPACE[ch];
        } else {
            result += ch;  // 如果没有匹配，保持原字符
        }
    }
    return result;
}

// 将半角字符转换为全角
std::wstring halfwidth_to_fullwidth(const std::wstring& input) {
    std::wstring result;
    for (wchar_t ch : input) {
        if (H2F_ASCII_LETTERS.count(ch)) {
            result += H2F_ASCII_LETTERS[ch];
        } else if (H2F_DIGITS.count(ch)) {
            result += H2F_DIGITS[ch];
        } else if (H2F_PUNCTUATIONS.count(ch)) {
            result += H2F_PUNCTUATIONS[ch];
        } else if (H2F_SPACE.count(ch)) {
            result += H2F_SPACE[ch];
        } else {
            result += ch;  // 如果没有匹配，保持原字符
        }
    }
    return result;
}

// 正则表达式匹配非拼音的汉字字符串（根据支持 UCS4 的不同情况）
std::wregex RE_NSW(L"[^\\u3007\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]+");

}  // namespace text_normalization

// int main() {
//#ifdef _WIN32
//    SetConsoleOutputCP(CP_UTF8);
//    // 使用系统默认区域设置
//    std::wcout.imbue(std::locale(""));
//#endif
//    // 初始化字符映射
//    initialize_constant_maps();
//
//    // 示例: 全角转半角
//    std::wstring input_fullwidth = L"ＡＢＣ１２３！＄％";
//    std::wstring result_halfwidth = fullwidth_to_halfwidth(input_fullwidth);
//    std::wcout << L"全角转半角: " << result_halfwidth << std::endl;
//
//    // 示例: 半角转全角
//    std::wstring input_halfwidth = L"ABC123!$%";
//    std::wstring result_fullwidth = halfwidth_to_fullwidth(input_halfwidth);
//    std::wcout << L"半角转全角: " << result_fullwidth << std::endl;
//
//    return 0;
//}