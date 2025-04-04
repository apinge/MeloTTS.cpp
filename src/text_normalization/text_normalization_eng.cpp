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
#include "text_normalization_eng.h"

#include <cmath>
#include <iostream>
#include <regex>
#include <unordered_map>

namespace text_normalization {

// normalize_numbers
const std::regex decimal_number_re(R"(([0-9]+\.[0-9]+))");
const std::regex number_re(R"(-?[0-9]+)");
const std::regex ordinal_re(R"([0-9]+(st|nd|rd|th))");
const std::regex comma_number_re(R"(\b\d{1,3}(,\d{3})+\b)");

const std::vector<std::string> belowTwenty = {
    "",    "one",    "two",    "three",    "four",     "five",    "six",     "seven",     "eight",    "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const std::vector<std::string> tens =
    {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const std::vector<std::string> thousands = {"", "thousand", "million"};
static std::string int_to_words_helper(int num) {
    if (num == 0)
        return "";
    if (num < 20)
        return belowTwenty[num] + " ";
    if (num < 100)
        return tens[num / 10] + (num % 10 ? " " + belowTwenty[num % 10] : "");
    return belowTwenty[num / 100] + " hundred" + (num % 100 ? " " + int_to_words_helper(num % 100) : "");
}
static std::string int_to_words(int num) {
    if (num == 0)
        return "zero";
    std::string result;
    int i = 0;
    while (num > 0) {
        if (num % 1000) {
            result = int_to_words_helper(num % 1000) + (thousands[i].empty() ? "" : " " + thousands[i]) +
                     (result.empty() ? "" : " " + result);
        }
        num /= 1000;
        i++;
    }
    return result;
}

static std::string decimal_to_words(double decimalPart) {
    if (decimalPart == 0)
        return "";
    std::ostringstream oss;
    oss << std::fixed << decimalPart;
    std::string decimalStr = oss.str().substr(2);
    std::string result = "point";
    for (char c : decimalStr) {
        result += " " + belowTwenty[c - '0'];
    }
    return result;
}
// we assume that if the string size exceeds 10, the number is likely too large and may cause errors with stod or stoi
inline bool is_number_too_large(const std::string& num_str) {
    return (num_str.size() > 10);
}

// We assume there are no negative numbers - the situation is too complex otherwise.
static std::string number_to_words(double num) {
    double x = std::floor(num);
    int intPart = static_cast<int>(x);
    double decimalPart = num - intPart;
    std::string words = int_to_words(intPart);
    std::string decimalWords = decimal_to_words(decimalPart);
    return words + (decimalWords.empty() ? "" : " " + decimalWords);
}

static std::string number_to_words(const std::string& s) {
    if (is_number_too_large(s)) {
        std::string res = "";
        for (const auto& ch : s) {
            if (ch == '.')
                res += "point ";
            else if (ch <= '9' && ch >= '0')
                res += (ch == '0') ? "zero " : belowTwenty[ch - '0'] + " ";
        }
        return res;
    }
    return number_to_words(std::stod(s));
}
// TODO: Fix secenario where 211st, 202nd, etc. are not converted correctly
static std::string convert_ordinal(int num) {
    static const std::unordered_map<int, std::string> ordinal_map = {
        {1, "first"},      {2, "second"},       {3, "third"},       {4, "fourth"},      {5, "fifth"},
        {6, "sixth"},      {7, "seventh"},      {8, "eighth"},      {9, "ninth"},       {10, "tenth"},
        {11, "eleventh"},  {12, "twelfth"},     {13, "thirteenth"}, {14, "fourteenth"}, {15, "fifteenth"},
        {16, "sixteenth"}, {17, "seventeenth"}, {18, "eighteenth"}, {19, "nineteenth"}, {20, "twentieth"},
        {30, "thirtieth"}, {40, "fortieth"},    {50, "fiftieth"},   {60, "sixtieth"},   {70, "seventieth"},
        {80, "eightieth"}, {90, "ninetieth"}};

    if (ordinal_map.contains(num)) {
        return ordinal_map.at(num);
    } else {
        // Handle cases like 21, 22, 23, etc.
        int tens = num / 10 * 10;  // Get the tens place
        int ones = num % 10;       // Get the ones place

        // Handle cases like 21st, 22nd, 23rd, 31st, 32nd, 33rd, etc.
        if (ordinal_map.contains(tens) && ordinal_map.contains(ones)) {
            return ordinal_map.at(tens) + " " + ordinal_map.at(ones);
        } else if (ordinal_map.contains(ones)) {
            return number_to_words(tens) + " " + ordinal_map.at(ones);
        }
        // For other cases, default to using number_to_words
        return number_to_words(num) + (ones == 1 ? "st" : ones == 2 ? "nd" : ones == 3 ? "rd" : "th");
    }
}

// TODO: Add __expand_currency
std::string normalize_numbers(const std::string& text) {
    std::string result = text;
    std::smatch match;
    // First, convert the commas to facilitate further conversion of decimals and other subsequent numbers.
    while (std::regex_search(result, match, comma_number_re)) {
        std::string no_commas = match.str(0);
        no_commas.erase(std::remove(no_commas.begin(), no_commas.end(), ','), no_commas.end());
        result.replace(match.position(0), match.length(0), no_commas);
    }
    while (std::regex_search(result, match, ordinal_re)) {
        int num = std::stoi(match.str(0));
        result.replace(match.position(0), match.length(0), convert_ordinal(num));
    }
    while (std::regex_search(result, match, decimal_number_re)) {
        result.replace(match.position(0), match.length(0), number_to_words(match.str(0)));
    }
    while (std::regex_search(result, match, number_re)) {
        result.replace(match.position(0), match.length(0), number_to_words(match.str(0)));
    }

    return result;
}

// expand_abbrevations
// List of (regular expression, replacement) pairs for abbreviations in English
const std::vector<std::pair<std::regex, std::string>> abbreviations_en = {
    {std::regex("\\bMrs\\.", std::regex_constants::icase), "misess"},
    {std::regex("\\bMr\\.", std::regex_constants::icase), "mister"},
    {std::regex("\\bDr\\.", std::regex_constants::icase), "doctor"},
    {std::regex("\\bSt\\.", std::regex_constants::icase), "saint"},
    {std::regex("\\bCo\\.", std::regex_constants::icase), "company"},
    {std::regex("\\bJr\\.", std::regex_constants::icase), "junior"},
    {std::regex("\\bMaj\\.", std::regex_constants::icase), "major"},
    {std::regex("\\bGen\\.", std::regex_constants::icase), "general"},
    {std::regex("\\bDrs\\.", std::regex_constants::icase), "doctors"},
    {std::regex("\\bRev\\.", std::regex_constants::icase), "reverend"},
    {std::regex("\\bLt\\.", std::regex_constants::icase), "lieutenant"},
    {std::regex("\\bHon\\.", std::regex_constants::icase), "honorable"},
    {std::regex("\\bSgt\\.", std::regex_constants::icase), "sergeant"},
    {std::regex("\\bCapt\\.", std::regex_constants::icase), "captain"},
    {std::regex("\\bEsq\\.", std::regex_constants::icase), "esquire"},
    {std::regex("\\bLtd\\.", std::regex_constants::icase), "limited"},
    {std::regex("\\bCol\\.", std::regex_constants::icase), "colonel"},
    {std::regex("\\bFt\\.", std::regex_constants::icase), "fort"}};
std::string expand_abbreviations(const std::string& text) {
    std::string result = text;
    for (const auto& [regex, replacement] : abbreviations_en) {
        result = std::regex_replace(result, regex, replacement);
    }
    return result;
}

// expand_time_english
const std::regex time_re(R"(((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3])):([0-5][0-9])\s*(a\.m\.|am|pm|p\.m\.|a\.m|p\.m)?)",
                         std::regex_constants::icase);

static std::string _expand_time_english(const std::smatch& match) {
    int hour = std::stoi(match.str(1));
    bool past_noon = hour >= 12;
    std::string result;

    if (hour > 12) {
        hour -= 12;
    } else if (hour == 0) {
        hour = 12;
        past_noon = true;
    }
    result += number_to_words(hour);

    int minute = std::stoi(match.str(6));
    if (minute > 0) {
        if (minute < 10) {
            result += " oh";
        }
        result += " " + number_to_words(minute);
    }

    std::string am_pm = match.str(7);
    if (am_pm.empty()) {
        result += past_noon ? " p m " : " a m ";
    } else {
        for (char c : am_pm) {
            if (c != '.') {
                result += " ";
                result += c;
            }
        }
        result += " ";
    }
    return result;
}

std::string expand_time_english(const std::string& text) {
    std::string result = text;
    std::smatch match;
    while (std::regex_search(result, match, time_re)) {
        // std::cout << "match\n";
        // for (std::size_t i = 0; i < match.size(); ++i)
        //     std::cout << i << ": " << match[i] << '\n';
        result.replace(match.position(0), match.length(0), _expand_time_english(match));
    }
    return result;
}

}  // namespace text_normalization