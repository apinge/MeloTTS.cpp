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
#ifndef ENGLISH_H
#define ENGLISH_H
#include "cmudict.h"
#include "language_module_base.h"
#include "mini-bart-g2p/mini-bart-g2p.h"
namespace melo {
class English : public AbstractLanguageModule {
public:
    English(std::unique_ptr<ov::Core>& core_ptr, const std::filesystem::path& data_folder);
    virtual ~English() = default;
    // Grapheme to Phoneme conversion
    virtual std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p(
        const std::string& segment,
        std::shared_ptr<OpenVinoTokenizer>& tokenizer) override;
    virtual std::string text_normalize(const std::string& text) override;
    virtual inline int64_t symbol_to_id(const std::string& symbol) override {
        return symbol_to_id_mp.at(symbol);
    }
    virtual inline std::string get_language_name() {
        return "EN";
    };

private:
    std::shared_ptr<CMUDict> cmudict;
    std::shared_ptr<MiniBartG2P> bart_g2p;
    const std::unordered_map<std::string, int> symbol_to_id_mp = {
        {"_", 0},     {"\"", 1},    {"(", 2},     {")", 3},    {"*", 4},    {"/", 5},    {":", 6},     {"AA", 7},
        {"E", 8},     {"EE", 9},    {"En", 10},   {"N", 11},   {"OO", 12},  {"Q", 13},   {"V", 14},    {"[", 15},
        {"\\", 16},   {"]", 17},    {"^", 18},    {"a", 19},   {"a:", 20},  {"aa", 21},  {"ae", 22},   {"ah", 23},
        {"ai", 24},   {"an", 25},   {"ang", 26},  {"ao", 27},  {"aw", 28},  {"ay", 29},  {"b", 30},    {"by", 31},
        {"c", 32},    {"ch", 33},   {"d", 34},    {"dh", 35},  {"dy", 36},  {"e", 37},   {"e:", 38},   {"eh", 39},
        {"ei", 40},   {"en", 41},   {"eng", 42},  {"er", 43},  {"ey", 44},  {"f", 45},   {"g", 46},    {"gy", 47},
        {"h", 48},    {"hh", 49},   {"hy", 50},   {"i", 51},   {"i0", 52},  {"i:", 53},  {"ia", 54},   {"ian", 55},
        {"iang", 56}, {"iao", 57},  {"ie", 58},   {"ih", 59},  {"in", 60},  {"ing", 61}, {"iong", 62}, {"ir", 63},
        {"iu", 64},   {"iy", 65},   {"j", 66},    {"jh", 67},  {"k", 68},   {"ky", 69},  {"l", 70},    {"m", 71},
        {"my", 72},   {"n", 73},    {"ng", 74},   {"ny", 75},  {"o", 76},   {"o:", 77},  {"ong", 78},  {"ou", 79},
        {"ow", 80},   {"oy", 81},   {"p", 82},    {"py", 83},  {"q", 84},   {"r", 85},   {"ry", 86},   {"s", 87},
        {"sh", 88},   {"t", 89},    {"th", 90},   {"ts", 91},  {"ty", 92},  {"u", 93},   {"u:", 94},   {"ua", 95},
        {"uai", 96},  {"uan", 97},  {"uang", 98}, {"uh", 99},  {"ui", 100}, {"un", 101}, {"uo", 102},  {"uw", 103},
        {"v", 104},   {"van", 105}, {"ve", 106},  {"vn", 107}, {"w", 108},  {"x", 109},  {"y", 110},   {"z", 111},
        {"zh", 112},  {"zy", 113},  {"~", 114},   {"¡", 115},  {"¿", 116},  {"æ", 117},  {"ç", 118},   {"ð", 119},
        {"ø", 120},   {"ŋ", 121},   {"œ", 122},   {"ɐ", 123},  {"ɑ", 124},  {"ɒ", 125},  {"ɔ", 126},   {"ɕ", 127},
        {"ə", 128},   {"ɛ", 129},   {"ɜ", 130},   {"ɡ", 131},  {"ɣ", 132},  {"ɥ", 133},  {"ɦ", 134},   {"ɪ", 135},
        {"ɫ", 136},   {"ɬ", 137},   {"ɭ", 138},   {"ɯ", 139},  {"ɲ", 140},  {"ɵ", 141},  {"ɸ", 142},   {"ɹ", 143},
        {"ɾ", 144},   {"ʁ", 145},   {"ʃ", 146},   {"ʊ", 147},  {"ʌ", 148},  {"ʎ", 149},  {"ʏ", 150},   {"ʑ", 151},
        {"ʒ", 152},   {"ʝ", 153},   {"ʲ", 154},   {"ˈ", 155},  {"ˌ", 156},  {"ː", 157},  {"̃", 158},    {"̩", 159},
        {"β", 160},   {"θ", 161},   {"ᄀ", 162},  {"ᄁ", 163}, {"ᄂ", 164}, {"ᄃ", 165}, {"ᄄ", 166},  {"ᄅ", 167},
        {"ᄆ", 168},  {"ᄇ", 169},  {"ᄈ", 170},  {"ᄉ", 171}, {"ᄊ", 172}, {"ᄋ", 173}, {"ᄌ", 174},  {"ᄍ", 175},
        {"ᄎ", 176},  {"ᄏ", 177},  {"ᄐ", 178},  {"ᄑ", 179}, {"ᄒ", 180}, {"ᅡ", 181},  {"ᅢ", 182},   {"ᅣ", 183},
        {"ᅤ", 184},   {"ᅥ", 185},   {"ᅦ", 186},   {"ᅧ", 187},  {"ᅨ", 188},  {"ᅩ", 189},  {"ᅪ", 190},   {"ᅫ", 191},
        {"ᅬ", 192},   {"ᅭ", 193},   {"ᅮ", 194},   {"ᅯ", 195},  {"ᅰ", 196},  {"ᅱ", 197},  {"ᅲ", 198},   {"ᅳ", 199},
        {"ᅴ", 200},   {"ᅵ", 201},   {"ᆨ", 202},   {"ᆫ", 203},  {"ᆮ", 204},  {"ᆯ", 205},  {"ᆷ", 206},   {"ᆸ", 207},
        {"ᆼ", 208},   {"ㄸ", 209},  {"!", 210},   {"?", 211},  {"…", 212},  {",", 213},  {".", 214},   {"'", 215},
        {"-", 216},   {"SP", 217},  {"UNK", 218}};
};
}  // namespace melo
#endif  // ENGLISH_H