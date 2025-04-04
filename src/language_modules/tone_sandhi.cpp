/**
 * Copyright      2024    Tong Qiu (tong.qiu@intel.com)   Haofan Rong (haofan.rong@hp.com)
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
#include "tone_sandhi.h"

#include <algorithm>

#include "utils.h"

namespace melo {
namespace ToneSandhi {
const std::unordered_set<char> punctuations = {
    ',',
    '.',
    '!',
    '?',
    ';',
    '-',
    '\''};  // After filtering, only these punctuation marks are accepted. same as in chinese_ix

std::vector<std::pair<std::string, std::string>> pre_merge_for_modify(
    std::vector<std::pair<std::string, std::string>>& seg) {
    auto seg1 = _merge_yi(seg);
    return _merge_chinese_patterns(seg1);
}
/**
 * @brief This function combines the logic of three Python functions:
 * _merge_bu, _merge_er, and _merge_reduplication.
 */
std::vector<std::pair<std::string, std::string>> _merge_chinese_patterns(
    std::vector<std::pair<std::string, std::string>>& seg) {
    std::vector<std::pair<std::string, std::string>> new_seg;
#ifdef MELO_DEBUG
    std::cout << "origin seg\n";
    for (const auto& [word, _] : seg) {
        std::cout << word << '|';
    }
    std::cout << std::endl;
#endif

    for (auto& [word, pos] : seg) {
        //_merge_reduplication and _merge_bu
        // Here, two consecutive punctuation marks are prevented from being combined within the same word segmentation.
        if (new_seg.size() &&
            (word == new_seg.back().first && !punctuations.contains(word.front()) || new_seg.back().first == "不"))
            new_seg.back().first += word;
        else if (new_seg.size() && word == "儿")  //_merge_er
            new_seg.back().first += "儿";
        else
            new_seg.emplace_back(std::move(word), std::move(pos));
    }
    if (new_seg.size() && new_seg.back().first == "不")
        new_seg.back().second = "d";
#ifdef MELO_DEBUG
    std::cout << "_merge_chinese_patterns:";
    for (const auto& [word, _] : new_seg)
        std::cout << word << '|';
    std::cout << std::endl;
#endif
    return new_seg;
}
// merge "不" and the word behind it
// if don't merge, "不" sometimes appears alone according to jieba, which may occur sandhi error
// std::vector<std::pair<std::string, std::string>> _merge_bu(std::vector<std::pair<std::string, std::string>>& seg) {
//    std::vector<std::pair<std::string, std::string>> new_seg;
//    std::string last_word;
//    for (auto& [word, pos] : seg) {
//        if (last_word == "不")
//            word = last_word+word;
//        if(word!= "不")
//            new_seg.emplace_back(word,pos);
//         last_word = word;
//    }
//    if(last_word == "不")
//        new_seg.emplace_back(last_word,"d");
//    return new_seg;
//}
/* function 1: merge "一" and reduplication words in it's left and right, e.g. "听","一","听" ->"听一听"
  function 2: merge single  "一" and the word behind it
  if don't merge, "一" sometimes appears alone according to jieba, which may occur sandhi error
  e.g.
input seg : [('听', 'v'), ('一', 'm'), ('听', 'v')]
output seg : [['听一听', 'v']]*/
std::vector<std::pair<std::string, std::string>> _merge_yi(std::vector<std::pair<std::string, std::string>>& seg) {
    std::vector<std::pair<std::string, std::string>> new_seg;
    int n = seg.size();
    // function 1 and function2
    bool if_continue = false;  // skip second "听", of "听一听"
    for (int i = 0; i < n; ++i) {
        if (if_continue) {
            if_continue = false;
            continue;
        }
        auto& [word, pos] = seg[i];
        if (i >= 1 && word == "一" && i + 1 < n && seg[i - 1].first == seg[i + 1].first && seg[i - 1].second == "v") {
            new_seg.back().first += "一" + seg[i + 1].first;
            if_continue = true;
        } else if (new_seg.size() && new_seg.back().first == "一") {
            new_seg.back().first += word;
        } else
            new_seg.emplace_back(word, pos);
    }
#ifdef MELO_DEBUG
    std::cout << "_merge_yi:";
    for (const auto& [word, _] : new_seg)
        std::cout << word << ' ';
    std::cout << std::endl;
#endif
    return new_seg;
}

// std::vector<std::pair<std::string, std::string>> _merge_er(std::vector<std::pair<std::string, std::string>>& seg) {
//     std::vector<std::pair<std::string, std::string>> new_seg;
//     int n = seg.size();
//     for (auto& [word, pos] : seg) {
//         if (new_seg.size() && word == "儿") {
//             new_seg.back().first += "儿";
//         }
//         else
//             new_seg.emplace_back(std::move(word),std::move(pos));
//     }
//     return new_seg;
// }
// std::vector<std::pair<std::string, std::string>> _merge_reduplication(std::vector<std::pair<std::string,
// std::string>>& seg) {
//     std::vector<std::pair<std::string, std::string>> new_seg;
//     int n = seg.size();
//     for (auto& [word, pos]:seg) {
//         if (new_seg.size() && word == new_seg.back().first)
//             new_seg.back().first += word;
//         else
//             new_seg.emplace_back(std::move(word),std::move(pos));
//     }
//     return new_seg;
// }
/**
 * Adjusts the tones of Chinese characters based on the given word and tag (part of speech).
 *
 * @param word The input Chinese word whose tones need to be adjusted.
 * @param tag The part of speech associated with the input word, which influences the tone modification.
 * @param sub_finals: 韵母
 */
void modified_tone(const std::string& word,
                   const std::string& tag,
                   const std::shared_ptr<cppjieba::Jieba>& jieba,
                   std::vector<std::string>& sub_finals) {
    // 此处需要 汉语分字 假设这里进入的是utf-8纯汉字无标点
    std::vector<std::string> chinese_characters = split_utf8_chinese(word);
    _bu_sandhi(chinese_characters, sub_finals);
    _yi_sandhi(chinese_characters, sub_finals);
    _neural_sandhi(word, chinese_characters, tag, jieba, sub_finals);
    _three_sandhi(word, chinese_characters, jieba, sub_finals);
}
void _bu_sandhi(const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals) {
    if (chinese_characters.size() == 3 && chinese_characters[1] == "不") {
        sub_finals[1].back() = '5';
        return;
    }
    for (size_t i = 0; i < chinese_characters.size(); i++) {
        auto& c = chinese_characters[i];
        if (c == "不" && i != chinese_characters.size() - 1 && sub_finals[i + 1].back() == '4') {
            sub_finals[i].back() = '2';
        }
    }
}
void _yi_sandhi(const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals) {
    if (is_numeric(chinese_characters)) {
        return;
    }
    if (chinese_characters.size() == 3 && chinese_characters[1] == "一" &&
        chinese_characters[0] == chinese_characters[2]) {
        sub_finals[1].back() = '5';
        return;
    }
    if (chinese_characters[0] == "第" && chinese_characters[1] == "一")  // word.startswith("第一"):
        return;

    for (size_t i = 0; i < chinese_characters.size(); i++) {
        auto& c = chinese_characters[i];
        if (c == "一" && i != chinese_characters.size() - 1) {
            if ((sub_finals[i + 1].back()) == '4') {
                sub_finals[i].back() = '2';
            } else {
                sub_finals[i].back() = '1';
            }
        }
    }
}
void _neural_sandhi(const std::string& word,
                    const std::vector<std::string>& chinese_characters,
                    const std::string& pos,
                    const std::shared_ptr<cppjieba::Jieba>& jieba,
                    std::vector<std::string>& sub_finals) {
    int n = chinese_characters.size();
    if (n < 2 || must_not_neural_tone_words.contains(word))
        return;

    std::string temp = chinese_characters[n - 2] + chinese_characters.back();

    if (must_neural_tone_words.contains(temp)) {
        sub_finals.back().back() = '5';
    }
    for (size_t i = 1; i < n; i++) {
        auto& c = chinese_characters[i];
        if (chinese_characters[i] == chinese_characters[i - 1] && (pos[0] == 'n' || pos[0] == 'v' || pos[0] == 'a')) {
            sub_finals[i].back() = '5';
        }
    }
    const static std::unordered_set<std::string> nenuralCharSet = {"吧", "呢", "啊", "呐", "噻", "嘛", "吖", "嗨",
                                                                   "呐", "哦", "哒", "额", "滴", "哩", "哟", "喽",
                                                                   "啰", "耶", "喔", "诶", "的", "地", "得"};
    static const std::unordered_set<std::string> st1 = {"上", "下", "进", "出", "回", "过", "起", "开"};
    static const std::unordered_set<std::string> st2 = {"几", "有", "两", "半", "多", "各", "整", "每", "做", "是"};
    if (nenuralCharSet.contains(chinese_characters.back())) {
        sub_finals.back().back() = '5';
    } else if ((chinese_characters.back() == "们" || chinese_characters.back() == "字") &&
               (pos[0] == 'n' || pos[0] == 'r')) {
        sub_finals.back().back() = '5';
    } else if ((chinese_characters.back() == "上" || chinese_characters.back() == "下" ||
                chinese_characters.back() == "里")  // e.g. 桌上, 地下, 家里
               && (pos[0] == 's' || pos[0] == 'l' || pos[0] == 'f')) {
        sub_finals.back().back() = '5';
    } else if ((chinese_characters.back() == "来" || chinese_characters.back() == "去") &&
               st1.contains(chinese_characters[n - 2])) {  // e.g. 上来, 下去
        sub_finals.back().back() = '5';
    }
    auto it = std::find(chinese_characters.begin(), chinese_characters.end(), "个");
    // "个"做量词
    if (it != chinese_characters.end() && it != chinese_characters.begin() &&
        (is_numeric(*(it - 1)) || st2.contains(*(it - 1)))) {
        sub_finals[it - chinese_characters.begin()].back() = '5';
    }
    auto slices = _split_word(word, jieba);
    if (slices == 2) {
        auto temp = chinese_characters.front() + chinese_characters[1];
        if (must_neural_tone_words.contains(temp)) {
            sub_finals[slices - 1].back() = '5';
        }
    }
}
auto _all_tone_three = [](const std::vector<std::string>& finals) {
    return std::all_of(finals.begin(), finals.end(), [](const std::string& f) {
        return f.back() == '3';
    });
};
void _three_sandhi(const std::string& word,
                   const std::vector<std::string>& chinese_characters,
                   const std::shared_ptr<cppjieba::Jieba>& jieba,
                   std::vector<std::string>& sub_finals) {
    if (chinese_characters.size() == 2 && sub_finals[0].back() == '3' && sub_finals[1].back() == '3') {
        sub_finals[0].back() = '2';
        return;
    }
    if (chinese_characters.size() == 3) {
        size_t slices = _split_word(word, jieba);
        if (_all_tone_three(sub_finals)) {
            if (slices == 2) {
                sub_finals[0].back() = sub_finals[1].back() = '2';
            } else if (slices == 1) {
                sub_finals[1].back() = '2';
            }
        } else {
            if (sub_finals[1].back() != '3') {
                return;
            }
            if (sub_finals[0].back() != '3' && sub_finals[2].back() != '3') {
                return;
            }
            if (sub_finals[0].back() == '3') {
                sub_finals[0].back() = '2';
                return;
            }
            if (slices == 1) {
                sub_finals[1].back() = '2';
            }
        }
    }

    if (chinese_characters.size() == 4) {
        if (sub_finals[0].back() == '3' && sub_finals[1].back() == '3') {
            sub_finals[0].back() = '2';
        }
        if (sub_finals[2].back() == '3' && sub_finals[3].back() == '3') {
            sub_finals[2].back() = '2';
        }
    }
}
}  // namespace ToneSandhi
}  // namespace melo