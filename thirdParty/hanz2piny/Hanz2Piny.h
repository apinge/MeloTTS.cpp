// last modified 

#ifndef HANZ2PINY_H
#define HANZ2PINY_H
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <unordered_set>
/**
 * This class is from https://github.com/yangyangwithgnu/hanz2piny
 * and is used to convert pure Chinese characters to UTF-8 Pinyin strings.
 *  This class has been modified to meet the project's requirements.  Modifications made by qiu, tong(tong.qiu@intel.com)
 *
 * Note:
 * 1. This library uses tone 5 for neutral tones, e.g., “了” => le5.
 * 2. The handling of polyphonic characters is limited, for instance, the character “不”
 *    in “不要” and “不能” is always marked with tone 4.
 * 3. Changed ü from the original representation u: to v, This is to ensure consistency with the pypinyin library.
 * 4. To align with pypinyin, cases where the initias are 'y' and 'w' are not taken into account.
 */
class Hanz2Piny 
{
    public:
        typedef unsigned short Unicode;
        enum Polyphone {all, name, noname};

    public:
        Hanz2Piny ();
        
        bool isHanziUnicode (const Unicode unicode) const;
        std::vector<std::string> toPinyinFromUnicode (const Unicode hanzi_unicode, const bool with_tone = true) const;
        
        bool isUtf8 (const std::string& s) const;
        std::vector<std::pair<bool, std::vector<std::string>>> toPinyinFromUtf8 ( const std::string& s,
                                                                                  const bool with_tone = true,
                                                                                  const bool replace_unknown = false,
                                                                                  const std::string& replace_unknown_with = "" ) const;
        
        bool isUtf8File (const std::string& file_path) const;
        
        bool isStartWithBom (const std::string& s) const;
        //Change ü from the original representation u: to v, This is to ensure consistency with the pypinyin library.
        inline std::string replaceUWithV(const std::string& input) const  {
            std::string result = input;
            std::string target = "u:";   
            std::string replacement = "v"; 
            size_t pos = 0;
            while ((pos = result.find(target, pos)) != std::string::npos) {
                result.replace(pos, target.length(), replacement);
                pos += replacement.length(); 
            }

            return result;
        }
        // @brief This function returns the initials and finals,e.g. bian1 -> "b" + "ian1"
        // This function is essentially the same as the pypinyin.lazy_pinyin function, but it retains the initials 'y' and 'w' 

        inline std::pair<std::string,std::string> split_initials_finals(const std::string& pinyin) const {
            int n = pinyin.length();
            if(n==0) return{};
            //check compound_initials
            if (n > 2 && compound_initials.contains(pinyin.substr(0, 2))) {
                return { pinyin.substr(0, 2) , replaceUWithV(pinyin.substr(2))};
            }
            else if (simple_initials.contains(pinyin.front())) {
                return {pinyin.substr(0,1), replaceUWithV(pinyin.substr(1))};
            }
            else {
                //有些字没有声母 比如 玉 鹅
                return { "", replaceUWithV(pinyin)};
            }
            return {};
        }
        /*
        * @brief This function returns the initials and finals, corresponding to the Python function of the same name.
        * https://github.com/zhaohb/MeloTTS-OV/blob/main/melo/text/chinese.py#L80
        */
        inline std::pair<std::vector<std::string>, std::vector<std::string>> _get_initials_finals(const std::string& input) const {
            std::vector<std::string> initials, finals;
            std::vector<std::pair<bool, std::vector<std::string>>> pinyin_list_list = toPinyinFromUtf8(input);
            for (const auto& e : pinyin_list_list) {
                const bool ok = e.first;
                auto pinyin_list = e.second;
                if (pinyin_list.size() == 1) {          // 单音字
                    auto pinyin = pinyin_list[0];
                    //std::cout << pinyin << std::endl;
                    auto [c,v] = split_initials_finals(pinyin);
                    //std::cout << c <<' '<< v << std::endl;
                    if(c=="y"||c=="w") c =""; //cases where the initias are 'y' and 'w' are not taken into account.
                    initials.emplace_back(c);
                    finals.emplace_back(v);
                }
                else if (pinyin_list.size() > 1) {    // 多音字

                    auto pinyin = pinyin_list[0];
                   // std::cout <<  pinyin<< std::endl;
                    auto [c, v] = split_initials_finals(pinyin);
                    //std::cout << c << ' ' << v << std::endl;
                    if (c == "y" || c == "w") c = "";//cases where the initias are 'y' and 'w' are not taken into account.
                    initials.emplace_back(c);
                    finals.emplace_back(v);
                }
                else {                                // 该 UTF-8 编码并无对应汉字，相应也就不存在拼音
                    std::cerr << "The given UTF-8 encoding does not correspond to any Chinese character, and therefore, no Pinyin exists for it.";
                }
            }
            return { initials, finals };
        }
        /**
         * @brief Directly converts a UTF-8 string to a Pinyin string.
         */
        inline std::string to_pinyin_str_from_utf8(const std::string &input, bool camel = false, const Hanz2Piny::Polyphone polyphone = Hanz2Piny::noname) const  {
            std::vector<std::pair<bool, std::vector<std::string>>> pinyin_list_list = toPinyinFromUtf8(input);
            std::stringstream ss;
            for (const auto& e : pinyin_list_list) {
                const bool ok = e.first;
                auto pinyin_list = e.second;

                if (pinyin_list.size() == 1) {          // 单音字
                    auto pinyin = pinyin_list[0];
                    if (ok && camel) {
                        pinyin[0] = (char)toupper(pinyin[0]);
                    }
                    ss << pinyin;
                }
                else if (pinyin_list.size() > 1) {    // 多音字
                    switch (polyphone) {
                    case Hanz2Piny::all: {
                        ss << '<';
                        for (auto pinyin : pinyin_list) {
                            if (ok && camel) {
                                pinyin[0] = (char)toupper(pinyin[0]);
                            }
                            ss << pinyin << ", ";
                        }
                        ss << "\b\b>";
                    }
                                       break;

                    case Hanz2Piny::name: {
                        auto pinyin = pinyin_list[0];
                        if (ok && camel) {
                            pinyin[0] = (char)toupper(pinyin[0]);
                        }
                        ss << pinyin;
                    }
                                        break;

                    case Hanz2Piny::noname: {
                        auto pinyin = pinyin_list[0];
                        if (ok && camel) {
                            pinyin[0] = (char)toupper(pinyin[0]);
                        }
                        ss << pinyin;
                    }
                                          break;
                    }
                }
                else {                                // 该 UTF-8 编码并无对应汉字，相应也就不存在拼音
                    std::cerr<< "The given UTF-8 encoding does not correspond to any Chinese character, and therefore, no Pinyin exists for it.";
                }
            }
                return replaceUWithV(ss.str());
        }


    private:
        static const Unicode begin_hanzi_unicode_, end_hanzi_unicode_;
        static const char* pinyin_list_with_tone_[];
        // 声母表 Refer to https://github.com/mozillazg/python-pinyin/blob/master/pypinyin/style/_constants.py
        static const std::unordered_set<char> simple_initials;//单声母 这些声母由一个字母组成
        static const std::unordered_set<std::string> compound_initials;// 复合声母, 在汉语拼音中，它们表示的是由两个字母组合而成的声母
};

#endif

