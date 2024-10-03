// last modified 

#ifndef HANZ2PINY_H
#define HANZ2PINY_H
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
/**
 * This class is from https://github.com/yangyangwithgnu/hanz2piny
 * and is used to convert pure Chinese characters to UTF-8 Pinyin strings.
 *
 * Note:
 * 1. This library uses tone 5 for neutral tones, e.g., “了” => le5.
 * 2. The handling of polyphonic characters is limited, for instance, the character “不”
 *    in “不要” and “不能” is always marked with tone 4.
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

        /**
         * @brief Directly converts a UTF-8 string to a Pinyin string.
         */
        inline std::string to_pinyin_str_from_utf8(const std::string &input, bool camel = false, const Hanz2Piny::Polyphone polyphone = Hanz2Piny::noname) const  {
            std::vector<std::pair<bool, std::vector<std::string>>> pinyin_list = toPinyinFromUtf8(input);
            std::stringstream ss;
            for (const auto& e : pinyin_list) {
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
                        auto pinyin = pinyin_list[1];
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
                return ss.str();
        }


    private:
        static const Unicode begin_hanzi_unicode_, end_hanzi_unicode_;
        static const char* pinyin_list_with_tone_[];
};

#endif

