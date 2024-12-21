#include "english.h"

void printVec(const auto& vec, const std::string& vecName) {
    std::cout << vecName << ":\n";
    for (const auto& row : vec) {

       std::cout << row << " ";
     

    }
    std::cout << std::endl;
    }

namespace melo{
    // Constructor
    English::English(const std::filesystem::path& data_folder) {
        //english pronounciation dict
        auto cmudict_path = data_folder / "cmudict_cache.txt";


        if (!std::filesystem::exists(cmudict_path)) {
            std::cerr << "[ERROR] ChineseMix::file does not exists: " << std::filesystem::absolute(cmudict_path) << "\n";
        }

        cmudict = std::make_shared<CMUDict>(cmudict_path.string());
        std::cout << "[INFO] Init English language Module Succeed!\n";
    }

	std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> English::g2p(const std::string& sentence, std::shared_ptr<OpenVinoTokenizer>& tokenizer) {
        std::vector<std::string> phones_list{ "_" };
        std::vector<int64_t> tones_list{ 0 };
        std::vector<int> word2ph{ 1 };

        std::vector<std::string> tokenized = tokenizer->word_segment(const_cast<std::string&>(sentence));
        std::vector<std::vector<std::string>> ph_groups;
        //remove ## in suffix
        for (auto& token : tokenized) {
            if (token.front() == '#') {
                if (!ph_groups.size()) {
                    std::cerr << "[ERROR] English::g2p: Suffix should has Prefix\n";
                    continue;
                }
                ph_groups.back().emplace_back(token.substr(2));
            }
            else
                ph_groups.push_back({token});
        }
        bool cmudict_unfound = false;
        for (auto& group : ph_groups) {
            int phone_len = 0;
            int word_len = group.size();
            std::string w = std::accumulate(group.begin(), group.end(), std::string{});
            auto syllables = cmudict->find(w);
#ifdef MELO_DEBUG
            if (syllables.has_value()) {
                for (std::cout << "token:" << token << ":"; auto & vec:syllables.value().get()) {
                    for (auto& x : vec)
                        std::cout << x << ' ';
                    std::cout << std::endl;
                }
            }
#endif
            // if not has value
            if (syllables.has_value()) {
                auto [phones, tones] = refine_syllables(syllables.value().get());
                phone_len += phones.size();
                phones_list.insert(phones_list.end(), phones.begin(), phones.end());
                tones_list.insert(tones_list.end(), tones.begin(), tones.end());
            }
            else {
                cmudict_unfound = true;
                std::cout << "[WARNNING] cmudict cannot find:" << w << " in " << sentence << std::endl;
                continue;
            }
            //std::cout << "phone_len" << phone_len << ' ' << "word_len" << word_len << std::endl;
            auto aaa = distribute_phone(phone_len, word_len);
            //for (std::cout << "aaa:"; auto & x:aaa) std::cout << x << std::endl;
            std::ranges::copy(aaa.begin(), aaa.end(), std::back_inserter(word2ph));
        }
        // workaround for abbreviation
        // We consider English words with a length of <= 5 as abbreviations if their existing tokens are not found in cmudict
        //if (word.length() <= 5 && cmudict_unfound) {
        //    // some abbreviation may be slit to sevaral tokens (some token can be found in cmudict) . clean them all first.
        //    phone_len = 0;
        //    phones_list.clear();
        //    tones_list.clear();
        //    for (const char& ch : word) {
        //        auto syllables = cmudict->find(std::string(1, ch));
        //        if (syllables.has_value()) {
        //            auto [phones, tones] = refine_syllables(syllables.value().get());
        //            phone_len += phones.size();
        //            phones_list.insert(phones_list.end(), phones.begin(), phones.end());
        //            tones_list.insert(tones_list.end(), tones.begin(), tones.end());
        //        }
        //    }
        //    if (phones_list.size())
        //        std::cout << "[INFO] " << word << " is treated as an abbravation, where each letter is pronounced individually.\n";
        //}
       
        phones_list.emplace_back("_");
        tones_list.emplace_back(0);
        word2ph.emplace_back(1);

        //printVec(phones_list, "phones_list");
        //printVec(tones_list, "tones_list");
        //printVec(word2ph, "word2ph");

        return { phones_list, tones_list, word2ph };
	}
    // TODO: implement the function
    std::string English::text_normalize(const std::string& text) {
        std::string norm_text = text;
        std::for_each(norm_text.begin(), norm_text.end(), [](auto& ch) {
            if (ch <= 'Z' && ch >= 'A')
                ch = ch + 'a' - 'A';
            });
        return norm_text;
    }

}