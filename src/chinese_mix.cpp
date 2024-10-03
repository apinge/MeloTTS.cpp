#include "chinese_mix.h"
namespace melo {
    namespace chinese_mix {
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::strong_ordering& segment, std::shared_ptr<Tokenizer>& tokenized) {
            std::vector<int64_t> phones_list, tones_list;
            std::vector<int > word2ph;
            return { phones_list, tones_list, word2ph };
        }
    }

}