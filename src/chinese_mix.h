#pragma once
#ifndef CHINESE_MIX_H
#define CHINESE_MIX_H
#include <memory>
#include "tokenizer.h"
namespace melo {
    namespace chinese_mix {
        std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::strong_ordering& segment, std::shared_ptr<Tokenizer>& tokenized);
    }
    
}
#endif