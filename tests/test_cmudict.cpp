#define CRT_
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


#include <iostream>
#include <cassert>
#include "CMUDict.h"

auto print_result = [](const auto& result) {
    if (result.has_value()) {
        const auto& vec = result.value().get();  // 获取引用
        for (const auto& inner_vec : vec) {
            for (const auto& str : inner_vec) {
                std::cout << str << ".";
            }
            std::cout << ",";
        }
    }
    else {
        std::cout << "Key not found" << std::endl;
    }
    std::cout << std::endl;
    };

int main() {
    melo::CMUDict dict("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\cmudict_cache.txt");



    //print dict
    //std::cout << dict;
    auto result =  dict.find("compiler");
    print_result(result);

    result = dict.find("engineer");
    print_result(result);

#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
}