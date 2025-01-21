#define CRT_
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include <iostream>
#include <cassert>
#include <filesystem>
#include "../src/language_modules/cmudict.h"

#define OV_MODEL_PATH "ov_models"


auto print_result = [](const auto& result) {
    if (result.has_value()) {
        const auto& vec = result.value().get();  // get the ref
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
    std::filesystem::path ov_model = "C:\\Users\\gta\\source\\repos\\MeloTTS.cpp\\ov_models\\";
    std::filesystem::path file_dir = ov_model / "cmudict_cache.txt";
    melo::CMUDict dict(file_dir.string());



    //print dict
    //std::cout << dict;
    auto result =  dict.find("compiler");
    print_result(result);

    result = dict.find("engineer");
    print_result(result);


}