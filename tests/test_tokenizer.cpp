#include <iostream>
#include <locale>

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

#include "tokenizer.h"
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    melo::Tokenizer tokenizer("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\vocab_bert.txt");
    //std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    //std::string text = "一千五百二十九年一月九日";
    std::string text = "Hello 世界";
    std::cout << text << std::endl;
    std::vector<std::string> tokens;
    std::vector<int64_t> token_ids;
    tokenizer.Tokenize(text, tokens, token_ids);
    for (const auto& word : tokens) std::cout << word << " ";
    std::cout << std::endl;
    for (const auto& id : token_ids) std::cout << id << " ";
    std::cout << std::endl;
   /* tensor([[ 101, 6784, 7984, 2693, 85065, 33719, 1817, 3295, 2415, 6990,
        1776, 2160, 4270, 3203, 2383, 18958, 59242, 4108, 3259, 6805,
        2981, 5975, 4767, 4508, 3203, 2383, 79947, 20849, 59242, 102]] ), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0]] ), 'attention_mask' : tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1]] )
}*/


    return 0;
}