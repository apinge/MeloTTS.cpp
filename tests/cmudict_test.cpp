#include "CMUDict.h"
#include <iostream>

int main() {
    melo::CMUDict dict("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\cmudict_cache.txt");



    // 或者使用流输出
    std::cout << dict;

    return 0;
}