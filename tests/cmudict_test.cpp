#include "CMUDict.h"
#include <iostream>

int main() {
    melo::CMUDict dict("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\cmudict_cache.txt");



    //print dict
    std::cout << dict;

    return 0;
}