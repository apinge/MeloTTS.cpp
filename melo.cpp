
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <sstream>
#include "src/openvoice2_processor.h"


int main()
{
    // set ONEDNN_CACHE_CAPACITY 
    auto status = _putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=0");
    if (status == 0) {
        char* onednn_kernel_capacity = std::getenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY");
        std::cout << "set ONEDNN_PRIMITIVE_CACHE_CAPACITY: " << onednn_kernel_capacity << "\n";
        int y;
        std::stringstream ss(onednn_kernel_capacity);
        ss >> y;
        //std::cout << y << std::endl;
        assert((y==0) && "[ERROR] Set ONEDNN_PRIMITIVE_CACHE_CAPACITY fails!");
    }
    melo::MeloTTSProcessor* tts_processor =  new melo::MeloTTSProcessor();

    //fp16 model
    //std::string zh_tts_path = "thirdParty/tts_ov/tts_zn_mix_en.xml";
    //std::string zh_bert_path = "thirdParty/tts_ov/bert.xml";

    //int8 model
    std::string zh_tts_path = "thirdParty/tts_ov/tts_int8.xml";
    std::string zh_bert_path = "thirdParty/tts_ov/bert_int8_static.xml";

    // init tokenizer
    std::string vocab_bert_path = "thirdParty/tts_ov/vocab_bert.txt";

    tts_processor->LoadTTSModel(zh_tts_path, zh_bert_path, vocab_bert_path);
    std::vector<float> addit_param = { 0.2, 0.6, 1.0, 0.80 };

    std::string convert_text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    std::vector<float> wav_data;

    tts_processor->Process(convert_text, 0, addit_param, wav_data);

    //16000 origin
    tts_processor->WriteWave("melo_tts_int8.wav", 44100, wav_data.data(), wav_data.size());
    std::cout << "finish to generate wav" << std::endl;
    //return 0;
#ifdef CRT_
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
#endif
}
