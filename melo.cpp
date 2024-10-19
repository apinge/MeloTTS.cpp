
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
//#include <codecvt>
//#include <fcntl.h>
//#include <io.h>
#include <windows.h>
#endif

//#define DEBUG_MEMORY
#if defined(_WIN32) && defined(DEBUG_MEMORY)
#define PSAPI_VERSION 1 // PrintMemoryInfo 
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfod
#include "processthreadsapi.h"
#endif



#include "utils.h"
#include "tts.h"
#include "chinese_mix.h"

#if defined(_WIN32) && defined(DEBUG_MEMORY)
// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header)
{
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
    {
        //The following printout corresponds to the value of Resource Memory, respectively
        printf("%s:\tCommit \t\t\t=  0x%08X- %u (KB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / 1024);
        printf("%s:\tWorkingSetSize\t\t\t=  0x%08X- %u (KB)\n", header, pmc.WorkingSetSize, pmc.WorkingSetSize / 1024);
        printf("%s:\tPrivateWorkingSetSize\t\t\t=  0x%08X- %u (KB)\n", header, pmc.PrivateWorkingSetSize, pmc.PrivateWorkingSetSize / 1024);
    }
}
#endif
int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    ConfigureOneDNNCache();

    //fp32 model
    //std::filesystem::path zh_tts_path = "ov_models/tts_ZH_fp32.xml";
    //std::filesystem::path zh_bert_path = "ov_models/bert_ZH_fp32.xml";
    //fp16 model
    /*std::filesystem::path zh_tts_path = "ov_models/tts_zn_mix_en.xml";
    std::filesystem::path zh_bert_path = "ov_models/bert_zn_mix_en.xml";*/

    //int8 model
    std::filesystem::path zh_tts_path = "ov_models/tts_zn_mix_en_int8.xml"; //fp16 to int8
    //std::filesystem::path zh_tts_path = "ov_models/tts_ZH_int8.xml"; //fp32 to int8
    std::filesystem::path zh_bert_path = "ov_models/bert_zn_mix_en_int8.xml";
    //std::filesystem::path zh_bert_path = "ov_models/bert_int8_ZH.xml";// model in python repo

    // init tokenizer
    std::filesystem::path vocab_bert_path = "ov_models/vocab_bert.txt";

    // punctuation dict
    std::filesystem::path punc_dict_path = "ov_models/punc.dic";

    // dict folder for cppjieba
   std::filesystem::path cppjieba_dict = "thirdParty/cppjieba/dict";

   // cppinyin
   std::filesystem::path cppinyin_resource = "thirdParty/cppinyin/resources/cpp_pinyin.raw";

   //dict
   std::filesystem::path cmudict_path = "ov_models/cmudict_cache.txt";

   //pinyin_to_symbol_map
   std::filesystem::path pinyin_to_symbol_map_path = "ov_models/opencpop-strict.txt";

    //outputpath
    std::filesystem::path output_path = "audio.wav";

    //Init lanugage module
    melo::chinese_mix::cmudict = std::make_shared<melo::CMUDict>(cmudict_path.string());
    melo::chinese_mix::jieba = std::make_shared<cppjieba::Jieba>(cppjieba_dict);
    melo::chinese_mix::pinyin_to_symbol_map = melo::chinese_mix::readPinyinFile(pinyin_to_symbol_map_path);
    melo::chinese_mix::pinyin = std::make_shared<cppinyin::PinyinEncoder>(cppinyin_resource);
    std::cout <<"Init language Module\n";

    // Init core
    std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
    auto startTime = Time::now();
    melo::TTS model(core_ptr, zh_tts_path,"CPU",zh_bert_path,"CPU",vocab_bert_path, punc_dict_path, "ZH", false);
    auto initTime = get_duration_ms_till_now(startTime);
    std::cout << "model init time is" << initTime <<" ms" << std::endl;
#if defined(_WIN32) && defined(DEBUG_MEMORY)
    DebugMemoryInfo("Memory after model loading");
#endif 


    std::vector<std::string> texts = {
        "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments",
        "我最近在学习machine learning, 希望能够在未来的artificial intelligence领域有所建树",
        "我家门口有很多柳树,这儿也有 那儿也有", //This example are different with or without bert
        "早就听闻阿勒泰的秋色绝美，真正看到时才知道是多么震撼。白桦林的风光真美。",
        "升级 pavilion laptop 硬盘的步骤是什么",
        "在很久很久以前，有一个国王，他把他的国家治理得非常好。国家不大，但百姓们丰衣足食，安居乐业，十分幸福。",
        "今天我太高兴了！我爸爸妈妈竟然让我挑一个地方玩，以示对我前阶段进步的鼓励！一大早，我就早早醒来，开始了我们的快乐之旅。",
    };
    for(int i = 0;i<1;++i){
        startTime = Time::now();
        model.tts_to_file(texts, output_path, 1, 0.95);
        auto inferTime = get_duration_ms_till_now(startTime);
        std::cout << "model infer time:" << inferTime << " ms"<< std::endl;
    }
#if defined(_WIN32) && defined(DEBUG_MEMORY)
   for (int i = 0; i < 50; ++i) {
#endif 

#if defined(_WIN32) && defined(DEBUG_MEMORY)
        DebugMemoryInfo(std::format("memory after infer {} round", i).c_str());
    }
#endif 



    //tts_processor->WriteWave("melo_tts_CPU.wav", 44100, wav_data.data(), wav_data.size());
    //std::cout << "finish to generate wav" << std::endl;
#if defined(_WIN32) && defined(DEBUG_MEMORY)
    DebugMemoryInfo("memory afer save wav");
#endif 
    //return 0;
#ifdef CRT_
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
#endif
}
