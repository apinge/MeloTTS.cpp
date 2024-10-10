
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


    //fp16 model
    std::filesystem::path zh_tts_path = "ov_models/tts_zn_mix_en.xml";
    std::filesystem::path zh_bert_path = "ov_models/bert_zn_mix_en.xml";

    //int8 model
    //std::filesystem::path zh_tts_path = "ov_models/tts_zn_mix_en_int8.xml";
    //std::filesystem::path zh_bert_path = "ov_models/bert_zn_mix_en_int8.xml";

    // init tokenizer
    std::filesystem::path vocab_bert_path = "ov_models/vocab_bert.txt";

    // dict folder for cppjieba
   std::filesystem::path cppjieba_dict = "thirdParty/cppjieba/dict";

   // cppinyin
   std::filesystem::path cppinyin_resource = "thirdParty/cppinyin/resources/cpp_pinyin.raw";

   //dict
   std::filesystem::path cmudict_path = "ov_models/cmudict_cache.txt";

   //pinyin_to_symbol_map
   std::filesystem::path pinyin_to_symbol_map_path = "ov_models/opencpop-strict.txt";

    //outputpath
    std::filesystem::path output_path = "MeloTTS_ov.wav";

    //Init lanugage module
    melo::chinese_mix::cmudict = std::make_shared<melo::CMUDict>(cmudict_path.string());
    melo::chinese_mix::jieba = std::make_shared<cppjieba::Jieba>(cppjieba_dict);
    melo::chinese_mix::pinyin_to_symbol_map = melo::chinese_mix::readPinyinFile(pinyin_to_symbol_map_path);
    melo::chinese_mix::pinyin = std::make_shared<cppinyin::PinyinEncoder>(cppinyin_resource);
    std::cout <<"Init language Module\n";

    // Init core
    std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
    auto startTime = Time::now();
    melo::TTS model(core_ptr, zh_tts_path,"CPU",zh_bert_path,"CPU",vocab_bert_path, "ZH");
    auto initTime = get_duration_ms_till_now(startTime);
    std::cout << "model init time is" << initTime <<" ms" << std::endl;
#if defined(_WIN32) && defined(DEBUG_MEMORY)
    DebugMemoryInfo("Memory after model loading");
#endif 
    //std::vector<float> addit_param = { 0.2f, 0.6f, 1.0f, 0.80f };

    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    startTime = Time::now();
    model.tts_to_file(text,1,output_path,0.8);
    auto inferTime = get_duration_ms_till_now(startTime);
    std::cout << "model infer time is" << inferTime << " ms"<< std::endl;
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
