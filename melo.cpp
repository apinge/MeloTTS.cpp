
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


#include <iostream>
#include <fstream>
//#include <sstream>
#include <string>
//#include <cassert>
#include <sstream>
#include <cstdlib>


//#define DEBUG_MEMORY

#if defined(_WIN32) && defined(DEBUG_MEMORY)
#define PSAPI_VERSION 1 // PrintMemoryInfo 
#include <windows.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfod
#include "processthreadsapi.h"
#endif

#include "src/openvoice2_processor.h"
#include "utils.h"

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
    ConfigureOneDNNCache();

    melo::MeloTTSProcessor* tts_processor =  new melo::MeloTTSProcessor();

    //fp16 model
    std::string zh_tts_path = "thirdParty/tts_ov/tts_zn_mix_en.xml";
    std::string zh_bert_path = "thirdParty/tts_ov/bert_zn_mix_en.xml";

    //int8 model
    //std::string zh_tts_path = "thirdParty/tts_ov/tts_zn_mix_en_int8.xml";
    //std::string zh_bert_path = "thirdParty/tts_ov/bert_zn_mix_en_int8.xml";

    // init tokenizer
    std::string vocab_bert_path = "thirdParty/tts_ov/vocab_bert.txt";
    auto startTime = Time::now();
    tts_processor->LoadTTSModel(zh_tts_path, zh_bert_path, vocab_bert_path);
    auto execTime = get_duration_ms_till_now(startTime);

    std::cout << "Model Load Time:" << execTime << " milliseconds\n";
#if defined(_WIN32) && defined(DEBUG_MEMORY)
    DebugMemoryInfo("Memory after model loading");
#endif 
    std::vector<float> addit_param = { 0.2, 0.6, 1.0, 0.80 };

    std::string convert_text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    std::vector<float> wav_data;
#if defined(_WIN32) && defined(DEBUG_MEMORY)
   for (int i = 0; i < 50; ++i) {
#endif 
        startTime = Time::now();
        tts_processor->Process(convert_text, 0, addit_param, wav_data);
        execTime = get_duration_ms_till_now(startTime);
        std::cout << "OV infer Time:" << execTime << " milliseconds\n";
#if defined(_WIN32) && defined(DEBUG_MEMORY)
        DebugMemoryInfo(std::format("memory after infer {} round", i).c_str());
    }
#endif 


    //16000 origin
    tts_processor->WriteWave("melo_tts_CPU.wav", 44100, wav_data.data(), wav_data.size());
    std::cout << "finish to generate wav" << std::endl;
#if defined(_WIN32) && defined(DEBUG_MEMORY)
    DebugMemoryInfo("memory afer save wav");
#endif 
    //return 0;
#ifdef CRT_
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
#endif
}
