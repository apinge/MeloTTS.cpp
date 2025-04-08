/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

//#define DEBUG_MEMORY
#if defined(_WIN32) && defined(DEBUG_MEMORY)
#define PSAPI_VERSION 1  // PrintMemoryInfo
#include <psapi.h>
#pragma comment(lib, "psapi.lib")  // PrintMemoryInfod
#include "processthreadsapi.h"
#endif

#include "language_modules/chinese_mix.h"
#include "parse_args.h"
#include "tts.h"
#include "utils.h"

#if defined(_WIN32) && defined(DEBUG_MEMORY)
// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header) {
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // The following printout corresponds to the value of Resource Memory, respectively
        printf("%s:\tCommit \t\t\t=  0x%08X- %u (KB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / 1024);
        printf("%s:\tWorkingSetSize\t\t\t=  0x%08X- %u (KB)\n", header, pmc.WorkingSetSize, pmc.WorkingSetSize / 1024);
        printf("%s:\tPrivateWorkingSetSize\t\t\t=  0x%08X- %u (KB)\n",
               header,
               pmc.PrivateWorkingSetSize,
               pmc.PrivateWorkingSetSize / 1024);
    }
}
#endif
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    ConfigureOneDNNCache();
    SetOneDNN_CPU_MAX_ISA();
    Args args = parse_args(argc, argv);

    std::filesystem::path input_path = args.input_file;
    std::string output_filename = args.output_filename;

    // Init core
    std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
    auto startTime = Time::now();
    melo::TTS model(core_ptr,
                    args.model_dir,
                    args.language,
                    args.tts_device,
                    args.quantize,
                    args.bert_device,
                    args.disable_bert,
#ifdef USE_DEEPFILTERNET
                    args.nf_ir_path,
                    args.nf_device,
                    args.disable_nf
#endif
    );
    
    auto initTime = get_duration_ms_till_now(startTime);
    std::cout << "model init time is" << initTime << " ms" << std::endl;

    std::vector<std::string> texts = read_file_lines(input_path);
    // TODO: make speaker id in args
    for (auto& [speaker_id, style_name] : melo::TTS::speaker_ids.at(args.language)) {
        startTime = Time::now();
        model.tts_to_file(texts, std::format("{}_{}.wav", output_filename, style_name), speaker_id, args.speed);
        auto inferTime = get_duration_ms_till_now(startTime);
        std::cout << "model infer time:" << inferTime << " ms" << std::endl;
    }
}
