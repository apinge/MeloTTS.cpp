/*
 * Licensed under the Apache License, Version 2.0.
 * See the LICENSE file for more information.
 */
#pragma once
#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H
#include <filesystem>
#include <iostream>
#include <unordered_set>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include <codecvt>
#endif
struct Args {
    std::filesystem::path model_dir = "ov_models";
    std::string tts_device = "CPU";
    std::string bert_device = "CPU";
    std::string nf_device = "CPU";
    std::string input_file = "inputs.txt";
    std::string output_filename = "audio";
    float speed = 1.0;
    bool quantize = true; // quantize for tts
    bool disable_bert = false;
    bool disable_nf = false;
    std::string language = "EN";

    void generate_init_file_paths();
    const std::unordered_set<std::string> supported_languages = {"ZH", "EN"};

    std::filesystem::path tts_path;   // tts_model
    std::filesystem::path bert_path;  // bert_model
    std::filesystem::path punc_dict_path;          // // punctuation dict
    std::filesystem::path nf_ir_path;
};

inline void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  --model_dir             Specifies the folder containing the model files, dictionary files, and "
                 "third-party resource files. \n"
              << "  --tts_device            Specifies the OpenVINO device to be used for the TTS model (Supported "
                 "devices include CPU, and GPU; default: CPU).\n"
              << "  --bert_device           Specifies the OpenVINO device to be used for the BERT model (Supported "
                 "devices include CPU, GPU, and NPU; default: CPU).\n"
#    ifdef USE_DEEPFILTERNET
              << "  --nf_device             Specifies the OpenVINO device to be used for the DeepfilterNet model "
                 "(Supported devices include CPU, GPU, and NPU; default: CPU).\n"
#    endif  // USE_DEEPFILTERNET
              << "  --input_file            Specifies the input text file to be processed.\n"
              << "  --output_filename       Specifies the output audio filename to be generated in the format "
                 "{output_filename}_{language_style}.wav. For example, if the language is Chinese and the output_filen "
                 "is \"audio\", the file will be saved as audio_ZH-MIX-EN.wav\n"
              << "  --speed                 Specifies the speed of output audio (default: 1.0).\n"
              << "  --quantize              Indicates whether to use an int8 quantized tts model (default: true, use int8 "
                 "model by default).\n"
              << "  --disable_bert          Indicates whether to disable the BERT model inference (default: false).\n"
#    ifdef USE_DEEPFILTERNET
              << "  --disable_nf            Indicates whether to disable the DeepfilterNet model inference (default: "
                 "false).\n"
#    endif  // USE_DEEPFILTERNET
              << "  --language              Specifies the language for TTS (default: EN).\n";
}

static bool to_bool(const std::string& s) {
    bool res;
    std::istringstream(s) >> std::boolalpha >> res;
    return res;
}

inline Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "--model_dir") {
            args.model_dir = argv[++i];
        } else if (arg == "--tts_device") {
            args.tts_device = argv[++i];
        } else if (arg == "--bert_device") {
            args.bert_device = argv[++i];
        } else if (arg == "--nf_device") {
            args.nf_device = argv[++i];
        } else if (arg == "--input_file") {
            args.input_file = argv[++i];
        } else if (arg == "--output_filename") {
            args.output_filename = argv[++i];
        } else if (arg == "--speed") {
            args.speed = std::stof(argv[++i]);
        } else if (arg == "--disable_bert") {
            args.disable_bert = to_bool(argv[++i]);
        } else if (arg == "--disable_nf") {
            args.disable_nf = to_bool(argv[++i]);
        } else if (arg == "--quantize") {
            args.quantize = to_bool(argv[++i]);
        } else if (arg == "--language") {
            args.language = argv[++i];
        } else {
            usage(argv[0]);
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    args.generate_init_file_paths();
    return args;
}

inline Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#    ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#    else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#    endif

    return parse_args(argv_vec);
}

inline void Args::generate_init_file_paths() {
#    ifdef USE_DEEPFILTERNET
    // nf_df2 model path
    nf_ir_path = model_dir;
#    endif  // USE_DEEPFILTERNET
}

#endif
