/*
 * Licensed under the Apache License, Version 2.0.
 * See the LICENSE file for more information.
 */
#pragma once
#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H
#include <filesystem>
#include <iostream>
#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif
struct Args
{
    std::filesystem::path model_dir = "ov_models";
    std::string tts_device = "CPU";
    std::string bert_device = "CPU";
    std::string input_file = "inputs.txt";
    std::string output_file = "audio.wav";
    float speed = 1.0;
    bool quantize = false;
    bool disable_bert = false;
    std::string language = "ZH";

    void generate_init_file_paths();

 
    std::filesystem::path zh_tts_path; //tts_model
    std::filesystem::path zh_bert_path; //bert_model
    std::filesystem::path vocab_bert_path;// init tokenizer
    std::filesystem::path punc_dict_path;// // punctuation dict
    std::filesystem::path cppjieba_dict;// dict folder for cppjieba
    std::filesystem::path cppinyin_resource; // cppinyin
    std::filesystem::path cmudict_path; // Carnegie Mellon University Pronouncing Dictionary, used for english pronunciation 
    std::filesystem::path pinyin_to_symbol_map_path;//pinyin_to_symbol_map
};

inline void usage(const std::string& prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  --model_dir             Specifies the folder containing the model files, dictionary files, and third-party resource files. \n"
        << "  --tts_device            Specifies the OpenVINO device to be used for the TTS model (default: CPU).\n"
        << "  --bert_device           Specifies the OpenVINO device to be used for the BERT model (default: CPU).\n"
        << "  --input_file            Specifies the input text file to be processed.\n"
        << "  --output_file           Specifies the output audio file to be generated.\n"
        << "  --speed                 Specifies the speed of output audio (default: 1.0).\n"
        << "  --quantize              Indicates whether to use an int8 quantized model (default: false, use fp16 model by default).\n"
        << "  --disable_bert          Indicates whether to disable the BERT model inference (default: false).\n"
        << "  --language              Specifies the language for TTS (default: ZH).\n";
}

static bool to_bool(const std::string& s) {
    bool res;
    std::istringstream(s) >> std::boolalpha >> res;
    return res;
}

inline Args parse_args(const std::vector<std::string>& argv)
{
    Args args;

    for (size_t i = 1; i < argv.size(); i++)
    {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "--model_dir")
        {
            args.model_dir = argv[++i];
        }
        else if (arg == "--tts_device")
        {
            args.tts_device = argv[++i];
        }
        else if (arg == "--bert_device")
        {
            args.bert_device = argv[++i];
        }
        else if (arg == "--input_file")
        {
            args.input_file = argv[++i];
        }
        else if (arg == "--output_file")
        {
            args.output_file = argv[++i];
        }
        else if (arg == "--speed")
        {
            args.speed = std::stof(argv[++i]);
        }
        else if (arg == "--disable_bert")
        {
            args.disable_bert = to_bool(argv[++i]);
        }
        else if (arg == "--quantize")
        {
            args.quantize = to_bool(argv[++i]);
        }
        else if (arg == "--language")
        {
            args.language = argv[++i];
        }
        else
        {
            usage(argv[0]);
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    args.generate_init_file_paths();
    return args;
}

inline Args parse_args(int argc, char** argv)
{
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++)
    {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++)
    {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

inline void Args::generate_init_file_paths() {
    zh_bert_path = model_dir / "bert_ZH_int8.xml";
    if (quantize) {
        zh_tts_path = model_dir / "tts_zn_mix_en_int8.xml";
        //zh_bert_path = model_dir / "bert_ZH_int8.xml";
    }
    else {
        //fp16 model
        zh_tts_path = model_dir / "tts_zn_mix_en.xml";
        //zh_bert_path = model_dir / "bert_ZH.xml";
    }

    // init tokenizer
    vocab_bert_path = model_dir / "vocab_bert.txt";

    // punctuation dict
    punc_dict_path = model_dir / "punc.dic";

    //dict
    cmudict_path = model_dir / "cmudict_cache.txt";

    //pinyin_to_symbol_map
    pinyin_to_symbol_map_path = model_dir / "opencpop-strict.txt";

    // These two folders should ideally belong to the thirdParty directory.
    // However, for convenience, they are placed under model_dir here.
    // dict folder for cppjieba
    cppjieba_dict = model_dir / "cppjieba/dict";
    // cppinyin
    cppinyin_resource = model_dir / "cppinyin/cpp_pinyin.raw";
}


#endif 
