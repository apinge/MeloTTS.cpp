#include <iostream>
#include <locale>
#include <filesystem>
#include <utility>
#include <memory>
#include <gtest/gtest.h>
#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif
#include <openvino/openvino.hpp>
#include "openvino_tokenizer.h"

//#include <openvino/op/transpose.hpp>
//#include <openvino/core/node.hpp>
#include "utils.h"
#define OV_MODEL_PATH "ov_models"

class TokenizerTestSuit : public ::testing::Test {
public:
    TokenizerTestSuit() {
        std::filesystem::path model_path = "C:\\Users\\gta\\source\\repos\\MeloTTS.cpp\\ov_models";
        std::filesystem::path zh_bert_subword_tokenizer = model_path / "bert-base-multilingual-uncased" / "bert_subword" / "bert_subword_tokenizer.xml";
        std::filesystem::path zh_bert_subword_detokenizer = model_path / "bert-base-multilingual-uncased" / "bert_subword" / "bert_subword_detokenizer.xml";
        std::filesystem::path en_bert_subword_tokenizer = model_path / "bert-base-uncased" /  "bert_subword_tokenizer.xml";
        std::filesystem::path en_bert_subword_detokenizer = model_path / "bert-base-uncased" /  "bert_subword_detokenizer.xml";
#ifdef _WIN32 
        std::filesystem::path dll_path = "openvino_tokenizers.dll";
#elif  __linux__
        std::filesystem::path dll_path = "libopenvino_tokenizers.so";
#else
        std::cerr << "[ERROR] Unsupported Operating System.\n"
#endif
        core = std::make_unique<ov::Core>();
        zh_tokenizer = melo::OpenVinoTokenizer(core,dll_path,zh_bert_subword_tokenizer,zh_bert_subword_detokenizer);
        en_tokenizer = melo::OpenVinoTokenizer(core,dll_path,en_bert_subword_tokenizer,en_bert_subword_detokenizer);

    }
  

protected:
    std::unique_ptr<ov::Core> core;
    melo::OpenVinoTokenizer zh_tokenizer, en_tokenizer;
    //ov::InferRequest zh_subword_tokenizer_infer, zh_subword_detokenizer_infer, en_subword_tokenizer_infer, en_subword_detokenizer_infer;


};


TEST_F(TokenizerTestSuit, ZH_BertSubwordTokenizer) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    std::cout << text << std::endl;
    std::vector<std::string> tokens;
    std::vector<int64_t> token_ids;
    auto startTime = Time::now();
    ov::Tensor res = zh_tokenizer.tokenize_tensor(std::move(text));
    auto execTime = get_duration_ms_till_now(startTime);
    auto vec = melo::OpenVinoTokenizer::get_output_vec<int64_t>(res);
    std::cout << "[INFO] subword_tokenize takes "<< execTime<<"ms\n";
    const std::vector<int64_t> correct_ids = { 101, 6784, 7984, 2693, 85065, 33719, 1817, 3295, 2415, 6990, 1776, 2160, 4270, 3203, 2383, 18958, 59242, 4108, 3259, 6805,
        2981, 5975, 4767, 4508, 3203, 2383, 79947, 20849, 59242, 102 };

    EXPECT_EQ(vec, correct_ids);
}

TEST_F(TokenizerTestSuit, ZH_BertSubwordDeTokenizer) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    //system("chcp 65001"); //Using UTF-8 Encoding
    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";

    std::vector<int64_t> input_ids = { 6784, 7984, 2693, 85065, 33719, 1817, 3295, 2415, 6990, 1776, 2160, 4270, 3203, 2383, 18958, 59242, 4108, 3259, 6805,
        2981, 5975, 4767, 4508, 3203, 2383, 79947, 20849, 59242 };
    std::vector<std::string> correct_subwords = {
    "编", "译", "器", "comp", "##iler", "会", "尽", "可", "能",
    "从", "函", "数", "实", "参", "function", "arguments",
    "推", "导", "缺", "失", "的", "模", "板", "实", "参",
    "temp", "##late", "arguments" };
    std::vector<std::string> res, res1;

    for (auto& id : input_ids) {
        res.emplace_back(*zh_tokenizer.detokenize(std::move(id)));
    }
    auto startTime = Time::now();
    res1 = zh_tokenizer.detokenize(std::move(input_ids), input_ids.size());
    auto execTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] detokenize takes " << execTime << "ms\n";
    EXPECT_EQ(res, correct_subwords);
    EXPECT_EQ(res1, correct_subwords);
}
//https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
TEST_F(TokenizerTestSuit, ZH_Subword_tokenization) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";

    std::vector<std::string> correct_subwords = {
    "编", "译", "器", "comp", "##iler", "会", "尽", "可", "能",
    "从", "函", "数", "实", "参", "function", "arguments",
    "推", "导", "缺", "失", "的", "模", "板", "实", "参",
    "temp", "##late", "arguments" };
    auto startTime = Time::now();
    std::vector<std::string> res = zh_tokenizer.word_segment(text);
    auto execTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] split subword takes " << execTime << "ms\n";

    EXPECT_EQ(res, correct_subwords);
}

//https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
TEST_F(TokenizerTestSuit, EN_BertSubwordDeTokenizer) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string text = "I've been learning machine learning recently and hope to make contributions in the field of artificial intelligence in the future.";
    std::cout << text << std::endl;
    std::vector<std::string> tokens;
    std::vector<int64_t> token_ids;
    auto startTime = Time::now();
    ov::Tensor res = en_tokenizer.tokenize_tensor(std::move(text));
    auto execTime = get_duration_ms_till_now(startTime);
    auto vec =melo::OpenVinoTokenizer::get_output_vec<int64_t>(res);
    std::cout << "[INFO] subword_tokenize takes " << execTime << "ms\n";
    const std::vector<int64_t> correct_ids = { 101, 1045, 1005, 2310, 2042, 4083, 3698, 4083, 3728, 1998, 3246, 2000,
         2191, 5857, 1999, 1996, 2492, 1997, 7976, 4454, 1999, 1996, 2925, 1012,
          102 };

    EXPECT_EQ(vec, correct_ids);
}

//https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
TEST_F(TokenizerTestSuit, EN_Subword_tokenization) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string text = "I have installed a fortran compiler";

    std::vector<std::string> correct_subwords = {
   "i", "have", "installed", "a", "fort", "##ran", "compiler"};
    auto startTime = Time::now();
    std::vector<std::string> res = en_tokenizer.word_segment(text);
    auto execTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] split subword takes " << execTime << "ms\n";

    EXPECT_EQ(res, correct_subwords);
}