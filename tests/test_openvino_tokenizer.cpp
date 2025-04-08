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
        std::filesystem::path zh_tokenizer_path = model_path / "bert-base-multilingual-uncased" ;
        std::filesystem::path en_tokenizer_path = model_path / "bert-base-uncased";

        zh_tokenizer = melo::OpenVinoTokenizer(zh_tokenizer_path);
        en_tokenizer = melo::OpenVinoTokenizer(en_tokenizer_path);
    }

protected:
    melo::OpenVinoTokenizer zh_tokenizer, en_tokenizer;

};


TEST_F(TokenizerTestSuit, ZH_BertTokenize) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    std::cout << text << std::endl;
    std::vector<std::string> tokens;
    std::vector<int64_t> token_ids;
    auto startTime = Time::now();
    std::vector<int64_t> res = zh_tokenizer.tokenize(text);
    auto execTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] subword_tokenize takes "<< execTime<<"ms\n";
    const std::vector<int64_t> correct_ids = { 101, 6784, 7984, 2693, 85065, 33719, 1817, 3295, 2415, 6990, 1776, 2160, 4270, 3203, 2383, 18958, 59242, 4108, 3259, 6805,
        2981, 5975, 4767, 4508, 3203, 2383, 79947, 20849, 59242, 102 };

    EXPECT_EQ(res, correct_ids);
}

//TEST_F(TokenizerTestSuit, ZH_BertSubwordDeTokenizer) {
//#ifdef _WIN32
//    SetConsoleOutputCP(CP_UTF8);
//#endif
//    //system("chcp 65001"); //Using UTF-8 Encoding
//    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
//
//    std::vector<int64_t> input_ids = { 6784, 7984, 2693, 85065, 33719, 1817, 3295, 2415, 6990, 1776, 2160, 4270, 3203, 2383, 18958, 59242, 4108, 3259, 6805,
//        2981, 5975, 4767, 4508, 3203, 2383, 79947, 20849, 59242 };
//    std::vector<std::string> correct_subwords = {
//    "编", "译", "器", "comp", "##iler", "会", "尽", "可", "能",
//    "从", "函", "数", "实", "参", "function", "arguments",
//    "推", "导", "缺", "失", "的", "模", "板", "实", "参",
//    "temp", "##late", "arguments" };
//    std::vector<std::string> res;
//    
//    ov::Tensor input_ids_tensor(ov::element::i64, ov::Shape{1, input_ids.size()});
//    std::string res1;
//
//    for (int i = 0;i< input_ids.size();++i) {
//        input_ids_tensor.data<int64_t>()[i] = input_ids[i];
//    }
//    auto startTime = Time::now();
//    res1 = zh_tokenizer.detokenize(input_ids_tensor);
//    auto execTime = get_duration_ms_till_now(startTime);
//    std::cout << "[INFO] detokenize takes " << execTime << "ms\n";
//    EXPECT_EQ(res, correct_subwords);
//    EXPECT_EQ(res1, correct_subwords);
//}
//https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
TEST_F(TokenizerTestSuit, ZH_word_segmentation) {
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
    for (const auto& x : res)
        std::cout << x << ' ';
    std::cout << std::endl;
    EXPECT_EQ(res, correct_subwords);
}

//https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
TEST_F(TokenizerTestSuit, EN_Tokenize) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string text = "I've been learning machine learning recently and hope to make contributions in the field of artificial intelligence in the future.";
    std::cout << text << std::endl;
    std::vector<std::string> tokens;
    std::vector<int64_t> token_ids;
    auto startTime = Time::now();
    token_ids = en_tokenizer.tokenize(text);
    auto execTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] subword_tokenize takes " << execTime << "ms\n";
    const std::vector<int64_t> correct_ids = { 101, 1045, 1005, 2310, 2042, 4083, 3698, 4083, 3728, 1998, 3246, 2000,
         2191, 5857, 1999, 1996, 2492, 1997, 7976, 4454, 1999, 1996, 2925, 1012,
          102 };

    EXPECT_EQ(token_ids, correct_ids);
}

//https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
TEST_F(TokenizerTestSuit, EN_word_segmentation) {
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
    for (const auto& x : res)
        std::cout << x << ' ';
    std::cout << std::endl;
    EXPECT_EQ(res, correct_subwords);
}