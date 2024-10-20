
//#define CRT_
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

#include "Jieba.hpp"
#define OV_MODEL_PATH "ov_models"

std::filesystem::path model_dir = OV_MODEL_PATH;

std::filesystem::path DICT_PATH = model_dir / "cppjieba\\dict\\jieba.dict.utf8";
std::filesystem::path HMM_PATH = model_dir / "cppjieba\\dict\\hmm_model.utf8";
std::filesystem::path USER_DICT_PATH = model_dir / "cppjieba\\dict\\user.dict.utf8";
std::filesystem::path IDF_PATH = model_dir / "cppjieba\\dict\\idf.utf8";
std::filesystem::path STOP_WORD_PATH = model_dir / "cppjieba\\dict\\stop_words.utf8";

std::filesystem::path cppjieba_dict = model_dir / "cppjieba\\dict";
auto printVec = [&](const std::vector<std::string>& arr, std::ostream& os) {
        for(const auto &w:arr)  os<< w <<"|"; 
        os <<"\n";
    };
auto printTagres = [&](const std::vector< std::pair<std::string, std::string>>& tagres, std::ostream& os) {
    for (const auto& [word,pos] : tagres)  os << word<<','<<pos << "|";
    os << "\n";
    };
int main() {

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    if(!std::filesystem::exists(DICT_PATH) || !std::filesystem::exists(HMM_PATH) || !std::filesystem::exists(USER_DICT_PATH) || !std::filesystem::exists(IDF_PATH) || 
        !std::filesystem::exists(STOP_WORD_PATH))
        std::cerr <<"input dict does not exit!\n";
    //cppjieba::Jieba jieba(DICT_PATH.string(),
    //    HMM_PATH.string(),
    //    USER_DICT_PATH.string(),
    //    IDF_PATH.string(),
    //    STOP_WORD_PATH.string());
    cppjieba::Jieba jieba(cppjieba_dict);
    std::vector<std::string> words;
    std::vector<cppjieba::Word> jiebawords;
    std::string s;
    std::string result;

    s = "compiler engineer";
    std::cout << s << std::endl;
    std::cout << "[demo] Cut With HMM" << std::endl;
    jieba.Cut(s, words, true);
    printVec(words, std::cout);

    s = "intel CPU 规格中列出了最大睿频和处理器基本频率";
    std::cout<< s << std::endl;
    std::cout<< "[demo] Cut With HMM" << std::endl;
    jieba.Cut(s, words, true);
    printVec(words,std::cout);

    std::cout << "[demo] Tagging" << std::endl;
    std::vector< std::pair<std::string, std::string> > tagres;
    s = "乌鹊南飞";
    jieba.Tag(s, tagres);
    printTagres(tagres, std::cout);
    //std::cout << tagres << std::endl;


     std::cout<< "[demo] Cut Without HMM " << std::endl;
    jieba.Cut(s, words, false);
    printVec(words, std::cout);

    s = "我来到北京清华大学";
     std::cout<< s << std::endl;
     std::cout<< "[demo] CutAll" << std::endl;
    jieba.CutAll(s, words);
    printVec(words, std::cout);

    s = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造";
     std::cout<< s << std::endl;
     std::cout<< "[demo] CutForSearch" << std::endl;
    jieba.CutForSearch(s, words);
     std::cout<< limonp::Join(words.begin(), words.end(), "/") << std::endl;

     std::cout<< "[demo] Insert User Word" << std::endl;
    jieba.Cut("男默女泪", words);
     std::cout<< limonp::Join(words.begin(), words.end(), "/") << std::endl;
    jieba.InsertUserWord("男默女泪");
    jieba.Cut("男默女泪", words);
     std::cout<< limonp::Join(words.begin(), words.end(), "/") << std::endl;

     std::cout<< "[demo] CutForSearch Word With Offset" << std::endl;
    jieba.CutForSearch(s, jiebawords, true);
     std::cout<< jiebawords << std::endl;

     std::cout<< "[demo] Lookup Tag for Single Token" << std::endl;
    const int DemoTokenMaxLen = 32;
    char DemoTokens[][DemoTokenMaxLen] = { "拖拉机", "CEO", "123", "。" };
    std::vector< std::pair<std::string, std::string> > LookupTagres(sizeof(DemoTokens) / DemoTokenMaxLen);
    std::vector< std::pair<std::string, std::string> >::iterator it;
    for (it = LookupTagres.begin(); it != LookupTagres.end(); it++) {
        it->first = DemoTokens[it - LookupTagres.begin()];
        it->second = jieba.LookupTag(it->first);
    }
     std::cout<< LookupTagres << std::endl;

     std::cout<< "[demo] Tagging" << std::endl;
    //std::vector< std::pair<std::string, std::string> > tagres;
    s = "我是拖拉机学院手扶拖拉机专业的。不用多久，我就会升职加薪，当上CEO，走上人生巅峰。";
    jieba.Tag(s, tagres);
    printVec(words, std::cout);
     std::cout<< tagres << std::endl;

     std::cout<< "[demo] Keyword Extraction" << std::endl;
    const size_t topk = 5;
    std::vector<cppjieba::KeywordExtractor::Word> keywordres;
    jieba.extractor.Extract(s, keywordres, topk);
    printVec(words, std::cout);
     std::cout<< keywordres << std::endl;
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

}