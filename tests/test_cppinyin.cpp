//#define CRT_
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include "cppinyin.h"
#define OV_MODEL_PATH "ov_models"

int main() {
    system("chcp 65001"); //Using UTF-8 Encoding
    std::filesystem::path vocab_path = std::filesystem::path(OV_MODEL_PATH) / "cppinyin\\cpp_pinyin.raw";
    cppinyin::PinyinEncoder processor(vocab_path.string());

    //std::string str = "于谦参加了会计培训班 虽然他的会计基础参差不齐，但他仍然会努力学习, 于谦有头驴";
    //std::string str = "明月几时有";
    std::string str = "乌鹊南飞";
    std::ostringstream oss;
    std::vector<std::string> pieces;

    processor.Encode(str, &pieces);
    for (auto piece : pieces) {
        oss << piece << " ";
    }
    std::cout << oss.str() << std::endl;
    std::cout << "succeed!\n";
    /*
    result:
    yu2 qian1 can1 jia1 le5 kuai4 ji4 pei2 xun4 ban1 sui1 ran2 ta1 de5 kuai4 ji4 ji1 chu3 cen1 ci1 bu4 qi2 ， dan4 ta1 reng2 ran2 hui4 nu3 li4 xue2 xi2
    */
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
}