#define CRT_
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include <iostream>
#include <vector>
#include <numeric>
#include <darts.h>
#include <string>
#include <algorithm>
#include <sstream>
#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif
#include "darts.h"

Darts::DoubleArray da;

std::vector<std::string> split_sentences_into_pieces(const std::string& text, bool quiet = false) {
    std::vector<std::string> pieces;
    int n = text.length();
    int MAX_HIT = n;
    int left = 0;
   
    for (int right = 0; right < n;) {
        const char* query = text.data()+right;
        std::vector<Darts::DoubleArray::result_pair_type> results(MAX_HIT);
        size_t num_matches = da.commonPrefixSearch(query, results.data(), MAX_HIT);
        if (!num_matches) {
            ++right ;
            continue;
        }
        pieces.emplace_back(text.substr(left,right-left));
        right += results[0].length;
        left = right;
    }
    if(left!=n) 
        pieces.emplace_back(text.substr(left,n-left));
    if (!quiet) {
        std::cout << " > Text split to sentences." << std::endl;
        for (const auto& piece : pieces) {
            std::cout <<"   "<<piece << std::endl;
        }
        std::cout << " > ===========================" << std::endl;
    }
    return pieces;
}
// this implementation only for english puncts whose length is only 1
std::vector<std::string> splitTextByPunctuation(const std::string& text, bool quiet = false) {
    std::vector<std::string> pieces;
    std::string delimiters = "，。！？、；：“”‘’（）【】《》——……·,.;:\"'()[]<>-...";
    std::string piece;
    std::istringstream stream(text);

    while (std::getline(stream, piece)) {
        size_t start = 0, end = 0;
        while ((end = piece.find_first_of(delimiters, start)) != std::string::npos) {
            if (end != start) {
                pieces.push_back(piece.substr(start, end - start));
            }
            start = end + 1;
        }
        if (start < piece.size()) {
            pieces.push_back(piece.substr(start));
        }
    }
    if (!quiet) {
        std::cout << " > Text split to sentences." << std::endl;
        for (const auto& piece : pieces) {
            std::cout << "   " << piece << std::endl;
        }
        std::cout << " > ===========================" << std::endl;
    }
    return pieces;
}
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    system("chcp 65001"); //Using UTF-8 Encoding
#endif
    da.open("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\ov_models\\punc.dic");
    std::cout << "open dict\n";
    auto res = split_sentences_into_pieces("我最近在学习machine learning, 希望能够在未来的artificial intelligence领域有所建树");

    auto res1= splitTextByPunctuation("我最近在学习machine learning, 希望能够在未来的artificial intelligence领域有所建树");

#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
}