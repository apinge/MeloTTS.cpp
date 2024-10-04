
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include "Hanz2Piny.h"

#ifdef _WIN32
#include "windows.h"
#endif
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    //SetConsoleOutputCP(LGRPID_SIMPLIFIED_CHINESE);
#endif
    //std::string line = "链接装载和库";
    //std::string line = "编译器";
    std::string line = "律法";//lu:4fa3
    const Hanz2Piny hanz2piny;
    // origin version
    std::cout << hanz2piny.to_pinyin_str_from_utf8(line) << std::endl;

    

    //std::cout << "split initials and finals for single pinyin\n";
    //auto [x,y] = hanz2piny.split_initials_finals("lu:4");
    //std::cout << x <<' '<< y << std::endl;

    std::cout << "split initials and finals for a string\n";
    auto [sub_initials, sub_finals] = hanz2piny._get_initials_finals("玉天鹅");
    for(const auto&x:sub_initials) std::cout << x <<' ';
    std::cout << std::endl;
    for (const auto& y : sub_finals) std::cout << y << ' ';
    std::cout << std::endl;

    sub_initials.clear();sub_finals.clear();
    std::cout << "split initials and finals for a string\n";
    std::tie(sub_initials, sub_finals) = hanz2piny._get_initials_finals("编译器");
    for (const auto& x : sub_initials) std::cout << x << ' ';
    std::cout << std::endl;
    for (const auto& y : sub_finals) std::cout << y << ' ';
    std::cout << std::endl;

 


#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

}
