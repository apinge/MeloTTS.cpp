
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
    std::string line = "链接装载和库";
    const Hanz2Piny hanz2piny;
    std::cout << hanz2piny.to_pinyin_str_from_utf8(line) << std::endl;


#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

}
