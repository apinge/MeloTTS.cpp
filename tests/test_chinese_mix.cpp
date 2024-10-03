#include <iostream>
#include "chinese_mix.h"
int main() {
    for(const auto &x:melo::chinese_mix::distribute_phone(7,2))
        std::cout << x <<' ';
    std::cout << std::endl;
}