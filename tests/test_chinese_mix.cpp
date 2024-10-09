#include <iostream>
#include <vector>
#include <string>
#include "chinese_mix.h"


int main() {
    //test  distribute_phone
    for(const auto &x:melo::chinese_mix::distribute_phone(7,2))
        std::cout << x <<' ';
    std::cout << std::endl;

     for(const auto&x: melo::chinese_mix::split_utf8_chinese("右值的生命周期"))
	 std::cout << x << ',';
	std::cout << std::endl;
}