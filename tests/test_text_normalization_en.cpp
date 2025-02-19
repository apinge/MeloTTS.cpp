#include <iostream>
#include "text_normalization_eng.h"


int main() {
	std::string num = "I have 1234567.893 in text and 201th in echo 253,235,365";
	std::string words = text_normalization::normalize_numbers(num);
	std::cout << words << std::endl;

	num = "23 sheep";
	words = text_normalization::normalize_numbers(num);
	std::cout << words << std::endl;


	std::string input = "Dr. Smith went to St. John's Church with Mr. Brown";
	std::string output = text_normalization::expand_abbreviations(input);
	std::cout << output << std::endl;


	input = "Meet me at 03:15 p.m. to 12:28 for coffee.";
	 output = text_normalization::expand_time_english(input);
	std::cout << output << std::endl;

	input = "At 10:03 ";
	output = text_normalization::expand_time_english(input);
	std::cout << output << std::endl;

	return 0;
}