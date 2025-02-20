#include <iostream>
#include "text_normalization_eng.h"


int main() {
	std::string input = "I have 1234567.893 in text and 201th in echo 253,235,365";
	std::string output = text_normalization::normalize_numbers(input);
	std::cout << output << std::endl;

	input = "64000.30 sheep";
	output = text_normalization::normalize_numbers(input);
	std::cout << output << std::endl;


	input = "Dr. Smith went to St. John's Church with Mr. Brown";
	output = text_normalization::expand_abbreviations(input);
	std::cout << output << std::endl;


	input = "Meet me at 03:15 p.m. to 12:28 for coffee.";
	output = text_normalization::expand_time_english(input);
	std::cout << output << std::endl;

	input = "At 10:03 ";
	output = text_normalization::expand_time_english(input);
	std::cout << output << std::endl;

	return 0;
}