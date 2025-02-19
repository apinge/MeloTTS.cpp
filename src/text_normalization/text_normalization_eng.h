#ifndef NUM_TO_ENG_H
#define NUM_TO_ENG_H
#include <vector>
#include <string>
#include <sstream>

namespace text_normalization {

	// @brief convert number to words (e.g. 11 -> eleven, 11st -> eleventh)
    std::string normalize_numbers(const std::string& text);

    // @brief replace abbreviations (e.g. mrs. -> misess) while ignoring case.
    std::string expand_abbreviations(const std::string& text);

	// @brief expand time in English (e.g. 03:15 p.m. -> three fifteen p m)
    std::string expand_time_english(const std::string& text);
	


}
#endif