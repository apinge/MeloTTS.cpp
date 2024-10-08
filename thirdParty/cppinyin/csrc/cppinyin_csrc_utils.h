#ifndef CPPINYIN_CSRC_UTILS_H_
#define CPPINYIN_CSRC_UTILS_H_

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace cppinyin {

constexpr auto HEADER = "__kcppinyinw__";

size_t ReadUint32(std::ifstream &ifile, uint32_t *data);

size_t WriteUint32(std::ofstream &ofile, uint32_t data);

size_t ReadFloat(std::ifstream &ifile, float *data);

size_t WriteFloat(std::ofstream &ofile, float data);

size_t ReadString(std::ifstream &ifile, std::string *data);

size_t WriteString(std::ofstream &ofile, const std::string &data);

size_t ReadHeader(std::ifstream &ifile, std::string *data);

size_t WriteHeader(std::ofstream &ofile);

} // namespace cppinyin

#endif // CPPINYIN_CSRC_UTILS_H_
