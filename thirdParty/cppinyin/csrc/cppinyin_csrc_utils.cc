#include "cppinyin_csrc_utils.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace cppinyin {

size_t ReadUint32(std::ifstream &ifile, uint32_t *data) {
  ifile.read(reinterpret_cast<char *>(data), sizeof(uint32_t));
  return sizeof(uint32_t);
}

size_t WriteUint32(std::ofstream &ofile, uint32_t data) {
  ofile.write(reinterpret_cast<const char *>(&data), sizeof(uint32_t));
  return sizeof(uint32_t);
}

size_t ReadFloat(std::ifstream &ifile, float *data) {
  ifile.read(reinterpret_cast<char *>(data), sizeof(float));
  return sizeof(float);
}

size_t WriteFloat(std::ofstream &ofile, float data) {
  ofile.write(reinterpret_cast<const char *>(&data), sizeof(float));
  return sizeof(float);
}

size_t ReadString(std::ifstream &ifile, std::string *data) {
  uint32_t size;
  ifile.read(reinterpret_cast<char *>(&size), sizeof(uint32_t));
  data->resize(size);
  ifile.read(const_cast<char *>(data->c_str()), size);
  return sizeof(uint32_t) + size;
}

size_t WriteString(std::ofstream &ofile, const std::string &data) {
  uint32_t size = data.size();
  ofile.write(reinterpret_cast<const char *>(&size), sizeof(uint32_t));
  ofile.write(&data[0], size);
  return sizeof(uint32_t) + size;
}

size_t ReadHeader(std::ifstream &ifile, std::string *data) {
  std::string header(HEADER);
  data->resize(header.size());
  ifile.read(const_cast<char *>(data->c_str()), header.size());
  return header.size();
}

size_t WriteHeader(std::ofstream &ofile) {
  std::string header(HEADER);
  ofile.write(&header[0], header.size());
  return header.size();
}

} // namespace cppinyin
