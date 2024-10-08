/**
 * Copyright      2024    Wei Kang (wkang@pku.edu.cn)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPPINYIN_CSRC_CPPINYIN_H_
#define CPPINYIN_CSRC_CPPINYIN_H_

#include <cstdlib>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>
#include <cassert>
#include <iostream>
#include "cppinyin.h"
#include "darts.h"
#include "threadpool.h"
#include "cppinyin_csrc_utils.h"

namespace cppinyin {

class PinyinEncoder {
  // <token score, index into input str, index into tokens>
  using DagItem = std::tuple<float, int32_t, int32_t>;
  using DagType = std::vector<std::vector<DagItem>>;

public:
  PinyinEncoder(const std::string &vocab_path,
                int32_t num_threads = std::thread::hardware_concurrency()) {
    pool_ = std::make_unique<ThreadPool>(num_threads);

    Load(vocab_path);
    std::cout <<"cppinyin::PinyinEncoder Constructed!\n";
  }
  PinyinEncoder(const std::filesystem::path& vocab_path,
      int32_t num_threads = std::thread::hardware_concurrency()) {
      pool_ = std::make_unique<ThreadPool>(num_threads);
      assert(std::filesystem::exists(vocab_path)&&"cppinyin resources does not exit!");
      Load(vocab_path.string());
      std::cout << "cppinyin::PinyinEncoder Constructed!\n";
  }
  PinyinEncoder(int32_t num_threads = std::thread::hardware_concurrency()) {
    pool_ = std::make_unique<ThreadPool>(num_threads);
  }
  
  ~PinyinEncoder() {}

  void Encode(const std::string &str, std::vector<std::string> *ostrs,
              bool tone = true, bool partial = false) const;

  void Encode(const std::vector<std::string> &strs,
              std::vector<std::vector<std::string>> *ostrs, bool tone = true,
              bool partial = false) const;

  void Load(const std::string &model_path);

  void Save(const std::string &model_path) const;

private:
  void Build(const std::string &vocab_path);

  void LoadVocab(const std::string &vocab_path);

  void EncodeBase(const std::string &str, std::vector<DagItem> *route) const;

  void EncodeBase(const std::string &str, std::vector<std::string> *ostrs,
                  bool tone, bool partial) const;

  void GetDag(const std::string &str, DagType *dag) const;

  void CalcDp(const std::string &str, const DagType &dag,
              std::vector<DagItem> *route) const;

  void Cut(const std::string &str, const std::vector<DagItem> &route, bool tone,
           bool partial, std::vector<std::string> *ostrs) const;

  std::string GetInitial(const std::string &s) const;

  std::string RemoveTone(const std::string &s) const;

  size_t SaveValues(const std::string &model_path) const;
  size_t LoadValues(std::ifstream &ifile);

  // Note: zh ch sh not included
  // Treat y w as initials
  std::string initials_ = "bpmfdtnlgkhjqxrzcsyw";
  std::string phonetics_ = "āáǎàēéěèōóǒòīíǐìūúǔùǖǘǚǜńňǹm̄ḿm̀";
  std::unordered_map<std::string, std::string> phonetics_map_ = {
      {"ā", "a"}, {"á", "a"}, {"ǎ", "a"}, {"à", "a"}, {"ē", "e"}, {"é", "e"},
      {"ě", "e"}, {"è", "e"}, {"ō", "o"}, {"ó", "o"}, {"ǒ", "o"}, {"ò", "o"},
      {"ī", "i"}, {"í", "i"}, {"ǐ", "i"}, {"ì", "i"}, {"ū", "u"}, {"ú", "u"},
      {"ǔ", "u"}, {"ù", "u"}, {"ǖ", "ü"}, {"ǘ", "ü"}, {"ǚ", "ü"}, {"ǜ", "ü"},
      {"ń", "n"}, {"ň", "n"}, {"ǹ", "n"}, {"m̄", "m"}, {"ḿ", "m"}, {"m̀", "m"}};
  std::vector<std::string> tokens_;
  std::vector<float> scores_;
  std::vector<std::vector<std::string>> values_;
  std::unique_ptr<ThreadPool> pool_;
  Darts::DoubleArray da_;
};

} // namespace cppinyin

#endif // CPPINYIN_CSRC_CPPINYIN_H_
