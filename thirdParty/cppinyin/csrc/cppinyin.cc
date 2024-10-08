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

#include "cppinyin.h"
#include "cppinyin_csrc_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace cppinyin {

void PinyinEncoder::Build(const std::string &vocab_path) {
  LoadVocab(vocab_path);

  std::vector<const char *> keys(tokens_.size());
  std::vector<size_t> length(tokens_.size());
  std::vector<int32_t> values(tokens_.size());

  std::iota(values.begin(), values.end(), 0);

  std::stable_sort(values.begin(), values.end(),
                   [&tokens = tokens_](size_t i1, size_t i2) {
                     return tokens[i1] < tokens[i2];
                   });

  for (int32_t i = 0; i < values.size(); ++i) {
    keys[i] = tokens_[values[i]].c_str();
    length[i] = tokens_[values[i]].size();
  }

  da_.build(keys.size(), keys.data(), length.data(), values.data());
  tokens_.clear();
}

void PinyinEncoder::GetDag(const std::string &str, DagType *dag) const {
  dag->resize(str.size());
  for (int32_t i = 0; i < str.size(); ++i) {
    int32_t MAX_HIT = str.size() - i;
    const char *query = str.data() + i;
    std::vector<Darts::DoubleArray::result_pair_type> results(MAX_HIT);
    std::size_t num_matches =
        da_.commonPrefixSearch(query, results.data(), MAX_HIT);
    std::vector<DagItem> items;
    for (int32_t j = 0; j < num_matches; ++j) {
      int32_t idx = results[j].value;
      int32_t length = results[j].length;
      items.push_back(std::make_tuple(scores_[idx], i + length, idx));
    }
    (*dag)[i] = items;
  }
}

void PinyinEncoder::CalcDp(const std::string &str, const DagType &dag,
                           std::vector<DagItem> *route) const {
  route->resize(str.size() + 1);
  (*route)[str.size()] = std::make_tuple(0.0, 0, 0);
  for (int32_t i = str.size() - 1; i >= 0; i--) {
    float max_score = -std::numeric_limits<float>::infinity();
    int32_t max_idx = -1;
    int32_t index = 0;
    for (const auto &item : dag[i]) {
      float score =
          std::get<0>(item) + std::get<0>((*route)[std::get<1>(item)]);
      if (score > max_score) {
        max_score = score;
        max_idx = std::get<1>(item);
        index = std::get<2>(item);
      } else if (score == max_score) {
        if (max_idx >= std::get<1>(item)) {
          max_idx = std::get<1>(item);
          index = std::get<2>(item);
        }
      } else {
        continue;
      }
    }
    (*route)[i] = std::make_tuple(
        max_score == -std::numeric_limits<float>::infinity() ? 0 : max_score,
        max_idx, index);
  }
}

void PinyinEncoder::Cut(const std::string &str,
                        const std::vector<DagItem> &route, bool tone,
                        bool partial, std::vector<std::string> *ostrs) const {
  ostrs->clear();
  int32_t i = 0;
  int32_t fail_bytes = 0;
  while (i < str.size()) {
    int32_t next_index = std::get<1>(route[i]);
    if (next_index == -1) {
      fail_bytes += 1;
      i += 1;
    } else {
      if (fail_bytes != 0) {
        ostrs->emplace_back(str.substr(i - fail_bytes, fail_bytes));
      }
      fail_bytes = 0;
      for (const auto &value : values_[std::get<2>(route[i])]) {
        if (partial) {
          auto initial = GetInitial(value);
          auto final_t = value.substr(initial.size());
          if (!tone) {
            final_t = RemoveTone(final_t);
          }
          if (!initial.empty()) {
            ostrs->push_back(initial);
          }
          ostrs->push_back(final_t);
        } else {
          if (!tone) {
            ostrs->push_back(RemoveTone(value));
          } else {
            ostrs->push_back(value);
          }
        }
      }
      i = next_index;
    }
  }
  if (fail_bytes != 0) {
    ostrs->emplace_back(str.substr(i - fail_bytes, fail_bytes));
  }
}

void PinyinEncoder::EncodeBase(const std::string &str,
                               std::vector<DagItem> *route) const {
  DagType dag;
  GetDag(str, &dag);
  CalcDp(str, dag, route);
}

void PinyinEncoder::EncodeBase(const std::string &str,
                               std::vector<std::string> *ostrs, bool tone,
                               bool partial) const {
  std::vector<DagItem> route;
  EncodeBase(str, &route);
  Cut(str, route, tone, partial, ostrs);
}

void PinyinEncoder::Encode(const std::string &str,
                           std::vector<std::string> *ostrs, bool tone /*=true*/,
                           bool partial /*=false*/) const {
  ostrs->clear();
  std::vector<std::string> substrs;
  std::string word;
  std::istringstream iss(str);
  while (iss >> word) {
    substrs.push_back(word);
  }
  std::vector<std::vector<std::string>> subostrs(substrs.size());
  std::vector<std::future<void>> results;
  for (int32_t i = 0; i < subostrs.size(); ++i) {
    results.emplace_back(
        pool_->enqueue([this, i, &substrs, &subostrs, tone, partial] {
          return this->EncodeBase(substrs[i], &(subostrs[i]), tone, partial);
        }));
  }
  for (auto &&result : results) {
    result.get();
  }
  for (int32_t i = 0; i < subostrs.size(); ++i) {
    ostrs->insert(ostrs->end(), subostrs[i].begin(), subostrs[i].end());
  }
}

void PinyinEncoder::Encode(const std::vector<std::string> &strs,
                           std::vector<std::vector<std::string>> *ostrs,
                           bool tone /*=true*/, bool partial /*=false*/) const {
  ostrs->resize(strs.size());
  std::vector<std::future<void>> results;
  for (int32_t i = 0; i < strs.size(); ++i) {
    results.emplace_back(pool_->enqueue([this, i, &strs, ostrs]() {
      return this->Encode(strs[i], &((*ostrs)[i]));
    }));
  }

  for (auto &&result : results) {
    result.get();
  }
}

void PinyinEncoder::LoadVocab(const std::string &vocab_path) {
  std::ifstream is(vocab_path);
  if (!is) {
    std::cerr << "Open vocab file failed : " << vocab_path.c_str();
    exit(-1);
  }
  tokens_.clear();
  scores_.clear();
  values_.clear();
  std::string line;
  std::string token;
  std::string value;
  float score;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> token >> score;
    tokens_.push_back(token);
    scores_.push_back(score);
    std::vector<std::string> values;
    while (iss >> value) {
      values.push_back(value);
    }
    if (values.empty()) {
      std::cerr << "Each line in vocab should contain at lease three items "
                   "(seperate by space), "
                   "the first one is Chinese word/character, the second one is "
                   "score, the others are the corresponding pinyins given : "
                << line.c_str() << std::endl;
      exit(-1);
    }
    values_.emplace_back(std::move(values));
  }
}

std::string PinyinEncoder::GetInitial(const std::string &s) const {
  std::size_t pos = s.find_first_of(initials_);
  if (pos == 0) {
    std::size_t npos = s.find_first_not_of(initials_, 1);
    if (npos == std::string::npos) {
      return s.substr(0);
    } else {
      // Handle m̄ : 0x6d 0xcc 0x7c and m̀ : 0x6d 0xcc 0x80
      // m is 0x6d
      const uint8_t *p = reinterpret_cast<const uint8_t *>(s.c_str());
      if (p[pos] == 0x6d && p[pos + 1] == 0xcc) {
        return std::string();
      } else {
        return s.substr(0, npos - pos);
      }
    }
  }
  return std::string();
}

std::string PinyinEncoder::RemoveTone(const std::string &s) const {
  std::string phonetic;
  std::size_t len;
  std::size_t pos = s.find_first_of(phonetics_);
  if (pos == std::string::npos) {
    return s;
  } else {
    std::size_t npos = s.find_first_not_of(phonetics_, pos + 1);
    if (npos == std::string::npos) {
      phonetic = s.substr(pos);
      len = s.size() - pos;
    } else {
      len = npos - pos;
      phonetic = s.substr(pos, len);
    }
  }
  std::string phone = phonetics_map_.at(phonetic);

  std::ostringstream oss;
  oss << s.substr(0, pos) << phone << s.substr(pos + len);
  return oss.str();
}

size_t PinyinEncoder::SaveValues(const std::string &model_path) const {
  std::ofstream of(model_path, std::ofstream::binary);
  size_t offset = 0;

  // Write header
  offset += WriteHeader(of);

  // Save scores_
  offset += WriteUint32(of, scores_.size());
  for (const auto score : scores_) {
    offset += WriteFloat(of, score);
  }
  // Save values_
  offset += WriteUint32(of, values_.size());
  for (const auto &value : values_) {
    offset += WriteUint32(of, value.size());
    for (const auto &v : value) {
      offset += WriteString(of, v);
    }
  }
  of.close();
  return offset;
}

size_t PinyinEncoder::LoadValues(std::ifstream &ifile) {
  size_t offset = 0;
  uint32_t size;

  // Load scores_
  offset += ReadUint32(ifile, &size);
  scores_.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    offset += ReadFloat(ifile, &(scores_[i]));
  }

  // Load values_
  offset += ReadUint32(ifile, &size);
  values_.resize(size);
  uint32_t sub_size;
  for (uint32_t i = 0; i < size; ++i) {
    offset += ReadUint32(ifile, &sub_size);
    values_[i].resize(sub_size);
    for (uint32_t j = 0; j < sub_size; ++j) {
      offset += ReadString(ifile, &values_[i][j]);
    }
  }
  return offset;
}

void PinyinEncoder::Load(const std::string &model_path) {
  std::ifstream ifile(model_path, std::ifstream::binary);

  std::string value;
  ReadHeader(ifile, &value);

  if (HEADER != value) {
    ifile.close();
    return Build(model_path);
  }

  size_t offset = LoadValues(ifile) + value.size();
  ifile.close();
  da_.open(model_path.c_str(), "rb", offset);
}

void PinyinEncoder::Save(const std::string &model_path) const {
  SaveValues(model_path);
  da_.save(model_path.c_str(), "ab", 0);
}

} // namespace cppinyin
