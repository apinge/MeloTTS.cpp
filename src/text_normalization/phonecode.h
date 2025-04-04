/**
 * Copyright      2025    Alex G Chen (alex.g.chen@intel.com)
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
#pragma once
#ifndef PHONECODE_H
#define PHONECODE_H

#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace text_normalization {
extern std::wregex re_mobile_phone;
extern std::wregex re_telephone;
extern std::wregex re_national_uniform_number;
extern std::wregex re_country_code;

std::wstring phone2str(const std::wstring& phone_string, bool mobile = true);
std::wstring replace_phone(const std::wsmatch& match);
std::wstring replace_mobile(const std::wsmatch& match);
std::wstring process_mobile_number(const std::wstring& phone);
std::wstring process_landline_number(const std::wstring& phone);
std::wstring process_uniform_number(const std::wstring& phone);
bool is_valid_phone_number(const std::wstring& text, const std::wsmatch& match);
}  // namespace text_normalization

#endif