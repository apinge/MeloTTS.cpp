/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com)
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
#ifndef CMUDICT_H
#define CMUDICT_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace melo {
class CMUDict {
public:
    explicit CMUDict(const std::string& filename);
    ~CMUDict() = default;

    CMUDict() = delete;
    CMUDict(const CMUDict&) = delete;
    CMUDict(CMUDict&&) = delete;
    CMUDict& operator=(const CMUDict&) = delete;
    CMUDict& operator=(CMUDict&&) = delete;

    inline std::optional<std::reference_wrapper<const std::vector<std::string>>> find(const std::string& key) const {
        if (dict_.contains(key)) {
            return std::cref(dict_.at(key));
        } else {
            return std::nullopt;
        }
    }
    // Friend function for overloading the operator<<
    friend std::ostream& operator<<(std::ostream& os, const CMUDict& dict);

private:
    std::unordered_map<std::string, std::vector<std::string>> dict_;
};
}  // namespace melo

#endif  // CMUDICT_H
