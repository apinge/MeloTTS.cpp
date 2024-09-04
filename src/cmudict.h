#ifndef CMUDICT_H
#define CMUDICT_H

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

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

     
        inline  std::optional<std::reference_wrapper<const std::vector<std::vector<std::string>>>> find(const std::string& key) const {
            if ( dict_.contains(key)) {
                return std::cref(dict_.at(key));
            }
            else {
                return std::nullopt;
            }
        }
    // Friend function for overloading the operator<<
    friend std::ostream& operator<<(std::ostream& os, const CMUDict& dict);
    private:
        std::unordered_map<std::string, std::vector<std::vector<std::string>>> dict_;
    };
}

#endif // CMUDICT_H
