#include "english.h"
namespace melo{
    // Constructor
    English::English(const std::filesystem::path& data_folder) {
        //english pronounciation dict
        auto cmudict_path = data_folder / "cmudict_cache.txt";


        if (!std::filesystem::exists(cmudict_path)) {
            std::cerr << "[ERROR] ChineseMix::file does not exists: " << std::filesystem::absolute(cmudict_path) << "\n";
        }

        cmudict = std::make_shared<CMUDict>(cmudict_path.string());
        std::cout << "[INFO] Init English language Module Succeed!\n";
    }

	std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> English::g2p(const std::string& segment, std::shared_ptr<OpenVinoTokenizer>& tokenizer) {
        return {};
	}
    // TODO: implement the function
    std::string English::text_normalize(const std::string& text) {
        return text;
    }

}