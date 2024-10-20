#include <filesystem>
#include "bert.h"
#define OV_MODEL_PATH "ov_models"
int main() {
    std::filesystem::path model_dir = OV_MODEL_PATH;
    std::filesystem::path zh_bert_path = model_dir / "bert_zn_mix_en.xml";
    std::cout << std::filesystem::absolute(zh_bert_path) << std::endl;
    std::cout << zh_bert_path.string() << std::endl;

    std::shared_ptr<melo::Tokenizer> tokenizer_ptr = std::make_shared<melo::Tokenizer>((model_dir / "vocab_bert.txt"));

    //std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
    std::shared_ptr<ov::Core> core_ptr = std::make_shared<ov::Core>();
    melo::Bert zh_bert(core_ptr, zh_bert_path.string(), "CPU", "ZH", tokenizer_ptr);


    //std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    //std::vector<int> word2ph{ 3, 4, 4, 4, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 14, 20, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 6, 20, 2 };
    std::string text = "今天的meeting真的是超级productive";
    std::vector<int> word2ph{ 3, 4, 4, 4, 10, 4, 4, 4, 4, 4, 10, 8, 2 };
    std::vector<std::vector<float>> berts;
    zh_bert.get_bert_feature(text, word2ph, berts);
    std::cout << berts.size() << " "<< berts.front().size() << std::endl;

    return 0;
}