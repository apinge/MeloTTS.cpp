#include <filesystem>
#include "bert.h"
#include "openvoice_tts.h"
std::vector<std::vector<float>> prepare_bert() {
    std::filesystem::path zh_bert_path = "C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\bert_zn_mix_en.xml";
    std::cout << std::filesystem::absolute(zh_bert_path) << std::endl;
    std::cout << zh_bert_path.string() << std::endl;

    std::shared_ptr<melo::Tokenizer> tokenizer_ptr = std::make_shared<melo::Tokenizer>("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\vocab_bert.txt");

    //std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
    std::shared_ptr<ov::Core> core_ptr = std::make_shared<ov::Core>();
    melo::Bert zh_bert(core_ptr, zh_bert_path.string(), "CPU", "ZH", tokenizer_ptr);


    std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
    std::vector<int> word2ph{ 3, 4, 4, 4, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 14, 20, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 6, 20, 2 };
   
    std::vector<std::vector<float>> berts;
    zh_bert.get_bert_feature(text, word2ph, berts);
    std::cout << "berts.size():"<<  berts.size() << " " << berts.front().size() << std::endl;

    return berts;
}
int main() {
    
    std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
    std::filesystem::path zh_bert_path = "C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\thirdParty\\tts_ov\\tts_zn_mix_en.xml";
    melo::OpenVoiceTTS model(core_ptr, zh_bert_path.string(),"CPU", "ZH");

    std::vector<std::vector<float>> phone_level_feature = prepare_bert();

    std::vector<int64_t> phones_ids{ 0,  0,  0, 19,  0, 44,  0, 99,  0, 40,  0, 73,  0, 40,  0, 57,  0, 12,
          0, 60,  0, 71,  0, 18,  0, 59,  0, 32,  0, 37,  0, 89,  0, 55,  0, 49,
          0, 57,  0, 26,  0, 62,  0, 31,  0, 21,  0, 67,  0, 37,  0, 14,  0, 77,
          0, 82,  0, 77,  0, 52,  0, 21,  0, 14,  0, 34,  0, 12,  0, 63,  0, 57,
          0, 77,  0, 12,  0, 62,  0, 10,  0, 74,  0, 35,  0, 99,  0, 12,  0, 60,
          0, 12,  0, 62,  0, 78,  0, 76,  0, 78,  0, 89,  0, 23,  0, 16,  0, 73,
          0, 95,  0, 77,  0, 52,  0, 23,  0, 26,  0, 60,  0, 82,  0, 19,  0, 14,
          0, 77,  0, 52,  0, 21,  0, 14,  0, 78,  0, 28,  0, 60,  0, 71,  0, 59,
          0, 12,  0, 78,  0, 10,  0, 74,  0, 35,  0, 99,  0, 12,  0, 60,  0, 12,
          0, 62,  0, 78,  0, 76,  0,  0,  0 };
    std::vector<int64_t> lang_ids{ 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,

     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
     0, 3, 0 };
    std::vector <int64_t> tones{ 0, 0, 0, 1, 0, 1, 0, 4, 0, 4, 0, 4, 0, 4, 0, 7, 0, 8, 0, 7, 0, 7, 0, 9,
     0, 7, 0, 8, 0, 4, 0, 4, 0, 4, 0, 4, 0, 3, 0, 3, 0, 2, 0, 2, 0, 2, 0, 2,
     0, 2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 1, 0, 1, 0, 7, 0, 9, 0, 7, 0, 7,
     0, 7, 0, 8, 0, 7, 0, 9, 0, 7, 0, 7, 0, 7, 0, 8, 0, 7, 0, 8, 0, 7, 0, 7,
     0, 7, 0, 1, 0, 1, 0, 3, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 5, 0, 5, 0, 2,
     0, 2, 0, 3, 0, 3, 0, 2, 0, 2, 0, 1, 0, 1, 0, 7, 0, 9, 0, 7, 0, 7, 0, 7,
     0, 8, 0, 7, 0, 9, 0, 7, 0, 7, 0, 7, 0, 8, 0, 7, 0, 8, 0, 7, 0, 7, 0, 7,
     0, 0, 0 };
     std::vector<float> wav_data = model.tts_infer(phones_ids, tones, phone_level_feature, lang_ids);
     std::cout << "wav infer ok\n";
     model.write_wave("C:\\Users\\gta\\source\\develop\\MeloTTS.cpp.current\\test_openvoice_tts.wav", 44100, wav_data.data(), wav_data.size());
     std::cout << "write wav ok\n";
    //word2ph.push_back({ 3, 4, 4, 4, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 14, 20, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 6, 20, 2 });

    return 0;
}