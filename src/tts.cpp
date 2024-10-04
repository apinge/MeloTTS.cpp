#include <cassert>
#include <fstream>
#include "tts.h"
#include "info_data.h"
#include "chinese_mix.h"
namespace melo {
    TTS::TTS(std::unique_ptr<ov::Core>& core, const std::filesystem::path & tts_ir_path, const std::string & tts_device,
        const std::filesystem::path& bert_ir_path, const std::string& bert_device, const std::filesystem::path& tokenizer_data_path,
        const std::string language):_language(language),
        tts_model(core,tts_ir_path,tts_device,language), tokenizer(std::make_shared<Tokenizer>(tokenizer_data_path.string())){

        assert((core.get() != nullptr) && "core should not be null!");
        assert((std::filesystem::exists(tts_ir_path) && std::filesystem::exists(bert_ir_path)  && std::filesystem::exists(tokenizer_data_path))
            && "ir files or vocab_bert does not exit!");
        //init bert 
        bert_model = Bert(core,bert_ir_path, bert_device,language, tokenizer);
        std::cout << "TTS:TTS:init bert_model\n";
    }

    void TTS::tts_to_file(const std::string& text, const int& speaker_id, const std::filesystem::path& output_path,
        const float& sdp_ratio, const float& noise_scale, const float& noise_scale_w , const float& speed){
        try {
            // structured binding
            auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(text);
            std::vector<float> wav_data = tts_model.tts_infer(phones_ids, tones, lang_ids, phone_level_feature);
            write_wave(output_path.string(), wav_data);
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }
    std::tuple<std::vector<std::vector<float>>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> 
        TTS::get_text_for_tts_infer(const std::string& text) {
        try {
            auto [phones_list, tones_list, word2ph_] = chinese_mix::_g2p_v2("编译器", tokenizer);
            std::string text = "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
            std::vector<int> word2ph{ 3, 4, 4, 4, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 14, 20, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 6, 20, 2 };
            std::vector<std::vector<float>> phone_level_feature;
            bert_model.get_bert_feature(text,word2ph,phone_level_feature);
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
            return { phone_level_feature, phones_ids, tones, lang_ids };
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }

    void TTS::write_wave(const std::filesystem::path& output_path, const std::vector<float>& wave, const int32_t& sampling_rate) {
        try {
            size_t n = wave.size();
            melo::WaveHeader header;
            header.chunk_id = 0x46464952;     // FFIR
            header.format = 0x45564157;       // EVAW
            header.subchunk1_id = 0x20746d66; // "fmt "
            header.subchunk1_size = 16;       // 16 for PCM
            header.audio_format = 1;          // PCM =1

            int32_t num_channels = 1;
            int32_t bits_per_sample = 16; // int16_t
            header.num_channels = num_channels;
            header.sample_rate = sampling_rate;
            header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
            header.block_align = num_channels * bits_per_sample / 8;
            header.bits_per_sample = bits_per_sample;
            header.subchunk2_id = 0x61746164; // atad
            header.subchunk2_size = n * num_channels * bits_per_sample / 8;

            header.chunk_size = 36 + header.subchunk2_size;

            std::vector<int16_t> samples_int16(n);
            for (int32_t i = 0; i != n; ++i)
            {
                samples_int16[i] = wave[i] * 32676;
            }

            std::ofstream os(output_path.string(), std::ios::binary);
            if (!os)
            {
                std::string msg = "Failed to create " + output_path.string();
                /* MELO_LOG(MELO_ERROR) << msg;
                 MELO_ERROR_RETURN(msg);*/
            }

            os.write(reinterpret_cast<const char*>(&header), sizeof(header));
            os.write(reinterpret_cast<const char*>(samples_int16.data()),
                samples_int16.size() * sizeof(int16_t));

            if (!os)
            {
                std::string msg = "Write " + output_path.string() + " failed.";
                /* MELO_LOG(MELO_ERROR) << msg;
                 MELO_ERROR_RETURN(msg);*/
            }
            std::cout << "write wav to " << output_path.string() << std::endl;
            return;
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }
}
