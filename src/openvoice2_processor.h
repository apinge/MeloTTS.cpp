// Copyright (C) 2019 FaceUnity Inc. All rights reserved.

#ifndef MELO_OPENVOICE_OPENVOICE_PROCESSOR_H_
#define MELO_OPENVOICE_OPENVOICE_PROCESSOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "status.h"
#include "openvino_model.h"
#include "info_data.h"
#include "tokenizer.h"
//#include "utils.h"
namespace melo
{

    class MemoryInfo
    {
    public:
        double total;
        double free;
        double used;
    };

    class TTSOpenVoiceProcessor
    {
    public:
        TTSOpenVoiceProcessor() = default;
        ~TTSOpenVoiceProcessor();

        /// @brief save generate audio to file
        /// @param text input text
        /// @param sid speaker id , -1 for single speaker; >= 0 for multi-speaker
        /// @param addit_param size==3, 0-noise_scale; 1-length_scale ;
        /// 2-noise_scale_w
        /// @param out_audio_save_path save audio path
        /// @return successs or failure
        Status Process(const std::string &text, const int sid,
                       const std::vector<float> &addit_param, 
                       std::vector<float> &out_audio_buffer);

        Status LoadTTSModel(const std::string &zh_tts_path, const std::string &zh_bert_path, const std::string& tokenizer_data_path);

        Status WriteWave(const std::string &filename, int32_t sampling_rate,
                         const float *samples, int32_t n);
        void SetLanguage(const std::string &language) { language_ = language; }
        int GetSampleRate() { return 16000; }

    private:
        melo::Tokenizer tokenizer;
        Status get_berts(const std::vector<int64_t> &phones,
                         const std::string &language, const std::vector<int> &word2ph,
                         const std::vector<int64_t> &input_ids,
                         const std::vector<int64_t> &attention_mask,
                         const std::vector<int64_t> &token_type_id,
                         std::vector<std::vector<float>> &berts);

        Status bert_infer(const std::vector<int64_t> &input_ids,
                          const std::vector<int64_t> &attention_mask,
                          const std::vector<int64_t> &token_type_ids,
                          std::vector<std::vector<float>> &bert_feats);

        Status tts_infer(const std::vector<int64_t> &target_seq, const int64_t speakers,
                         const std::vector<int64_t> &tones,
                         const std::vector<int64_t> &lang_ids,
                         const std::vector<float> &bert,
                         const std::vector<float> &ja_bert, std::vector<float> &wavs);

        Status converter_infer(const std::vector<float> &spec,
                               const std::vector<float> &source_se,
                               const std::vector<float> &target_se,
                               std::vector<float> &wavs);

        std::shared_ptr<OpenvinoModel> openvoice_zh_tts_model_;
        std::shared_ptr<OpenvinoModel> openvoice_en_tts_model_;
        std::shared_ptr<OpenvinoModel> openvoice_zh_bert_model_;
        std::shared_ptr<OpenvinoModel> openvoice_en_bert_model_;
        std::shared_ptr<OpenvinoModel> openvoice_bert_model_;
        std::shared_ptr<OpenvinoModel> openvoice_cv_model_;
        std::shared_ptr<OpenvinoModel> openvoice_tts_model_;


        std::string language_ = "ZH";

        std::vector<std::vector<float>> speaker_emb;

        float sdp_ration_ = 0.2;
        float noise_scale_ = 0.6;
        float length_scale_ = 1.00;
        float noise_scale_w_ = 0.80;

    };

} // namespace melo

#endif // MELO_OPENVOICE_OPENVOICE_PROCESSOR_H_
