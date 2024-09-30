
#include "openvoice2_processor.h"
#include "librosa.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <list>
#include <memory>
#include <sstream>
#include <utility>
#include <complex>
#include <chrono>

namespace melo
{

    MeloTTSProcessor::~MeloTTSProcessor()
    {

        if (openvoice_zh_tts_model_)
        {
            openvoice_zh_tts_model_ = nullptr;
        }
        if (openvoice_tts_model_)
        {
            openvoice_tts_model_ = nullptr;
        }
        if (openvoice_zh_bert_model_)
        {
            openvoice_zh_bert_model_ = nullptr;
        }
        if (openvoice_bert_model_)
        {
            openvoice_bert_model_ = nullptr;
        }

    }

    Status MeloTTSProcessor::LoadTTSModel(const std::string &zh_tts_path,  const std::string &zh_bert_path, const std::string & tokenizer_data_path)
    {
        std::string device_name = "CPU";
        MELO_LOG(MELO_TRACE) << "LoadTTSModel start";
        openvoice_zh_tts_model_ = std::make_shared<OpenvinoModel>(core_ptr);
        openvoice_zh_tts_model_->Init(zh_tts_path, device_name);


        openvoice_zh_bert_model_ = std::make_shared<OpenvinoModel>(core_ptr);
        openvoice_zh_bert_model_->Init(zh_bert_path, device_name);

        //init tokenizer
        tokenizer = melo::Tokenizer(tokenizer_data_path);
        tokenizer_ptr = std::make_shared<melo::Tokenizer>(tokenizer_data_path);
        zh_bert = melo::Bert(core_ptr, zh_bert_path, "CPU", "ZH", tokenizer_ptr);

        bool flag = false;
        if (openvoice_zh_bert_model_ == nullptr)
        {
            std::string msg =
                "zh bert model " + std::string(GetErrorMsg(MELO_INIT_MODEL_ERROR));
            MELO_LOG(MELO_ERROR) << msg;
            flag = true;
        }


        if (openvoice_zh_tts_model_ == nullptr)
        {
            std::string msg =
                "zh tts model " + std::string(GetErrorMsg(MELO_INIT_MODEL_ERROR));
            MELO_LOG(MELO_ERROR) << msg;
            flag = true;
        }

        if (flag)
        {
            MELO_LOG(MELO_ERROR) << GetErrorMsg(MELO_INIT_MODEL_ERROR);
            std::string msg = GetErrorMsg(MELO_INIT_MODEL_ERROR);
            MELO_ERROR_RETURN(msg);
        }
        MELO_LOG(MELO_TRACE) << "LoadTTSModel end";
        return Status::OK();
    }

    Status MeloTTSProcessor::Process(const std::string &text, const int sid,
                                          const std::vector<float> &addit_param,
                                          std::vector<float> &out_audio_buffer)
    {
        MELO_LOG(MELO_DEBUG) << "TTSProcessor::Process start";
        auto time_bb = std::chrono::steady_clock::now();

        sdp_ration_ = addit_param[0];
        noise_scale_ = addit_param[1];
        length_scale_ = addit_param[2];
        noise_scale_w_ = addit_param[3];

        auto ReadText = [&](const std::string& path, std::string& content)
        {
            std::ifstream fin(path);
            if (!fin.is_open())
            {
                MELO_LOG(MELO_ERROR) << "Open file faild! filename: " << path;
                return;
            }
            content = std::string(std::istreambuf_iterator<char>(fin),
                std::istreambuf_iterator<char>());
            fin.close();
        };

        auto readSpeakerSe = [&](const std::string &target_se_path, std::vector<std::vector<float>> &emb)
        {
            std::string target_se_string;
            ReadText(target_se_path, target_se_string);
            std::istringstream file(target_se_string);
            std::string line;
            std::vector<float> tmp_spk;
            while (getline(file, line))
            {
                float value;
                std::istringstream iss(line);
                if (!(iss >> value))
                {
                    std::cerr << "parse error" << std::endl;
                    continue;
                }
                tmp_spk.push_back(value);
                if (tmp_spk.size() == 256)
                {
                    emb.push_back(tmp_spk);
                    tmp_spk.clear();
                }
            }
        };

        auto readInt64VecrFromFile = [&](const std::string save_path,
                                         std::vector<std::vector<int64_t>> &saveVec)
        {
            std::ifstream FILE(save_path, std::ios::in | std::ifstream::binary);

            int size = 0;
            FILE.read(reinterpret_cast<char *>(&size), sizeof(size));
            saveVec.resize(size);
            for (int n = 0; n < size; ++n)
            {
                int size2 = 0;
                FILE.read(reinterpret_cast<char *>(&size2), sizeof(size2));
                saveVec[n].resize(size2);
                FILE.read(reinterpret_cast<char *>(saveVec[n].data()), sizeof(int64_t) * size2);
            }
        };

        auto readIntVecFromFile = [&](const std::string save_path,
                                      std::vector<std::vector<int>> &saveVec)
        {
            std::ifstream FILE(save_path, std::ios::in | std::ifstream::binary);

            int size = 0;
            FILE.read(reinterpret_cast<char *>(&size), sizeof(size));
            saveVec.resize(size);
            for (int n = 0; n < size; ++n)
            {
                int size2 = 0;
                FILE.read(reinterpret_cast<char *>(&size2), sizeof(size2));
                saveVec[n].resize(size2);
                FILE.read(reinterpret_cast<char *>(saveVec[n].data()), sizeof(int) * size2);
            }
        };

        //Bert input
        std::vector<std::vector<int64_t>> input_ids;
        std::vector<std::vector<int64_t>> attention_mask;
        std::vector<std::vector<int64_t>> token_type_id;

        //TTS input
        std::vector<std::vector<int64_t>> phones_ids;
        std::vector<std::vector<int>> word2ph;
        std::vector<std::vector<int64_t>> tones;
        std::vector<std::vector<int64_t>> lang_ids;


        //readInt64VecrFromFile(data_path + "phones_ids.bin", phones_ids);
        //readIntVecFromFile(data_path + "word2ph.bin", word2ph);
        //readInt64VecrFromFile(data_path + "input_ids.bin", input_ids);
        //readInt64VecrFromFile(data_path + "attention_mask.bin", attention_mask);
        //readInt64VecrFromFile(data_path + "token_type_id.bin", token_type_id);
        //readInt64VecrFromFile(data_path + "tones.bin", tones);
        //readInt64VecrFromFile(data_path + "lang_ids.bin", lang_ids);

        //example 1
        //input_ids.push_back({ 101, 4348, 9416, 1659, 3530, 5675, 9074, 2082,  102 });
        //attention_mask.push_back({ 1, 1, 1, 1, 1, 1, 1, 1, 1 });
        //token_type_id.push_back({ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
        //phones_ids.push_back({ 0, 0, 0, 22, 0, 90, 0, 34, 0, 31, 0, 19, 0, 82, 0, 23, 0, 82,
        //    0, 99, 0, 93, 0, 60, 0, 30, 0, 35, 0, 86, 0, 0, 0 });
        //lang_ids.push_back({ 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
        //0, 3, 0, 3, 0, 3, 0, 3, 0 });
        //tones.push_back({ 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2l, 0, 4, 0, 4, 0, 4, 0, 4, 0, 2,
        //0, 2, 0, 1, 0, 1, 0, 0, 0 });
        //word2ph.push_back({ 3, 4, 4, 4, 4, 4, 4, 4, 2 });


        //example 2
        
        // "编译器compiler会尽可能从函数实参function arguments推导缺失的模板实参template arguments";
        std::vector<int64_t> ids;
        std::vector<std::string> strs;
        tokenizer.Tokenize(text,strs,ids);
        input_ids.push_back(ids);
        int n = input_ids.front().size();
        attention_mask.push_back(std::vector<int64_t>(n,1));
        token_type_id.push_back(std::vector<int64_t>(n, 0));
        phones_ids.push_back({ 0,  0,  0, 19,  0, 44,  0, 99,  0, 40,  0, 73,  0, 40,  0, 57,  0, 12,
          0, 60,  0, 71,  0, 18,  0, 59,  0, 32,  0, 37,  0, 89,  0, 55,  0, 49,
          0, 57,  0, 26,  0, 62,  0, 31,  0, 21,  0, 67,  0, 37,  0, 14,  0, 77,
          0, 82,  0, 77,  0, 52,  0, 21,  0, 14,  0, 34,  0, 12,  0, 63,  0, 57,
          0, 77,  0, 12,  0, 62,  0, 10,  0, 74,  0, 35,  0, 99,  0, 12,  0, 60,
          0, 12,  0, 62,  0, 78,  0, 76,  0, 78,  0, 89,  0, 23,  0, 16,  0, 73,
          0, 95,  0, 77,  0, 52,  0, 23,  0, 26,  0, 60,  0, 82,  0, 19,  0, 14,
          0, 77,  0, 52,  0, 21,  0, 14,  0, 78,  0, 28,  0, 60,  0, 71,  0, 59,
          0, 12,  0, 78,  0, 10,  0, 74,  0, 35,  0, 99,  0, 12,  0, 60,  0, 12,
          0, 62,  0, 78,  0, 76,  0,  0,  0 });
        lang_ids.push_back({ 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,

         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0 });
        tones.push_back({ 0, 0, 0, 1, 0, 1, 0, 4, 0, 4, 0, 4, 0, 4, 0, 7, 0, 8, 0, 7, 0, 7, 0, 9,
         0, 7, 0, 8, 0, 4, 0, 4, 0, 4, 0, 4, 0, 3, 0, 3, 0, 2, 0, 2, 0, 2, 0, 2,
         0, 2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 1, 0, 1, 0, 7, 0, 9, 0, 7, 0, 7,
         0, 7, 0, 8, 0, 7, 0, 9, 0, 7, 0, 7, 0, 7, 0, 8, 0, 7, 0, 8, 0, 7, 0, 7,
         0, 7, 0, 1, 0, 1, 0, 3, 0, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 5, 0, 5, 0, 2,
         0, 2, 0, 3, 0, 3, 0, 2, 0, 2, 0, 1, 0, 1, 0, 7, 0, 9, 0, 7, 0, 7, 0, 7,
         0, 8, 0, 7, 0, 9, 0, 7, 0, 7, 0, 7, 0, 8, 0, 7, 0, 8, 0, 7, 0, 7, 0, 7,
         0, 0, 0 });
        word2ph.push_back({ 3, 4, 4, 4, 8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 14, 20, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 6, 20, 2 });
        
        /*printVec(input_ids,"input_ids");
        printVec(attention_mask, "attention_mask");
        printVec(token_type_id,"token_type_id_.bin");
        printVec(phones_ids, "phones_ids");
        printVec(lang_ids, "lang_ids");
        printVec(tones,"tones");
        printVec(word2ph, "word2ph");
        

        system("pause");*/
        int target_speaker_id = sid;
        if (sid < 0 || sid >= speaker_emb.size())
        {
            target_speaker_id = 0;
        }
        int source_speaker_id = 1;

        if ((phones_ids.size() == tones.size()) &&
            (input_ids.size() == token_type_id.size()))
        {
            int n = 1;

            for (int text_idx = 0; text_idx < n; ++text_idx)
            {
                std::vector<int64_t> input_phone_ids = phones_ids[text_idx];
                std::vector<int> input_word2ph = word2ph[text_idx];
                std::vector<int64_t> input_input_ids = input_ids[text_idx];
                std::vector<int64_t> input_token_type_id = token_type_id[text_idx];
                std::vector<int64_t> input_attention_mask = attention_mask[text_idx];
                std::vector<int64_t> input_tones = tones[text_idx];
                std::vector<int64_t> input_lang_ids = lang_ids[text_idx];
                MELO_LOG(MELO_DEBUG)
                    << "< " << text_idx
                    << " ---------------------------------------------- phones size : "
                    << input_phone_ids.size() << " ->";
                std::vector<std::vector<float>> ja_bert, jb_bert;
                std::vector<float> bert;
                std::vector<float> jabert;
                zh_bert.get_bert_feature(text, input_word2ph, ja_bert);
                //MELO_RETURN_IF_ERROR(get_berts(input_phone_ids, language_, input_word2ph,
                //                               input_input_ids, input_token_type_id,
                //                               input_attention_mask, ja_bert));
                std::cout << "ja_bert" << ja_bert.size() << " "<<ja_bert.front().size() << std::endl;
                std::cout << "input_phone_ids.size()" << input_phone_ids.size() << std::endl;
               // std::cout << "jb_bert" << jb_bert.size() << " " << jb_bert.front().size() << std::endl;
                for (int j = 0; j < phones_ids[text_idx].size(); ++j)
                {
                    std::vector<float> tmp(1024, 0.0);
                    bert.insert(bert.end(), tmp.begin(), tmp.end());
                }
                int row = ja_bert.size();
                int col = ja_bert[0].size();
                // jabert.resize(col * row);
                for (int k = 0; k < col; ++k)
                {
                    for (int j = 0; j < row; ++j)
                    {
                        jabert.push_back(ja_bert[j][k]);
                    }
                }

                std::vector<float> wavs;
                if (language_ == "EN")
                {
                    source_speaker_id = 0;
                }

                auto time_b = std::chrono::steady_clock::now();
                MELO_RETURN_IF_ERROR(tts_infer(input_phone_ids, source_speaker_id,
                                               input_tones, input_lang_ids, bert, jabert,
                                               wavs));

                int n_fft = 1024;
                int hop_length = 256;
                int win_length = 1024;

                std::vector<std::vector<std::complex<float>>> X =
                    melo::Feature::stft(wavs, n_fft, hop_length, "hann", true, "reflect");

                row = X.size();
                col = X[0].size();
                std::vector<float> spec(row * col, 0.0);

                for (int k = 0; k < row; ++k)
                {
                    for (int j = 0; j < col; ++j)
                    {
                        spec[k * col + j] = std::sqrt(std::pow(std::abs(X[k][j]), 2) + 1e-6);
                    }
                }

               /* std::vector<float> cv_wavs;

                MELO_RETURN_IF_ERROR(converter_infer(spec, source_se_[source_speaker_id],
                                                     speaker_emb[sid], cv_wavs));

                out_audio_buffer.insert(out_audio_buffer.end(), cv_wavs.begin(),
                                        cv_wavs.end());*/
                out_audio_buffer.insert(out_audio_buffer.end(), wavs.begin(), wavs.end());
                MELO_LOG(MELO_DEBUG) << "<---------------------------------------------- "
                                     << text_idx << " >";
                // out_audio_buffer.insert(out_audio_buffer.end(), wavs.begin(),
                // wavs.end());
            }
        }
        else
        {
            std::string msg = "Error error input_ids, phones and other data";
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }
        // release infer buffer after inference
        openvoice_zh_tts_model_->ReleaseInferMemory();
        openvoice_zh_bert_model_->ReleaseInferMemory();

        return Status::OK();
    }

    Status MeloTTSProcessor::get_berts(
        const std::vector<int64_t> &phones, const std::string &language,
        const std::vector<int> &word2ph, const std::vector<int64_t> &input_ids,
        const std::vector<int64_t> &attention_mask,
        const std::vector<int64_t> &token_type_id,
        std::vector<std::vector<float>> &berts)
    {
        std::vector<std::vector<float>> bert_feats;
        //if (language == "ZH" || language == "EN")
        if(language == "ZH")
        {
            bert_feats.clear();
            std::vector<std::vector<float>> zh_bert_feats;
            bert_infer(input_ids, attention_mask, token_type_id, zh_bert_feats);
            for (int i = 0; i < word2ph.size(); ++i)
            {
                for (int j = 0; j < word2ph[i]; ++j)
                {
                    bert_feats.push_back(zh_bert_feats[i]);
                }
            }
        }
        else
        {
            std::string msg =
                "input invalid language ( " + language + " ) to get bert features";
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }
        berts.insert(berts.end(), bert_feats.begin(), bert_feats.end());
        return Status::OK();
    }

    Status MeloTTSProcessor::bert_infer(
        const std::vector<int64_t> &input_ids,
        const std::vector<int64_t> &attention_mask,
        const std::vector<int64_t> &token_type_ids,
        std::vector<std::vector<float>> &bert_feats)
    {

        int input_ids_length = static_cast<int>(input_ids.size());
        if (language_ == "ZH")
        {
            openvoice_bert_model_ = openvoice_zh_bert_model_;
        }
        else
        {
            std::string msg =
                "Input invalid language type : " + language_ + " to get bert features";
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }
#ifdef MELO_DEBUG
        openvoice_bert_model_->PrintInputNames();
#endif
        openvoice_bert_model_->ResizeInputTensor(0, {1, input_ids_length});
        openvoice_bert_model_->SetInputData(0, input_ids.data());

       

        int attention_mask_length = static_cast<int>(attention_mask.size());
        openvoice_bert_model_->ResizeInputTensor(2, {1, attention_mask_length});
        openvoice_bert_model_->SetInputData(2, attention_mask.data());


        int token_type_ids_length = static_cast<int>(token_type_ids.size());
        openvoice_bert_model_->ResizeInputTensor(1, { 1, token_type_ids_length });
        openvoice_bert_model_->SetInputData(1, token_type_ids.data());


  
#ifdef MELO_DEBUG
        std::cout << "bert data ok\n";
#endif

        try
        {
            openvoice_bert_model_->Run();
            const float *output = static_cast<const float *>(openvoice_bert_model_->GetOutputData(0));
            size_t output_size = openvoice_bert_model_->GetOutputTensorSize(0);
            int frame_num = static_cast<int>(output_size / 768);
            bert_feats.resize(frame_num, std::vector<float>(768, 0.0));
            for (int i = 0; i < frame_num; ++i)
            {
                for (int j = 0; j < 768; ++j)
                {
                    bert_feats[i][j] = output[i * 768 + j];
                }
            }
        }
        catch (std::exception &e)
        {
            std::string msg = "encoder inference error : " + std::string(e.what());
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }
        return Status::OK();
    }

    Status MeloTTSProcessor::tts_infer(const std::vector<int64_t> &target_seq,
                                            const int64_t speakers,
                                            const std::vector<int64_t> &tones,
                                            const std::vector<int64_t> &lang_ids,
                                            const std::vector<float> &bert,
                                            const std::vector<float> &ja_bert,
                                            std::vector<float> &wavs)
    {
        if (language_ == "ZH")
        {
            openvoice_tts_model_ = openvoice_zh_tts_model_;
        }
        //else if (language_ == "EN")
        //{
        //    openvoice_tts_model_ = openvoice_en_tts_model_;
        //}
        else
        {
            std::string msg =
                "Input invalid language type : " + language_ + " to inference audio";
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }
#ifdef MELO_DEBUG
        openvoice_tts_model_->PrintInputNames();
#endif
        int target_seq_length = static_cast<int>(target_seq.size());
        //phones
        openvoice_tts_model_->ResizeInputTensor(0, {1, target_seq_length});
        openvoice_tts_model_->SetInputData(0, target_seq.data());

        //phones_length
        int64_t target_seq_size = static_cast<int64_t>(target_seq_length);
        openvoice_tts_model_->SetInputData(1, &target_seq_size);

        //speakers
        openvoice_tts_model_->SetInputData(2, &speakers);

        //tones
        int tones_length = static_cast<int>(tones.size());
        openvoice_tts_model_->ResizeInputTensor(3, {1, tones_length});
        openvoice_tts_model_->SetInputData(3, tones.data());

        //lang_ids
        int lang_ids_length = static_cast<int>(lang_ids.size());
        openvoice_tts_model_->ResizeInputTensor(4, {1, lang_ids_length});
        openvoice_tts_model_->SetInputData(4, lang_ids.data());

        int bert_length = static_cast<int>(bert.size() / 1024);
        openvoice_tts_model_->ResizeInputTensor(5, {1, 1024, bert_length});
        openvoice_tts_model_->SetInputData(5, bert.data());

        int ja_bert_length = static_cast<int>(ja_bert.size() / 768);
        openvoice_tts_model_->ResizeInputTensor(6, {1, 768, ja_bert_length});
        openvoice_tts_model_->SetInputData(6, ja_bert.data());

        openvoice_tts_model_->SetInputData(7, &noise_scale_);
        openvoice_tts_model_->SetInputData(8, &length_scale_);
        openvoice_tts_model_->SetInputData(9, &noise_scale_w_);
        openvoice_tts_model_->SetInputData(10, &sdp_ration_);
#ifdef MELO_DEBUG
        std::cout << "data prepare ok\n";
#endif
        try
        {
            openvoice_tts_model_->Run();
#ifdef MELO_DEBUG
            std::cout << "infer ok\n";
#endif
            const float *output =
                static_cast<const float *>(openvoice_tts_model_->GetOutputData(0));
            size_t output_size = openvoice_tts_model_->GetOutputTensorSize(0);
            wavs.resize(output_size);
            memcpy(wavs.data(), output, output_size * sizeof(float));
        }
        catch (std::exception &e)
        {
            std::string msg = "openvoice inference error : " + std::string(e.what());
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }
        return Status::OK();
    }



    Status MeloTTSProcessor::WriteWave(const std::string &filename,
                                            int32_t sampling_rate,
                                            const float *samples, int32_t n)
    {
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
            samples_int16[i] = samples[i] * 32676;
        }

        std::ofstream os(filename, std::ios::binary);
        if (!os)
        {
            std::string msg = "Failed to create " + filename;
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }

        os.write(reinterpret_cast<const char *>(&header), sizeof(header));
        os.write(reinterpret_cast<const char *>(samples_int16.data()),
                 samples_int16.size() * sizeof(int16_t));

        if (!os)
        {
            std::string msg = "Write " + filename + " failed.";
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }

        return Status::OK();
    }

} // namespace melo
