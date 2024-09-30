#pragma once
#ifndef BERT_H
#define BERT_H
#include <string>
#include <memory>
#include "tokenizer.h"
#include "openvino_model.h"
namespace melo {
    class Bert : public AbstractOpenvinoModel {
    public:
        Bert(std::unique_ptr<ov::Core>& core_ptr, const std::string& model_path, const std::string& device,
            std::string language, std::shared_ptr<Tokenizer> tokenizer) :
            AbstractOpenvinoModel(core_ptr, model_path, device), _language(language), _tokenizer(tokenizer){}
        Bert(std::unique_ptr<ov::Core>& core_ptr, const std::filesystem::path& model_path, const std::string& device,
            std::string language, std::shared_ptr<Tokenizer> tokenizer) :
            AbstractOpenvinoModel(core_ptr, model_path, device), _language(language), _tokenizer(tokenizer) {}
        Bert(std::shared_ptr<ov::Core>& core_ptr, const std::string& model_path, const std::string& device,
            std::string language, std::shared_ptr<Tokenizer> tokenizer) :
            AbstractOpenvinoModel(core_ptr, model_path, device), _language(language), _tokenizer(tokenizer) {}
        Bert() = default;
        void get_bert_feature(const std::string& text, const std::vector<int>& word2ph, std::vector<std::vector<float>>& berts);
        virtual void ov_infer() override;
        virtual void get_output(const std::vector<int>& word2ph, std::vector<std::vector<float>>& phone_level_feature);
        virtual void get_output(std::vector<std::vector<float>>&);
        //virtual void get_output(std::vector<std::any>& output) {};

        inline std::string get_language() { return _language; }
        static constexpr size_t BATCH_SIZE = 1;
    private:

        std::string _language;
        std::shared_ptr<Tokenizer> _tokenizer;
        std::vector<int64_t> _input_ids, _attention_mask, _token_type_ids;
        

    };
}
#endif // !BERT_H
