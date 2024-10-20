/**
 * Copyright      2024    Tong Qiu (tong.qiu@intel.com)
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
#ifndef BERT_H
#define BERT_H
#include <string>
#include <memory>
#include "tokenizer.h"
#include "openvino_model_base.h"
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
#endif // BERT_H
