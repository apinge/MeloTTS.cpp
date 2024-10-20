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
#include <cassert>
#include "bert.h"
#include "utils.h"
namespace melo {
    void Bert::get_bert_feature(const std::string& text, const std::vector<int>& word2ph, std::vector<std::vector<float>>& berts) {
        //clear previous result
        _input_ids.clear();
        _attention_mask.clear();
        _token_type_ids.clear();

        //get token ids
        std::vector<int64_t> ids;
        std::vector<std::string> strs;
        _tokenizer->Tokenize(text, strs, _input_ids);
        size_t n = _input_ids.size();
        _attention_mask = std::vector<int64_t>(n, 1);
        _token_type_ids = std::vector<int64_t>(n, 0);
#ifdef MELO_DEBUG
        for (const auto& word : strs) std::cout << word << " ";
        std::cout << std::endl;
        for (const auto& id :_input_ids) std::cout << id << " ";
        std::cout << std::endl;    
#endif
        ov_infer();
 
       get_output(word2ph, berts);
       //get_output(berts);

    }

    void Bert::ov_infer() {
        std::cout << "ov_infer begin\n";
        size_t n = _input_ids.size();
       
        // set input tensor
        ov::Tensor input_ids(ov::element::i64, { BATCH_SIZE, n }, _input_ids.data());
        ov::Tensor token_type_ids(ov::element::i64, { BATCH_SIZE, n}, _token_type_ids.data());
        ov::Tensor attention_mask(ov::element::i64, { BATCH_SIZE, n}, _attention_mask.data());
#ifdef MELO_DEBUG
        std::cout << "ov_infer begin"<< n << std::endl;
        std::cout << input_ids.get_shape() << " "<< input_ids.get_byte_size() << std::endl;
        std::cout << token_type_ids.get_shape() << " "<< token_type_ids.get_byte_size() << std::endl;
        std::cout << attention_mask.get_shape() << " "<< attention_mask.get_byte_size() << std::endl;
        //_infer_request->set_input_tensors({ input_ids,token_type_id,attention_mask });
#endif
        _infer_request->set_input_tensor(0, input_ids);
        _infer_request->set_input_tensor(1, token_type_ids);
        _infer_request->set_input_tensor(2, attention_mask);

        _infer_request->infer();
        std::cout <<"infer ok\n";
    }

    void Bert::get_output(const std::vector<int>& word2ph, std::vector<std::vector<float>>& phone_level_feature) {
        const ov::Tensor& output_tensor = _infer_request->get_output_tensor(0);
        const float* output_data = _infer_request->get_output_tensor(0).data<const float>();
        //size_t output_size = _input_ids.size();//_infer_request->GetOutputTensorSize(0);
        size_t frame_num = output_tensor.get_shape()[0];

        //const ov::Tensor& output_tensor = _infer_request->get_output_tensor(0);
        //const float* output_data = _infer_request->get_output_tensor(0).data<const float>();
        //ov::Shape output_tensor_shape = output_tensor.get_shape();
       // int frame_num = output_tensor_shape[0];
        assert(frame_num == _input_ids.size() && "[ERROR] Should be frame_num == _input_ids.size()");
        //std::cout << " output_tensor_shape"<<  output_tensor_shape << std::endl;
        std::vector<std::vector<float>> res(frame_num, std::vector<float>(768, 0.0));
        for (int i = 0; i < frame_num; ++i)
        {
            for (int j = 0; j < 768; ++j)
            {
                res[i][j] = output_data[i * 768 + j];
            }
        }
#ifdef MELO_DEBUG
        print_mean_variance("jb_bert", res);
#endif
         /*Corresponding Python code :
         phone_level_feature = []
         for i in range(len(word2phone)):
             repeat_feature = res[i].repeat(word2phone[i], 1)
             phone_level_feature.append(repeat_feature)
         phone_level_feature = torch.cat(phone_level_feature, dim=0)*/
        for (int i = 0; i < word2ph.size(); ++i)
        {
            for (int j = 0; j < word2ph[i]; ++j)
            {
                phone_level_feature.push_back(res[i]);
            }
        }
    }

    void Bert::get_output(std::vector<std::vector<float>>& res) {
        const ov::Tensor& output_tensor = _infer_request->get_output_tensor();
        const float* output_data = _infer_request->get_output_tensor().data<const float>();
        ov::Shape output_tensor_shape = output_tensor.get_shape();
        size_t frame_num = output_tensor_shape[0];
        assert(frame_num == _input_ids.size() && "[ERROR] Should be frame_num == _input_ids.size()");
        std::cout << " output_tensor_shape" << output_tensor_shape << std::endl;
        res.clear();
        res.resize(frame_num, std::vector<float>(768, 0.0));
        for (int i = 0; i < frame_num; ++i)
        {
            for (int j = 0; j < 768; ++j)
            {
                res[i][j] = output_data[i * 768 + j];
            }
        }
    }


}