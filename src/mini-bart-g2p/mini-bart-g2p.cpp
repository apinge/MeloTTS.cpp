/**
 * Copyright      2024-2025    Tong Qiu (tong.qiu@intel.com)  Haofan Rong (haofan.rong@hp.com)
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
#include "mini-bart-g2p.h"
namespace melo {
MiniBartG2P::MiniBartG2P(std::unique_ptr<ov::Core>& core_ptr,
                         const std::filesystem::path& model_folder_path,
                         const std::string& device_,
                         bool use_past_)
    : device(device_),
      use_past(use_past_) {
    std::filesystem::path encoder_model_path = model_folder_path / "openvino_encoder_model.xml";
    std::filesystem::path decoder_model_path = model_folder_path / "openvino_decoder_model.xml";
    std::filesystem::path decoder_model_with_past_path = model_folder_path / "openvino_decoder_with_past_model.xml";
    core_ptr->set_property("CPU", {{"CPU_RUNTIME_CACHE_CAPACITY", "0"}});
    std::cout << "Set CPU_RUNTIME_CACHE_CAPACITY 0\n";
    encoder_model = std::make_unique<ov::CompiledModel>(
        core_ptr->compile_model(encoder_model_path.string(), device, set_ov_config(device)));
    encoder_req = std::make_unique<ov::InferRequest>(encoder_model->create_infer_request());
    decoder_model = std::make_unique<ov::CompiledModel>(
        core_ptr->compile_model(decoder_model_path.string(), device, set_ov_config(device)));
    decoder_req = std::make_unique<ov::InferRequest>(decoder_model->create_infer_request());
    if (use_past_ && std::filesystem::exists(decoder_model_with_past_path)) {
        decoder_with_past_model = std::make_unique<ov::CompiledModel>(
            core_ptr->compile_model(decoder_model_with_past_path.string(), device, set_ov_config(device)));
        decoder_with_past_req = std::make_unique<ov::InferRequest>(decoder_with_past_model->create_infer_request());
        std::cout << "[INFO] MiniBartG2P: use_past is true.\n";
    } else
        std::cout << "[INFO] MiniBartG2P: use_past is false.\n";
    std::cout << "[INFO] Construct MiniBartG2P succeeded.\n";
    get_ov_info(core_ptr, device);
}
std::vector<std::string> MiniBartG2P::forward(const std::string& input) {
    _input_ids.clear();
    _attention_mask.clear();
    _decoder_input_ids.clear();
    std::vector<std::string> res;
    try {
        std::string text = filter(input);
        unsigned long long n = text.length();
        _attention_mask.resize(n + 2, 1);
        _input_ids.emplace_back(0);  //<s>
        for (auto& ch : text)
            _input_ids.emplace_back(tokenizer.at(ch));
        _input_ids.emplace_back(2);  //</s>
#ifdef MELO_DEBUG
        for (std::cout << "input_ids"; auto& x : _input_ids)
            std::cout << x << ' ';
        std::cout << std::endl;
        print_input_names(encoder_model.get());
#endif
        /*
        * encoder
        * 0 input_ids
         1 attention_mask
        */
        ov::Tensor input_ids(ov::element::i64, {BATCH_SIZE, n + 2}, _input_ids.data());
        ov::Tensor attention_mask(ov::element::i64, {BATCH_SIZE, n + 2}, _attention_mask.data());
        encoder_req->set_input_tensor(0, input_ids);
        encoder_req->set_input_tensor(1, attention_mask);
        encoder_req->start_async();
        encoder_req->wait();

        ov::Tensor last_hidden_state = encoder_req->get_output_tensor(0);
        const float* output_data = encoder_req->get_output_tensor(0).data<const float>();
        // size_t output_size = _input_ids.size();//_infer_request->GetOutputTensorSize(0);
        size_t frame_num = last_hidden_state.get_size();
#ifdef MELO_DEBUG
        std::cout << "Encoder last_hidden_state shape:" << last_hidden_state.get_shape() << std::endl;
        std::cout << "Encoder last_hidden_state.get_size():" << frame_num << std::endl;
#endif
        std::vector<float> last_hidden_state_data(frame_num, 0);
        for (int i = 0; i < frame_num; ++i)
            last_hidden_state_data[i] = output_data[i];
        /*
        * decoder
        0 encoder_attention_mask
        1 input_ids
        2 encoder_hidden_states*/
        _decoder_input_ids = {2};
        for (;;) {
            // for (std::cout << "_decoder_input_ids:"; auto & x:_decoder_input_ids) std::cout << x << ' ';
            // std::cout << std::endl;
            ov::Tensor encoder_attention_mask(ov::element::i64,
                                              {BATCH_SIZE, n + 2},
                                              _attention_mask.data());  // TODO deduplicate
            ov::Tensor decoder_input_ids(ov::element::i64,
                                         {BATCH_SIZE, _decoder_input_ids.size()},
                                         _decoder_input_ids.data());
            ov::Tensor encoder_hidden_states(ov::element::f32,
                                             last_hidden_state.get_shape(),
                                             last_hidden_state_data.data());
            decoder_req->set_input_tensor(0, encoder_attention_mask);
            decoder_req->set_input_tensor(1, decoder_input_ids);
            decoder_req->set_input_tensor(2, encoder_hidden_states);
            decoder_req->start_async();
            decoder_req->wait();
            const float* logits_data = decoder_req->get_output_tensor(0).data<const float>();
            // std::cout << "Decoder logits.get_size()/vocab_size:" <<
            // decoder_req->get_output_tensor(0).get_size()/vocab_size << std::endl;
            int logits_row = decoder_req->get_output_tensor(0).get_size() / vocab_size;
            float maxLogits = 0.f;
            int maxArg = -1;

            for (int i = (logits_row - 1) * vocab_size, j = 0; i < logits_row * vocab_size && j < vocab_size;
                 ++i, ++j) {
                if (logits_data[i] > maxLogits) {
                    maxLogits = logits_data[i];
                    maxArg = j;
                }
            }
            _decoder_input_ids.emplace_back(maxArg);
            // for (std::cout << "_decoder_input_ids"; auto & x:_decoder_input_ids) std::cout << detokenizer.at(x) << '
            // '; std::cout << "\n";
            if (maxArg == 2)
                break;
        }
        // detokenize
        for (auto& id : _decoder_input_ids) {
            if (id == 0 || id == 2)  //<s> or </s>
                continue;
            res.emplace_back(_to_lower(detokenizer.at(id)));
        }
        release_memory();

    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General exception: " << e.what() << std::endl;
    }
    return res;
}
/*
 * space is '<unk>':3, ref: https://huggingface.co/cisco-ai/mini-bart-g2p/blob/main/vocab.json
 */
const std::map<char, int64_t> MiniBartG2P::tokenizer = {
    {' ', 3},  {'e', 5},  {'a', 6},  {'s', 7},  {'i', 8},  {'r', 9},  {'n', 10}, {'o', 12}, {'t', 14},
    {'l', 15}, {'c', 21}, {'d', 22}, {'u', 24}, {'m', 26}, {'h', 29}, {'g', 30}, {'p', 31}, {'b', 34},
    {'y', 40}, {'k', 41}, {'f', 44}, {'w', 46}, {'v', 48}, {'z', 63}, {'j', 76}, {'x', 79}, {'q', 85}};
const std::unordered_map<int, std::string> MiniBartG2P::detokenizer = {
    {0, "<s>"},  {2, "</s>"}, {11, "AH0"}, {13, "N"},   {16, "S"},   {17, "L"},    {18, "T"},   {19, "R"},
    {20, "K"},   {23, "D"},   {25, "IH0"}, {27, "M"},   {28, "Z"},   {32, "ER0"},  {33, "IY0"}, {35, "B"},
    {36, "P"},   {37, "EH1"}, {38, "AE1"}, {39, "AA1"}, {42, "IH1"}, {43, "F"},    {45, "G"},   {47, "V"},
    {49, "NG"},  {51, "IY1"}, {52, "EY1"}, {53, "HH"},  {54, "W"},   {55, "SH"},   {56, "OW1"}, {57, "AO1"},
    {58, "OW0"}, {59, "AH1"}, {60, "UW1"}, {61, "AY1"}, {62, "JH"},  {64, "CH"},   {65, "Y"},   {66, "AA0"},
    {67, "ER1"}, {68, "EH2"}, {69, "IH2"}, {70, "TH"},  {71, "AY2"}, {72, "AE2"},  {73, "EY2"}, {74, "AA2"},
    {75, "EH0"}, {77, "AW1"}, {78, "OW2"}, {80, "IY2"}, {81, "UW0"}, {82, "AO2"},  {83, "UH1"}, {84, "AE0"},
    {86, "AO0"}, {87, "AH2"}, {88, "UW2"}, {89, "AY0"}, {90, "OY1"}, {92, "EY0"},  {93, "DH"},  {94, "AW2"},
    {95, "ER2"}, {96, "ZH"},  {97, "UH2"}, {98, "AW0"}, {99, "UH0"}, {100, "OY2"}, {101, "OY0"}};
}  // namespace melo