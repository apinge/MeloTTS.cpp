#include "openvino_tokenizer.h"

namespace melo{
    OpenVinoTokenizer::OpenVinoTokenizer(std::unique_ptr<ov::Core>& core, const std::filesystem::path& runtime_path, const std::filesystem::path& tokenize_path, const std::filesystem::path& detokenize_path) {
        core->add_extension(runtime_path.string().c_str());
        tokenizer_infer = core->compile_model(tokenize_path.string()).create_infer_request();
        detokenizer_infer = core->compile_model(detokenize_path.string()).create_infer_request();
        auto res = this->tokenize("warm up");//Workaround for warming up the tokenizer to save inference time.
    }
    OpenVinoTokenizer::OpenVinoTokenizer(std::unique_ptr<ov::Core>& core, const std::filesystem::path& runtime_path, const std::filesystem::path& tokenizer_model_folder) {
        core->add_extension(runtime_path.string().c_str());
        auto tokenize_path = tokenizer_model_folder / "bert_subword_tokenizer.xml";
        auto detokenize_path = tokenizer_model_folder / "bert_subword_detokenizer.xml";
        tokenizer_infer = core->compile_model(tokenize_path.string()).create_infer_request();
        detokenizer_infer = core->compile_model(detokenize_path.string()).create_infer_request();
        auto res = this->tokenize("warm up");//Workaround for warming up the tokenizer to save inference time.
        std::cout << "[INFO] OpenVinoTokenizer Initialized!\n";
    }


    ov::Tensor OpenVinoTokenizer::tokenize_tensor(std::string&& prompt) {
        tokenizer_infer.set_input_tensor(ov::Tensor{ ov::element::string, {1}, &prompt });
        tokenizer_infer.infer();
        return  tokenizer_infer.get_tensor("input_ids");
    }
    std::vector<int64_t> OpenVinoTokenizer::tokenize(std::string&& prompt) {
        tokenizer_infer.set_input_tensor(ov::Tensor{ ov::element::string, {1}, &prompt });
        tokenizer_infer.infer();
        return get_output_vec<int64_t>(tokenizer_infer.get_tensor("input_ids"));
    }


    std::vector<std::string> OpenVinoTokenizer::detokenize(ov::Tensor& token_ids) {
        detokenizer_infer.set_input_tensor(token_ids);
        detokenizer_infer.infer();
        size_t len = detokenizer_infer.get_output_tensor().get_shape()[0];
        std::vector<std::string> res;
        std::string* ptr = detokenizer_infer.get_output_tensor().data<std::string>();
        for (int i = 0; i < len; ++i) {
            // ignore the [CLS] token at the beginning and the [SEP] token at the end.
            if (*(ptr + i) == "[CLS]" || *(ptr + i) == "[SEP]")
                continue;
            res.emplace_back(*(ptr + i));
        }
        return res;
    }
    std::vector<std::string> OpenVinoTokenizer::detokenize(std::vector<int64_t>&& token_id, size_t size) {
        constexpr size_t BATCH_SIZE = 1;
        detokenizer_infer.set_input_tensor(ov::Tensor{ ov::element::i64, {BATCH_SIZE,size}, token_id.data() });
        detokenizer_infer.infer();
        size_t len = detokenizer_infer.get_output_tensor().get_shape()[0];
        std::vector<std::string> res;
        std::string* ptr = detokenizer_infer.get_output_tensor().data<std::string>();
        for (int i = 0; i < len; ++i)
            res.emplace_back(*(ptr + i));//subword_detokenizer_infer.get_output_tensor().data<std::string>()[i]
        return res;
    }

    std::string* OpenVinoTokenizer::detokenize(int64_t&& token_id) {
        constexpr size_t BATCH_SIZE = 1;
        detokenizer_infer.set_input_tensor(ov::Tensor{ ov::element::i64, {BATCH_SIZE,1}, &token_id });
        detokenizer_infer.infer();
        return detokenizer_infer.get_output_tensor().data<std::string>();
    }
    //https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
    std::vector<std::string> OpenVinoTokenizer::word_segment(std::string& prompt) {
        ov::Tensor token_ids = tokenize_tensor(std::move(prompt));
        return detokenize(token_ids);
    }
}