#include <iostream>
#include <filesystem>
#include <gtest/gtest.h>
#include "bert.h"
#include "utils.h"
#define MELO_TEST
#define OV_MODEL_PATH "ov_models"
class BertTestSuit : public ::testing::Test {
public:
    BertTestSuit() {
        model_dir = OV_MODEL_PATH;
        en_bert_path = model_dir / "bert_EN.xml";
        std::cout << "en_bert_path:" << std::filesystem::absolute(en_bert_path) << std::endl;
        std::unique_ptr<ov::Core> core_ptr = std::make_unique<ov::Core>();
        std::filesystem::path en_bert_subword_tokenizer = model_dir / "bert-base-uncased" / "bert_subword_tokenizer.xml";
        std::filesystem::path en_bert_subword_detokenizer = model_dir / "bert-base-uncased" / "bert_subword_detokenizer.xml";
        std::filesystem::path runtime_path = "C:\\Users\\gta\\source\\repos\\openvino_tokenizers_windows_2024.6.0.0_x86_64\\runtime\\bin\\intel64\\Release\\openvino_tokenizers.dll";


        ov_tokenizer_ptr = std::make_shared<melo::OpenVinoTokenizer>(core_ptr, runtime_path, en_bert_subword_tokenizer, en_bert_subword_detokenizer);
        en_bert = melo::Bert(core_ptr, en_bert_path.string(), "CPU", "EH", ov_tokenizer_ptr);
        //en_bert.set_static_shape();
    }
protected:
    std::filesystem::path model_dir;
    std::filesystem::path en_bert_path;
    std::unique_ptr<ov::Core> core_ptr;
    melo::Bert en_bert;
    std::shared_ptr<melo::OpenVinoTokenizer> ov_tokenizer_ptr;
   
};


TEST_F(BertTestSuit, DynamicShapeModel){
#if  WIN32
    system("chcp 65001"); //Using UTF-8 Encoding
#endif //  WIN32    
    
    std::string text = "i am absolutely thrilled to share this incredible news with everyone";
    std::vector<int> word2ph{ 3, 2, 4, 18, 10, 4, 6, 6, 20, 6, 6, 14, 2 };
    std::vector<std::vector<float>> berts;

    en_bert.get_bert_feature(text, word2ph, berts);
    std::cout << berts.size() <<' '<< berts.front().size()<<  std::endl;
    EXPECT_EQ(berts.size(), 101);
    EXPECT_EQ(berts.front().size(),768);
    
}
//
//TEST_F(BertTestSuit, TestEachRow_Static) {
//#if  WIN32
//    system("chcp 65001"); //Using UTF-8 Encoding
//#endif //  WIN32    
//
//    std::string text = "今天的meeting真的是超级productive";
//    std::vector<int64_t> token_ids = { 101,  1773,  2975,  5975, 17829,  6032,  5975,  4353,  8224,  6709, 20058, 12899, 102};
//    std::vector<int> word2ph{ 3, 4, 4, 4, 10, 4, 4, 4, 4, 4, 10, 8, 2 };
//    std::vector<std::vector<float>> res;
//    zh_bert.set_input_tensors(token_ids, true);
//    zh_bert.ov_infer();
//    zh_bert.get_output(res);
//    int n = res.size(), m = res.front().size();
//    std::vector<float> mean,variance;
//    for (int i =0;i<token_ids.size();++i) {
//        auto& row = res[i];
//        float row_mean =  calculate_mean(row);
//        float row_variance = calculate_variance(row,row_mean);
//        mean.emplace_back(std::move(row_mean));
//        variance.emplace_back(std::move(row_variance));
//    }
//    EXPECT_EQ(n, melo::Bert::NPU_BERT_STATIC_SHAPE_SIZE);
//    EXPECT_EQ(m, 768);
//    //pytorch result
//    std::vector<float> correct_mean{ - 0.0217, -0.0180, -0.0163, -0.0172, -0.0341, -0.0222, -0.0174, -0.0205, -0.0158, -0.0197, -0.0197, -0.0210, 0.0017};
//    std::vector<float> correct_variance{ 0.8274, 0.7458, 0.6663, 0.6998, 1.3672, 0.8105, 0.7077, 0.7894, 0.6316, 0.7086, 0.7134, 0.8239, 0.2795 };
//    for (size_t i = 0; i < mean.size(); ++i) {
//        EXPECT_NEAR(mean[i], correct_mean[i], 0.01);
//        EXPECT_NEAR(variance[i], correct_variance[i], 0.01);
//    }
//}


//int main(int argc, char** argv) {
//    ::testing::InitGoogleTest(&argc, argv); // INT Google Test
//    return RUN_ALL_TESTS();
//}