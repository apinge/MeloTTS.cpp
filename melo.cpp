
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "src/openvoice2_processor.h"


int main()
{
    melo::MeloTTSProcessor* tts_processor =  new melo::MeloTTSProcessor();

    //fp16 model
    //std::string zh_tts_path = "../thirdParty/tts_ov/tts_zn_mix_en.xml";
   // std::string zh_bert_path = "../thirdParty/tts_ov/bert.xml";

    //int8 model
    std::string zh_tts_path = "../thirdParty/tts_ov/tts_int8.xml";
    std::string zh_bert_path = "../thirdParty/tts_ov/bert_int8.xml";

    // init tokenizer
    std::string vocab_bert_path = "../thirdParty/tts_ov/vocab_bert.txt";

    tts_processor->LoadTTSModel(zh_tts_path, zh_bert_path, vocab_bert_path);
    std::vector<float> addit_param = { 0.2, 0.6, 1.0, 0.80 };
    /*sdp_ration_ = addit_param[0];
        noise_scale_ = addit_param[1];
        length_scale_ = addit_param[2];
        noise_scale_w_ = addit_param[3];*/
    //std::vector<float> addit_param = { 0.2, 0.667, 1.0, 0.80 };
    std::string convert_text = "";
    std::vector<float> wav_data;
    //loop for time
    //while(true){
        wav_data.clear();
      tts_processor->Process(convert_text, 0, addit_param, wav_data);
    //}
    //16000 origin
    tts_processor->WriteWave("../melo_tts_int8.wav", 44100, wav_data.data(), wav_data.size());
    std::cout << "finish to generate wav" << std::endl;
    return 0;
}
