
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "src/openvoice2_processor.h"


int main()
{
    melo::TTSOpenVoiceProcessor* tts_processor =  new melo::TTSOpenVoiceProcessor();

    std::string zh_tts_path = "../thirdParty/tts_ov/tts_zn_mix_en.xml";
    std::string zh_bert_path = "../thirdParty/tts_ov/bert.xml";

    tts_processor->LoadTTSModel(zh_tts_path, zh_bert_path);
    std::vector<float> addit_param = { 0.2, 0.6, 1.0, 0.80 };
    /*sdp_ration_ = addit_param[0];
        noise_scale_ = addit_param[1];
        length_scale_ = addit_param[2];
        noise_scale_w_ = addit_param[3];*/
    //std::vector<float> addit_param = { 0.2, 0.667, 1.0, 0.80 };
    std::string convert_text = "";
    std::vector<float> wav_data;
    std::string data_path = "../thirdParty/binData/";
    //loop for time
    for (int k = 0; k < 1; ++k) {
      tts_processor->Process(convert_text, 0, addit_param, data_path, wav_data);
      //16000 origin
      tts_processor->WriteWave("../tts_intel.wav", 44100,
                               wav_data.data(), wav_data.size());
      std::cout << "finish to generate wav" << std::endl;
    }

    return 0;
}
