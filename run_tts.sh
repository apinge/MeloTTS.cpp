#!/bin/bash
echo "./build/RmeloTTS_ov --model_dir ov_models --input_file inputs.txt --output_file audio.wav --tts_device CPU --bert_device CPU --quantize false --disable_bert true"
./build/meloTTS_ov --model_dir ov_models --input_file inputs.txt --output_file audio.wav --tts_device CPU --bert_device CPU --quantize false --disable_bert true