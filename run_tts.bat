@echo off
echo ".\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs.txt  --output_file audio.wav --tts_device CPU --bert_device CPU --quantize false --disable_bert false"
.\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs.txt  --output_file audio.wav --tts_device CPU --bert_device CPU --quantize false --disable_bert false