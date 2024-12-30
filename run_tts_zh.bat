@echo off
setlocal
echo ".\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs_zh.txt --output_filename audio --language ZH --speed 0.95 --tts_device CPU --bert_device CPU --quantize false --disable_bert false --disable_nf false"
.\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs_zh.txt --output_filename audio --language ZH --speed 0.95 --tts_device CPU --bert_device CPU --quantize false --disable_bert false --disable_nf false
endlocal
