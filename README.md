# MeloTTS.cpp

This repository primarily offers a pure C++ implementation of [meloTTS](https://github.com/myshell-ai/MeloTTS), which is a high-quality, multilingual Text-to-Speech (TTS) library released by MyShell.ai that supports English, Chinese (mixed with English), and various other languages. This implementation is fully integrated with OpenVINO. Currently, this repository is limited to supporting Chinese mixed with English. Support for English is planned for future releases.

Project Owner: Tong Qiu (tong.qiu@intel.com)

## Setup and Execution Guide

### 1. Download OpenVINO C++ Package

To download the OpenVINO C++ package for Windows, please refer to the following link: [Install OpenVINO for Windows]( https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html).
For Linux, you can download the C++ package from this link: [Install OpenVINO for Linux](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html).

For additional versions and more information about OpenVINO, visit the official OpenVINO Toolkit page: [OpenVINO Toolkit Overview](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html).

### 2. Clone the Repository
```
git install lfs
git clone https://github.com/apinge/MeloTTS.cpp.git
```

### 3. Build and Run
#### 3.1 Windows Build and Run
```
<OpenVINO_DIR>\setupvars.bat
cd MeloTTS.cpp 
cmake -S . -B build && cmake --build . --config Release
.\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs.txt  --output_file audio.wav --tts_device CPU --bert_device CPU --quantize false --disable_bert false
```
#### 3.2 Linux Build and Run
```
source <OpenVINO_DIR>/setupvars.sh
cd MeloTTS.cpp 
cmake -S . -B build && cmake --build . --config Release
./build/meloTTS_ov --model_dir ov_models --input_file inputs.txt --output_file audio.wav --tts_device CPU --bert_device CPU --quantize false --disable_bert false
```
### 4. Arguments Description
You can use `run_tts.bat` or `run_tts.sh` as sample scripts to run the models. Below are the meanings of the arguments you can use with these scripts:

- `--model_dir`: Specifies the folder containing the model files, dictionary files, and third-party resource files.
- `--tts_device`: Specifies the OpenVINO device to be used for the TTS model. The default device is CPU.
- `--bert_device`: Specifies the OpenVINO device to be used for the BERT model. The default device is CPU.
- `--input_file`: Specifies the input text file to be processed.
- `--output_file`: Specifies the output audio file to be generated.
- `--quantize`: Indicates whether to use an int8 quantized model. The default is false, meaning an fp16 model is used by default.
- `--disable_bert`: Indicates whether to disable the BERT model inference. The default is false.
- `--language`: Specifies the language for TTS. The default language is Chinese (`ZH`).

## Supported Versions
- **Operating System**: Windows, Ubuntu 
- **CPU Architecture**: Metor Lake,  Lunar Lake
- **GPU Architecture**: Intel® Arc™ Graphics (Intel Xe, including iGPU)
- **C++ Version**: >=C++20


We have also successfully performed inference on other platforms, such as Xeon. However, we do not provide regular testing environments for these platforms and cannot guarantee the performance.
