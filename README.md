# MeloTTS.cpp
<p>
  <b>< English</b> | <a href='./README.zh-CN.md'>简体中文</a> | <a href='./README.zh-TW.md'>繁體中文</a> >
</p>


This repository offers a C++ implementation of [meloTTS](https://github.com/myshell-ai/MeloTTS), which is a high-quality, multilingual Text-to-Speech (TTS) library released by MyShell.ai that supports English, Chinese (mixed with English), and various other languages. This implementation is fully integrated with OpenVINO, supporting seamless deployment on CPU, GPU, and NPU devices. Currently, this repository supports both Chinese with mixed English and English. Support for [Japanese model](https://huggingface.co/myshell-ai/MeloTTS-Japanese) is coming next.

## 🔀 Branch Usage Guide

This repository supports multilingual text-to-speech inference. Please switch to the appropriate branch based on your use case:

- **`EN` branch**:  
  For **English-only** speech inference.

- **`ZH_MIX_EN` branch**:  
  For **Chinese-only** speech inference, designed to handle Mandarin with embedded English words.

- **`multilang-develop` branch**:  
  For **multilingual** speech inference 
  Supports both Mandarin-English mixed speech and English-only processing.


## Pipeline Design



The pipeline design is largely consistent with the [orignal pytorch Version](https://github.com/myshell-ai/MeloTTS), comprising three models (BERT, TTS, and DeepFilterNet), with DeepFilterNet added as an additional component.


<img src="images/melotts_design.png" alt="Pipeline Design" title="Pipeline Design" width="800" style="display: block">


#### Legend / Terminology
1. tokenizer and BERT: The tokenizer and BERT model are `bert-base-multilingual-uncased` for Chinese and `bert-base-uncased` for English 
2. g2p: Grapheme-to-Phoneme conversion. For English g2p, [mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p) is used to generate phonemes. See [Enable mini-bart-g2p for OpenVINO](https://github.com/apinge/MeloTTS.cpp/blob/multilang-develop/experimental/mini-bart-g2p/README.md) for details.
3. phones and tones: represented as pinyin with four tones for Chinese and phonemes with stress marks for English
4. tone_sandi: class used for handling Chinese scenarios, correcting tokenization and phones
5. DeepFilterNet: used for denoising (background noise introduced by int8 quantization)


### Model-Device Compatibility Table
The table below outlines the supported devices for each model:
| Model Name       | CPU Support | GPU Support | NPU Support |
|------------------|-------------|-------------|-------------|
| BERT (Preprocessing) | ✅           | ✅           | ✅           |
| TTS (Inference)      | ✅           | ✅           | ❌           |
| DeepFilterNet (Post-processing) | ✅           | ✅           | ✅           |

## Setup and Execution Guide

### 1. Download OpenVINO™ GenAI C++ Package

To download the OpenVINO GenAI C++ package, please refer to the following link: [Install OpenVINO™ GenAI](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html).
For **OpenVINO™ GenAI 2025.1** on Windows, you can run the command line in the command prompt (cmd).
```
curl -O https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.1/windows/openvino_genai_windows_2025.1.0.0_x86_64.zip
tar -xzvf openvino_genai_windows_2025.1.0.0_x86_64.zip
```
For Linux, you can download the C++ package from https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.1/linux/ and unzip the package.

For additional versions and more information about OpenVINO, visit the official OpenVINO Toolkit page: [OpenVINO Toolkit Overview](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html).

### 2. Clone the Repository
```
git lfs install
git clone https://github.com/apinge/MeloTTS.cpp.git
```

### 3. Build and Run
#### 3.1 Windows Build and Run
```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd MeloTTS.cpp
cmake -S . -B build && cmake --build build --config Release
.\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs_en.txt  --output_filename audio
```
#### 3.2 Linux Build and Run
```
source <OpenVINO_GenAI_DIR>/setupvars.sh
cd MeloTTS.cpp 
cmake -S . -B build && cmake --build build --config Release
./build/meloTTS_ov --model_dir ov_models --input_file inputs_en.txt --output_filename audio
```
#### 3.3 Enabling and Disabling DeepFilterNet
DeepFilterNet functionality is currently supported on both Windows and Linux and is used to filter noise from int8 quantized models. By default, it is enabled, but you can disable it during the CMake stage using the `-DUSE_DEEPFILTERNET` option.

For example, to disable the feature, you can use the following line during the CMake generation process:
```
cmake -S . -B build -DUSE_DEEPFILTERNET=OFF
```
For more information, please refer to [DeepFilterNet.cpp](https://github.com/apinge/MeloTTS.cpp/blob/develop/src/deepfilternet/README.md).

### 4. Arguments Description
You can use `run_tts.bat` or `run_tts.sh` as sample scripts to run the models. Below are the meanings of all the arguments you can use with these scripts:

- `--model_dir`: Specifies the folder containing the model files, dictionary files, and third-party resource files, which is `ov_models` folder within the repo. You may need to adjust the relative path based on your current working directory.
- `--tts_device`: Specifies the OpenVINO device to be used for the TTS model. Supported devices include CPU and GPU (default: CPU).
- `--bert_device`: Specifies the OpenVINO device to be used for the BERT model. Supported devices include CPU, GPU, and NPU (default: CPU).
- `--nf_device`: Specifies the OpenVINO device to be used for the DeepfilterNet model. Supported devices include CPU, GPU, and NPU (default: CPU).
- `--input_file`: Specifies the input text file to be processed. Make sure that the text is in **UTF-8** format.
- `--output_filename`: Specifies the output audio filename to be generated in the format {output_filename}_{language_style}.wav. For example, if the language is Chinese and the output_filename is "audio", the file will be saved as audio_ZH-MIX-EN.wav"
- `--speed`: Specifies the speed of output audio. The default is 1.0.
- `--quantize`: Indicates whether to use a quantized tts model. The default is `true`, meaning int8 quantized model is used by default.
- `--disable_bert`: Indicates whether to disable the BERT model inference. The default is `false`.
- `--disable_nf`: Indicates whether to disable the DeepfilterNet model inference (default: `false`).
- `--language`: Specifies the language for TTS. The default language is English (`EN`).

## NPU Device Support
The BERT and DeepFilterNet models in the pipeline support NPU as the inference device, utilizing the integrated NPUs in Meteor Lake and Lunar Lake.

Below are the methods to enable this feature and the usage details:
<details>
  <summary>Click here to expand/collapse content</summary>
  <ul>
   <li><strong>How to Build</strong></li>
   To enable the BERT model on NPU, an additional CMake option <code>-DUSE_BERT_NPU=ON</code> is required during the CMake generation. For example:
    <pre><code>cmake -DUSE_BERT_NPU=ON -B build -S .</code></pre>
   To enable DeepFilterNet on NPU, no additional compilation steps are required.
   <li><strong>How to Set Arguments</strong></li>
        To set arguments for models on NPU, use <code>--bert_device NPU</code> for the BERT model and  <code>--nf_device NPU</code> for the DeepFilterNet model respectively. For example:
        <pre><code>build\Release\meloTTS_ov.exe --bert_device NPU --nf_device NPU --model_dir ov_models --input_file inputs.txt  --output_file audio.wav</code></pre>
    
</ul>
</details>

## Supported Versions
- **Operating System**: Windows, Linux
- **CPU Architecture**: Metor Lake, Lunar Lake, and most Intel CPUs
- **GPU Architecture**: Intel® Arc™ Graphics (Intel Xe, including iGPU)
- **NPU  Architecture**: NPU 4,  NPU in Meteor Lake or Lunar Lake
- **OpenVINO Version**: >=2024.4
- **C++ Version**: >=C++20

If you're using an AI PC notebook with Windows, GPU and NPU drivers are typically pre-installed. However, Linux users or Windows users who prefer to update to the latest drivers should follow the guidelines below:

- **For GPU**: If using GPU, please refer to [Configurations for Intel® Processor Graphics (GPU) with OpenVINO™](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html) to install the GPU driver.

- **For NPU**: If using NPU, please refer to [NPU Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) to ensure the NPU driver is correctly installed. 

Note that all the drivers differs between Windows and Linux, so make sure to follow the instructions for your specific operating system.

## Future Development Plan
Here are some features and improvements planned for future releases:

1. **Add Japanese language TTS support**: 
   - Enable Japanese text-to-speech (TTS) functionality, but tokenization for Japanese language input is not yet implemented.
   
2. **Enhancing Quality in Quantized TTS Models**:
   - The current INT8 quantized model exhibits slight background noise. As a workaround, we integrated DeepFilterNet for post-processing. Moving forward, we aim to address the noise issue more effectively by the quantization techniques.

## Python Version
The Python version of this repository (MeloTTS integrated with OpenVINO) is provided in [MeloTTS-OV](https://github.com/zhaohb/MeloTTS-OV/tree/speech-enhancement-and-npu). The Python version includes methods to convert the model into OpenVINO IR.

## Third-Party Code
This repository includes third-party code and libraries for Chinese word segmentation and pinyin processing.

- [cppjieba](https://github.com/yanyiwu/cppjieba)
    - A Chinese text segmentation library.
- [cppinyin](https://github.com/pkufool/cppinyin)
    - A C++ library supporting conversion between Chinese characters and pinyin
- [libtorch](https://github.com/pytorch/pytorch/blob/main/docs/libtorch.rst)
   - Used to integrate DeepFilterNet


