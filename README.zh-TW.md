# MeloTTS.cpp
<p>
   <b>< <a href='./README.md'>English</a></b> | <a href='./README.zh-CN.md'>简体中文</a> | <b>繁體中文</b> >
</p>


**MeloTTS.cpp** 是[MeloTTS](https://github.com/myshell-ai/MeloTTS) 的 C++ 實現，**MeloTTS**是由MyShell.ai 發佈的一個高品質、多語言的文字轉語音(Text To Speech) 函式庫，支援英語、中文以及其他多種語言。這個函式庫是基於**OpenVINO**並且支援在 CPU、GPU 和 NPU 邊緣設備上的部署。目前，僅支援中文(混合英文)及英文。對於 [日文模型](https://huggingface.co/myshell-ai/MeloTTS-Japanese) 的支援計畫推出。


## 🔀 分支使用指南
此存儲庫支持多語言文本到語音的推理。請根據您的使用情況切換到適當的分支：
 - `EN` 分支：
用於僅限英語的語音推理。
- `ZH_MIX_EN` 分支：
專為普通話-英語混合語音而設計。
- `multilang-develop` 分支：
用於多語言語音推理，支持普通話-英語混合語音和僅限英語的處理。

## Pipeline Design




MeloTTS.cpp的設計與[原始PyTorch 版本](https://github.com/myshell-ai/MeloTTS) 基本架構是一致的。由三個模型組成（BERT、TTS 和DeepFilterNet），其中 DeepFilterNet 是額外新增的模型。


<img src="images/melotts_design.png" alt="Pipeline Design" title="Pipeline Design" width="800" style="display: block">


#### 圖例
1. Tokenizer and BERT: Tokenizer 和 BERT 模型為中文使用 `bert-base-multilingual-uncased`，英文使用 `bert-base-uncased`
2. g2p: 字母到音素的轉換。對於英語的g2p，使用[mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p)來生成音素。詳情請參見[Enable mini-bart-g2p for OpenVINO](https://github.com/apinge/MeloTTS.cpp/blob/multilang-develop/experimental/mini-bart-g2p/README.md)。
3. phones and tones: 中文表示為拼音和四聲，英文表示為音標和重音
4. tone_sandi: 修正分詞和音素的類別（僅用於中文）
5. DeepFilterNet: 用於降噪（用於 int8 量化所引入的背景噪音）


### Model-Device Compatibility Table
下表概述了每個模型支援的XPU：
| Model Name       | CPU Support | GPU Support | NPU Support |
|------------------|-------------|-------------|-------------|
| BERT (Preprocessing) | ✅           | ✅           | ✅           |
| TTS (Inference)      | ✅           | ✅           | ❌           |
| DeepFilterNet (Post-processing) | ✅           | ✅           | ✅           |

## Setup and Execution Guide

### 1. 下載 OpenVINO C++ Package

若要下載 OpenVINO GenAI C++ 套件，請參考以下連結：[Install OpenVINO™ GenAI](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html)。
對於 **OpenVINO™ GenAI 2025.2** 在 Windows 上的安裝，可以在cmd中執行命令列。

```
curl -O https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/windows/openvino_genai_windows_2025.2.0.0_x86_64.zip
tar -xzvf openvino_genai_windows_2025.2.0.0_x86_64.zip
```
對於 OpenVINO 2025.2 在 Linux 上的安裝，只需要從 https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/linux/ 下載並解壓縮該套件。

對於 OpenVINO 2025.2 在 MacOS 上的安裝，只需要從 https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2025.2/macos/ 下載並解壓縮該套件。

有關其他版本和更多 OpenVINO 的資訊，請參考 OpenVINO 官方工具包頁面：[OpenVINO Toolkit Overview](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

### 2. Clone Git Repository
```
git lfs install
git clone https://github.com/apinge/MeloTTS.cpp.git
```

### 3. 編譯與執行
#### 3.1 Windows 編譯與執行
```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd MeloTTS.cpp
cmake -S . -B build && cmake --build build --config Release
.\build\Release\meloTTS_ov.exe --model_dir ov_models --input_file inputs_en.txt  --output_filename audio
```
#### 3.2 Linux 編譯與執行
```
source <OpenVINO_GenAI_DIR>/setupvars.sh
cd MeloTTS.cpp 
cmake -S . -B build && cmake --build build --config Release
./build/meloTTS_ov --model_dir ov_models --input_file inputs_en.txt --output_filename audio
```
#### 3.3 在 cmake build 裡啟用或停用 DeepFilterNet
DeepFilterNet 功能目前有技援 Windows 和 Linux平台，用於消除由於 int8 TTS 量化模型中所產生的雜訊。在預設情況下，該功能是啟用的，但可以在 CMake build 階段使用 `-DUSE_DEEPFILTERNET` 選項來啟用或停用它。

例如，若要停用此功能，可以在 CMake build 產生過程中使用下列命令：
```
cmake -S . -B build -DUSE_DEEPFILTERNET=OFF
```
有關更多資訊，請參考 [DeepFilterNet.cpp](https://github.com/apinge/MeloTTS.cpp/blob/develop/src/deepfilternet/README.md).

### 4. 參數說明

可以使用 `run_tts.bat` 或 `run_tts.sh` 作為範例腳本來執行模型。以下為參數說明：


- `--model_dir`: 指定包含模型檔案、字典檔案和第三方資源檔案的資料夾，該資料夾為函式庫中的 `ov_models` 資料夾。需要根據目前工作目錄調整相對工作路徑。
- `--tts_device`: 指定用於 TTS 模型的 OpenVINO 裝置。支援的設備包括 CPU 和 GPU（預設：CPU）。
- `--bert_device`: 指定用於 BERT 模型的 OpenVINO 裝置。支援的設備包括 CPU、GPU 和 NPU（預設：CPU）。
- `--nf_device`: 指定用於 DeepfilterNet 模型的 OpenVINO 裝置。支援的設備包括 CPU、GPU 和 NPU（預設：CPU）。
- `--input_file`: 指定要處理的輸入文字檔。確保文字是 **UTF-8** 格式。
- `--output_filename`: 指定生成的輸出音訊檔案名稱，格式為 {output_filename}_{language_style}.wav。例如，如果語言為中文且 output_filename 為 "audio"，檔案將儲存為 audio_ZH-MIX-EN.wav。
- `--speed`: 指定輸出音訊的速度。預設值為 1.0。
- `--quantize`: 指定是否使用 tts的量化模型。預設值為 `true`，表示預設使用 int8 模型。
- `--disable_bert`: 指定是否停用 BERT 模型推理。預設值為 `false`。
- `--disable_nf`: 指定是否停用 DeepfilterNet 模型推理（預設：`false`）。
- `--language`: 指定 TTS 的語言。預設語言為英文（`EN`）。

## NPU設備支援
BERT 和 DeepFilterNet 模型支援將 NPU 作為推理設備，利用 Meteor Lake 和 Lunar Lake 中整合的 NPU。


以下是啟用的方法:
<details>
  <summary>Click here to expand/collapse content</summary>
  <ul>
   <li><strong>在 CMake 生成階段</strong></li>
   要在 NPU 上啟用 BERT 模型，在 CMake build 階段需要額外的 CMake 選項 <code>-DUSE_BERT_NPU=ON</code>。例如：
    <pre><code>cmake -DUSE_BERT_NPU=ON -B build -S .</code></pre>
   在 NPU 上啟用 DeepFilterNet，無需額外的編譯步驟。
   <li><strong>設定參數</strong></li>
        若要設定 NPU 上的模型參數，分別使用 <code>--bert_device NPU</code>來設定 BERT 模型，使用 <code>--nf_device NPU</code> 來設定 DeepFilterNet 模型。例如：
        <pre><code>build\Release\meloTTS_ov.exe --bert_device NPU --nf_device NPU --model_dir ov_models --input_file inputs.txt  --output_file audio.wav</code></pre>
    
</ul>
</details>

## 版本支援
- **Operating System**: Windows, Linux, MacOS
- **CPU Architecture**: Metor Lake, Lunar Lake, Arrow Lake, 大多數 Intel CPUs
- **GPU Architecture**: Intel® Arc™ Graphics (Intel Xe, 包括iGPU)
- **NPU  Architecture**: NPU 4, Meteor Lake, Lunar Lake, Arrow Lake中整合的NPU
- **OpenVINO Version**: >=2024.4
- **C++ Version**: >=C++20

如果使用的 Windows 的 AI PC 筆記型電腦，GPU 和 NPU 的驅動程式通常是預先安裝的。然而，Linux 使用者或希望更新到最新驅動程式的 Windows 使用者請參考以下連結操作:

- **安裝/更新 GPU 驅動程式**: 請參考 [Configurations for Intel® Processor Graphics (GPU) with OpenVINO™](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html) 

- **安裝/更新 NPU 驅動程式**: 請參考 [NPU Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) 

請注意，Windows 和 Linux 的驅動程式不同，因此請確保按照特定作業系統的說明進行操作。

## 開發計劃

1. **實現MeloTTS日文版本**: 
   
2. **提高量化品質**:
   - 目前的INT8量化模型表現出輕微的背景雜訊。我們整合了 DeepFilterNet 進行後處理。未來的目標是透過量化技術解決噪音問題。

## Jupyter Notebook版本

我們已經更新了[Voice tone cloning with OpenVoice2 and MeloTTS for Text-to-Speech by OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvoice2-and-melotts)，其中包含了如何將torch版本轉換為OpenVINO IR的步驟。這個倉庫的原始Python版本[MeloTTS-OV](https://github.com/zhaohb/MeloTTS-OV/tree/speech-enhancement-and-npu) 現在已經停止維護。

## 第三方函式庫

參考了以下函式庫:

- [cppjieba](https://github.com/yanyiwu/cppjieba)
    - C++中文分詞庫
- [cppinyin](https://github.com/pkufool/cppinyin)
    - C++漢語拼音庫，我們移除了 python 部分並且整合在程式碼中
- [libtorch](https://github.com/pytorch/pytorch/blob/main/docs/libtorch.rst)
   - 用於 DeepFilterNet


