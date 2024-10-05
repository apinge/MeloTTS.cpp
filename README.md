# MeloTTS.cpp

[meloTTS](https://github.com/myshell-ai/MeloTTS), released by MyShell.ai, is a high-quality, multilingual Text-to-Speech (TTS) library that supports English, Chinese(mixed English), and various other languages.This repo delivers a pure C++ build of meloTTS fully integrated with OpenVINO. As of now, this repository is limited to supporting Chinese mixed with English.

## How to build and run

### Windows
```
cd openvino_package
setupvars.bat
cd MeloTTS.cpp 
cmake -S . -B build && cmake --build . --config Release
build\Release\meloTTS_ov.exe
```
### Linux
```
cd openvino_package
source setupvars.sh
cd MeloTTS.cpp 
cmake -S . -B build && cmake --build . --config Release
build\meloTTS_ov
```

## Supported Versions
- **Operating System**: Ubuntu, Windows 11
- **CPU Architecture**: Metor Lake,  Lunar Lake
- **GPU Architecture**: Intel® Arc™ Graphics (Intel Xe)
- **C++ Version**: >=C++20


