# Copyright (C) 2024-2025 Tong Qiu (tong.qiu@intel.com)
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.25)
project(MeloTTS)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# for env witout setting utf-8 for Chinese 
if(WIN32)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>") #https://learn.microsoft.com/en-us/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8?view=msvc-170
    #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif()

if(WIN32) #default compiler on win is msvc. I've not verifed it with clang-cl yet.
    add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
    add_compile_options(/Zc:__cplusplus) # Add /Zc:__cplusplus flag for Visual Studio to properly set the __cplusplus macro
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Choose the configuration types" FORCE)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL") # refer to https://developercommunity.visualstudio.com/t/debug-build-works-but-release-build-failsas-well-a/428160
endif()

# Workaround for an MSVC compiler issue in some versions of Visual Studio 2022.
# The issue involves a null dereference to a mutex. For details, refer to link https://github.com/microsoft/STL/wiki/Changelog#vs-2022-1710
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1930 AND MSVC_VERSION LESS 1941)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

find_package(OpenVINOGenAI REQUIRED
    PATHS
        ${INTEL_OPENVINO_DIR} # get the environment variable
    NO_CMAKE_FIND_ROOT_PATH
)
 

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCE_FILES
    melo.cpp
    src/openvino_model_base.cpp
    src/openvino_tokenizer.cpp
    src/utils.cpp
    src/bert.cpp
    src/openvoice_tts.cpp
    src/tts.cpp
    src/language_modules/cmudict.cpp
    src/language_modules/chinese_mix.cpp
    src/language_modules/english.cpp
    src/language_modules/tone_sandhi.cpp
    src/language_modules/language_module_base.cpp
    src/text_normalization/text_normalization.cpp   
    src/text_normalization/char_convert.cpp     
    src/text_normalization/chronology.cpp     
    src/text_normalization/constants.cpp     
    src/text_normalization/num.cpp     
    src/text_normalization/phonecode.cpp     
    src/text_normalization/quantifier.cpp     
    ${CMAKE_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin.cc
    ${CMAKE_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin_csrc_utils.cc
    src/text_normalization/text_normalization.cpp
    src/text_normalization/char_convert.cpp
    src/text_normalization/chronology.cpp
    src/text_normalization/constants.cpp
    src/text_normalization/num.cpp
    src/text_normalization/phonecode.cpp
    src/text_normalization/quantifier.cpp
    src/deepfilternet/noisefilter.cpp
    src/deepfilternet/deepfilter.cpp
    src/deepfilternet/dfnet_model.cpp
    src/deepfilternet/multiframe.cpp
    src/mini-bart-g2p/mini-bart-g2p.cpp
    src/text_normalization/text_normalization_eng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin_csrc_utils.cc
)

# Header files
set(HEADER_FILES
    src/parse_args.h
    src/openvino_model_base.h
    src/info_data.h
    src/openvino_tokenizer.h
    src/utils.h
    src/bert.h
    src/openvoice_tts.h
    src/tts.h
    src/language_modules/cmudict.h
    src/language_modules/chinese_mix.h
    src/language_modules/english.h
    src/language_modules/tone_sandhi.h
    src/language_modules/language_module_base.h
    src/text_normalization/text_normalization.h
    src/text_normalization/char_convert.h
    src/text_normalization/chronology.h
    src/text_normalization/constant.h
    src/text_normalization/number.h
    src/text_normalization/phonecode.h
    src/text_normalization/quantifier.h
    src/text_normalization/text_normalization_eng.h
    ${CMAKE_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin.h
    ${CMAKE_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin_csrc_utils.h
    src/deepfilternet/noisefilter.h
    src/deepfilternet/deepfilter.h
    src/deepfilternet/multiframe.h
    src/deepfilternet/dfnet_model.h
    src/mini-bart-g2p/mini-bart-g2p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppinyin/csrc/cppinyin_csrc_utils.h
)
# Define the executable
add_executable(meloTTS_ov ${SOURCE_FILES} ${HEADER_FILES})

# Whether use deep filter net; We do not support this feature on Linux now.
option(USE_DEEPFILTERNET "Enable DeepFilterNet support" ON)

if(USE_DEEPFILTERNET)
    add_compile_definitions(USE_DEEPFILTERNET)
    message(STATUS "DeepFilterNet is enabled")
else()
    message(STATUS "DeepFilterNet is disabled")
endif()


if(USE_DEEPFILTERNET AND WIN32)
    add_compile_definitions(USE_DEEPFILTERNET)
    # unzip libtorch
    set(LIBTORCH_DIR  ${CMAKE_SOURCE_DIR}/thirdParty/libtorch)
    if(NOT EXISTS ${LIBTORCH_DIR})
        message(STATUS "libtorch directory does not exist, will perform unzip.")
        add_custom_target(unZip ALL)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf libtorch.tar.gz 
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/thirdParty
        )
    else()
        message(STATUS "libtorch folder already exists, skipping unzip.")
    endif()
    # COPY torch runtime to exe folder
    set(RELEASE_DIR ${CMAKE_BINARY_DIR}/Release)
    #set(DEBUG_DIR ${CMAKE_BINARY_DIR}/Debug)
    file(MAKE_DIRECTORY ${RELEASE_DIR})

    set(DLLS torch.dll torch_cpu.dll asmjit.dll fbgemm.dll c10.dll uv.dll libiomp5md.dll)

    foreach(dll ${DLLS})
        message("Copying ${dll} from ${LIBTORCH_DIR}/lib to ${RELEASE_DIR}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBTORCH_DIR}/lib/${dll} ${RELEASE_DIR}
            #COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBTORCH_DIR}/lib/${dll} ${DEBUG_DIR}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endforeach()
    # Find libraries
    find_library(TORCH_LIB torch PATHS ${LIBTORCH_DIR}/lib)
    if (NOT TORCH_LIB)
        message(FATAL_ERROR "torch.lib not found")
    endif()

    find_library(C10_LIB c10 PATHS ${LIBTORCH_DIR}/lib)
    if (NOT C10_LIB)
        message(FATAL_ERROR "c10.lib not found")
    endif()

    find_library(TORCH_CPU_LIB torch_cpu PATHS ${LIBTORCH_DIR}/lib)
    if (NOT TORCH_CPU_LIB)
        message(FATAL_ERROR "torch_cpu.lib not found")
    endif()
    target_link_directories(meloTTS_ov PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/libtorch/lib")
    target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/libtorch/include)
    target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/libtorch/include/torch/csrc/api/include)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        set(ADDITIONAL_LIBRARY_DEPENDENCIES "torch.lib" "c10.lib" "torch_cpu.lib")
        target_link_libraries(meloTTS_ov PRIVATE ${ADDITIONAL_LIBRARY_DEPENDENCIES})
    endif()

endif() # end USE_DEEPFILTERNET AND WIN32

if(USE_DEEPFILTERNET AND UNIX)
    message(STATUS "Checking if libtorch is already downloaded and extracted")

    set(LIBTORCH_ZIP ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/libtorch-cxx11-abi-shared-with-deps-2.6.0*cpu.zip)
    set(LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/libtorch)

    if(NOT EXISTS ${LIBTORCH_ZIP} AND NOT EXISTS ${LIBTORCH_DIR})
        message(STATUS "Downloading and extracting libtorch for Linux")

        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty)

        execute_process(
            COMMAND wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty
            RESULT_VARIABLE download_result
        )
        if(NOT download_result EQUAL 0)
            message(FATAL_ERROR "Failed to download libtorch.")
        endif()

        execute_process(
            COMMAND unzip libtorch-cxx11-abi-shared-with-deps-2.6.0*cpu.zip
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty
            RESULT_VARIABLE unzip_result
        )
        if(NOT unzip_result EQUAL 0)
            message(FATAL_ERROR "Failed to unzip libtorch.")
        endif()

        message(STATUS "libtorch downloaded and extracted successfully")
    else()
        message(STATUS "libtorch already downloaded and extracted, skipping download")
    endif()
    
    set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
    find_package(Torch REQUIRED)
    target_link_libraries(meloTTS_ov PRIVATE "${TORCH_LIBRARIES}")

    # Copy torch runtime to executable folder
    file(GLOB_RECURSE LIBSO_FILES ${LIBTORCH_DIR}/lib/*.so)
    set(RELEASE_DIR ${CMAKE_BINARY_DIR})
    foreach(SO_FILE ${LIBSO_FILES})
        message("Copying ${SO_FILE} from  ${LIBTORCH_DIR}/lib to ${RELEASE_DIR}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBTORCH_DIR}/lib/${dll} ${RELEASE_DIR}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endforeach()

endif() # end USE_DEEPFILTERNET AND UNIX
   



# Define DEBUG macro for Debug configuration
target_compile_definitions(meloTTS_ov PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>")
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_SOURCE_DIR}/thirdParty/cppjieba)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_SOURCE_DIR}/thirdParty/cppjieba/include)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_SOURCE_DIR}/thirdParty/cppinyin/csrc)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba/include)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppinyin/csrc)


target_link_libraries(meloTTS_ov
    PRIVATE openvino::genai  # Link OpenVINO.GenAI Runtime privately
)

if (UNIX)
    target_link_libraries(meloTTS_ov PRIVATE pthread)
endif()


option(USE_BERT_NPU "Set Bert on NPU" OFF)
if(USE_BERT_NPU)
    message(STATUS "USE BERT on NPU: We need a static shape of the model.")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/ov_models/bert_ZH_int8.bin
                ${CMAKE_SOURCE_DIR}/ov_models/bert_ZH_static_int8.bin
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    message("Copying bert_ZH_int8.bin to bert_ZH_int8.bin if bert_ZH_static_int8.bin does not exist")
endif()


option(ENABLE_TEST "Enable tests" OFF) # Disable the tests by default

if (${ENABLE_TEST})
    enable_testing()
    add_subdirectory(tests)
endif()

