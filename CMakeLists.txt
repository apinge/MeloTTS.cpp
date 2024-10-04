cmake_minimum_required(VERSION 3.10)
project(MeloTTS)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# for env witout setting utf-8 for Chinese 
if(WIN32)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif()


find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)


# Source files
set(SOURCE_FILES
    melo.cpp
    src/openvino_model.cpp
    src/logging.cpp
    src/tokenizer.cpp
    src/cmudict.cpp
    src/utils.cpp
    src/bert.cpp
    src/chinese_mix.cpp
    src/openvoice_tts.cpp
    src/tts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/hanz2piny/Hanz2Piny.cpp
)

# Header files
set(HEADER_FILES
    src/cmudict.h
    src/openvino_model.h
    src/logging.h
    src/info_data.h
    src/status.h
    src/librosa.h
    src/tokenizer.h
    src/utils.h
    src/bert.h
    src/chinese_mix.h
    src/openvoice_tts.h
    src/tts.h
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/eigen)

# Define the executable
add_executable(meloTTS_ov ${SOURCE_FILES} ${HEADER_FILES})

# Define DEBUG macro for Debug configuration
target_compile_definitions(meloTTS_ov PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>")
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/eigen)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba/include)
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/hanz2piny)

target_link_libraries(meloTTS_ov
    PRIVATE openvino::runtime  # Link OpenVINO Runtime privately
)

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()


#CTest
# https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html
option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})
    #add_subdirectory(tests)
    add_executable(test_tokenizer tests/test_tokenizer.cpp src/tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE openvino::runtime) 
    target_include_directories(test_tokenizer PRIVATE src/tokenizer.h)
    #set_target_properties(bert_tokenizer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

    add_executable(test_cmudict tests/test_cmudict.cpp src/cmudict.cpp)
    target_include_directories(test_cmudict PRIVATE src/cmudict.h)

    add_executable(test_bert tests/test_bert.cpp src/tokenizer.cpp src/openvino_model.cpp src/bert.cpp  src/utils.cpp src/logging.cpp)
    target_link_libraries(test_bert PRIVATE openvino::runtime) 
    target_include_directories(test_bert PRIVATE src/tokenizer.h src/openvino_model.h src/bert.h  src/logging.h src/utils.h)

    add_executable(test_openvoice_tts tests/test_openvoice_tts.cpp src/openvino_model.cpp src/openvoice_tts.cpp  src/utils.cpp src/logging.cpp src/tokenizer.cpp src/bert.cpp src/logging.cpp)
    target_link_libraries(test_openvoice_tts PRIVATE openvino::runtime) 
    target_include_directories(test_openvoice_tts PRIVATE src/openvoice_tts.h src/openvino_model.h src/logging.h src/utils.h src/tokenizer.h src/bert.h src/info_data.h )

    add_executable(test_chinese_mix tests/test_chinese_mix.cpp src/chinese_mix.cpp src/tokenizer.cpp src/utils.cpp)
    target_include_directories(test_chinese_mix PRIVATE src/chinese_mix.h src/tokenizer.h src/tokenizer.h src/logging.h src/utils.h
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba 
            ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba/include)


    add_executable(test_hanz2piny tests/test_hanz2piny.cpp ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/hanz2piny/Hanz2Piny.cpp )
    target_include_directories(test_hanz2piny PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/hanz2piny)

    add_executable(test_cppjieba tests/test_cppjieba.cpp)
    target_include_directories(test_cppjieba PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba 
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba/include)
                                                    #${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/cppjieba/include)
endif()
