cmake_minimum_required(VERSION 3.10)
project(MeloTTS)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCE_FILES
    melo.cpp
    src/openvoice2_processor.cpp
    src/openvino_model.cpp
    src/logging.cpp
)

# Header files
set(HEADER_FILES
    src/openvoice2_processor.h
    src/openvino_model.h
    src/logging.h
    src/info_data.h
    src/status.h
    src/librosa.h
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/eigen)




# Define the executable
add_executable(meloTTS_ov ${SOURCE_FILES} ${HEADER_FILES})

# Define DEBUG macro for Debug configuration
target_compile_definitions(meloTTS_ov PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>")
target_include_directories(meloTTS_ov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/eigen)



target_link_libraries(meloTTS_ov
    PRIVATE openvino::runtime  # Link OpenVINO Runtime privately
)

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()



