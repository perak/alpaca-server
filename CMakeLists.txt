cmake_minimum_required(VERSION 3.23)
project(alpaca_server)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(LLAMA_ALL_WARNINGS            "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY  "llama: enable all compiler warnings in 3rd party libs" OFF)

option(LLAMA_SANITIZE_THREAD         "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS        "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED      "llama: enable undefined sanitizer" OFF)


if (APPLE)
    option(LLAMA_NO_ACCELERATE       "llama: disable Accelerate framework" OFF)
    option(LLAMA_NO_AVX              "llama: disable AVX" OFF)
    option(LLAMA_NO_AVX2             "llama: disable AVX2" OFF)
    option(LLAMA_NO_FMA              "llama: disable FMA" OFF)
endif()

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=undefined")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    endif()
endif()

if (APPLE AND NOT LLAMA_NO_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(LLAMA_EXTRA_LIBS  ${LLAMA_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
        set(LLAMA_EXTRA_FLAGS ${LLAMA_EXTRA_FLAGS} -DGGML_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wshadow                        \
            -Wcast-qual                     \
            -Wstrict-prototypes             \
            -Wpointer-arith                 \
            -Wno-unused-function            \
        ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wcast-qual                     \
        ")
    else()
        # todo : msvc
    endif()
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
else()
    message(STATUS "x86 detected")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
    else()
        if(NOT LLAMA_NO_AVX)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        if(NOT LLAMA_NO_AVX2)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        if(NOT LLAMA_NO_FMA)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
    endif()
endif()

# if (LLAMA_PERF)
#     set(LLAMA_EXTRA_FLAGS ${LLAMA_EXTRA_FLAGS} -DGGML_PERF)
# endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-httplib/ ${CMAKE_CURRENT_SOURCE_DIR}/external/json/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(alpaca_server main.cpp src/utils.cpp src/llama.cpp)
add_library(ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml.c ${CMAKE_CURRENT_SOURCE_DIR}/include/ggml.h)

target_compile_definitions(ggml PUBLIC ${LLAMA_EXTRA_FLAGS})
target_compile_definitions(alpaca_server PUBLIC ${LLAMA_EXTRA_FLAGS})
target_link_libraries(ggml PRIVATE ${LLAMA_EXTRA_LIBS})
target_link_libraries(alpaca_server PRIVATE ggml)
