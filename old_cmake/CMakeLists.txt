# Copyright 2018-2020, Collabora, Ltd.
# SPDX-License-Identifier: BSL-1.0

# If building for android, must call CMake with something like:
# cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake .. -DANDROID_PLATFORM=24
#
# You need the toolchain file to cause CMake to build for Android
# You need a new enough platform level to get Vulkan headers - 24 known to work

cmake_minimum_required(VERSION 3.10.2)


# We use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wall -Wextra -Wno-unused-parameter -Werror=incompatible-pointer-types")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined")
endif()

# Default to PIC code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Ceres REQUIRED)


find_package(Eigen3 3 REQUIRED)



find_package(StereoKitC REQUIRED)
# message("Found StereoKitC " ${StereoKitC_Found})






add_executable(helloworld src/hworld.cpp)
target_link_libraries(helloworld PRIVATE StereoKitC::StereoKitC Ceres::ceres) # aux_os aux_util)

set_property(TARGET helloworld PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(helloworld PRIVATE ${EIGEN3_INCLUDE_DIR})

# add_sanitizers(helloworld)

install(TARGETS helloworld)
