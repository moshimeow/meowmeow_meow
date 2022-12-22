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
	set(CMAKE_BUILD_TYPE DEBUG)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -O0 -fno-omit-frame-pointer -pedantic -Wall -Wextra -Wno-unused-parameter -Werror=incompatible-pointer-types")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0 -fno-omit-frame-pointer -Wall -Wextra -Wno-unused-parameter")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined")
endif()

# cmake .. 
#-DCMAKE_BUILD_TYPE=Debug -GNinja -DCMAKE_C_FLAGS="-g3 -O0 -fno-omit-frame-pointer" 
#-DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer"

# Default to PIC code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Make sure we have pretty colours
option(DISABLE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF)

if(NOT DISABLE_COLORED_OUTPUT)
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options(-fdiagnostics-color=always)
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		add_compile_options(-fcolor-diagnostics)
	endif()
endif()

find_package(Ceres REQUIRED)


find_package(Eigen3 3 REQUIRED)



find_package(StereoKitC REQUIRED)
# message("Found StereoKitC " ${StereoKitC_Found})



add_library(xrt-external-tinyceres INTERFACE)
target_include_directories(
	xrt-external-tinyceres SYSTEM INTERFACE src/tinyceres/include
	)


add_executable(helloworld src/hworld.cpp)
target_link_libraries(helloworld PRIVATE StereoKitC::StereoKitC xrt-external-tinyceres) # aux_os aux_util)

set_property(TARGET helloworld PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(helloworld PRIVATE ${EIGEN3_INCLUDE_DIR})

# add_sanitizers(helloworld)

install(TARGETS helloworld)
