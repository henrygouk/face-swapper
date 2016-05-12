cmake_minimum_required(VERSION 2.8)

project(face-swapper CXX)

find_package(OpenCV REQUIRED)

add_definitions(-std=c++11)

add_executable(
	face-swapper
	source/main.cpp
	source/FaceSwapper.cpp
)

target_link_libraries(
	face-swapper
	${OpenCV_LIBS}
	pthread
	blas
	lapack
)
