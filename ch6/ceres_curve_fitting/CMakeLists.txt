cmake_minimum_required( VERSION 2.8 )
project( ceres_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(CMAKE_CXX_STANDARD 11)

# ceres 라이브러리를 사용하기 위해 cmake 모듈 추가
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# Ceres 라이브러리를 찾아 헤더를 추가
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting main.cpp )
# Ceres 및 OpenCV와 연결
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )
