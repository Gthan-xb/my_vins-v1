cmake_minimum_required(VERSION 2.8.3)
project(camera_model)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# 手动设置 camodocal 路径（避免 find_package(CAMODOCAL) 可能失败）
set(CAMODOCAL_INCLUDE_DIR "/usr/local/include/camodocal")
set(CAMODOCAL_LIBRARY_DIR "/usr/local/lib/camodocal")

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES camera_model
    CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    include
    ${CAMODOCAL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

link_directories(${CAMODOCAL_LIBRARY_DIR})

# 如果你原来的包有一个“camera_model”库，且你的其他节点依赖它，
# 那么保留这个库，但源文件应该使用你实际存在的封装代码，
# 而不是 “your_own_code.cpp”。如果没有任何封装，删除整个 add_library 块。
# 假设你有一个简单的封装文件 src/camera_model.cpp（请确认是否存在），
# 否则注释掉以下两行。

add_library(camera_model
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)
target_link_libraries(camera_model
    ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}
)

# 可执行文件
add_executable(Calibration src/intrinsic_calib.cc)
target_link_libraries(Calibration
    ${catkin_LIBRARIES}
    camera_model
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)