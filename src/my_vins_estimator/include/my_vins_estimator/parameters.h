#pragma once
#include <ros/ros.h>
#include <Eigen/Dense>
#include <string>



const int WINDOW_SIZE=10;
const double FOCAL_LENGTH=460.0;
const int NUM_OF_CAM=1;
const int NUM_OF_F=1000;
//#define UNIT_SPHERE_ERROR
extern double MIN_PARALLAX;
extern double INIT_DEPTH;

extern double ACC_N;
extern double GYR_N;
extern double ACC_W;
extern double GYR_W;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern Eigen::Vector3d G;
extern Eigen::Vector3d TIC[1];
extern Eigen::Matrix3d RIC[1];


extern std::string IMU_TOPIC;
extern double TD;
extern int ROLLING_SHUTTER;
void readParameters(ros::NodeHandle& n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};