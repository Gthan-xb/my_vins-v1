#include "my_vins_estimator/parameters.h"

#include <vector>

double MIN_PARALLAX;
double INIT_DEPTH;

double ACC_N;
double GYR_N;
double ACC_W;
double GYR_W;

double SOLVER_TIME;
int NUM_ITERATIONS;

Eigen::Vector3d G;

Eigen::Vector3d TIC[NUM_OF_CAM];
Eigen::Matrix3d RIC[NUM_OF_CAM];

std::string IMU_TOPIC;
double TD;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
double TR;
double ROW;
double COL;

void readParameters(ros::NodeHandle& n)
{
    n.param("min_parallax", MIN_PARALLAX, 10.0);
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    n.param("init_depth", INIT_DEPTH, 5.0);

    n.param("acc_n", ACC_N, 0.02);
    n.param("gyr_n", GYR_N, 0.015);
    n.param("acc_w", ACC_W, 0.0002);
    n.param("gyr_w", GYR_W, 0.00015);

    n.param("max_solver_time", SOLVER_TIME, 0.04);
    n.param("max_num_iterations", NUM_ITERATIONS, 8);

    n.param("imu_topic", IMU_TOPIC, std::string("/imu0"));

    n.param("td", TD, 0.0);
    n.param("estimate_td", ESTIMATE_TD, 0);

    n.param("rolling_shutter", ROLLING_SHUTTER, 0);
    n.param("rolling_shutter_tr", TR, 0.0);

    n.param("image_height", ROW, 480.0);
    n.param("image_width", COL, 752.0);

    ROS_INFO("td initial value: %.6f, estimate_td: %d", TD, ESTIMATE_TD);
    ROS_INFO("image size: ROW = %.0f, COL = %.0f, rolling shutter = %d, TR = %.6f",
            ROW, COL, ROLLING_SHUTTER, TR);
  
    
    std::vector<double> g_vec;
    if (n.getParam("gravity", g_vec) && g_vec.size() == 3)
    {
        G = Eigen::Vector3d(g_vec[0], g_vec[1], g_vec[2]);
    }
    else
    {
        G = Eigen::Vector3d(0.0, 0.0, 9.8);
        ROS_WARN("gravity not set correctly, use default [0, 0, 9.8]");
    }

    std::vector<double> tic_vec;
    if (n.getParam("tic", tic_vec) && tic_vec.size() == 3)
    {
        TIC[0] = Eigen::Vector3d(tic_vec[0], tic_vec[1], tic_vec[2]);
    }
    else
    {
        TIC[0].setZero();
        ROS_WARN("tic not set correctly, use zero");
    }

    std::vector<double> ric_vec;
    if (n.getParam("ric", ric_vec) && ric_vec.size() == 9)
    {
        RIC[0] << ric_vec[0], ric_vec[1], ric_vec[2],
                  ric_vec[3], ric_vec[4], ric_vec[5],
                  ric_vec[6], ric_vec[7], ric_vec[8];
    }
    else
    {
        RIC[0].setIdentity();
        ROS_WARN("ric not set correctly, use identity");
    }

    ROS_INFO("Parameters loaded.");
}