#pragma once
#include <vector>
#include <map>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <Eigen/Dense>
#include "my_vins_estimator/parameters.h"
#include "my_vins_estimator/utility.h"
#include "my_vins_estimator/integration_base.h"
#include "my_vins_estimator/feature_manager.h"
#include "my_vins_estimator/initial_alignment.h"
#include "my_vins_estimator/solve_5pts.h"
#include "my_vins_estimator/initial_sfm.h"
#include "my_vins_estimator/marginalization_factor.h"

// #include "my_vins_estimator/imu_factor.h"
// #include "my_vins_estimator/projection_factor.h"
// #include "my_vins_estimator/pose_local_parameterization.h"
class IntegrationBase;

class Estimator
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Estimator();

    void clearState();
    void setParameter();

    void processIMU(double dt, const Vector3d &linear_acceleration , const Vector3d &angular_velocity);

    void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
        const std_msgs::Header &header);

    void vector2double();
    void double2vector();

    void solveOdometry();
    void optimization();    
    void slideWindow();

    bool failureDetection();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
public:
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

public:
    // =========================
    // 1. 滑窗状态变量
    // ========================= 
    Eigen::Vector3d Ps[WINDOW_SIZE + 1];
    Eigen::Matrix3d Rs[WINDOW_SIZE + 1];
    Eigen::Vector3d Vs[WINDOW_SIZE + 1];
    Eigen::Vector3d Bas[WINDOW_SIZE + 1];
    Eigen::Vector3d Bgs[WINDOW_SIZE + 1];
    Eigen::Matrix3d last_R;
    Eigen::Vector3d last_P;

    // =========================
    // 外参，第一版固定
    // =========================
    Eigen::Matrix3d ric[NUM_OF_CAM];
    Eigen::Vector3d tic[NUM_OF_CAM];

    

     // =========================
    // IMU 预积分
    // =========================

    // pre_integrations[i]:
    // 表示第 i-1 帧到第 i 帧之间的 IMU 预积分
    IntegrationBase *pre_integrations[WINDOW_SIZE + 1];

    // 临时预积分，原版用于图像帧之间实时累计
    IntegrationBase *tmp_pre_integration;

    // 保存每一帧间的 IMU 原始数据
    std::vector<double> dt_buf[WINDOW_SIZE + 1];
    std::vector<Eigen::Vector3d> linear_acceleration_buf[WINDOW_SIZE + 1];
    std::vector<Eigen::Vector3d> angular_velocity_buf[WINDOW_SIZE + 1];
    
    // =========================
    // 当前 IMU 上一时刻数据
    // =========================

    bool first_imu;
    bool failure_check_initialized;

    Eigen::Vector3d acc_0; 
    Eigen::Vector3d gyr_0;

     // =========================
    // 4. 滑窗控制变量
    // =========================

    int frame_count; 
    std_msgs::Header Headers[WINDOW_SIZE + 1];

    std::map<double, ImageFrame> all_image_frame;
    double initial_timestamp;


     // =========================

    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;


    // =========================
    // 特征管理
    // =========================
    FeatureManager f_manager;
    MotionEstimator m_estimator;
    // =========================
    // Ceres 参数块
    // =========================
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    
    MarginalizationInfo *last_marginalization_info;
    std::vector<double *> last_marginalization_parameter_blocks;
    
    Eigen::Vector3d g;
    double td;
};
