#include "my_vins_estimator/estimator.h"
#include "my_vins_estimator/projection_factor.h"
#include"my_vins_estimator/pose_local_parameterization.h"
#include "my_vins_estimator/imu_factor.h"
#include "my_vins_estimator/parameters.h"
#include "my_vins_estimator/initial_sfm.h"
#include "my_vins_estimator/initial_alignment.h"
#include "my_vins_estimator/solve_5pts.h"
#include "my_vins_estimator/utility.h"
#include <iostream>
#include<ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace Eigen;

Estimator::Estimator():f_manager(Rs)
{
    ROS_INFO("my_vins_estimator init begins");
    
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    
    td = TD;
    g = G;
}

void Estimator::clearState()
{

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i].setZero();
        Rs[i].setIdentity();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();

        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i]!=nullptr)
        {
            delete pre_integrations[i];
            pre_integrations[i] = nullptr;
        }
    }

    if (tmp_pre_integration != nullptr)
    {
        delete tmp_pre_integration;
        tmp_pre_integration = nullptr;
    }

    for(int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i].setZero();
        ric[i].setIdentity();
    }
    
    if (last_marginalization_info != nullptr)
    {
        delete last_marginalization_info;
        last_marginalization_info = nullptr;
    }

    last_marginalization_parameter_blocks.clear();


    acc_0.setZero();
    gyr_0.setZero();
    last_R.setIdentity();
    last_P.setZero();
    first_imu = false;
    failure_check_initialized = false;
    frame_count = 0;
    solver_flag = INITIAL;
    marginalization_flag = MARGIN_OLD;
    
    g=G;
    
    td = TD;

    f_manager.clearState();

    for (auto &it : all_image_frame)
   {
      if (it.second.pre_integration != nullptr)
      {
         delete it.second.pre_integration;
         it.second.pre_integration = nullptr;
      }
   }
   all_image_frame.clear();

   initial_timestamp = 0.0;
}


void Estimator::processIMU(double dt, const Vector3d &linear_acceleration , const Vector3d &angular_velocity)
{
    // ==========================================
    // 1. 处理第一帧 IMU
    // ==========================================
    if (!first_imu)
    {
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        first_imu = true;
        return;
    }

    // ==========================================
    // 2. 如果当前帧的预积分对象还没创建，则创建
    // ==========================================
    if (pre_integrations[frame_count] == nullptr)
    {
        
        pre_integrations[frame_count] =new IntegrationBase(acc_0, gyr_0,Bas[frame_count],Bgs[frame_count]);
    }


    if (tmp_pre_integration == nullptr)
    {
        tmp_pre_integration = new IntegrationBase(acc_0, gyr_0,Bas[frame_count],Bgs[frame_count]);
    }

    
    if (frame_count != 0)
    {
      pre_integrations[frame_count]->push_back(
          dt,
          linear_acceleration,
          angular_velocity
      );

      tmp_pre_integration->push_back(
          dt,
          linear_acceleration,
          angular_velocity
      );
    dt_buf[frame_count].push_back(dt);
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    angular_velocity_buf[frame_count].push_back(angular_velocity);
  
    int j = frame_count;
    Vector3d un_acc_0= Rs[j] * (acc_0 - Bas[j]) - g;
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] = Rs[j] * Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] *(linear_acceleration - Bas[j]) - g;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Vs[j] = Vs[j] + un_acc * dt;
    Ps[j] = Ps[j] + Vs[j] * dt + 0.5 * un_acc * dt * dt;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}


void Estimator::processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    const std_msgs::Header &header)
{
    ROS_INFO("new image coming, stamp:%.6f", header.stamp.toSec());
    ROS_INFO("feature size:% lu", image.size());


     /*
     * 1. 把当前图像帧的特征放进 FeatureManager
     *
     * 原版这里会根据视差决定：
     *   MARGIN_OLD
     *   MARGIN_SECOND_NEW
     *
     * 第一版我们暂时只保留接口和主流程。
     */
    bool is_keyframe = f_manager.addFeatureCheckParallax(frame_count, image, td);

    if (is_keyframe)
    {
        marginalization_flag = MARGIN_OLD;
        ROS_INFO("current frame is treated as keyframe");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        ROS_INFO("current frame is treated as non-keyframe");
    }

    /*
     * 2. 保存当前图像帧 Header
     */
     Headers[frame_count] = header;

     /*
     * 3. 当前图像帧到来后，说明上一段 IMU 预积分已经归属于当前图像帧。
     *
     */

   ImageFrame imageframe(image, header.stamp.toSec());
   imageframe.pre_integration = tmp_pre_integration;
   all_image_frame.insert(std::make_pair(header.stamp.toSec(), imageframe));

   tmp_pre_integration =new IntegrationBase(acc_0,gyr_0,Bas[frame_count],Bgs[frame_count]);
    
    /*
     * 当 frame_count < WINDOW_SIZE:
     *   继续往后走
     */
    if (frame_count < WINDOW_SIZE)
   {
      frame_count++;
   }
   else
   {
    /*
     * 窗口已经满了。
     *
     * 原版逻辑：
     * 1. 如果还没初始化，先尝试 initialStructure()
     * 2. 初始化成功后，切换到 NON_LINEAR
     * 3. 后续每帧走 solveOdometry() + slideWindow()
     */
      if (solver_flag == INITIAL)
     {
        ROS_INFO("solver_flag == INITIAL, try initialization");

        bool result = initialStructure();

        if (result)
        {
            ROS_INFO("initialization success, switch to NON_LINEAR");

            solver_flag = NON_LINEAR;

            /*
             * 初始化成功后，先做一次非线性优化。
             *
             * 注意：
             * visualInitialAlign() 已经给 Ps/Rs/Vs/Bgs/g/深度提供了合理初值。
             * 这里再调用 solveOdometry()，相当于用 IMUFactor + ProjectionFactor
             * 做一次完整滑窗优化。
             */
            solveOdometry();
            if (failureDetection())
          {
            ROS_WARN("failure detected right after initialization, reset system");

            clearState();
            setParameter();

            return;
          }
            
            f_manager.removeFailures();

            last_R = Rs[WINDOW_SIZE];
            last_P = Ps[WINDOW_SIZE];

            slideWindow();

            ROS_INFO("initialization finished and slideWindow executed");
        }
        else
       {
          ROS_WARN("initialization failed, slideWindow and wait for next try");

            /*
             * 初始化失败时：
             * 原版会滑窗继续等待更好的运动和视差。
             *
             * 第一版没有边缘化，所以这里直接 slideWindow()。
             */
          slideWindow();
       }
     }
     else
    {
        /*
         * 已经初始化完成，正常 VIO 非线性优化阶段。
         */
        ROS_INFO("solver_flag == NON_LINEAR, solve odometry");

        solveOdometry();
        
        if (failureDetection())
        {
            ROS_WARN("estimator failure detected, reset system");
            clearState();
            setParameter();
            return;
        }
        f_manager.removeFailures();

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];

        slideWindow();

        ROS_INFO("nonlinear optimization and slideWindow executed");
            
    }
   }

  ROS_INFO("frame_count: %d, feature_count: %d, all_image_frame: %lu, solver_flag: %d",frame_count,f_manager.getFeatureCount(),all_image_frame.size(),solver_flag);
    
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // ==========================
        // pose: P + Q
        // ==========================

        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();

        Eigen::Quaterniond q(Rs[i]);

        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        // ==========================
        // speed + bias
        // ==========================

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // ==========================
        // extrinsic pose: tic + ric
        // ==========================

        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();

        Eigen::Quaterniond q(ric[i]);

        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // ==========================
    // feature inverse depth
    // ==========================

    Eigen::VectorXd dep = f_manager.getDepthVector();

    for (int i = 0; i < dep.size(); i++)
    {
        para_Feature[i][0] = dep(i);
    }
} 


void Estimator::double2vector()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // ==========================
        // pose
        // ==========================

        Ps[i] = Eigen::Vector3d(para_Pose[i][0],
                                para_Pose[i][1],
                                para_Pose[i][2]);

        Eigen::Quaterniond q(para_Pose[i][6],
                             para_Pose[i][3],
                             para_Pose[i][4],
                             para_Pose[i][5]);

        Rs[i] = q.normalized().toRotationMatrix();

        // ==========================
        // speed + bias
        // ==========================

        Vs[i] = Eigen::Vector3d(para_SpeedBias[i][0],
                                para_SpeedBias[i][1],
                                para_SpeedBias[i][2]);

        Bas[i] = Eigen::Vector3d(para_SpeedBias[i][3],
                                 para_SpeedBias[i][4],
                                 para_SpeedBias[i][5]);

        Bgs[i] = Eigen::Vector3d(para_SpeedBias[i][6],
                                 para_SpeedBias[i][7],
                                 para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Eigen::Vector3d(para_Ex_Pose[i][0],
                                 para_Ex_Pose[i][1],
                                 para_Ex_Pose[i][2]);

        Eigen::Quaterniond q(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]);

        ric[i] = q.normalized().toRotationMatrix();
    }

    // ==========================
    // feature inverse depth -> depth
    // ==========================

    Eigen::VectorXd dep = f_manager.getDepthVector();

    for (int i = 0; i < dep.size(); i++)
    {
        dep(i) = para_Feature[i][0];
    }

    f_manager.setDepth(dep);
}


void Estimator::optimization()
{
    /*
     * 第一版：滑窗没满，不优化。
     *
     * 原版 VINS-Mono 也是在 solveOdometry() 里：
     * if (frame_count < WINDOW_SIZE)
     *     return;
     */

     if (frame_count < WINDOW_SIZE)
     {
         ROS_WARN("optimization skipped: frame_count = %d < WINDOW_SIZE", frame_count);
         return;
     }

      /*
     * 1. 用当前位姿和外参三角化特征深度初值
     *
     * 注意：
     * FeatureManager::triangulate() 只会给 estimated_depth <= 0 的特征三角化。
     */

    

      /*
     * 2. 把 Eigen 状态转成 Ceres double 参数块
     */

     vector2double();
    
     ceres::Problem problem;

     ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

     /*
     * 3. 添加滑窗内 pose 和 speedbias 参数块
     */

     for(int i = 0; i <= WINDOW_SIZE; i++)
     {
         ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();

         problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);

         problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
     }

     /*
     * 4. 第一版固定第一帧 pose
     *
     * 原版完整系统依靠初始化、边缘化先验和 yaw 对齐处理 gauge。
     * 我们第一版没有边缘化，所以先固定 para_Pose[0]。
     */
     problem.SetParameterBlockConstant(para_Pose[0]);

      /*
     * 5. 添加外参参数块，并固定外参
     *
     * 第一版目标是固定外参，不做 ESTIMATE_EXTRINSIC。
     */

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();

        problem.AddParameterBlock( para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
    
     /*
     * 6. 添加上一轮边缘化留下来的 prior。
     */
    if (last_marginalization_info)
    {
        MarginalizationFactor *marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);

        problem.AddResidualBlock(
            marginalization_factor,
            nullptr,
            last_marginalization_parameter_blocks);

        ROS_INFO("add last marginalization prior, residual size = %d, parameter block num = %lu",
                 last_marginalization_info->n,
                 last_marginalization_parameter_blocks.size());
    }

    /*
     * 7. 添加 IMUFactor
     *
     * pre_integrations[j] 表示第 j-1 帧到第 j 帧之间的 IMU 预积分。
     */

    int imu_factor_count = 0;

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;

        if (pre_integrations[j] == nullptr)
        {
            continue;
        }

        if (pre_integrations[j]->sum_dt > 10.0)
        {
            continue;
        }

        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);

        problem.AddResidualBlock( imu_factor, nullptr, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);

        imu_factor_count++;
    }

     /*
     * 8. 添加 ProjectionFactor
     *
     * 每个有效 feature：
     *   以第一次观测帧 imu_i 为 anchor
     *   与后续每个观测帧 imu_j 构造一个视觉重投影残差
     */
    int feature_count = f_manager.getFeatureCount();
    int visual_factor_count = 0;
    int feature_index = -1;

    for (auto &it_per_id : f_manager.feature)
  {
        it_per_id.used_num =
        static_cast<int>(it_per_id.feature_per_frame.size());

        if (!f_manager.isFeatureValidForOptimization(it_per_id))
        {
            continue;
        }

        feature_index++;

        if (feature_index >= feature_count)
        {
            ROS_WARN("optimization: feature_index %d >= feature_count %d",
                     feature_index,
                     feature_count);
            break;
        }

        if (!std::isfinite(para_Feature[feature_index][0]) ||
            para_Feature[feature_index][0] <= 1e-6)
        {
            para_Feature[feature_index][0] = 1.0 / INIT_DEPTH;
        }

        problem.AddParameterBlock(para_Feature[feature_index], SIZE_FEATURE);

        problem.SetParameterLowerBound(para_Feature[feature_index], 0, 1.0 / 100.0);

        problem.SetParameterUpperBound(para_Feature[feature_index], 0, 1.0 / 0.1);

        int imu_i = it_per_id.start_frame;
        int imu_j = imu_i - 1;

        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            if (imu_i == imu_j)
            {
                continue;
            }

            if (imu_i < 0 || imu_i > WINDOW_SIZE ||imu_j < 0 || imu_j > WINDOW_SIZE)
            {
                ROS_WARN("optimization: bad imu index imu_i=%d imu_j=%d",imu_i, imu_j);
                continue;
            }

            Eigen::Vector3d pts_j = it_per_frame.point;

            ProjectionFactor *projection_factor = new ProjectionFactor(pts_i, pts_j);

            problem.AddResidualBlock(
                projection_factor,
                loss_function,
                para_Pose[imu_i],
                para_Pose[imu_j],
                para_Ex_Pose[0],
                para_Feature[feature_index]);

            visual_factor_count++;
        }
   }

    ROS_INFO("optimization factors: imu = %d, visual = %d, feature = %d",
             imu_factor_count,
             visual_factor_count,
             feature_index + 1);

     /*
     * 如果没有视觉约束，暂时不优化。
     * 否则只有 IMU 传播，优化意义不大，还可能数值不稳定。
     */
    if (visual_factor_count == 0)
    {
        ROS_WARN("optimization skipped: no visual factor");
        return;
    }

    /*
     * 9. 设置 Ceres 求解器
     *
     * 原版使用 DENSE_SCHUR + DOGLEG。
     */
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = SOLVER_TIME;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    ROS_INFO("ceres brief report: %s",
             summary.BriefReport().c_str());

    /*
     * 10. 把 Ceres 优化后的 double 参数块写回 Estimator 状态
     */
    double2vector();

    std::unordered_map<long, double *> addr_shift;

    for (int i = 1; i <= WINDOW_SIZE; i++)
    {
        addr_shift[reinterpret_cast<long>(para_Pose[i])] =
            para_Pose[i - 1];

        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - 1];
    }

    addr_shift[reinterpret_cast<long>(para_Ex_Pose[0])] =
        para_Ex_Pose[0];

    for (int i = 0; i < NUM_OF_F; i++)
    {
        addr_shift[reinterpret_cast<long>(para_Feature[i])] =
            para_Feature[i];
    }
    
    if (marginalization_flag == MARGIN_OLD)
    {
        ROS_INFO("start marginalization: MARGIN_OLD");

        MarginalizationInfo *marginalization_info =
            new MarginalizationInfo();

        /*
        * 1. 先把上一轮留下的 prior 加入新的边缘化。
        *
        * 如果旧 prior 中包含 para_Pose[0] 或 para_SpeedBias[0]，
        * 那么这两个参数块要在本轮被 drop。
        */
        if (last_marginalization_info)
        {
            std::vector<int> drop_set;

            for (int i = 0;
                i < static_cast<int>(last_marginalization_parameter_blocks.size());
                i++)
            {
                if (last_marginalization_parameter_blocks[i] ==
                        para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] ==
                        para_SpeedBias[0])
                {
                    drop_set.push_back(i);
                }
            }

            MarginalizationFactor *marginalization_factor =
                new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info =
                new ResidualBlockInfo(
                    marginalization_factor,
                    nullptr,
                    last_marginalization_parameter_blocks,
                    drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        /*
        * 2. 加入第 0 帧到第 1 帧之间的 IMU 约束。
        *
        * 参数块顺序：
        *   0: pose0
        *   1: speedbias0
        *   2: pose1
        *   3: speedbias1
        *
        * 要 drop：
        *   pose0, speedbias0
        */
        if (pre_integrations[1] &&
            pre_integrations[1]->sum_dt < 10.0)
        {
            IMUFactor *imu_factor =
                new IMUFactor(pre_integrations[1]);

            ResidualBlockInfo *residual_block_info =
                new ResidualBlockInfo(
                    imu_factor,
                    nullptr,
                    std::vector<double *>{para_Pose[0],
                                        para_SpeedBias[0],
                                        para_Pose[1],
                                        para_SpeedBias[1]},
                    std::vector<int>{0, 1});

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        /*
        * 3. 加入和第 0 帧相关的视觉重投影因子。
        *
        * 对于 start_frame == 0 的特征：
        *   它的逆深度定义在第 0 帧。
        *   当第 0 帧被边缘化时，这个特征深度也要一起 drop。
        */
        int feature_index = -1;
        int margin_visual_factor_num = 0;

        int feature_count =
            f_manager.getFeatureCount();

        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num =
                static_cast<int>(it_per_id.feature_per_frame.size());

            if (!f_manager.isFeatureValidForOptimization(it_per_id))
            {
                continue;
            }

            feature_index++;

            if (feature_index >= feature_count)
            {
                ROS_WARN("marg old: feature_index %d >= feature_count %d",
                        feature_index, feature_count);
                break;
            }

            if (it_per_id.start_frame != 0)
            {
                continue;
            }

            if (!std::isfinite(para_Feature[feature_index][0]) ||
                para_Feature[feature_index][0] <= 1e-6)
            {
                continue;
            }

            int imu_i =
                it_per_id.start_frame;

            int imu_j =
                imu_i - 1;

            Eigen::Vector3d pts_i =
                it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;

                if (imu_i == imu_j)
                {
                    continue;
                }

                if (imu_j < 0 || imu_j > WINDOW_SIZE)
                {
                    continue;
                }

                Eigen::Vector3d pts_j =
                    it_per_frame.point;

                ProjectionFactor *projection_factor =
                    new ProjectionFactor(pts_i, pts_j);

                /*
                * 参数块顺序：
                *   0: pose_i，也就是 pose0
                *   1: pose_j
                *   2: ex_pose
                *   3: feature inverse depth
                *
                * drop_set:
                *   pose0
                *   feature inverse depth
                */
                ResidualBlockInfo *residual_block_info =
                    new ResidualBlockInfo(
                        projection_factor,
                        new ceres::HuberLoss(1.0),
                        std::vector<double *>{para_Pose[imu_i],
                                            para_Pose[imu_j],
                                            para_Ex_Pose[0],
                                            para_Feature[feature_index]},
                        std::vector<int>{0, 3});

                marginalization_info->addResidualBlockInfo(residual_block_info);

                margin_visual_factor_num++;
            }
        }

        ROS_INFO("MARGIN_OLD marginalization factors: visual = %d",
                margin_visual_factor_num);

        /*
        * 4. 执行边缘化。
        */
        marginalization_info->preMarginalize();
        marginalization_info->marginalize();

        std::vector<double *> parameter_blocks =
            marginalization_info->getParameterBlocks(addr_shift);

        /*
        * 5. 更新 last_marginalization_info。
        */
        if (last_marginalization_info)
        {
            delete last_marginalization_info;
            last_marginalization_info = nullptr;
        }

        last_marginalization_info =
            marginalization_info;

        last_marginalization_parameter_blocks =
            parameter_blocks;

        ROS_INFO("MARGIN_OLD marginalization done, keep block num = %lu",
                last_marginalization_parameter_blocks.size());
    }
    else
    {
        ROS_INFO("start marginalization: MARGIN_SECOND_NEW");

        if (last_marginalization_info)
        {
            MarginalizationInfo *marginalization_info =
                new MarginalizationInfo();

            std::vector<int> drop_set;

            for (int i = 0;
                i < static_cast<int>(last_marginalization_parameter_blocks.size());
                i++)
            {
                if (last_marginalization_parameter_blocks[i] ==
                    para_Pose[WINDOW_SIZE - 1])
                {
                    drop_set.push_back(i);
                }
            }

            /*
            * 如果旧 prior 里面不包含 para_Pose[WINDOW_SIZE - 1]，
            * 那么这次就没有东西需要边缘化。
            */
            if (!drop_set.empty())
            {
                MarginalizationFactor *marginalization_factor =
                    new MarginalizationFactor(last_marginalization_info);

                ResidualBlockInfo *residual_block_info =
                    new ResidualBlockInfo(
                        marginalization_factor,
                        nullptr,
                        last_marginalization_parameter_blocks,
                        drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);

                marginalization_info->preMarginalize();
                marginalization_info->marginalize();

                /*
                * MARGIN_SECOND_NEW 的地址映射不同：
                *
                * 0 ... WINDOW_SIZE - 2 保持不变
                * WINDOW_SIZE 复制到 WINDOW_SIZE - 1
                */
                std::unordered_map<long, double *> addr_shift_second_new;

                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    if (i == WINDOW_SIZE - 1)
                        continue;

                    if (i == WINDOW_SIZE)
                    {
                        addr_shift_second_new[
                            reinterpret_cast<long>(para_Pose[i])] =
                            para_Pose[i - 1];

                        addr_shift_second_new[
                            reinterpret_cast<long>(para_SpeedBias[i])] =
                            para_SpeedBias[i - 1];
                    }
                    else
                    {
                        addr_shift_second_new[
                            reinterpret_cast<long>(para_Pose[i])] =
                            para_Pose[i];

                        addr_shift_second_new[
                            reinterpret_cast<long>(para_SpeedBias[i])] =
                            para_SpeedBias[i];
                    }
                }

                addr_shift_second_new[
                    reinterpret_cast<long>(para_Ex_Pose[0])] =
                    para_Ex_Pose[0];

                for (int i = 0; i < NUM_OF_F; i++)
                {
                    addr_shift_second_new[
                        reinterpret_cast<long>(para_Feature[i])] =
                        para_Feature[i];
                }

                std::vector<double *> parameter_blocks =
                    marginalization_info->getParameterBlocks(
                        addr_shift_second_new);

                if (last_marginalization_info)
                {
                    delete last_marginalization_info;
                    last_marginalization_info = nullptr;
                }

                last_marginalization_info =
                    marginalization_info;

                last_marginalization_parameter_blocks =
                    parameter_blocks;

                ROS_INFO("MARGIN_SECOND_NEW marginalization done, keep block num = %lu",
                        last_marginalization_parameter_blocks.size());
            }
            else
            {
                ROS_INFO("MARGIN_SECOND_NEW: no matching block in prior, skip");

                delete marginalization_info;
                marginalization_info = nullptr;
            }
        }
        else
        {
            ROS_INFO("MARGIN_SECOND_NEW: no last marginalization info, skip");
        }
    }
    

    
    ROS_INFO("after optimization: P[%d] = %.3f %.3f %.3f, V = %.3f %.3f %.3f, Ba norm = %.4f, Bg norm = %.4f",
         WINDOW_SIZE,
         Ps[WINDOW_SIZE].x(),
         Ps[WINDOW_SIZE].y(),
         Ps[WINDOW_SIZE].z(),
         Vs[WINDOW_SIZE].x(),
         Vs[WINDOW_SIZE].y(),
         Vs[WINDOW_SIZE].z(),
         Bas[WINDOW_SIZE].norm(),
         Bgs[WINDOW_SIZE].norm());

}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
    {
        ROS_INFO("solveOdometry skipped: frame_count = %d < WINDOW_SIZE",
                 frame_count);
        return;
    }

    if (solver_flag != NON_LINEAR)
    {
        ROS_INFO("solveOdometry skipped: solver is not NON_LINEAR");
        return;
    }

    ROS_INFO("solveOdometry begins");

    f_manager.triangulate(Ps, tic, ric);

    ROS_INFO("triangulation finished, feature count: %d",
             f_manager.getFeatureCount());

    optimization();

    ROS_INFO("solveOdometry finished");
}

void Estimator::slideWindow()
{
    ROS_INFO("slideWindow begin, marginalization_flag = %d",
             marginalization_flag);

    f_manager.removeFailures();

    if (marginalization_flag == MARGIN_OLD)
    {
        /*
         * 删除最老帧：
         * [1 ... WINDOW_SIZE] 左移到 [0 ... WINDOW_SIZE - 1]
         */

        if (pre_integrations[0] != nullptr)
        {
            delete pre_integrations[0];
            pre_integrations[0] = nullptr;
        }

        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            Ps[i] = Ps[i + 1];
            Rs[i] = Rs[i + 1];
            Vs[i] = Vs[i + 1];
            Bas[i] = Bas[i + 1];
            Bgs[i] = Bgs[i + 1];

            Headers[i] = Headers[i + 1];

            dt_buf[i] = dt_buf[i + 1];
            linear_acceleration_buf[i] = linear_acceleration_buf[i + 1];
            angular_velocity_buf[i] = angular_velocity_buf[i + 1];

            pre_integrations[i] = pre_integrations[i + 1];
        }

        Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
        Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
        Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
        Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
        Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

        Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();

        pre_integrations[WINDOW_SIZE] = nullptr;

        pre_integrations[WINDOW_SIZE] =
            new IntegrationBase(acc_0,
                                gyr_0,
                                Bas[WINDOW_SIZE],
                                Bgs[WINDOW_SIZE]);

        f_manager.removeBack();

        /*
         * all_image_frame 同步删除最老图像帧。
         */
        while (all_image_frame.size() > WINDOW_SIZE + 1)
        {
            auto it = all_image_frame.begin();

            if (it->second.pre_integration != nullptr)
            {
                delete it->second.pre_integration;
                it->second.pre_integration = nullptr;
            }

            all_image_frame.erase(it);
        }
    }
    else
    {
        /*
         * MARGIN_SECOND_NEW：
         * 当前帧不是关键帧，删除倒数第二帧。
         *
         * 窗口中：
         * WINDOW_SIZE - 1 是倒数第二帧
         * WINDOW_SIZE 是最新帧
         *
         * 要把最新帧状态覆盖到 WINDOW_SIZE - 1，
         * 并把 newest 这段 IMU 预积分合并到 pre_integrations[WINDOW_SIZE - 1]。
         */

        if (pre_integrations[WINDOW_SIZE - 1] != nullptr &&
            pre_integrations[WINDOW_SIZE] != nullptr)
        {
            for (int i = 0;
                 i < static_cast<int>(dt_buf[WINDOW_SIZE].size());
                 i++)
            {
                pre_integrations[WINDOW_SIZE - 1]->push_back(
                    dt_buf[WINDOW_SIZE][i],
                    linear_acceleration_buf[WINDOW_SIZE][i],
                    angular_velocity_buf[WINDOW_SIZE][i]);

                dt_buf[WINDOW_SIZE - 1].push_back(dt_buf[WINDOW_SIZE][i]);
                linear_acceleration_buf[WINDOW_SIZE - 1].push_back(
                    linear_acceleration_buf[WINDOW_SIZE][i]);
                angular_velocity_buf[WINDOW_SIZE - 1].push_back(
                    angular_velocity_buf[WINDOW_SIZE][i]);
            }
        }

        Ps[WINDOW_SIZE - 1] = Ps[WINDOW_SIZE];
        Rs[WINDOW_SIZE - 1] = Rs[WINDOW_SIZE];
        Vs[WINDOW_SIZE - 1] = Vs[WINDOW_SIZE];
        Bas[WINDOW_SIZE - 1] = Bas[WINDOW_SIZE];
        Bgs[WINDOW_SIZE - 1] = Bgs[WINDOW_SIZE];

        Headers[WINDOW_SIZE - 1] = Headers[WINDOW_SIZE];

        if (pre_integrations[WINDOW_SIZE] != nullptr)
        {
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = nullptr;
        }

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();

        Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
        Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
        Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
        Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
        Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

        pre_integrations[WINDOW_SIZE] =
            new IntegrationBase(acc_0,
                                gyr_0,
                                Bas[WINDOW_SIZE],
                                Bgs[WINDOW_SIZE]);

        /*
         * 删除 FeatureManager 中倒数第二帧的观测。
         */
        f_manager.removeFront(frame_count);

        /*
         * 删除 all_image_frame 中倒数第二帧。
         *
         * Headers[WINDOW_SIZE - 1] 已经被最新帧覆盖，
         * 所以这里根据时间删除原来的倒数第二帧比较麻烦。
         * 第一版简单做法：
         * 只限制 all_image_frame 长度。
         */
        while (all_image_frame.size() > WINDOW_SIZE + 1)
        {
            auto it = all_image_frame.begin();

            if (it->second.pre_integration != nullptr)
            {
                delete it->second.pre_integration;
                it->second.pre_integration = nullptr;
            }

            all_image_frame.erase(it);
        }
    }

    ROS_INFO("slideWindow end");
}


bool Estimator::relativePose(Eigen::Matrix3d &relative_R,
                             Eigen::Vector3d &relative_T,
                             int &l)
{
    /*
     * 在滑窗中寻找一帧 i，使它和最新帧 WINDOW_SIZE：
     * 1. 有足够多共同特征
     * 2. 平均视差足够大
     * 3. 能估计出相对位姿
     *
     * 找到后：
     *   l = i
     *   relative_R / relative_T 作为 GlobalSFM 的初始两视图约束
     */

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres =
            f_manager.getCorresponding(i, WINDOW_SIZE);

        if (corres.size() <= 20)
        {
            continue;
        }

        double sum_parallax = 0.0;

        for (int j = 0; j < static_cast<int>(corres.size()); j++)
        {
            Eigen::Vector2d pts_0(
                corres[j].first.x(),
                corres[j].first.y());

            Eigen::Vector2d pts_1(
                corres[j].second.x(),
                corres[j].second.y());

            double parallax =
                (pts_0 - pts_1).norm();

            sum_parallax += parallax;
        }

        double average_parallax =
            sum_parallax / static_cast<double>(corres.size());

        /*
         * 原版判断：
         * average_parallax * 460 > 30
         *
         * 因为归一化平面坐标乘焦距后近似像素视差。
         */
        if (average_parallax * FOCAL_LENGTH > 30.0 &&
            m_estimator.solveRelativeRT(corres, relative_R, relative_T))
        {
            l = i;

            ROS_INFO("relativePose success: choose l = %d, corres = %lu, parallax = %.3f px",
                     l,
                     corres.size(),
                     average_parallax * FOCAL_LENGTH);

            return true;
        }
    }

    ROS_WARN("relativePose failed: not enough parallax or correspondence");

    return false;
}


bool Estimator::initialStructure()
{
    ROS_INFO("initialStructure begin");

    /*
     * 1. 检查 IMU excitation
     *
     * 原版会检查相邻图像帧之间 IMU 预积分平均加速度变化。
     * 如果运动太小，视觉-IMU初始化会退化。
     *
     * 第一版先保留这个检查和日志，但不直接 return false，
     * 避免一开始太严格导致永远初始化不了。
     */
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d sum_g;
        sum_g.setZero();

        for (frame_it = all_image_frame.begin();
             std::next(frame_it) != all_image_frame.end();
             frame_it++)
        {
            double dt = std::next(frame_it)->second.pre_integration ?
                        std::next(frame_it)->second.pre_integration->sum_dt :
                        0.0;

            if (dt > 0)
            {
                Eigen::Vector3d tmp_g =
                    std::next(frame_it)->second.pre_integration->delta_v / dt;

                sum_g += tmp_g;
            }
        }

        Eigen::Vector3d aver_g =
            sum_g / std::max(1, static_cast<int>(all_image_frame.size()) - 1);

        double var = 0.0;

        for (frame_it = all_image_frame.begin();
             std::next(frame_it) != all_image_frame.end();
             frame_it++)
        {
            double dt = std::next(frame_it)->second.pre_integration ?
                        std::next(frame_it)->second.pre_integration->sum_dt :
                        0.0;

            if (dt > 0)
            {
                Eigen::Vector3d tmp_g =
                    std::next(frame_it)->second.pre_integration->delta_v / dt;

                var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            }
        }

        var = std::sqrt(var / std::max(1, static_cast<int>(all_image_frame.size()) - 1));

        ROS_INFO("IMU excitation check: var = %.6f", var);

        if (var < 0.25)
        {
            ROS_WARN("IMU excitation not enough, but continue in first version");
        }
    }

    /*
     * 2. 寻找一对有足够视差的关键帧，计算相对位姿
     */
    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l = -1;

    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_WARN("initialStructure failed: relativePose failed");
        return false;
    }

    /*
     * 3. 构造 SfM 特征列表 sfm_f
     *
     * sfm_f 中每个元素对应一个 feature id，
     * observation 里存它在哪些帧被观测到，以及归一化平面坐标。
     */
    std::vector<SFMFeature> sfm_f;

    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;

        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d pts_3d = it_per_frame.point;

            Eigen::Vector2d pts_2d;
            pts_2d << pts_3d.x(), pts_3d.y();

            tmp_feature.observation.emplace_back(imu_j, pts_2d);
        }

        sfm_f.push_back(tmp_feature);
    }

    ROS_INFO("initialStructure: sfm feature size = %lu", sfm_f.size());

    /*
     * 4. 调用 GlobalSFM
     *
     * Q / T 是滑窗内视觉 SfM 位姿。
     * 注意：这里是视觉尺度，不是最终 IMU 尺度。
     */
    Eigen::Quaterniond Q[WINDOW_SIZE + 1];
    Eigen::Vector3d T[WINDOW_SIZE + 1];

    std::map<int, Eigen::Vector3d> sfm_tracked_points;

    GlobalSFM sfm;

    bool sfm_result =
        sfm.construct(WINDOW_SIZE + 1,
                      Q,
                      T,
                      l,
                      relative_R,
                      relative_T,
                      sfm_f,
                      sfm_tracked_points);

    if (!sfm_result)
    {
        ROS_WARN("initialStructure failed: GlobalSFM construct failed");
        return false;
    }

    ROS_INFO("GlobalSFM success, tracked point size = %lu",
             sfm_tracked_points.size());

    /*
     * 5. 把滑窗关键帧的 SfM 位姿写入 all_image_frame
     *
     * 原版中：
     * imageframe.R = Q[i].toRotationMatrix() * RIC[0].transpose()
     * imageframe.T = T[i]
     *
     * 含义：
     * Q/T 是 camera 位姿，转成 IMU/body 位姿时要考虑相机-IMU外参。
     */
    int i = 0;
    for (auto frame_it = all_image_frame.begin();
         frame_it != all_image_frame.end() && i <= WINDOW_SIZE;
         frame_it++)
    {
        double frame_time = frame_it->first;

        if (std::abs(frame_time - Headers[i].stamp.toSec()) < 1e-6)
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * ric[0].transpose();
            frame_it->second.T = T[i];

            i++;
        }
    }

    /*
     * 6. 给 all_image_frame 里非关键帧用 PnP 补齐位姿
     *
     * 这一步对应原版：一些 image frame 存在 all_image_frame 里，
     * 但不一定是滑窗关键帧 Headers[i]，需要用已有 SfM 3D 点和该帧2D观测做 PnP。
     *
     * 你当前第一版中 all_image_frame 基本就是滑窗帧，
     * 但保留这段逻辑，后面接近原版时会用到。
     */
    i = 0;

    for (auto frame_it = all_image_frame.begin();
         frame_it != all_image_frame.end();
         frame_it++)
    {
        if (i <= WINDOW_SIZE &&
            std::abs(frame_it->first - Headers[i].stamp.toSec()) < 1e-6)
        {
            frame_it->second.is_key_frame = true;
            i++;
            continue;
        }

        if (i > WINDOW_SIZE)
        {
            break;
        }

        Eigen::Matrix3d R_initial =
            (Q[i].inverse()).toRotationMatrix();

        Eigen::Vector3d P_initial =
            -R_initial * T[i];

        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;

        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;

            auto it = sfm_tracked_points.find(feature_id);

            if (it == sfm_tracked_points.end())
            {
                continue;
            }

            Eigen::Vector3d world_pts = it->second;

            pts_3_vector.emplace_back(world_pts.x(),
                                      world_pts.y(),
                                      world_pts.z());

            Eigen::Vector3d img_pts =
                id_pts.second[0].second.head<3>();

            pts_2_vector.emplace_back(img_pts.x(), img_pts.y());
        }

        if (pts_3_vector.size() < 6)
        {
            ROS_WARN("PnP failed: not enough points, size = %lu",
                     pts_3_vector.size());
            return false;
        }

        cv::Mat rvec;
        cv::Mat t;
        cv::Mat R_cv;

        cv::Mat K =
            (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                       0, 1, 0,
                                       0, 0, 1);

        cv::Mat D;

        cv::Mat tmp_R;
        cv::eigen2cv(R_initial, tmp_R);
        cv::Rodrigues(tmp_R, rvec);
        cv::eigen2cv(P_initial, t);

        bool pnp_succ =
            cv::solvePnP(pts_3_vector,
                         pts_2_vector,
                         K,
                         D,
                         rvec,
                         t,
                         true);

        if (!pnp_succ)
        {
            ROS_WARN("initialStructure: solvePnP failed for non-keyframe");
            return false;
        }

        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R_pnp;
        Eigen::Vector3d T_pnp;

        cv::cv2eigen(R_cv, R_pnp);
        cv::cv2eigen(t, T_pnp);

        /*
         * R_pnp / T_pnp 是 world -> camera。
         * 转成 body/IMU 位姿。
         */
        frame_it->second.R =
            R_pnp.transpose() * ric[0].transpose();

        frame_it->second.T =
            -frame_it->second.R * T_pnp;
    }

    ROS_INFO("initialStructure visual SfM part success");

  
    return visualInitialAlign();
}


bool Estimator::visualInitialAlign()
{
    ROS_INFO("visualInitialAlign begin");

    /*
     * x 的结构来自 VisualIMUAlignment：
     *
     * x = [v_0, v_1, ..., v_n, g, s]
     *
     * 其中：
     * 每个 v_i 是 3维
     * g 是 3维
     * s 是 1维
     */
    Eigen::VectorXd x;

    bool result =
        VisualIMUAlignment(all_image_frame, Bgs, g, x);

    if (!result)
    {
        ROS_WARN("visualInitialAlign failed: VisualIMUAlignment failed");
        return false;
    }

    int all_frame_count =
        static_cast<int>(all_image_frame.size());

    if (all_frame_count < WINDOW_SIZE + 1)
    {
        ROS_WARN("visualInitialAlign warning: all_image_frame size = %d",
                 all_frame_count);
    }

    int n_state =
        all_frame_count * 3 + 3 + 1;

    double s =
        x(n_state - 1);

    ROS_INFO("visualInitialAlign: scale = %.6f", s);

    if (!std::isfinite(s) || s <= 0.0)
    {
        ROS_WARN("visualInitialAlign failed: bad scale %.6f", s);
        return false;
    }

    /*
     * 1. 把 all_image_frame 中对应 Headers 的视觉位姿写回滑窗状态。
     *
     * initialStructure() 中已经给 all_image_frame 写入：
     *     R: body/IMU rotation
     *     T: visual translation
     */
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        double t = Headers[i].stamp.toSec();

        auto frame_it =
            all_image_frame.find(t);

        if (frame_it == all_image_frame.end())
        {
            ROS_WARN("visualInitialAlign failed: cannot find image frame %.9f", t);
            return false;
        }

        Rs[i] = frame_it->second.R;
        Ps[i] = frame_it->second.T;
    }

    /*
     * 2. 先清掉已有深度。
     *
     * 原版这里会把深度重置为 -1，
     * 然后用初始化后的 Ps/Rs 重新三角化。
     */
    Eigen::VectorXd dep =
        f_manager.getDepthVector();

    for (int i = 0; i < dep.size(); i++)
    {
        dep(i) = -1.0;
    }

    f_manager.clearDepth(dep);

    /*
     * 3. 用新的 Bgs 重新传播预积分。
     *
     * solveGyroscopeBias() 已经更新了 Bgs，
     * 这里保证滑窗 pre_integrations 也使用新 gyro bias。
     */
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        if (pre_integrations[i] != nullptr)
        {
            pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
        }
    }

    /*
     * 4. 用尺度 s 缩放视觉平移，并把第0帧移到原点。
     *
     * 注意：
     * Ps[i] 当前是视觉尺度下的 body 位姿平移。
     * VINS 原版会减去第一帧 body 位置，使窗口起点为世界原点。
     */
    Eigen::Vector3d first_body_pos =
        s * Ps[0] - Rs[0] * tic[0];

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i] =
            s * Ps[i] - Rs[i] * tic[0] - first_body_pos;
    }

    /*
     * 5. 从 x 中取出速度。
     *
     * x 中速度对应 all_image_frame 的顺序。
     * 我们当前 all_image_frame 只保留滑窗内 WINDOW_SIZE + 1 帧，
     * 所以这里直接按 Headers 顺序写回。
     */
    int frame_index = 0;

    for (auto frame_it = all_image_frame.begin();
         frame_it != all_image_frame.end() && frame_index <= WINDOW_SIZE;
         frame_it++, frame_index++)
    {
        Vs[frame_index] =
            frame_it->second.R *
            x.segment<3>(frame_index * 3);
    }

    /*
     * 6. 重力对齐。
     *
     * VisualIMUAlignment 求出的 g 在视觉初始化坐标系里。
     * 这里构造 rot_diff，把它旋转到世界系，使重力方向稳定。
     */
    Eigen::Matrix3d R0 =
        Utility::g2R(g);

    double yaw =
        Utility::R2ypr(R0 * Rs[0]).x();

    R0 =
        Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;

    g =
        R0 * g;

    Eigen::Matrix3d rot_diff =
        R0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i] =
            rot_diff * Ps[i];

        Rs[i] =
            rot_diff * Rs[i];

        Vs[i] =
            rot_diff * Vs[i];
    }

    /*
     * 7. 使用初始化后的位姿重新三角化特征深度。
     */
    f_manager.triangulate(Ps, tic, ric);

    initial_timestamp =
        Headers[0].stamp.toSec();

    ROS_INFO("visualInitialAlign success");
    ROS_INFO("initial gravity: %.6f %.6f %.6f, norm %.6f",
             g.x(), g.y(), g.z(), g.norm());

    return true;
}


bool Estimator::failureDetection()
{
    /*
     * 这个函数对应原版 VINS-Mono 的 failureDetection()。
     *
     * 作用：
     *   判断当前优化结果是否明显发散。
     *
     * 当前第一版做基础检查：
     *   1. 跟踪点太少
     *   2. 加速度 bias 太大
     *   3. 陀螺 bias 太大
     *   4. 位置突变太大
     *   5. z 方向突变太大
     *   6. 姿态突变太大
     */

    ROS_INFO("failureDetection check: track = %d, P = %.3f %.3f %.3f, V = %.3f %.3f %.3f",
         f_manager.last_track_num,
         Ps[WINDOW_SIZE].x(),
         Ps[WINDOW_SIZE].y(),
         Ps[WINDOW_SIZE].z(),
         Vs[WINDOW_SIZE].x(),
         Vs[WINDOW_SIZE].y(),
         Vs[WINDOW_SIZE].z());
        
    if (!failure_check_initialized)
    {
    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    failure_check_initialized = true;
    return false;
    }
    
    if (f_manager.last_track_num < 2)
    {
        ROS_WARN("failureDetection: too few tracked features: %d",
                 f_manager.last_track_num);
        return true;
    }

    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_WARN("failureDetection: acc bias too large: %.6f",
                 Bas[WINDOW_SIZE].norm());
        return true;
    }

    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_WARN("failureDetection: gyro bias too large: %.6f",
                 Bgs[WINDOW_SIZE].norm());
        return true;
    }

    Eigen::Vector3d tmp_P = Ps[WINDOW_SIZE];

    if ((tmp_P - last_P).norm() > 5.0)
    {
        ROS_WARN("failureDetection: position jump too large: %.6f",
                 (tmp_P - last_P).norm());
        return true;
    }

    if (std::abs(tmp_P.z() - last_P.z()) > 1.0)
    {
        ROS_WARN("failureDetection: z jump too large: %.6f",
                 std::abs(tmp_P.z() - last_P.z()));
        return true;
    }

    Eigen::Matrix3d tmp_R = Rs[WINDOW_SIZE];

    Eigen::Matrix3d delta_R =
        last_R.transpose() * tmp_R;

    Eigen::Quaterniond delta_Q(delta_R);

    double delta_angle =
        2.0 * std::acos(std::min(1.0, std::abs(delta_Q.w()))) * 180.0 / M_PI;

    if (delta_angle > 50.0)
    {
        ROS_WARN("failureDetection: rotation jump too large: %.6f deg",
                 delta_angle);
        return true;
    }

    return false;
}