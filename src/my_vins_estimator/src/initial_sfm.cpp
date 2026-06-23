#include "my_vins_estimator/initial_sfm.h"

#include <cassert>

#include <ros/ros.h>

using namespace Eigen;
using namespace std;

GlobalSFM::GlobalSFM()
{
    feature_num = 0;
}


void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                                 Eigen::Matrix<double, 3, 4> &Pose1,
                                 Eigen::Vector2d &point0,
                                 Eigen::Vector2d &point1,
                                 Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix;
    design_matrix.setZero();

    design_matrix.row(0) =
        point0[0] * Pose0.row(2) - Pose0.row(0);

    design_matrix.row(1) =
        point0[1] * Pose0.row(2) - Pose0.row(1);

    design_matrix.row(2) =
        point1[0] * Pose1.row(2) - Pose1.row(0);

    design_matrix.row(3) =
        point1[1] * Pose1.row(2) - Pose1.row(1);

    Eigen::Vector4d triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV)
            .matrixV()
            .rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial,
                                Eigen::Vector3d &P_initial,
                                int i,
                                std::vector<SFMFeature> &sfm_f)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;

    for (int j = 0; j < feature_num; j++)
    {
        if (!sfm_f[j].state)
        {
            continue;
        }

        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++)
        {
            if (sfm_f[j].observation[k].first == i)
            {
                Eigen::Vector2d img_pts =
                    sfm_f[j].observation[k].second;

                pts_2_vector.emplace_back(img_pts.x(), img_pts.y());

                pts_3_vector.emplace_back(sfm_f[j].position[0],
                                          sfm_f[j].position[1],
                                          sfm_f[j].position[2]);

                break;
            }
        }
    }

    if (static_cast<int>(pts_2_vector.size()) < 15)
    {
        ROS_WARN("solveFrameByPnP unstable, pts size: %lu",
                 pts_2_vector.size());

        if (static_cast<int>(pts_2_vector.size()) < 10)
        {
            return false;
        }
    }

    cv::Mat rvec;
    cv::Mat t;
    cv::Mat r;
    cv::Mat tmp_r;

    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);

    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                   0, 1, 0,
                                   0, 0, 1);

    cv::Mat D;

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
        ROS_WARN("solvePnP failed at frame %d", i);
        return false;
    }

    cv::Rodrigues(rvec, r);

    Eigen::Matrix3d R_pnp;
    Eigen::Vector3d T_pnp;

    cv::cv2eigen(r, R_pnp);
    cv::cv2eigen(t, T_pnp);

    R_initial = R_pnp;
    P_initial = T_pnp;

    return true;
}


void GlobalSFM::triangulateTwoFrames(int frame0,
                                     Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1,
                                     Eigen::Matrix<double, 3, 4> &Pose1,
                                     std::vector<SFMFeature> &sfm_f)
{
    assert(frame0 != frame1);

    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state)
        {
            continue;
        }

        bool has_0 = false;
        bool has_1 = false;

        Eigen::Vector2d point0;
        Eigen::Vector2d point1;

        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++)
        {
            if (sfm_f[j].observation[k].first == frame0)
            {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }

            if (sfm_f[j].observation[k].first == frame1)
            {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }

        if (has_0 && has_1)
        {
            Eigen::Vector3d point_3d;

            triangulatePoint(Pose0,
                             Pose1,
                             point0,
                             point1,
                             point_3d);

            sfm_f[j].state = true;

            sfm_f[j].position[0] = point_3d.x();
            sfm_f[j].position[1] = point_3d.y();
            sfm_f[j].position[2] = point_3d.z();
        }
    }
}


bool GlobalSFM::construct(int frame_num,
                          Eigen::Quaterniond *q,
                          Eigen::Vector3d *T,
                          int l,
                          const Eigen::Matrix3d relative_R,
                          const Eigen::Vector3d relative_T,
                          std::vector<SFMFeature> &sfm_f,
                          std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
    feature_num = static_cast<int>(sfm_f.size());

    if (feature_num == 0)
    {
        ROS_WARN("GlobalSFM construct failed: no feature");
        return false;
    }

    /*
     * 原版约定：
     * q[i], T[i] 是 cam 到 world 的位姿表达。
     *
     * 在 SFM 内部，为了 PnP 和 BA，会转换成 world 到 camera：
     * c_Rotation = q.inverse()
     * c_Translation = -c_Rotation * T
     */

    q[l] = Eigen::Quaterniond::Identity();
    T[l].setZero();

    q[frame_num - 1] =
        q[l] * Eigen::Quaterniond(relative_R);

    T[frame_num - 1] =
        relative_T;

    std::vector<Eigen::Matrix3d> c_Rotation(frame_num);
    std::vector<Eigen::Vector3d> c_Translation(frame_num);
    std::vector<Eigen::Quaterniond> c_Quat(frame_num);
    std::vector<Eigen::Matrix<double, 3, 4>> Pose(frame_num);

    std::vector<std::array<double, 4>> c_rotation(frame_num);
    std::vector<std::array<double, 3>> c_translation(frame_num);

    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1.0 * (c_Rotation[l] * T[l]);

    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[frame_num - 1] =
        q[frame_num - 1].inverse();

    c_Rotation[frame_num - 1] =
        c_Quat[frame_num - 1].toRotationMatrix();

    c_Translation[frame_num - 1] =
        -1.0 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);

    Pose[frame_num - 1].block<3, 3>(0, 0) =
        c_Rotation[frame_num - 1];

    Pose[frame_num - 1].block<3, 1>(0, 3) =
        c_Translation[frame_num - 1];

    /*
     * 1. 从 l 到最后一帧向前推：
     *    PnP 求中间帧位姿
     *    和最后一帧三角化特征点
     */
    for (int i = l; i < frame_num - 1; i++)
    {
        if (i > l)
        {
            Eigen::Matrix3d R_initial =
                c_Rotation[i - 1];

            Eigen::Vector3d P_initial =
                c_Translation[i - 1];

            if (!solveFrameByPnP(R_initial,
                                 P_initial,
                                 i,
                                 sfm_f))
            {
                ROS_WARN("GlobalSFM solveFrameByPnP failed, frame %d", i);
                return false;
            }

            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = Eigen::Quaterniond(c_Rotation[i]);

            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        triangulateTwoFrames(i,
                             Pose[i],
                             frame_num - 1,
                             Pose[frame_num - 1],
                             sfm_f);
    }

    /*
     * 2. 从 l 和中间帧继续三角化
     */
    for (int i = l + 1; i < frame_num - 1; i++)
    {
        triangulateTwoFrames(l,
                             Pose[l],
                             i,
                             Pose[i],
                             sfm_f);
    }

    /*
     * 3. 从 l 往前推：
     *    PnP 求前面帧位姿
     *    与 l 帧三角化特征
     */
    for (int i = l - 1; i >= 0; i--)
    {
        Eigen::Matrix3d R_initial =
            c_Rotation[i + 1];

        Eigen::Vector3d P_initial =
            c_Translation[i + 1];

        if (!solveFrameByPnP(R_initial,
                             P_initial,
                             i,
                             sfm_f))
        {
            ROS_WARN("GlobalSFM solveFrameByPnP failed, frame %d", i);
            return false;
        }

        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = Eigen::Quaterniond(c_Rotation[i]);

        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];

        triangulateTwoFrames(i,
                             Pose[i],
                             l,
                             Pose[l],
                             sfm_f);
    }

    /*
     * 4. 三角化剩余还没有求出的特征
     */
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state)
        {
            continue;
        }

        if (static_cast<int>(sfm_f[j].observation.size()) >= 2)
        {
            Eigen::Vector2d point0;
            Eigen::Vector2d point1;

            int frame_0 =
                sfm_f[j].observation.front().first;

            point0 =
                sfm_f[j].observation.front().second;

            int frame_1 =
                sfm_f[j].observation.back().first;

            point1 =
                sfm_f[j].observation.back().second;

            Eigen::Vector3d point_3d;

            triangulatePoint(Pose[frame_0],
                             Pose[frame_1],
                             point0,
                             point1,
                             point_3d);

            sfm_f[j].state = true;

            sfm_f[j].position[0] = point_3d.x();
            sfm_f[j].position[1] = point_3d.y();
            sfm_f[j].position[2] = point_3d.z();
        }
    }

    /*
     * 5. bundle adjustment
     */
    ceres::Problem problem;

    ceres::LocalParameterization *local_parameterization =
        new ceres::QuaternionParameterization();

    for (int i = 0; i < frame_num; i++)
    {
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();

        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();

        problem.AddParameterBlock(c_rotation[i].data(),
                                  4,
                                  local_parameterization);

        problem.AddParameterBlock(c_translation[i].data(),
                                  3);

        if (i == l)
        {
            problem.SetParameterBlockConstant(c_rotation[i].data());
        }

        if (i == l || i == frame_num - 1)
        {
            problem.SetParameterBlockConstant(c_translation[i].data());
        }
    }

    for (int i = 0; i < feature_num; i++)
    {
        if (!sfm_f[i].state)
        {
            continue;
        }

        for (int j = 0; j < static_cast<int>(sfm_f[i].observation.size()); j++)
        {
            int frame_id =
                sfm_f[i].observation[j].first;

            ceres::CostFunction *cost_function =
                ReprojectionError3D::Create(
                    sfm_f[i].observation[j].second.x(),
                    sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     c_rotation[frame_id].data(),
                                     c_translation[frame_id].data(),
                                     sfm_f[i].position);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ROS_INFO("GlobalSFM BA: %s", summary.BriefReport().c_str());

    /*
     * 6. BA 后写回 q / T
     *
     * c_Rotation / c_Translation 是 world 到 camera。
     * 所以：
     * q = c_Rotation^T
     * T = -q * c_Translation
     */
    for (int i = 0; i < frame_num; i++)
    {
        Eigen::Quaterniond q_tmp(
            c_rotation[i][0],
            c_rotation[i][1],
            c_rotation[i][2],
            c_rotation[i][3]);

        q_tmp.normalize();

        Eigen::Matrix3d R_tmp =
            q_tmp.toRotationMatrix();

        q[i] =
            Eigen::Quaterniond(R_tmp.transpose());

        Eigen::Vector3d T_tmp(
            c_translation[i][0],
            c_translation[i][1],
            c_translation[i][2]);

        T[i] =
            -1.0 * (q[i] * T_tmp);
    }

    /*
     * 7. 输出 SfM 3D 点
     */
    for (int i = 0; i < feature_num; i++)
    {
        if (sfm_f[i].state)
        {
            sfm_tracked_points[sfm_f[i].id] =
                Eigen::Vector3d(sfm_f[i].position[0],
                                sfm_f[i].position[1],
                                sfm_f[i].position[2]);
        }
    }

    return true;
}