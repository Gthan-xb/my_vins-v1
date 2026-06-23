#include"my_vins_estimator/projection_factor.h"
#include<iostream>
#include<cmath>
Eigen::Matrix2d ProjectionFactor::sqrt_info;
double ProjectionFactor::sum_t = 0.0;

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i,const Eigen::Vector3d &_pts_j)
    : pts_i(_pts_i), pts_j(_pts_j)
{
}

bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // ==============================
    // 1. 读取第 i 帧状态
    // ==============================

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
     // ==============================
    // 2. 读取第 j 帧状态
    // ==============================
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Qi.normalize();
    Qj.normalize();

    // ==============================
    // 3. 读取相机-IMU 外参
    // ==============================
    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    qic.normalize();

    // ==============================
    // 4. 读取特征逆深度
    // ==============================
    double inv_dep_i = parameters[3][0];
    if (!std::isfinite(inv_dep_i) || inv_dep_i <= 1e-8)
   {
    return false;
   }

    // ==============================
    // 5. 几何投影链路
    // ==============================
    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    // ==============================
    // 6. 计算 2 维重投影残差
    // ==============================
    Eigen::Map<Eigen::Vector2d> residual(residuals);

    double dep_j = pts_camera_j.z();
    if (!std::isfinite(dep_j) || std::abs(dep_j) <= 1e-8)
   {
    return false;
   }

    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();

    residual = sqrt_info * residual;

     // ==============================
    // 7. 计算解析 Jacobian
    // ==============================
    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();

        Eigen::Matrix<double, 2, 3> reduce;

        reduce << 1.0 / dep_j, 0.0, -pts_camera_j.x() / (dep_j * dep_j),
                  0.0, 1.0 / dep_j, -pts_camera_j.y() / (dep_j * dep_j);

        reduce =
            sqrt_info * reduce;

        
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
                jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.setZero();

            jaco_i.leftCols<3>() =
                ric.transpose() * Rj.transpose();

            jaco_i.rightCols<3>() =
                ric.transpose() *
                Rj.transpose() *
                Ri *
                -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() =
                reduce * jaco_i;

            jacobian_pose_i.rightCols<1>().setZero();
        }

        
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
                jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.setZero();

            jaco_j.leftCols<3>() =
                ric.transpose() * -Rj.transpose();

            jaco_j.rightCols<3>() =
                ric.transpose() *
                Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() =
                reduce * jaco_j;

            jacobian_pose_j.rightCols<1>().setZero();
        }

        
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
                jacobian_ex_pose(jacobians[2]);

            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.setZero();

            jaco_ex.leftCols<3>() =
                ric.transpose() *
                (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());

            Eigen::Matrix3d tmp_r =
                ric.transpose() * Rj.transpose() * Ri * ric;

            jaco_ex.rightCols<3>() =
                -tmp_r * Utility::skewSymmetric(pts_camera_i)
                + Utility::skewSymmetric(tmp_r * pts_camera_i)
                + Utility::skewSymmetric(
                      ric.transpose() *
                      (Rj.transpose() *
                           (Ri * tic + Pi - Pj)
                       - tic));

            jacobian_ex_pose.leftCols<6>() =
                reduce * jaco_ex;

            jacobian_ex_pose.rightCols<1>().setZero();
        }

        
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d>
                jacobian_feature(jacobians[3]);

            jacobian_feature =
                reduce *
                ric.transpose() *
                Rj.transpose() *
                Ri *
                ric *
                pts_i *
                (-1.0 / (inv_dep_i * inv_dep_i));
        }
    }

    return true;
}

void ProjectionFactor::check(double **parameters)
{
    double residuals[2];
    double *jacobians[4];
    double jacobian_pose_i[2 * 7];
    double jacobian_pose_j[2 * 7];
    double jacobian_ex_pose[2 * 7];
    double jacobian_feature[2];

    jacobians[0] = jacobian_pose_i;
    jacobians[1] = jacobian_pose_j;
    jacobians[2] = jacobian_ex_pose;
    jacobians[3] = jacobian_feature;

    Evaluate(parameters, residuals, jacobians);

    std::cout<<"ProjectionFactor residual:"<< Eigen::Map<Eigen::Vector2d>(residuals).transpose() <<std::endl;
}


