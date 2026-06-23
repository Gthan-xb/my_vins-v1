#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include<Eigen/Dense>

using namespace Eigen;

class Utility
{
public:
  // 反对称矩阵
  static Matrix3d skewSymmetric(const Vector3d& v)
   {
    Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
        -v(1), v(0), 0;
    return m;
   }
  // 小角度 -> 四元数
  static Quaterniond deltaQ(const Vector3d& theta)
    {
        Quaterniond dq;
        Vector3d half_theta = theta * 0.5;
        dq.w() = 1.0;
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }
  // 保证四元数w>0
  static Quaterniond positify(const Quaterniond& q)
    {
        if (q.w() < 0)
            return Quaterniond(-q.w(), -q.x(), -q.y(), -q.z());
        else
            return q;
    }
  // R -> ypr
  static Vector3d R2ypr(const Matrix3d& R)
  {
    Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;
        return ypr / M_PI * 180.0;
  }
  // ypr -> R
  static Matrix3d ypr2R(const Vector3d& ypr)
  {
    double y = ypr(0) / 180.0 * M_PI;
        double p = ypr(1) / 180.0 * M_PI;
        double r = ypr(2) / 180.0 * M_PI;

        Matrix3d Rz;
        Rz << cos(y), -sin(y), 0,
              sin(y), cos(y), 0,
              0, 0, 1;

        Matrix3d Ry;
        Ry << cos(p), 0., sin(p),
              0., 1., 0.,
             -sin(p), 0., cos(p);

        Matrix3d Rx;
        Rx << 1., 0., 0.,
              0., cos(r), -sin(r),
              0., sin(r), cos(r);

        return Rz * Ry * Rx;
  }

  static Eigen::Matrix4d Qleft(const Eigen::Quaterniond &q)
  {
    Eigen::Matrix4d ans;
    ans.setZero();

    ans(0, 0) = q.w();
    ans.block<1, 3>(0, 1) = -q.vec().transpose();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<3, 3>(1, 1) =
        q.w() * Eigen::Matrix3d::Identity() + skewSymmetric(q.vec());

    return ans;
  }

  static Eigen::Matrix4d Qright(const Eigen::Quaterniond &q)
  {
    Eigen::Matrix4d ans;
    ans.setZero();

    ans(0, 0) = q.w();
    ans.block<1, 3>(0, 1) = -q.vec().transpose();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<3, 3>(1, 1) =
        q.w() * Eigen::Matrix3d::Identity() - skewSymmetric(q.vec());

    return ans;
  }

  static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
 {
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2;
    ng2 << 0.0, 0.0, 1.0;

    Eigen::Matrix3d R0 =
        Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();

    double yaw = R2ypr(R0).x();

    R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;

    return R0;
 } 
};

 