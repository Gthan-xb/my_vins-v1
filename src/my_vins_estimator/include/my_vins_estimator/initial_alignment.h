#pragma once

#include <map>
#include <vector>

#include <Eigen/Dense>

#include "my_vins_estimator/integration_base.h"

class ImageFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImageFrame()
    {
        t = 0.0;
        R.setIdentity();
        T.setZero();
        pre_integration = nullptr;
        is_key_frame = false;
    }

    ImageFrame(
        const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &_points,
        double _t)
        : points(_points),
          t(_t),
          pre_integration(nullptr),
          is_key_frame(false)
    {
        R.setIdentity();
        T.setZero();
    }

    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;

    double t;

    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    IntegrationBase *pre_integration;

    bool is_key_frame;

};

void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame,
                        Eigen::Vector3d *Bgs);

Eigen::MatrixXd TangentBasis(Eigen::Vector3d &g0);

bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame,
                     Eigen::Vector3d &g,
                     Eigen::VectorXd &x);

void RefineGravity(std::map<double, ImageFrame> &all_image_frame,
                   Eigen::Vector3d &g,
                   Eigen::VectorXd &x);

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame,
                        Eigen::Vector3d *Bgs,
                        Eigen::Vector3d &g,
                        Eigen::VectorXd &x);