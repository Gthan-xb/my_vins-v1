#pragma once
#include<Eigen/Dense>
#include<ceres/ceres.h>
#include<ros/assert.h>
#include<ros/console.h>
#include "my_vins_estimator/parameters.h"
#include "my_vins_estimator/utility.h"

class ProjectionFactor : public ceres::SizedCostFunction<2,7,7,7,1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionFactor(const Eigen::Vector3d &_pts_i,const Eigen::Vector3d &_pts_j);

    virtual bool Evaluate(double const *const *parameters,double *residuals,double **jacobians) const override;
 
    void check(double **parameters);

public:
    Eigen::Vector3d pts_i;
    Eigen::Vector3d pts_j;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};