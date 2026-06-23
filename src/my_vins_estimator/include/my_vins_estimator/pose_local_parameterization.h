#pragma once    
#include<Eigen/Dense>
#include<ceres/ceres.h>
#include "my_vins_estimator/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{public:
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
    
    virtual bool ComputeJacobian(const double *x, double *jacobian) const override;

    virtual int GlobalSize() const override { return 7; }

    virtual int LocalSize() const override { return 6; }
};