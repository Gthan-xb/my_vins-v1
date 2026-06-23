#pragma once

#include <vector>
#include <unordered_map>
#include <numeric>
#include <cstring>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "my_vins_estimator/utility.h"

class MarginalizationInfo ;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      const std::vector<double *> &_parameter_blocks,
                      const std::vector<int> &_drop_set)
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(_parameter_blocks),
          drop_set(_drop_set),
          raw_jacobians(nullptr)
    {
    }

    void Evaluate();

    int localSize(int size) const
    {
        return size == 7 ? 6 : size;
    }

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;

    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;

    std::vector<Eigen::Matrix<double,
                              Eigen::Dynamic,
                              Eigen::Dynamic,
                              Eigen::RowMajor>>
        jacobians;

    Eigen::VectorXd residuals;
};

class MarginalizationInfo
{
public:
    MarginalizationInfo()
    {
        m = 0;
        n = 0;
        sum_block_size = 0;
    }

    ~MarginalizationInfo();

    int localSize(int size) const
    {
        return size == 7 ? 6 : size;
    }

    int globalSize(int size) const
    {
        return size == 6 ? 7 : size;
    }

    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);

    void preMarginalize();

    void marginalize();

    std::vector<double *> getParameterBlocks(
        std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;

    /*
     * m: 要被边缘化掉的变量维度
     * n: 保留下来的变量维度
     */
    int m;
    int n;

    /*
     * parameter_block_size:
     *     参数块地址 -> global size
     *
     * parameter_block_idx:
     *     参数块地址 -> local size 下的起始索引
     */
    std::unordered_map<long, int> parameter_block_size;
    std::unordered_map<long, int> parameter_block_idx;

    /*
     * 保存线性化点处的参数值。
     */
    std::unordered_map<long, double *> parameter_block_data;

    int sum_block_size;

    std::vector<int> keep_block_size;
    std::vector<int> keep_block_idx;
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;

    const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction
{
public:
    explicit MarginalizationFactor(MarginalizationInfo *_marginalization_info);

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const;

    MarginalizationInfo *marginalization_info;
};






