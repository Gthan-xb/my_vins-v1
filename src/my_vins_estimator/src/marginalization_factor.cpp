#include "my_vins_estimator/marginalization_factor.h"

#include <cmath>
#include <algorithm>

using namespace std;
using namespace Eigen;

void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes =
        cost_function->parameter_block_sizes();

    raw_jacobians = new double *[block_sizes.size()];

    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(),
                            block_sizes[i]);

        raw_jacobians[i] =
            jacobians[i].data();
    }

    cost_function->Evaluate(parameter_blocks.data(),
                            residuals.data(),
                            raw_jacobians);

    /*
     * 如果这个残差块使用了 robust loss，
     * 需要对 residual 和 jacobian 做一致缩放。
     */
    if (loss_function)
    {
        double sq_norm =
            residuals.squaredNorm();

        double rho[3];

        loss_function->Evaluate(sq_norm, rho);

        double residual_scaling;
        double alpha_sq_norm;

        double sqrt_rho1 =
            std::sqrt(rho[1]);

        if (sq_norm == 0.0 || rho[2] <= 0.0)
        {
            residual_scaling = sqrt_rho1;
            alpha_sq_norm = 0.0;
        }
        else
        {
            double D =
                1.0 + 2.0 * sq_norm * rho[2] / rho[1];

            double alpha =
                1.0 - std::sqrt(D);

            residual_scaling =
                sqrt_rho1 / (1.0 - alpha);

            alpha_sq_norm =
                alpha / sq_norm;
        }

        for (int i = 0;
             i < static_cast<int>(parameter_blocks.size());
             i++)
        {
            jacobians[i] =
                sqrt_rho1 *
                (jacobians[i] -
                 alpha_sq_norm *
                     residuals *
                     (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    for (auto &it : parameter_block_data)
    {
        delete[] it.second;
    }

    for (int i = 0; i < static_cast<int>(factors.size()); i++)
    {
        if (factors[i]->raw_jacobians != nullptr)
        {
            delete[] factors[i]->raw_jacobians;
            factors[i]->raw_jacobians = nullptr;
        }

        delete factors[i]->cost_function;

        /*
         * loss_function 通常由外部创建。
         * 原版这里不 delete loss_function。
         * 第一版保持一致。
         */

        delete factors[i];
    }
}


void MarginalizationInfo::addResidualBlockInfo(
    ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);

    std::vector<double *> &parameter_blocks =
        residual_block_info->parameter_blocks;

    std::vector<int> parameter_block_sizes =
        residual_block_info->cost_function->parameter_block_sizes();

    /*
     * 记录所有参与该 residual 的参数块大小。
     */
    for (int i = 0;
         i < static_cast<int>(parameter_blocks.size());
         i++)
    {
        double *addr =
            parameter_blocks[i];

        int size =
            parameter_block_sizes[i];

        parameter_block_size[reinterpret_cast<long>(addr)] =
            size;
    }

    /*
     * drop_set 里的参数块是这次要被边缘化掉的变量。
     * 先把它们放入 parameter_block_idx。
     * marginalize() 里会先给这些变量分配索引。
     */
    for (int i = 0;
         i < static_cast<int>(residual_block_info->drop_set.size());
         i++)
    {
        int drop_id =
            residual_block_info->drop_set[i];

        double *addr =
            parameter_blocks[drop_id];

        parameter_block_idx[reinterpret_cast<long>(addr)] =
            0;
    }
}



void MarginalizationInfo::preMarginalize()
{
    for (auto factor : factors)
    {
        factor->Evaluate();

        std::vector<int> block_sizes =
            factor->cost_function->parameter_block_sizes();

        for (int i = 0;
             i < static_cast<int>(block_sizes.size());
             i++)
        {
            long addr =
                reinterpret_cast<long>(factor->parameter_blocks[i]);

            int size =
                block_sizes[i];

            if (parameter_block_data.find(addr) ==
                parameter_block_data.end())
            {
                double *data =
                    new double[size];

                memcpy(data,
                       factor->parameter_blocks[i],
                       sizeof(double) * size);

                parameter_block_data[addr] =
                    data;
            }
        }
    }
}

void MarginalizationInfo::marginalize()
{
    /*
     * 1. 给要被 marginalize 的变量分配索引。
     */
    int pos = 0;

    for (auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    m = pos;

    /*
     * 2. 给保留下来的变量分配索引。
     */
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) ==
            parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    n = pos - m;

    ROS_INFO("marginalize: total local size = %d, marg size = %d, keep size = %d",
             pos, m, n);

    if (m == 0 || n == 0)
    {
        ROS_WARN("marginalize: bad size, m = %d, n = %d", m, n);
    }

    /*
     * 3. 构造 Hessian A 和 b。
     *
     * 对每个 residual:
     *     A += J^T J
     *     b += J^T r
     */
    MatrixXd A(pos, pos);
    VectorXd b(pos);

    A.setZero();
    b.setZero();

    for (auto factor : factors)
    {
        for (int i = 0;
             i < static_cast<int>(factor->parameter_blocks.size());
             i++)
        {
            long addr_i =
                reinterpret_cast<long>(factor->parameter_blocks[i]);

            int idx_i =
                parameter_block_idx[addr_i];

            int size_i =
                localSize(parameter_block_size[addr_i]);

            MatrixXd jacobian_i =
                factor->jacobians[i].leftCols(size_i);

            for (int j = i;
                 j < static_cast<int>(factor->parameter_blocks.size());
                 j++)
            {
                long addr_j =
                    reinterpret_cast<long>(factor->parameter_blocks[j]);

                int idx_j =
                    parameter_block_idx[addr_j];

                int size_j =
                    localSize(parameter_block_size[addr_j]);

                MatrixXd jacobian_j =
                    factor->jacobians[j].leftCols(size_j);

                if (i == j)
                {
                    A.block(idx_i, idx_j, size_i, size_j) +=
                        jacobian_i.transpose() * jacobian_j;
                }
                else
                {
                    MatrixXd tmp =
                        jacobian_i.transpose() * jacobian_j;

                    A.block(idx_i, idx_j, size_i, size_j) += tmp;

                    A.block(idx_j, idx_i, size_j, size_i) += tmp.transpose();
                }
            }

            b.segment(idx_i, size_i) +=
                jacobian_i.transpose() * factor->residuals;
        }
    }

    /*
     * 4. Schur Complement。
     *
     * A 被分成：
     *
     * [ Amm Amr ]
     * [ Arm Arr ]
     *
     * 消掉 m：
     *
     * A_prior = Arr - Arm * Amm^-1 * Amr
     * b_prior = brr - Arm * Amm^-1 * bmm
     */
    MatrixXd Amm =
        0.5 *
        (A.block(0, 0, m, m) +
         A.block(0, 0, m, m).transpose());

    SelfAdjointEigenSolver<MatrixXd> saes(Amm);

    VectorXd S =
        saes.eigenvalues();

    MatrixXd Amm_inv =
        saes.eigenvectors() *
        VectorXd((S.array() > eps)
                     .select(S.array().inverse(), 0))
            .asDiagonal() *
        saes.eigenvectors().transpose();

    VectorXd bmm =
        b.segment(0, m);

    MatrixXd Amr =
        A.block(0, m, m, n);

    MatrixXd Arm =
        A.block(m, 0, n, m);

    MatrixXd Arr =
        A.block(m, m, n, n);

    VectorXd brr =
        b.segment(m, n);

    MatrixXd A_prior =
        Arr - Arm * Amm_inv * Amr;

    VectorXd b_prior =
        brr - Arm * Amm_inv * bmm;

    /*
     * 5. 把 prior Hessian 分解为：
     *
     * A_prior = J^T J
     * b_prior = J^T r
     *
     * 得到：
     *     linearized_jacobians
     *     linearized_residuals
     */
    SelfAdjointEigenSolver<MatrixXd> saes2(A_prior);

    VectorXd S2 =
        saes2.eigenvalues();

    VectorXd S2_pos =
        VectorXd((S2.array() > eps)
                     .select(S2.array(), 0));

    VectorXd S2_inv =
        VectorXd((S2.array() > eps)
                     .select(S2.array().inverse(), 0));

    VectorXd S_sqrt =
        S2_pos.cwiseSqrt();

    VectorXd S_inv_sqrt =
        S2_inv.cwiseSqrt();

    linearized_jacobians =
        S_sqrt.asDiagonal() *
        saes2.eigenvectors().transpose();

    linearized_residuals =
        S_inv_sqrt.asDiagonal() *
        saes2.eigenvectors().transpose() *
        b_prior;

    ROS_INFO("marginalize done: residual size = %d, keep size = %d",
             static_cast<int>(linearized_residuals.size()),
             n);
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(
    std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;

    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            keep_block_size.push_back(
                parameter_block_size[it.first]);

            keep_block_idx.push_back(
                parameter_block_idx[it.first]);

            keep_block_data.push_back(
                parameter_block_data[it.first]);

            if (addr_shift.find(it.first) != addr_shift.end())
            {
                keep_block_addr.push_back(
                    addr_shift[it.first]);
            }
            else
            {
                ROS_WARN("getParameterBlocks: cannot find shifted address");
                keep_block_addr.push_back(nullptr);
            }
        }
    }

    sum_block_size =
        std::accumulate(keep_block_size.begin(),
                        keep_block_size.end(),
                        0);

    return keep_block_addr;
}


MarginalizationFactor::MarginalizationFactor(
    MarginalizationInfo *_marginalization_info)
    : marginalization_info(_marginalization_info)
{
    int cnt = 0;

    for (auto size : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(size);
        cnt += size;
    }

    set_num_residuals(marginalization_info->n);

    ROS_INFO("create MarginalizationFactor: residual size = %d, parameter global size sum = %d",
             marginalization_info->n,
             cnt);
}

bool MarginalizationFactor::Evaluate(double const *const *parameters,
                                     double *residuals,
                                     double **jacobians) const
{
    int n =
        marginalization_info->n;

    int m =
        marginalization_info->m;

    VectorXd dx(n);
    dx.setZero();

    /*
     * 对所有保留下来的参数块，计算当前值和线性化点的差。
     */
    for (int i = 0;
         i < static_cast<int>(marginalization_info->keep_block_size.size());
         i++)
    {
        int size =
            marginalization_info->keep_block_size[i];

        int idx =
            marginalization_info->keep_block_idx[i] - m;

        Eigen::Map<const VectorXd> x(parameters[i], size);

        Eigen::Map<const VectorXd> x0(
            marginalization_info->keep_block_data[i],
            size);

        if (size != 7)
        {
            dx.segment(idx, size) =
                x - x0;
        }
        else
        {
            /*
             * pose 参数块：
             * global size = 7
             * local size = 6
             *
             * para_Pose:
             *     p[0:3], q[x y z w]
             */
            dx.segment<3>(idx) =
                x.head<3>() - x0.head<3>();

            Quaterniond q0(
                x0(6),
                x0(3),
                x0(4),
                x0(5));

            Quaterniond q(
                x(6),
                x(3),
                x(4),
                x(5));

            Quaterniond dq =
                q0.inverse() * q;

            dq =
                Utility::positify(dq);

            dx.segment<3>(idx + 3) =
                2.0 * dq.vec();
        }
    }

    Eigen::Map<VectorXd>(residuals, n) =
        marginalization_info->linearized_residuals +
        marginalization_info->linearized_jacobians * dx;

    if (jacobians)
    {
        for (int i = 0;
             i < static_cast<int>(marginalization_info->keep_block_size.size());
             i++)
        {
            if (jacobians[i])
            {
                int size =
                    marginalization_info->keep_block_size[i];

                int local_size =
                    marginalization_info->localSize(size);

                int idx =
                    marginalization_info->keep_block_idx[i] - m;

                Eigen::Map<Eigen::Matrix<double,
                                         Eigen::Dynamic,
                                         Eigen::Dynamic,
                                         Eigen::RowMajor>>
                    jacobian(jacobians[i], n, size);

                jacobian.setZero();

                jacobian.leftCols(local_size) =
                    marginalization_info->linearized_jacobians
                        .middleCols(idx, local_size);
            }
        }
    }

    return true;
}