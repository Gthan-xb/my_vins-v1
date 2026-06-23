#include "my_vins_estimator/initial_alignment.h"
#include "my_vins_estimator/parameters.h"

#include <cmath>
#include <iterator>

#include <ros/ros.h>

using namespace Eigen;
using namespace std;

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame,
                        Vector3d *Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;

    A.setZero();
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;

    for (frame_i = all_image_frame.begin(), frame_j = std::next(frame_i);
         frame_j != all_image_frame.end();
         frame_i++, frame_j++)
    {
        if (frame_j->second.pre_integration == nullptr)
        {
            continue;
        }

        MatrixXd tmp_A =
            frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

        Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);

        Vector3d tmp_b =
            2.0 *
            (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();

        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }

    delta_bg = A.ldlt().solve(b);

    ROS_INFO("gyroscope bias initial calibration: %.6f %.6f %.6f",
             delta_bg.x(), delta_bg.y(), delta_bg.z());

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Bgs[i] += delta_bg;
    }

    for (frame_i = all_image_frame.begin(), frame_j = std::next(frame_i);
         frame_j != all_image_frame.end();
         frame_i++, frame_j++)
    {
        if (frame_j->second.pre_integration != nullptr)
        {
            frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
        }
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b;
    Vector3d c;

    Vector3d a = g0.normalized();

    Vector3d tmp(0, 0, 1);

    if (std::abs(a.dot(tmp)) > 0.99)
    {
        tmp << 1, 0, 0;
    }

    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);

    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;

    return bc;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame,
                     Vector3d &g,
                     VectorXd &x)
{
    int all_frame_count = static_cast<int>(all_image_frame.size());

    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A(n_state, n_state);
    VectorXd b(n_state);

    A.setZero();
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;

    int i = 0;

    for (frame_i = all_image_frame.begin(), frame_j = std::next(frame_i);
         frame_j != all_image_frame.end();
         frame_i++, frame_j++, i++)
    {
        if (frame_j->second.pre_integration == nullptr)
        {
            continue;
        }

        double dt = frame_j->second.pre_integration->sum_dt;

        if (dt <= 0)
        {
            continue;
        }

        Matrix<double, 6, Dynamic> tmp_A(6, n_state);
        Matrix<double, 6, 1> tmp_b;

        tmp_A.setZero();
        tmp_b.setZero();

        /*
         * position equation:
         *
         * delta_p_ij =
         * R_i^T (s(P_j - P_i) + 0.5 g dt^2 - R_i tic + R_j tic)
         * - v_i dt
         */

        tmp_A.block<3, 3>(0, i * 3) =
            -dt * Matrix3d::Identity();

        tmp_A.block<3, 3>(0, all_frame_count * 3) =
            frame_i->second.R.transpose() *
            dt * dt / 2.0 *
            Matrix3d::Identity();

        tmp_A.block<3, 1>(0, n_state - 1) =
            frame_i->second.R.transpose() *
            (frame_j->second.T - frame_i->second.T) / 100.0;

        tmp_b.block<3, 1>(0, 0) =
            frame_j->second.pre_integration->delta_p +
            frame_i->second.R.transpose() * frame_j->second.R * TIC[0] -
            TIC[0];

        /*
         * velocity equation:
         *
         * delta_v_ij =
         * R_i^T (v_j - v_i + g dt)
         */

        tmp_A.block<3, 3>(3, i * 3) =
            -Matrix3d::Identity();

        tmp_A.block<3, 3>(3, (i + 1) * 3) =
            frame_i->second.R.transpose() * frame_j->second.R;

        tmp_A.block<3, 3>(3, all_frame_count * 3) =
            frame_i->second.R.transpose() *
            dt *
            Matrix3d::Identity();

        tmp_b.block<3, 1>(3, 0) =
            frame_j->second.pre_integration->delta_v;

        Matrix<double, 6, 6> cov_inv;
        cov_inv.setIdentity();

        MatrixXd r_A =
            tmp_A.transpose() * cov_inv * tmp_A;

        VectorXd r_b =
            tmp_A.transpose() * cov_inv * tmp_b;

        A += r_A;
        b += r_b;
    }

    A *= 1000.0;
    b *= 1000.0;

    x = A.ldlt().solve(b);

    double s = x(n_state - 1) / 100.0;
    g = x.segment<3>(all_frame_count * 3);

    ROS_INFO("linear alignment result: g norm %.6f, scale %.6f",
             g.norm(), s);

    if (std::abs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        ROS_WARN("linear alignment failed: g norm %.6f, scale %.6f",
                 g.norm(), s);
        return false;
    }

    /*
     * 继续细化重力方向。
     */
    RefineGravity(all_image_frame, g, x);

    s = x(n_state - 1) / 100.0;
    x(n_state - 1) = s;

    ROS_INFO("refined gravity: %.6f %.6f %.6f, norm %.6f, scale %.6f",
             g.x(), g.y(), g.z(), g.norm(), s);

    if (s < 0)
    {
        ROS_WARN("refined scale is negative");
        return false;
    }

    return true;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame,
                   Vector3d &g,
                   VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();

    int all_frame_count = static_cast<int>(all_image_frame.size());

    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd lxly(3, 2);

    for (int k = 0; k < 4; k++)
    {
        MatrixXd A(n_state, n_state);
        VectorXd b(n_state);

        A.setZero();
        b.setZero();

        lxly = TangentBasis(g0);

        map<double, ImageFrame>::iterator frame_i;
        map<double, ImageFrame>::iterator frame_j;

        int i = 0;

        for (frame_i = all_image_frame.begin(), frame_j = std::next(frame_i);
             frame_j != all_image_frame.end();
             frame_i++, frame_j++, i++)
        {
            if (frame_j->second.pre_integration == nullptr)
            {
                continue;
            }

            double dt = frame_j->second.pre_integration->sum_dt;

            if (dt <= 0)
            {
                continue;
            }

            Matrix<double, 6, Dynamic> tmp_A(6, n_state);
            Matrix<double, 6, 1> tmp_b;

            tmp_A.setZero();
            tmp_b.setZero();

            tmp_A.block<3, 3>(0, i * 3) =
                -dt * Matrix3d::Identity();

            tmp_A.block<3, 2>(0, all_frame_count * 3) =
                frame_i->second.R.transpose() *
                dt * dt / 2.0 *
                Matrix3d::Identity() *
                lxly;

            tmp_A.block<3, 1>(0, n_state - 1) =
                frame_i->second.R.transpose() *
                (frame_j->second.T - frame_i->second.T) / 100.0;

            tmp_b.block<3, 1>(0, 0) =
                frame_j->second.pre_integration->delta_p +
                frame_i->second.R.transpose() * frame_j->second.R * TIC[0] -
                TIC[0] -
                frame_i->second.R.transpose() *
                dt * dt / 2.0 *
                g0;

            tmp_A.block<3, 3>(3, i * 3) =
                -Matrix3d::Identity();

            tmp_A.block<3, 3>(3, (i + 1) * 3) =
                frame_i->second.R.transpose() * frame_j->second.R;

            tmp_A.block<3, 2>(3, all_frame_count * 3) =
                frame_i->second.R.transpose() *
                dt *
                Matrix3d::Identity() *
                lxly;

            tmp_b.block<3, 1>(3, 0) =
                frame_j->second.pre_integration->delta_v -
                frame_i->second.R.transpose() *
                dt *
                g0;

            Matrix<double, 6, 6> cov_inv;
            cov_inv.setIdentity();

            MatrixXd r_A =
                tmp_A.transpose() * cov_inv * tmp_A;

            VectorXd r_b =
                tmp_A.transpose() * cov_inv * tmp_b;

            A += r_A;
            b += r_b;
        }

        A *= 1000.0;
        b *= 1000.0;

        VectorXd x_tmp =
            A.ldlt().solve(b);

        Vector2d dg =
            x_tmp.segment<2>(all_frame_count * 3);

        g0 =
            (g0 + lxly * dg).normalized() * G.norm();

        /*
         * 把 refine 后的结果转回原来的 x：
         * velocities: all_frame_count * 3
         * gravity: 3
         * scale: 1
         */
        VectorXd x_new(all_frame_count * 3 + 3 + 1);
        x_new.setZero();

        x_new.head(all_frame_count * 3) =
            x_tmp.head(all_frame_count * 3);

        x_new.segment<3>(all_frame_count * 3) =
            g0;

        x_new(all_frame_count * 3 + 3) =
            x_tmp(n_state - 1);

        x = x_new;
    }

    g = g0;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame,
                        Vector3d *Bgs,
                        Vector3d &g,
                        VectorXd &x)
{
    ROS_INFO("VisualIMUAlignment begin");

    if (all_image_frame.size() < 2)
    {
        ROS_WARN("VisualIMUAlignment failed: not enough image frames");
        return false;
    }

    solveGyroscopeBias(all_image_frame, Bgs);

    if (!LinearAlignment(all_image_frame, g, x))
    {
        ROS_WARN("VisualIMUAlignment failed: LinearAlignment failed");
        return false;
    }

    ROS_INFO("VisualIMUAlignment success");

    return true;
}