#include "my_vins_estimator/solve_5pts.h"
#include <ros/ros.h>

using namespace std;
using namespace Eigen;

bool MotionEstimator::solveRelativeRT(
    const vector<pair<Vector3d, Vector3d>> &corres,
    Matrix3d &Rotation,
    Vector3d &Translation)
{
    if (corres.size() < 15)
    {
        return false;
    }

    vector<cv::Point2f> ll;
    vector<cv::Point2f> rr;

    ll.reserve(corres.size());
    rr.reserve(corres.size());

    for (int i = 0; i < static_cast<int>(corres.size()); i++)
    {
        ll.emplace_back(corres[i].first.x(), corres[i].first.y());
        rr.emplace_back(corres[i].second.x(), corres[i].second.y());
    }

    cv::Mat mask;

    /*
     * 输入点已经是归一化平面坐标。
     * 原版这里使用 findFundamentalMat，阈值 0.3 / 460。
     */
    cv::Mat E = cv::findFundamentalMat(
        ll,
        rr,
        cv::FM_RANSAC,
        0.3 / 460.0,
        0.99,
        mask);

    if (E.empty())
    {
        ROS_WARN("findFundamentalMat failed");
        return false;
    }

    cv::Mat cameraMatrix =
        (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                   0, 1, 0,
                                   0, 0, 1);

    cv::Mat rot;
    cv::Mat trans;

    int inlier_cnt = cv::recoverPose(
        E,
        ll,
        rr,
        cameraMatrix,
        rot,
        trans,
        mask);

    if (inlier_cnt < 12)
    {
        ROS_WARN("recoverPose inliers too few: %d", inlier_cnt);
        return false;
    }

    Matrix3d R;
    Vector3d T;

    for (int i = 0; i < 3; i++)
    {
        T(i) = trans.at<double>(i, 0);

        for (int j = 0; j < 3; j++)
        {
            R(i, j) = rot.at<double>(i, j);
        }
    }

    Rotation = R.transpose();
    Translation = -R.transpose() * T;

    return true;
}


void MotionEstimator::decomposeE(
    cv::Mat E,
    cv::Mat_<double> &R1,
    cv::Mat_<double> &R2,
    cv::Mat_<double> &t1,
    cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);

    cv::Matx33d W(0, -1, 0,
                  1,  0, 0,
                  0,  0, 1);

    cv::Matx33d Wt(0, 1, 0,
                  -1, 0, 0,
                   0, 0, 1);

    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;

    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}


double MotionEstimator::testTriangulation(
    const vector<cv::Point2f> &l,
    const vector<cv::Point2f> &r,
    cv::Mat_<double> R,
    cv::Mat_<double> t)
{
    cv::Mat pointcloud;

    cv::Matx34f P =
        cv::Matx34f(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);

    cv::Matx34f P1 =
        cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                    R(1, 0), R(1, 1), R(1, 2), t(1),
                    R(2, 0), R(2, 1), R(2, 2), t(2));

    cv::triangulatePoints(P, P1, l, r, pointcloud);

    int front_count = 0;

    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);

        cv::Mat_<double> p_3d_l =
            cv::Mat(P) * (pointcloud.col(i) / normal_factor);

        cv::Mat_<double> p_3d_r =
            cv::Mat(P1) * (pointcloud.col(i) / normal_factor);

        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
        {
            front_count++;
        }
    }

    return 1.0 * front_count / pointcloud.cols;
}