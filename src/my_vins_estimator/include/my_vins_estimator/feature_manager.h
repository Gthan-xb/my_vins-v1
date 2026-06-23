#pragma once    
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <ros/assert.h>
#include "my_vins_estimator/parameters.h"

class FeaturePerFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double _td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);

        uv.x() = _point(3);
        uv.y() = _point(4);

        velocity.x() = _point(5);
        velocity.y() = _point(6);

        cur_td = _td;
        is_used = false;
    }

    double cur_td;
    Eigen::Vector3d point;
    Eigen::Vector2d uv;
    Eigen::Vector2d velocity;

    bool is_used;
    
};

class FeaturePerId
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FeaturePerId(int _feature_id, int _start_frame)
      :feature_id(_feature_id), 
       start_frame(_start_frame),
       used_num(0),
       estimated_depth(-1.0),
       solve_flag(0)
    {
    }

    int endFrame() const
    {
        return start_frame + static_cast<int>(feature_per_frame.size()) - 1;
    }

    const int feature_id;
    int start_frame;
    std::vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0: not solve, 1: solve, 2: solve fail
};

class FeatureManager
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FeatureManager(Eigen::Matrix3d _Rs[]);

        void clearState();

        void setRic(Eigen::Matrix3d _ric[]);

        int getFeatureCount();

        bool addFeatureCheckParallax(int frame_count,const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,double td);
        
        double compensatedParallax2(const FeaturePerId &it_per_id,int frame_count);
        
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

        Eigen::VectorXd getDepthVector();

        void setDepth(const Eigen::VectorXd &x);

        void clearDepth(const Eigen::VectorXd &x);

        bool isFeatureValidForOptimization(const FeaturePerId &it_per_id) const;
        
        void triangulate(Eigen::Vector3d Ps[] ,Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
        
        void removeFailures();

        void removeBack();

        void removeFront(int frame_count);
    public:
        std::list<FeaturePerId> feature;

        int last_track_num;

    private:
        const Eigen::Matrix3d *Rs;
        Eigen::Matrix3d ric[NUM_OF_CAM];
};