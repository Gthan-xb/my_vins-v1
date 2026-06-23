#include "my_vins_estimator/feature_manager.h"
#include <cmath>
FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[])
  : Rs(_Rs)
{
    last_track_num = 0;

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i].setIdentity();
    }
}

void FeatureManager::clearState()
{
    feature.clear();
    last_track_num = 0;
}

void FeatureManager::setRic(Eigen::Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;

    for(auto &it_per_id: feature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.feature_per_frame.size());
        if (isFeatureValidForOptimization(it_per_id))
        {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(
    int frame_count,
    const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    double td)
{
    ROS_INFO("input feature: %lu", image.size());

    last_track_num = 0;

    for (auto &id_pts : image)
    {
        int feature_id = id_pts.first;

        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        auto it = std::find_if(feature.begin(),
                               feature.end(),
                               [feature_id](const FeaturePerId &it)
                               {
                                   return it.feature_id == feature_id;
                               });

        if (it == feature.end())
        {
            feature.emplace_back(feature_id, frame_count);
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    /*
     * 原版逻辑：
     * 前两帧，或者跟踪点太少，直接认为是关键帧。
     */
    if (frame_count < 2 || last_track_num < 20)
    {
        ROS_INFO("keyframe: frame_count < 2 or track num too small");
        return true;
    }

    double parallax_sum = 0.0;
    int parallax_num = 0;

    for (auto &it_per_id : feature)
    {
        it_per_id.used_num =
            static_cast<int>(it_per_id.feature_per_frame.size());

        /*
         * 只有同时出现在：
         * frame_count - 2
         * frame_count - 1
         * 的特征，才能用来计算视差。
         */
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.endFrame() >= frame_count - 1)
        {
            parallax_sum +=
                compensatedParallax2(it_per_id, frame_count);

            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        ROS_INFO("keyframe: no parallax feature");
        return true;
    }

    double average_parallax =
        parallax_sum / parallax_num;

    ROS_INFO("parallax: %.6f, threshold: %.6f, parallax_num: %d",
             average_parallax,
             MIN_PARALLAX,
             parallax_num);

    return average_parallax >= MIN_PARALLAX;
}

Eigen::VectorXd FeatureManager::getDepthVector()
{
    Eigen::VectorXd dep_vec(getFeatureCount());

    int feature_index = -1;

    for (auto &it_per_id : feature)
    { 
        it_per_id.used_num = static_cast<int>(it_per_id.feature_per_frame.size());

        if (!isFeatureValidForOptimization(it_per_id))
        {
            continue;
        }
        feature_index++;
        dep_vec(feature_index) = 1/it_per_id.estimated_depth;
    }
    return dep_vec;
}

void FeatureManager::setDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;

    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = static_cast<int>(it_per_id.feature_per_frame.size());

        if (!isFeatureValidForOptimization(it_per_id))
        {
            continue;
        }

        feature_index++;

         double inv_depth = x(feature_index);

        if (!std::isfinite(inv_depth) || inv_depth <= 1e-6)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
            it_per_id.solve_flag = 2;
            continue;
        }

        double depth = 1.0 / inv_depth;

        if (!std::isfinite(depth) || depth < 0.1 || depth > 100.0)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
            it_per_id.solve_flag = 2;
        }
        else
        {
            it_per_id.estimated_depth = depth;
            it_per_id.solve_flag = 1;
        }
    }
}

void FeatureManager::clearDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;

    for (auto &it_per_id : feature)
    {
        it_per_id.used_num =
            static_cast<int>(it_per_id.feature_per_frame.size());

        if (!isFeatureValidForOptimization(it_per_id))
        {
            continue;
        }

        feature_index++;

        if (feature_index >= x.size())
        {
            ROS_WARN("clearDepth: feature_index %d >= x.size %ld",
                     feature_index,
                     x.size());
            break;
        }

        double inv_depth = x(feature_index);

        if (!std::isfinite(inv_depth) || inv_depth <= 1e-6)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
        else
        {
            it_per_id.estimated_depth = 1.0 / inv_depth;
        }
    }
}

void FeatureManager::triangulate(Eigen::Vector3d Ps[],
                                 Eigen::Vector3d tic[],
                                 Eigen::Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num =
            static_cast<int>(it_per_id.feature_per_frame.size());

        if (!(it_per_id.used_num >= 2 &&
              it_per_id.start_frame < WINDOW_SIZE - 2))
        {
            continue;
        }

        if (it_per_id.estimated_depth > 0)
        {
            continue;
        }

        int imu_i = it_per_id.start_frame;
        int imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);

        Eigen::MatrixXd svd_A(
            2 * static_cast<int>(it_per_id.feature_per_frame.size()),
            4);

        int svd_idx = 0;

        /*
         * 以特征第一次被观测到的相机坐标系作为参考系。
         *
         * 第一次观测帧：
         *   camera_i pose 被设成 [I | 0]
         */
        Eigen::Vector3d t0 =
            Ps[imu_i] + Rs[imu_i] * tic[0];

        Eigen::Matrix3d R0 =
            Rs[imu_i] * ric[0];

        Eigen::Matrix<double, 3, 4> P0;
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 =
                Ps[imu_j] + Rs[imu_j] * tic[0];

            Eigen::Matrix3d R1 =
                Rs[imu_j] * ric[0];

            /*
             * 把第 j 帧相机相对于第 i 帧相机的位姿写出来。
             *
             * t = R0^T * (t1 - t0)
             * R = R0^T * R1
             */
            Eigen::Vector3d t =
                R0.transpose() * (t1 - t0);

            Eigen::Matrix3d R =
                R0.transpose() * R1;

            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;

            Eigen::Vector3d f =
                it_per_frame.point.normalized();

            svd_A.row(svd_idx++) =
                f.x() * P.row(2) - f.z() * P.row(0);

            svd_A.row(svd_idx++) =
                f.y() * P.row(2) - f.z() * P.row(1);
        }

        ROS_ASSERT(svd_idx == svd_A.rows());

        Eigen::Vector4d svd_V =
            Eigen::JacobiSVD<Eigen::MatrixXd>(
                svd_A,
                Eigen::ComputeThinV)
                .matrixV()
                .rightCols<1>();

        double depth = INIT_DEPTH;
       
        if (std::abs(svd_V[3]) > 1e-8)
        {
            depth = svd_V[2] / svd_V[3];
        }

        if (!std::isfinite(depth) || depth < 0.1 || depth > 100.0)
        {
            depth = INIT_DEPTH;
        }
        
        it_per_id.estimated_depth = depth;

    }
}

// 删除 solve_flag 失败或深度非法的特征
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->solve_flag == 2)
        {
            feature.erase(it);
        }
    }
}

// 删除滑窗最旧帧之前的特征
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
        {
            it->start_frame--;
        }
        else
        {
            /*
             * 这个特征从最老帧开始被观测。
             * 滑窗删除最老帧时，也要删掉它的第一个观测。
             */
            it->feature_per_frame.erase(it->feature_per_frame.begin());

            if (it->feature_per_frame.empty())
            {
                feature.erase(it);
            }
        }
    }
}

// 更新 start_frame，删除滑窗最前帧之后的特征索引
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;

            if (it->endFrame() < frame_count - 1)
            {
                continue;
            }
            
            if (j < 0 || j >= static_cast<int>(it->feature_per_frame.size()))
            {
                ROS_WARN("removeFront: invalid erase index j=%d, size=%lu, start_frame=%d, endFrame=%d, frame_count=%d",
                         j,
                         it->feature_per_frame.size(),
                         it->start_frame,
                         it->endFrame(),
                         frame_count);
                continue;
            }

            it->feature_per_frame.erase(
                it->feature_per_frame.begin() + j);

            if (it->feature_per_frame.empty())
            {
                feature.erase(it);
            }
        }
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count_l &&
            it_per_id.endFrame() >= frame_count_r)
        {
            int idx_l = frame_count_l - it_per_id.start_frame;
            int idx_r = frame_count_r - it_per_id.start_frame;

            Eigen::Vector3d pts_l =
                it_per_id.feature_per_frame[idx_l].point;

            Eigen::Vector3d pts_r =
                it_per_id.feature_per_frame[idx_r].point;

            corres.emplace_back(pts_l, pts_r);
        }
    }

    return corres;
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count)
{
    /*
     * 原版含义：
     * 比较倒数第二帧 frame_count - 2
     * 和倒数第一帧 frame_count - 1
     * 中同一个特征的归一化平面视差。
     *
     * 第一版先不做旋转补偿，和原版中常见写法一致：
     * p_i_comp = p_i;
     */

    const FeaturePerFrame &frame_i =
        it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];

    const FeaturePerFrame &frame_j =
        it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    Eigen::Vector3d p_i = frame_i.point;
    Eigen::Vector3d p_j = frame_j.point;

    Eigen::Vector3d p_i_comp = p_i;

    double dep_i = p_i.z();
    double dep_j = p_j.z();
    double dep_i_comp = p_i_comp.z();

    if (std::abs(dep_i) < 1e-8 ||
        std::abs(dep_j) < 1e-8 ||
        std::abs(dep_i_comp) < 1e-8)
    {
        return 0.0;
    }

    double u_i = p_i.x() / dep_i;
    double v_i = p_i.y() / dep_i;

    double u_j = p_j.x() / dep_j;
    double v_j = p_j.y() / dep_j;

    double u_i_comp = p_i_comp.x() / dep_i_comp;
    double v_i_comp = p_i_comp.y() / dep_i_comp;

    double du = u_i - u_j;
    double dv = v_i - v_j;

    double du_comp = u_i_comp - u_j;
    double dv_comp = v_i_comp - v_j;

    double parallax =
        std::sqrt(std::min(du * du + dv * dv,
                           du_comp * du_comp + dv_comp * dv_comp));

    return parallax;
}

bool FeatureManager::isFeatureValidForOptimization(
    const FeaturePerId &it_per_id) const
{
    int used_num =
        static_cast<int>(it_per_id.feature_per_frame.size());

    if (used_num < 2)
        return false;

    if (it_per_id.start_frame >= WINDOW_SIZE - 2)
        return false;

    if (it_per_id.solve_flag == 2)
        return false;

    return true;
}