#include <queue>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <Eigen/Dense>

#include "my_vins_estimator/estimator.h"
#include "my_vins_estimator/parameters.h"

using namespace std;
using namespace Eigen;

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

mutex m_buf;
mutex m_estimator;
condition_variable con;

double current_time = -1.0;
double last_imu_t = 0.0;

ros::Publisher pub_odometry;
ros::Publisher pub_path;
nav_msgs::Path path;

string FEATURE_TOPIC = "/feature_tracker/feature";


void pubOdometry(const std_msgs::Header &header)
{
    nav_msgs::Odometry odom;

    odom.header = header;
    odom.header.frame_id = "map";
    odom.child_frame_id = "body";

    int index = estimator.frame_count;
    if (index > WINDOW_SIZE)
        index = WINDOW_SIZE;

    odom.pose.pose.position.x = estimator.Ps[index].x();
    odom.pose.pose.position.y = estimator.Ps[index].y();
    odom.pose.pose.position.z = estimator.Ps[index].z();

    Eigen::Quaterniond q(estimator.Rs[index]);

    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = estimator.Vs[index].x();
    odom.twist.twist.linear.y = estimator.Vs[index].y();
    odom.twist.twist.linear.z = estimator.Vs[index].z();

    pub_odometry.publish(odom);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odom.header;
    pose_stamped.pose = odom.pose.pose;

    path.header = odom.header;
    path.header.frame_id = "map";
    path.poses.push_back(pose_stamped);

    pub_path.publish(path);
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();

    if (t <= last_imu_t)
    {
        ROS_WARN("imu message disorder, current: %.9f, last: %.9f",
                 t, last_imu_t);
        return;
    }

    last_imu_t = t;

    {
        lock_guard<mutex> lock(m_buf);
        imu_buf.push(imu_msg);
    }

    con.notify_one();
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    {
        lock_guard<mutex> lock(m_buf);
        feature_buf.push(feature_msg);
    }

    con.notify_one();
}


vector<pair<vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    vector<pair<vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        double img_t = feature_buf.front()->header.stamp.toSec() + estimator.td;

        /*
         * 最新 IMU 还没有覆盖当前图像时间，等待更多 IMU。
         */
        if (imu_buf.back()->header.stamp.toSec() <= img_t)
        {
            return measurements;
        }

        /*
         * 最早 IMU 已经晚于图像时间，说明这张图像没有可用 IMU，丢弃。
         */
        if (imu_buf.front()->header.stamp.toSec() > img_t)
        {
            ROS_WARN("throw image, no imu before image time");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        vector<sensor_msgs::ImuConstPtr> IMUs;

        while (!imu_buf.empty() &&
               imu_buf.front()->header.stamp.toSec() < img_t)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        /*
         * 加入第一帧时间大于等于图像时间的 IMU。
         * 注意：这里不 pop，下一张图像还可能需要它做插值。
         */
        if (!imu_buf.empty())
        {
            IMUs.emplace_back(imu_buf.front());
        }

        if (IMUs.empty())
        {
            ROS_WARN("no imu between two image");
        }

        measurements.emplace_back(IMUs, img_msg);
    }

    return measurements;
}


void process()
{
    while (ros::ok())
    {
        vector<pair<vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

        {
            unique_lock<mutex> lock(m_buf);
            con.wait(lock, [&]
                     {
                         measurements = getMeasurements();
                         return !measurements.empty() || !ros::ok();
                     });
        }

        if (!ros::ok())
            break;

        lock_guard<mutex> estimator_lock(m_estimator);

        for (auto &measurement : measurements)
        {
            const vector<sensor_msgs::ImuConstPtr> &imus = measurement.first;
            sensor_msgs::PointCloudConstPtr img_msg = measurement.second;

            double img_t = img_msg->header.stamp.toSec() + estimator.td;

            double dx = 0.0;
            double dy = 0.0;
            double dz = 0.0;
            double rx = 0.0;
            double ry = 0.0;
            double rz = 0.0;

            for (size_t i = 0; i < imus.size(); i++)
            {
                sensor_msgs::ImuConstPtr imu_msg = imus[i];

                double t = imu_msg->header.stamp.toSec();

                double acc_x = imu_msg->linear_acceleration.x;
                double acc_y = imu_msg->linear_acceleration.y;
                double acc_z = imu_msg->linear_acceleration.z;

                double gyr_x = imu_msg->angular_velocity.x;
                double gyr_y = imu_msg->angular_velocity.y;
                double gyr_z = imu_msg->angular_velocity.z;

                if (current_time < 0)
                {
                    current_time = t;

                    dx = acc_x;
                    dy = acc_y;
                    dz = acc_z;

                    rx = gyr_x;
                    ry = gyr_y;
                    rz = gyr_z;

                    estimator.processIMU(
                        0.0,
                        Vector3d(dx, dy, dz),
                        Vector3d(rx, ry, rz));

                    continue;
                }

                if (t <= img_t)
                {
                    double dt = t - current_time;

                    if (dt < 0)
                    {
                        ROS_WARN("negative dt in imu processing");
                        continue;
                    }

                    current_time = t;

                    dx = acc_x;
                    dy = acc_y;
                    dz = acc_z;

                    rx = gyr_x;
                    ry = gyr_y;
                    rz = gyr_z;

                    estimator.processIMU(
                        dt,
                        Vector3d(dx, dy, dz),
                        Vector3d(rx, ry, rz));
                }
                else
                {
                    /*
                     * 当前 IMU 时间超过图像时间，需要插值到 img_t。
                     *
                     * 上一时刻 IMU: dx dy dz rx ry rz
                     * 当前 IMU: acc_x acc_y acc_z gyr_x gyr_y gyr_z
                     */
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;

                    if (dt_1 < 0 || dt_2 < 0 || dt_1 + dt_2 <= 0)
                    {
                        ROS_WARN("bad imu interpolation dt");
                        continue;
                    }

                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);

                    double acc_x_interp = w1 * dx + w2 * acc_x;
                    double acc_y_interp = w1 * dy + w2 * acc_y;
                    double acc_z_interp = w1 * dz + w2 * acc_z;

                    double gyr_x_interp = w1 * rx + w2 * gyr_x;
                    double gyr_y_interp = w1 * ry + w2 * gyr_y;
                    double gyr_z_interp = w1 * rz + w2 * gyr_z;

                    current_time = img_t;

                    estimator.processIMU(
                        dt_1,
                        Vector3d(acc_x_interp, acc_y_interp, acc_z_interp),
                        Vector3d(gyr_x_interp, gyr_y_interp, gyr_z_interp));

                    break;
                }
            }

            /*
             * 把 feature_tracker 输出的 PointCloud 转成 VINS 后端需要的 image map。
             *
             * 原版 feature_tracker 发布格式：
             * points[i].x/y/z: 归一化坐标 x y z
             * channels[0].values[i]: feature_id * NUM_OF_CAM + camera_id
             * channels[1].values[i]: pixel u
             * channels[2].values[i]: pixel v
             * channels[3].values[i]: velocity_x
             * channels[4].values[i]: velocity_y
             */
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;

            for (size_t i = 0; i < img_msg->points.size(); i++)
            {
                if (img_msg->channels.size() < 5)
                {
                    ROS_WARN("feature message channels size < 5");
                    break;
                }

                int v = static_cast<int>(img_msg->channels[0].values[i] + 0.5);

                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;

                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;

                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];

                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];

                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x,
                                   y,
                                   z,
                                   p_u,
                                   p_v,
                                   velocity_x,
                                   velocity_y;

                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            ROS_INFO("process image at %.9f, feature size: %lu",
                     img_msg->header.stamp.toSec(),
                     image.size());

            estimator.processImage(image, img_msg->header);

            pubOdometry(img_msg->header);
        }
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_vins_estimator");
    ros::NodeHandle n("~");

    readParameters(n);

    n.param("feature_topic", FEATURE_TOPIC, string("/feature_tracker/feature"));

    estimator.setParameter();

    pub_odometry = n.advertise<nav_msgs::Odometry>("/my_vins/odom", 1000);
    pub_path = n.advertise<nav_msgs::Path>("/my_vins/path", 1000);
    path.header.frame_id = "world";

    ros::Subscriber sub_imu =
        n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());

    ros::Subscriber sub_feature =
        n.subscribe(FEATURE_TOPIC, 2000, feature_callback);

    std::thread measurement_process{process};

    ROS_INFO("my_vins_estimator node started");
    ROS_INFO("subscribe imu topic: %s", IMU_TOPIC.c_str());
    ROS_INFO("subscribe feature topic: %s", FEATURE_TOPIC.c_str());

    ros::spin();

    con.notify_all();

    if (measurement_process.joinable())
    {
        measurement_process.join();
    }

    return 0;
}