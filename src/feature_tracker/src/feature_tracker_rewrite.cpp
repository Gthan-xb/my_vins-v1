#include "feature_tracker.h"

int FeatureTracker::n_id = 0;


bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;

    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);

    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}


void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;

    for (int i = 0; i < int(v.size()); i++)
    {
        if (status[i])
            v[j++] = v[i];
    }

    v.resize(j);
}


void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;

    for (int i = 0; i < int(v.size()); i++)
    {
        if (status[i])
            v[j++] = v[i];
    }

    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    struct Feature
    {
        cv::Point2f pt;
        int id;
        int track_cnt;
    };

    vector<Feature> features;

    for (size_t i = 0; i < forw_pts.size(); i++)
    {
        features.push_back({forw_pts[i], ids[i], track_cnt[i]});
    }

    sort(features.begin(), features.end(),
         [](const Feature &a, const Feature &b)
         {
             return a.track_cnt > b.track_cnt;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (const auto &f : features)
    {
        int x = cvRound(f.pt.x);
        int y = cvRound(f.pt.y);

        if (x >= 0 && x < mask.cols &&
            y >= 0 && y < mask.rows &&
            mask.at<uchar>(y, x) == 255)
        {
            forw_pts.push_back(f.pt);
            ids.push_back(f.id);
            track_cnt.push_back(f.track_cnt);

            cv::circle(mask, f.pt, MIN_DIST, 0, -1);
        }
    }
}


void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}


void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;

    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
    {
        img = _img;
    }

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;

        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(
            cur_img,
            forw_img,
            cur_pts,
            forw_pts,
            status,
            err,
            cv::Size(21, 21),
            3
        );

        for (int i = 0; i < int(forw_pts.size()); i++)
        {
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        }

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &cnt : track_cnt)
        cnt++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();

        ROS_DEBUG("set mask begins");
        TicToc t_m;

        setMask();

        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;

        int need_cnt = MAX_CNT - static_cast<int>(forw_pts.size());

        n_pts.clear();

        if (need_cnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;

            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;

            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            cv::goodFeaturesToTrack(
                forw_img,
                n_pts,
                need_cnt,
                0.01,
                MIN_DIST,
                mask
            );
        }

        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;

        addPoints();

        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    prev_img = cur_img;
    prev_pts = cur_pts;

    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    cur_img = forw_img;
    cur_pts = forw_pts;

    undistortedPoints();

    ROS_DEBUG("feature tracker whole time %fms", t_r.toc());
}


void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;

        vector<cv::Point2f> un_cur_pts(cur_pts.size());
        vector<cv::Point2f> un_forw_pts(forw_pts.size());

        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;

            m_camera->liftProjective(
                Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y),
                tmp_p
            );

            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(
                Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y),
                tmp_p
            );

            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;

        cv::findFundamentalMat(
            un_cur_pts,
            un_forw_pts,
            cv::FM_RANSAC,
            F_THRESHOLD,
            0.99,
            status
        );

        int size_before = cur_pts.size();

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG(
            "FM ransac: %d -> %lu: %f",
            size_before,
            forw_pts.size(),
            1.0 * forw_pts.size() / size_before
        );

        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}


bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;

        return true;
    }
    else
    {
        return false;
    }
}


void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}


void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));

    vector<Eigen::Vector2d> distortedp;
    vector<Eigen::Vector2d> undistortedp;

    for (int i = 0; i < COL; i++)
    {
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;

            m_camera->liftProjective(a, b);

            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    }

    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);

        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;

        if (pp.at<float>(1, 0) + 300 >= 0 &&
            pp.at<float>(1, 0) + 300 < ROW + 600 &&
            pp.at<float>(0, 0) + 300 >= 0 &&
            pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(
                pp.at<float>(1, 0) + 300,
                pp.at<float>(0, 0) + 300
            ) = cur_img.at<uchar>(
                distortedp[i].y(),
                distortedp[i].x()
            );
        }
    }

    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();

    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;

        m_camera->liftProjective(a, b);

        cv::Point2f un_pt(
            b.x() / b.z(),
            b.y() / b.z()
        );

        cur_un_pts.push_back(un_pt);
        cur_un_pts_map.insert(make_pair(ids[i], un_pt));
    }

    pts_velocity.clear();

    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;

        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                auto it = prev_un_pts_map.find(ids[i]);

                if (it != prev_un_pts_map.end() && dt > 0)
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;

                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                {
                    pts_velocity.push_back(cv::Point2f(0, 0));
                }
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }

    prev_un_pts_map = cur_un_pts_map;
}