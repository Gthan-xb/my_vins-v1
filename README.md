# my_vins-v1
第一版可以跑通的vins简略版

已完成：
主干：
    processIMU()
    processImage()
    IntegrationBase
    IMUFactor
    ProjectionFactor
    Ceres optimization()
    estimator_node.cpp
    /my_vins/odom
    /my_vins/path
    RViz 轨迹显示

初始化：
    ImageFrame
    all_image_frame
    getCorresponding()
    MotionEstimator::solveRelativeRT()
    relativePose()
    GlobalSFM
    initialStructure()
    VisualIMUAlignment()
    visualInitialAlign()
    INITIAL / NON_LINEAR 状态切换

滑窗：
    compensatedParallax2()
    MARGIN_OLD
    MARGIN_SECOND_NEW
    removeBack()
    removeFront()

稳定性：
    inverse depth 保护
    failureDetection()
    FeatureManager 深度向量边界保护
    固定 para_Pose[0] 保持 gauge 稳定

边缘化：
    ResidualBlockInfo
    MarginalizationInfo
    MarginalizationFactor
    last_marginalization_info
    last_marginalization_parameter_blocks
    MARGIN_OLD prior
    MARGIN_SECOND_NEW prior
    prior 已能加入下一轮优化

评估问题：
    Original VINS / My VINS / EuRoC GT 已完成时间同步和曲线对比，问题每当物体往一个方向运动到局部最远点（波峰或波谷），并开始往反方向运动（转向）时，误差就会达到最大


相对于原版没有做的：

1. 缺少ProjectionTdFactor

2. td固定，缺少时间偏移 td 在线估计

3. 缺少外参在线估计，固定tic / ric 

4. 缺少原版完整失败检测与重启逻辑你目前有基础版：

            
        有基础版failureDetection()
        
        但没有完整原版的：
        
        failure_occur 标志
        relocalization / restart 流程
        节点层 restart 通信
        完整异常状态恢复

5. 边缘化实现仍需要继续验证
        重点风险点包括：
        
        MARGIN_SECOND_NEW 的 addr_shift
        drop_set 是否完全正确
        保留下来的 parameter block 顺序
        pose global size 7 / local size 6 映射
        prior 的 gauge 处理
        
        已经通过实验确认：
        
        释放 para_Pose[0] 后误差明显增大；
        持续固定 para_Pose[0] 更稳定。

6. 原版更完整的特征管理细节
       已经有：
      
      addFeatureCheckParallax()
      compensatedParallax2()
      removeBack()
      removeFront()
      
      但需要继续确认是否和原版完全一致，尤其是：
      
      feature 的 start_frame 更新
      MARGIN_SECOND_NEW 删除观测后索引对应
      triangulate() 的深度筛选
      solve_flag 生命周期
      removeFailures() 时机

7. 初始化细节仍是简化版
        已经实现：
        
        relativePose()
        GlobalSFM
        VisualIMUAlignment()
        
        但solveRelativeRT() 用 OpenCV 简化了原版内部流程，且初始化成功后的局部状态质量可能仍和原版不同。
        
        初始化误差会影响后续：
        
        初始 bias
        初始重力方向
        初始尺度
        初始特征深度

8. 没有回环和重定位
        当前没有：
        
        loop closure
        pose graph optimization
        relocalization
        
        影响是：
        
        长期累计漂移无法被全局纠正
