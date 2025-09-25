import open3d as o3d
import numpy as np
import time  # 用于记录运行时间
from Parameter_matrix2 import Param
from rotate import row

def main(cloud_path, target_points_cam, visualize=False):  # 添加visualize参数控制是否可视化
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 获取外参（相机->世界的旋转矩阵R和平移向量t）
        R, t = Param.extri_param()  # R:3x3旋转矩阵，t:3x1平移向量
        
        # 加载点云
        point_cloud = o3d.io.read_point_cloud(cloud_path)
        
        if not point_cloud.has_points():
            raise ValueError("点云数据为空，请检查文件路径和内容")
        
        # 转换为NumPy数组并过滤无效点（相机坐标系）
        points_cam = np.asarray(point_cloud.points)  # 明确命名为相机坐标系下的点
        mask = ~np.isnan(points_cam).any(axis=1)
        points_cam = points_cam[mask]  # 过滤后仍为相机坐标系
        
        if points_cam.shape[0] == 0:
            raise ValueError("清理后点云数据为空")
        
        # 确保数据类型一致
        points_cam = points_cam.astype(np.float64)
        R = R.astype(np.float64)
        t = t.astype(np.float64)  
        
        # 应用旋转和平移变换：相机坐标系 -> 世界坐标系
        points_world = (R @ points_cam.T).T + t.T  # 变换后的世界坐标系点云

        points_world = row.rotate_points(points_world,-9.4,-25,20)#-9,-24,17

        new_point_cloud = o3d.geometry.PointCloud()  # 初始化新的点云对象
        new_point_cloud.points = o3d.utility.Vector3dVector(points_world)  # 赋值世界坐标系下的点
        points_world = np.asarray(new_point_cloud.points)
        
        # 仅在可视化模式下打印坐标对比
        if visualize:
            print(f"相机坐标系前5点:\n{points_cam[:5]}")
            print(f"世界坐标系前5点:\n{points_world[:5]}")
        
        # 将目标点从相机坐标系转换到世界坐标系
        target_points_world = (R @ target_points_cam.T).T + t.T
        if visualize:
            print(f"目标点（世界坐标系）:\n{target_points_world}")
        target_points_world = row.rotate_points(target_points_world,-9.4,-25,20)
        
        # 初始化颜色数组（绿色：默认颜色）
        colors = np.tile([0.0, 1.0, 0.0], (points_world.shape[0], 1))  # 基于世界坐标系点数量
        
        # 1. 标记目标点附近的点（黄色）
        for target_point in target_points_world:
            distances = np.linalg.norm(points_world - target_point, axis=1)
            nearest_indices = np.argsort(distances)[:200]  # 最近的200个点
            # 仅在可视化模式下标记颜色
            if visualize:
                colors[nearest_indices] = [1.0, 1.0, 0.0]  # 黄色
        
        # 2. 标记蓝点：x小于目标x（世界坐标系）且在y-z范围内
        target_x, target_y, target_z = target_points_world[0]  # 世界坐标系下的目标点坐标
        x_range = [target_x - 200, target_x + 200]
        z_range = [target_z - 230, target_z + 230]
        
        # 基于世界坐标系的点计算蓝点掩码
        blue_mask = (points_world[:, 1] > target_y) & \
                   (points_world[:, 0] >= x_range[0]) & (points_world[:, 0] <= x_range[1]) & \
                   (points_world[:, 2] >= z_range[0]) & (points_world[:, 2] <= z_range[1])
        
        # 仅在可视化模式下标记颜色
        if visualize:
            colors[blue_mask] = [0.0, 0.0, 1.0]  # 蓝色
        blue_points = points_world[blue_mask]  # 世界坐标系下的蓝点
        blue_x_values = blue_points[:, 1]  # 世界坐标系下的蓝点y值
        
        # 3. 分析蓝点的最大跨度（基于世界坐标系）
        span_info = {}  # 用于存储跨度信息
        if len(blue_x_values) > 1:
            sorted_blue_x = np.sort(blue_x_values)
            x_spans = np.diff(sorted_blue_x)
            max_span_idx = np.argmax(x_spans)
            
            max_span_start = sorted_blue_x[max_span_idx]
            max_span_end = sorted_blue_x[max_span_idx + 1]
            max_span = x_spans[max_span_idx]
            
            # 基于世界坐标系查找最大跨度的起点和终点
            eps = 1e-6
            start_mask = blue_mask & np.isclose(points_world[:, 1], max_span_start, atol=eps)
            end_mask = blue_mask & np.isclose(points_world[:, 1], max_span_end, atol=eps)
            
            if np.any(start_mask) and np.any(end_mask):
                start_point = points_world[start_mask][0]  # 世界坐标系下的起点
                end_point = points_world[end_mask][0]      # 世界坐标系下的终点
                
                # 仅在可视化模式下标记颜色
                if visualize:
                    # 标记起点和终点附近的点（红色）
                    start_distances = np.linalg.norm(points_world - start_point, axis=1)
                    start_indices = np.argsort(start_distances)[:600]
                    colors[start_indices] = [1.0, 0.0, 0.0]  # 红色
                    
                    end_distances = np.linalg.norm(points_world - end_point, axis=1)
                    end_indices = np.argsort(end_distances)[:600]
                    end_indices = [i for i in end_indices if i not in start_indices]  # 避免重复标记
                    colors[end_indices] = [1.0, 0.0, 0.0]  # 红色
                
                # 始终输出跨度信息
                print(
                    f"最大跨度坐标点（世界坐标系）: "
                    f"起点({start_point[0]:.2f}, {start_point[1]:.2f}, {start_point[2]:.2f}), "
                    f"终点({end_point[0]:.2f}, {end_point[1]:.2f}, {end_point[2]:.2f})"
                )
                print(f"最大跨度大小: {max_span:.2f}")
                span_info = {
                    "start_point": start_point,
                    "end_point": end_point,
                    "max_span": max_span
                }
            else:
                print("未找到最大跨度对应的点")
        elif len(blue_x_values) == 1:
            print("只有一个蓝点，无法计算跨度")
        else:
            print("没有蓝点被标记")
        
        # 仅在可视化模式下进行可视化操作
        if visualize:
            # 更新点云颜色
            new_point_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # 可视化
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1024, height=768, window_name="点云可视化（世界坐标系）")
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
            render_option.point_size = 3.0
            render_option.line_width = 1.0
            
            vis.add_geometry(new_point_cloud)
            
            # 添加坐标轴
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1000,  # 坐标轴长度
                origin=[0, 0, 0]  # 原点位置
            )
            # vis.add_geometry(coordinate_frame)
            
            # 更新几何
            vis.update_geometry(new_point_cloud)
            
            # 运行可视化
            print("启动可视化窗口...")
            vis.run()
            vis.destroy_window()
        
        # 计算并打印运行时长
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序运行完成，总时长: {elapsed_time:.2f}秒")
        
        return span_info  # 返回跨度信息供外部使用
        
    except Exception as e:
        print(f"程序出错: {str(e)}")
        return None

if __name__ == "__main__":
    # 目标点坐标（相机坐标系下）
    target_points_cam = np.array([x,y,z])
    # 作为主程序运行时，启用可视化和颜色标记
    main(r"point_cloud_path", target_points_cam, visualize=True)
    