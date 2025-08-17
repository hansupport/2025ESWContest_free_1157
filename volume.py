import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time

# --- [1] 사용자 설정: 관심 영역(ROI) 및 바닥 제거 ---
ROI_BOUNDS = {
    "min_x": -0.3,  # ROI의 왼쪽 경계
    "max_x": 0.3,   # ROI의 오른쪽 경계
    "min_y": -0.3,  # ROI의 위쪽 경계
    "max_y": 0.3,   # ROI의 아래쪽 경계
    "min_z": 0.5,   # ROI의 앞쪽(카메라에 가까운) 경계
    "max_z": 0.8    # ROI의 뒤쪽(카메라에서 먼) 경계
}
PLANE_DISTANCE_THRESHOLD = 0.01
MIN_PLANE_RATIO = 0.2

# --- CUDA 장치 설정 ---
if o3d.core.cuda.is_available():
    device = o3d.core.Device("CUDA:0")
    print("Open3D에서 CUDA를 사용합니다.")
else:
    device = o3d.core.Device("CPU:0")
    print("CUDA를 찾을 수 없어 CPU를 사용합니다.")

# --- [2] 함수 정의 영역 ---
def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def get_intrinsics(pipeline):
    profile = pipeline.get_active_profile()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    return o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )

def get_pointcloud(pipeline, intrinsic, align, filters):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        return None # 오류 발생 시 None 반환
    
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame:
        return None
        
    for f in filters:
        depth_frame = f.process(depth_frame)
        
    depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image, intrinsic, depth_scale=1000.0, depth_trunc=3.0
    )
    return pcd

# --- [3] 메인 실행 영역 ---
pipelines = []
vis = o3d.visualization.Visualizer()
is_vis_initialized = False

try:
    ctx = rs.context()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    if len(serials) < 2:
        raise RuntimeError("2대 이상의 RealSense 카메라가 필요합니다.")
    
    print("연결된 카메라:", serials)
    pipelines = [create_pipeline(s) for s in serials]
    depth_filters = [rs.decimation_filter(), rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]

    print("카메라 센서 안정화를 위해 잠시 대기합니다...")
    for _ in range(30):
        for pipe in pipelines:
            pipe.wait_for_frames()
    print("안정화 완료.")

    intrinsics = [get_intrinsics(p) for p in pipelines]
    align_to_color = rs.align(rs.stream.color)
    transform_matrix_cpu = np.load('transform_matrix.npy')
    transform_matrix_gpu = o3d.core.Tensor(transform_matrix_cpu, o3d.core.float64, device)
    
    print(f"transform_matrix.npy 파일을 성공적으로 불러왔습니다.")

    is_vis_initialized = vis.create_window(window_name="Real-time Size Estimation (3D BBox)", width=1280, height=720)
    if not is_vis_initialized:
        print("[경고] Open3D 시각화 창을 생성하지 못했습니다.")


    min_bound_cpu = [ROI_BOUNDS["min_x"], ROI_BOUNDS["min_y"], ROI_BOUNDS["min_z"]]
    max_bound_cpu = [ROI_BOUNDS["max_x"], ROI_BOUNDS["max_y"], ROI_BOUNDS["max_z"]]
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound_cpu, max_bound=max_bound_cpu)
    roi_bbox.color = (0, 1, 0) # ROI는 초록색

    roi_bbox_t = o3d.t.geometry.AxisAlignedBoundingBox.from_legacy(roi_bbox, device=device)

    while True:
        pcd1_cpu = get_pointcloud(pipelines[0], intrinsics[0], align_to_color, depth_filters)
        pcd2_cpu = get_pointcloud(pipelines[1], intrinsics[1], align_to_color, depth_filters)

        if is_vis_initialized:
            vis.clear_geometries()
            vis.add_geometry(roi_bbox)

        if pcd1_cpu is None or pcd2_cpu is None or pcd1_cpu.is_empty() or pcd2_cpu.is_empty():
            if is_vis_initialized and not vis.poll_events(): break
            if is_vis_initialized: vis.update_renderer()
            continue
        
        pcd1_t = o3d.t.geometry.PointCloud.from_legacy(pcd1_cpu, device=device)
        pcd2_t = o3d.t.geometry.PointCloud.from_legacy(pcd2_cpu, device=device)
        
        pcd2_t.transform(transform_matrix_gpu)
        merged_t = pcd1_t + pcd2_t
        pcd_cropped_t = merged_t.crop(roi_bbox_t)
        
        object_detected = False
        if len(pcd_cropped_t.point["positions"]) > 0:
            
            # 반복적 평면 제거 로직
            remaining_pcd_t = pcd_cropped_t
            min_points_in_plane = int(len(remaining_pcd_t.point["positions"]) * MIN_PLANE_RATIO)

            while len(remaining_pcd_t.point["positions"]) > min_points_in_plane:
                try:
                    _, inliers = remaining_pcd_t.segment_plane(
                        distance_threshold=PLANE_DISTANCE_THRESHOLD,
                        ransac_n=3,
                        num_iterations=100
                    )
                    if len(inliers) < min_points_in_plane:
                        break
                    remaining_pcd_t = remaining_pcd_t.select_by_index(inliers, invert=True)
                except Exception:
                    break
            object_pcd_t = remaining_pcd_t

            if len(object_pcd_t.point["positions"]) > 0:
                labels = object_pcd_t.cluster_dbscan(eps=0.02, min_points=10, print_progress=False)
                counts = np.bincount(labels.cpu().numpy()[labels.cpu().numpy() >= 0])
                if len(counts) > 0:
                    largest_cluster_label = counts.argmax()
                    
                    indices_np = np.where(labels.cpu().numpy() == largest_cluster_label)[0]
                    indices_t = o3d.core.Tensor(indices_np, dtype=o3d.core.int64, device=device)
                    final_object_pcd_t = object_pcd_t.select_by_index(indices_t)
                    
                    if len(final_object_pcd_t.point["positions"]) > 0:
                        final_object_pcd_cpu = final_object_pcd_t.to_legacy()
                        
                        # 3D 바운딩 박스 계산
                        oriented_bbox = final_object_pcd_cpu.get_oriented_bounding_box()
                        dims_sorted = sorted(oriented_bbox.extent, reverse=True)
                        length, width, height = dims_sorted[0], dims_sorted[1], dims_sorted[2]
                        print(f"\r📏 최종 물체 크기 (m): L={length:.3f}, W={width:.3f}, H={height:.3f}", end="")
                        
                        oriented_bbox.color = (1, 0, 0) # 바운딩 박스는 빨간색
                        
                        if is_vis_initialized:
                            vis.add_geometry(final_object_pcd_cpu)
                            vis.add_geometry(oriented_bbox)
                        object_detected = True
        
        if not object_detected:
            print("\r" + " " * 80, end="")

        if is_vis_initialized and not vis.poll_events():
            break
        if is_vis_initialized: vis.update_renderer()

except (KeyboardInterrupt, SystemExit):
    print("\n프로그램을 종료합니다.")
except Exception as e:
    import traceback
    print(f"\n오류 발생: {e}")
    traceback.print_exc()
finally:
    for pipe in pipelines:
        pipe.stop()
    if is_vis_initialized:
        vis.destroy_window()
    print("\n안전하게 종료되었습니다.")