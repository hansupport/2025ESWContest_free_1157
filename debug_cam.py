import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time
import cv2
from collections import deque

# --- [1] 사용자 설정 ---
# --- [1.1] 최종 디버깅을 위한 설정 ---
# 0: 정상 작동 (두 카메라 병합)
# 1: 1번 카메라만 보기
# 2: 2번 카메라만 보기
DEBUG_CAMERA_INDEX = 2

ROI_BOUNDS = { "min_x": -0.2, "max_x": 0.2, "min_y": -0.2, "max_y": 0.2, "min_z": 0.5, "max_z": 0.9 }
THRESHOLD_FILTER_MAX_DISTANCE = 1.2
PLANE_DISTANCE_THRESHOLD = 0.01
MIN_PLANE_RATIO = 0.2
STAT_OUTLIER_NEIGHBORS = 20
STAT_OUTLIER_STD_RATIO = 2.0
MOVING_AVERAGE_WINDOW = 5

# --- CUDA 장치 설정 ---
if o3d.core.cuda.is_available(): device = o3d.core.Device("CUDA:0")
else: device = o3d.core.Device("CPU:0")
print(f"Using device: {device}")

# --- [2] 함수 정의 영역 (변경 없음) ---
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
    return o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

def get_pointcloud(pipeline, intrinsic, align, filters):
    try: frames = pipeline.wait_for_frames(timeout_ms=1000)
    except RuntimeError: return None
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame: return None
    threshold_filter = rs.threshold_filter(0.1, THRESHOLD_FILTER_MAX_DISTANCE)
    depth_frame = threshold_filter.process(depth_frame)
    for f in filters: depth_frame = f.process(depth_frame)
    depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic, depth_scale=1000.0, depth_trunc=THRESHOLD_FILTER_MAX_DISTANCE)
    return pcd

# --- [3] 메인 실행 영역 ---
pipelines, vis = [], o3d.visualization.Visualizer()
is_vis_initialized = False
length_buffer, width_buffer, height_buffer = deque(maxlen=MOVING_AVERAGE_WINDOW), deque(maxlen=MOVING_AVERAGE_WINDOW), deque(maxlen=MOVING_AVERAGE_WINDOW)

try:
    ctx = rs.context()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    if len(serials) < 2: raise RuntimeError("2대 이상의 RealSense 카메라가 필요합니다.")
    print("연결된 카메라:", serials)
    pipelines = [create_pipeline(s) for s in serials]
    depth_filters = [rs.decimation_filter(), rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]
    print("카메라 센서 안정화 중...")
    for _ in range(30):
        for pipe in pipelines: pipe.wait_for_frames()
    print("안정화 완료.")
    intrinsics = [get_intrinsics(p) for p in pipelines]
    align_to_color = rs.align(rs.stream.color)
    transform_matrix_cpu = np.load('transform_matrix.npy')
    print("`transform_matrix.npy` 로드 완료.")
    is_vis_initialized = vis.create_window(window_name=f"[DEBUG MODE: Cam {DEBUG_CAMERA_INDEX if DEBUG_CAMERA_INDEX != 0 else 'Merged'}]", width=1280, height=720)
    if not is_vis_initialized: print("[경고] Open3D 창 생성 실패.")
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[ROI_BOUNDS["min_x"], ROI_BOUNDS["min_y"], ROI_BOUNDS["min_z"]], max_bound=[ROI_BOUNDS["max_x"], ROI_BOUNDS["max_y"], ROI_BOUNDS["max_z"]])
    roi_bbox.color = (0, 1, 0)

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
        
        # --- [3.1] 디버깅 로직 ---
        target_pcd = o3d.geometry.PointCloud()
        if DEBUG_CAMERA_INDEX == 1:
            target_pcd = pcd1_cpu
            target_pcd.paint_uniform_color([1, 0, 0])
        elif DEBUG_CAMERA_INDEX == 2:
            target_pcd = pcd2_cpu
            target_pcd.transform(transform_matrix_cpu)
            target_pcd.paint_uniform_color([0, 0, 1])
        else: # DEBUG_CAMERA_INDEX == 0
            pcd1_cpu.paint_uniform_color([1, 0, 0])
            pcd2_cpu.transform(transform_matrix_cpu)
            pcd2_cpu.paint_uniform_color([0, 0, 1])
            target_pcd = pcd1_cpu + pcd2_cpu
        # --- 디버깅 로직 종료 ---
        
        pcd_cropped = target_pcd.crop(roi_bbox)
        
        status_message = "물체 감지 대기 중..."
        if not pcd_cropped.has_points():
            status_message = "ROI 내에 포인트 없음"
        else:
            # 측정 로직은 pcd_cropped에 대해 동일하게 수행됩니다.
            # ... (이하 측정 로직은 이전과 동일하여 생략)
            try:
                plane_model, inliers = pcd_cropped.segment_plane(PLANE_DISTANCE_THRESHOLD, 3, 1000)
                object_pcd = pcd_cropped.select_by_index(inliers, invert=True)
                if object_pcd.has_points():
                    # (이후 클러스터링, 크기 계산 로직이 여기에 위치)
                    status_message = f"처리 중... (포인트 수: {len(pcd_cropped.points)})"
                else:
                    status_message = "바닥 제외 후 남은 포인트 없음"
            except Exception:
                 status_message = "평면 검출 실패"

        print(f"\r상태: {status_message}" + " " * 20, end="")

        if is_vis_initialized:
            # 어떤 모드이든 현재 처리 중인 포인트 클라우드를 보여줌
            vis.add_geometry(pcd_cropped)
            if not vis.poll_events(): break
            vis.update_renderer()

except Exception as e:
    import traceback
    print(f"\n치명적 오류 발생: {e}")
    traceback.print_exc()
finally:
    for pipe in pipelines: pipe.stop()
    if is_vis_initialized: vis.destroy_window()
    print("\n안전하게 종료되었습니다.")