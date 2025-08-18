import os
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import time

# --- [0] 환경 변수 설정 ---
# GUI 창을 띄우기 위한 DISPLAY 환경 변수를 스크립트 내에서 설정합니다.
# 'GLX: Failed to find a suitable GLXFBConfig' 오류를 해결하기 위함입니다.
os.environ['DISPLAY'] = ':0'

# --- [1] 사용자 설정 부분 ---
CAMERA_INDEX = 0

# ROI(관심 영역) 경계 설정 (단위: 미터)
ROI_BOUNDS = {
    "min_x": -0.3, "max_x": 0.3,
    "min_y": -0.3, "max_y": 0.3,
    "min_z": 0.4,  "max_z": 0.9
}

# 바닥 추정 설정
FIXED_FLOOR_DISTANCE = 0.8
FLOOR_TOLERANCE = 0.015

# --- CUDA 장치 설정 ---
device = o3d.core.Device("CUDA:0") if o3d.core.cuda.is_available() else o3d.core.Device("CPU:0")
print(f"Open3D에서 {device.get_type().name}를 사용합니다.")

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
    return o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

def get_pointcloud(pipeline, intrinsic, align, filters):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        return None

    aligned_frames = align.process(frames)
    depth_frame, color_frame = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
    if not depth_frame or not color_frame: return None

    for f in filters: depth_frame = f.process(depth_frame)
    
    depth_np = np.asanyarray(depth_frame.get_data())
    color_np = np.asanyarray(color_frame.get_data())

    # BGR 이미지를 RGB로 변환하여 'Unsupported image format' 오류 해결
    color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
    
    depth_image = o3d.geometry.Image(depth_np)
    color_image = o3d.geometry.Image(color_np_rgb)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic, depth_scale=1000.0, depth_trunc=3.0)
    return pcd

# --- [3] 메인 실행 영역 ---
pipeline, vis = None, None
try:
    ctx = rs.context()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    if not serials: raise RuntimeError("RealSense 카메라가 연결되지 않았습니다.")
    if CAMERA_INDEX >= len(serials): raise RuntimeError(f"카메라 인덱스({CAMERA_INDEX})가 범위를 벗어났습니다.")
    
    selected_serial = serials[CAMERA_INDEX]
    print(f"선택된 카메라: [인덱스 {CAMERA_INDEX}] {selected_serial}")
    
    pipeline = create_pipeline(selected_serial)
    depth_filters = [rs.decimation_filter(), rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]
    
    print("카메라 센서 안정화를 위해 잠시 대기합니다...")
    for _ in range(30): pipeline.wait_for_frames()
    print("안정화 완료.")

    intrinsic = get_intrinsics(pipeline)
    align_to_color = rs.align(rs.stream.color)
    
    vis = o3d.visualization.Visualizer()
    is_vis_initialized = vis.create_window(window_name="Object Size Estimation", width=1280, height=720)
    if not is_vis_initialized:
        print("\n[중요] Open3D 시각화 창 생성에 실패했습니다. 그래픽 드라이버 상태를 확인하세요.\n")
    
    min_bound = o3d.core.Tensor([ROI_BOUNDS["min_x"], ROI_BOUNDS["min_y"], ROI_BOUNDS["min_z"]], dtype=o3d.core.Dtype.Float64, device=device)
    max_bound = o3d.core.Tensor([ROI_BOUNDS["max_x"], ROI_BOUNDS["max_y"], ROI_BOUNDS["max_z"]], dtype=o3d.core.Dtype.Float64, device=device)
    roi_bbox_t = o3d.t.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

    while True:
        pcd_cpu = get_pointcloud(pipeline, intrinsic, align_to_color, depth_filters)

        if pcd_cpu is None or pcd_cpu.is_empty():
            if is_vis_initialized and not vis.poll_events(): break
            if is_vis_initialized: vis.update_renderer()
            continue

        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_cpu, device=device)
        pcd_cropped_t = pcd_t.crop(roi_bbox_t)
        object_detected = False

        if len(pcd_cropped_t.point["positions"]) > 10:
            object_indices = o3d.core.Tensor(
                np.where(pcd_cropped_t.point["positions"][:, 2].cpu().numpy() < (FIXED_FLOOR_DISTANCE - FLOOR_TOLERANCE))[0],
                dtype=o3d.core.int64, device=device)
            object_pcd_t = pcd_cropped_t.select_by_index(object_indices)

            if len(object_pcd_t.point["positions"]) > 10:
                labels = object_pcd_t.cluster_dbscan(eps=0.02, min_points=10, print_progress=False)
                valid_labels = labels.cpu().numpy()[labels.cpu().numpy() >= 0]
                
                if len(valid_labels) > 0:
                    largest_cluster_label = np.bincount(valid_labels).argmax()
                    indices_t = o3d.core.Tensor(np.where(labels.cpu().numpy() == largest_cluster_label)[0], dtype=o3d.core.int64, device=device)
                    final_object_pcd_t = object_pcd_t.select_by_index(indices_t)

                    if len(final_object_pcd_t.point["positions"]) > 10:
                        final_object_pcd_cpu = final_object_pcd_t.to_legacy()
                        oriented_bbox = final_object_pcd_cpu.get_oriented_bounding_box()
                        dims_sorted = sorted(oriented_bbox.extent, reverse=True)
                        length, width, height = dims_sorted[0], dims_sorted[1], dims_sorted[2]
                        
                        print(f"\r📏 물체 크기 (m): X={length:.3f}, Y={width:.3f}, Z={height:.3f}", end="")
                        object_detected = True

                        if is_vis_initialized:
                            vis.clear_geometries()
                            oriented_bbox.color = (1, 0, 0)
                            vis.add_geometry(final_object_pcd_cpu)
                            vis.add_geometry(oriented_bbox)

        if not object_detected and is_vis_initialized:
            vis.clear_geometries()

        if is_vis_initialized:
            if not vis.poll_events(): break
            vis.update_renderer()
        else:
            time.sleep(0.1)

except (KeyboardInterrupt, SystemExit): print("\n프로그램을 종료합니다.")
except Exception as e:
    import traceback
    print(f"\n오류 발생: {e}")
    traceback.print_exc()
finally:
    if pipeline: pipeline.stop()
    if vis: vis.destroy_window()
    print("\n안전하게 종료되었습니다.")
