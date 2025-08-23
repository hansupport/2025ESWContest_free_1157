import pyrealsense2 as rs
import cupoch as cph
import numpy as np
import time
import os
import sys
import traceback

# --- [1] 사용자 설정: 관심 영역(ROI) 및 바닥 제거 ---
ROI_BOUNDS = {
    "min_x": -0.3,
    "max_x":  0.3,
    "min_y": -0.3,
    "max_y":  0.3,
    "min_z":  0.5,
    "max_z":  0.8
}
PLANE_DISTANCE_THRESHOLD = 0.01   # 평면 분리 임계값 [m]
MIN_PLANE_RATIO = 0.2             # inliers가 전체 중 이 비율 미만이면 평면 제거 중단
DBSCAN_EPS = 0.02                 # [m]
DBSCAN_MIN_POINTS = 10

# --- CUDA 사용 여부 출력 ---
try:
    cuda_ok = cph.utility.is_cuda_available()
except Exception:
    cuda_ok = False
print("cupoch CUDA 사용 가능:", cuda_ok)

# --- [2] RealSense 파이프라인 유틸 ---
def create_pipeline(serial):
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    return pipe

def warmup(pipelines, n=30):
    for _ in range(n):
        for p in pipelines:
            p.wait_for_frames()

def get_pointcloud_from_depth(pipeline, align, filters, depth_scale=1000.0, depth_trunc=3.0):
    """
    RealSense depth → rs.pointcloud() → numpy Nx3 → cupoch PointCloud
    """
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        return None

    frames = align.process(frames)
    depth  = frames.get_depth_frame()
    if not depth:
        return None

    for f in filters:
        depth = f.process(depth)

    # RealSense 포인트클라우드로 바로 변환
    pc = rs.pointcloud()
    points = pc.calculate(depth)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # N x 3

    # 유효 Z만 남기기
    mask = np.isfinite(vtx).all(axis=1) & (vtx[:, 2] > 0) & (vtx[:, 2] < depth_trunc)
    vtx = vtx[mask]
    if vtx.size == 0:
        return None

    pcd = cph.geometry.PointCloud()
    pcd.points = cph.utility.Vector3fVector(vtx.astype(np.float32))
    return pcd

def merge_pointclouds(pcd_a, pcd_b):
    """
    cupoch PointCloud 두 개를 병합
    """
    a = np.asarray(pcd_a.points)
    b = np.asarray(pcd_b.points)
    merged = cph.geometry.PointCloud()
    merged.points = cph.utility.Vector3fVector(
        np.vstack([a, b]).astype(np.float32)
    )
    return merged

# --- [3] 메인 실행 ---
pipelines = []
vis = None
is_vis_initialized = False

try:
    # 2대 장치 열거
    if not hasattr(rs, "context"):
        raise AttributeError("pyrealsense2에 context가 없습니다. import 경로 충돌을 먼저 해결하세요.")
    ctx = rs.context()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    if len(serials) < 2:
        raise RuntimeError("2대 이상의 RealSense 카메라가 필요합니다.")
    print("연결된 카메라:", serials)

    # 파이프라인 시작
    pipelines = [create_pipeline(s) for s in serials]

    # 필터 준비
    depth_filters = [rs.decimation_filter(), rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]

    print("카메라 센서 안정화를 위해 잠시 대기합니다...")
    warmup(pipelines, n=30)
    print("안정화 완료.")

    # 컬러 기준 정렬
    align_to_color = rs.align(rs.stream.color)

    # 외부 행렬 로드 (cam2 → cam1)
    T = np.load('transform_matrix.npy')  # 4x4
    if T.shape != (4, 4):
        raise ValueError("transform_matrix.npy는 4x4 행렬이어야 합니다.")
    T = T.astype(np.float32)

    # cupoch 시각화
    vis = cph.visualization.Visualizer()
    is_vis_initialized = vis.create_window(window_name="Real-time Size Estimation (3D BBox)", width=1280, height=720)
    if not is_vis_initialized:
        print("[경고] 시각화 창 생성 실패")

    # ROI AABB
    min_bound = np.array([ROI_BOUNDS["min_x"], ROI_BOUNDS["min_y"], ROI_BOUNDS["min_z"]], dtype=np.float32)
    max_bound = np.array([ROI_BOUNDS["max_x"], ROI_BOUNDS["max_y"], ROI_BOUNDS["max_z"]], dtype=np.float32)
    roi_bbox = cph.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    roi_bbox.color = (0.0, 1.0, 0.0)

    if is_vis_initialized:
        vis.add_geometry(roi_bbox)

    while True:
        # 두 카메라에서 포인트클라우드 획득
        pcd1 = get_pointcloud_from_depth(pipelines[0], align_to_color, depth_filters)
        pcd2 = get_pointcloud_from_depth(pipelines[1], align_to_color, depth_filters)

        if is_vis_initialized:
            vis.clear_geometries()
            vis.add_geometry(roi_bbox)

        if pcd1 is None or pcd2 is None or pcd1.is_empty() or pcd2.is_empty():
            if is_vis_initialized and not vis.poll_events():
                break
            if is_vis_initialized:
                vis.update_renderer()
            continue

        # 두 번째 카메라 포인트클라우드 정렬
        pcd2_t = cph.geometry.PointCloud(pcd2)  # 복사
        pcd2_t.transform(T)

        # 병합 및 ROI 크롭
        merged = merge_pointclouds(pcd1, pcd2_t)
        cropped = merged.crop(roi_bbox)

        object_detected = False

        # 평면 반복 제거
        n_pts = len(np.asarray(cropped.points))
        if n_pts > 0:
            remaining = cropped
            min_inliers = int(n_pts * MIN_PLANE_RATIO)

            while True:
                total_pts = len(np.asarray(remaining.points))
                if total_pts < min_inliers or total_pts == 0:
                    break
                try:
                    plane_model, inliers = remaining.segment_plane(
                        distance_threshold=PLANE_DISTANCE_THRESHOLD,
                        ransac_n=3,
                        num_iterations=100
                    )
                    # inliers가 충분히 크지 않으면 중단
                    if len(inliers) < min_inliers:
                        break
                    remaining = remaining.select_by_index(inliers, invert=True)
                except Exception:
                    break

            object_pcd = remaining

            # 클러스터링으로 최대 군집만 선택
            if not object_pcd.is_empty():
                labels = np.asarray(object_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False))
                valid = labels >= 0
                if valid.any():
                    counts = np.bincount(labels[valid])
                    largest_label = counts.argmax()
                    indices = np.where(labels == largest_label)[0].astype(np.int64)
                    final_obj = object_pcd.select_by_index(indices)

                    if not final_obj.is_empty():
                        # 바운딩 박스 계산
                        try:
                            obox = final_obj.get_oriented_bounding_box()
                            extent = np.asarray(obox.extent, dtype=np.float32)
                            length, width, height = np.sort(extent)[::-1]
                            obox.color = (1.0, 0.0, 0.0)
                        except Exception:
                            # 일부 버전에서 OBB가 없을 수 있음 → AABB로 대체
                            abox = final_obj.get_axis_aligned_bounding_box()
                            extent = np.asarray(abox.get_extent(), dtype=np.float32)
                            length, width, height = np.sort(extent)[::-1]
                            abox.color = (1.0, 0.0, 0.0)
                            obox = abox

                        print(f"\r최종 물체 크기 (m): L={length:.3f}, W={width:.3f}, H={height:.3f}", end="")

                        if is_vis_initialized:
                            vis.add_geometry(final_obj)
                            vis.add_geometry(obox)

                        object_detected = True

        if not object_detected:
            # 이전 라인 지우기용 공백
            print("\r" + " " * 80, end="")

        if is_vis_initialized and not vis.poll_events():
            break
        if is_vis_initialized:
            vis.update_renderer()

except (KeyboardInterrupt, SystemExit):
    print("\n프로그램을 종료합니다.")
except Exception as e:
    print(f"\n오류 발생: {e}")
    traceback.print_exc()
finally:
    for p in pipelines:
        try:
            p.stop()
        except Exception:
            pass
    if is_vis_initialized and vis is not None:
        vis.destroy_window()
    print("\n안전하게 종료되었습니다.")
