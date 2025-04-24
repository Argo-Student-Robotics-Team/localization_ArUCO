import cv2
import numpy as np
import pyrealsense2 as rs
import cv2.aruco as aruco
import json
import time
from datetime import datetime

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
PARAMS = aruco.DetectorParameters()
MARKER_LENGTH = 0.05

def setup_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.min_distance):
        depth_sensor.set_option(rs.option.min_distance, 0.2)
    if depth_sensor.supports(rs.option.max_distance):
        depth_sensor.set_option(rs.option.max_distance, 5.0)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    cam_matrix = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs[:5])
    colorizer = rs.colorizer()
    
    return pipeline, align, cam_matrix, dist_coeffs, colorizer

def get_average_depth(depth_frame, x, y, window_size=2):
    width, height = depth_frame.get_width(), depth_frame.get_height()
    depths = []
    for dy in range(-window_size, window_size + 1):
        for dx in range(-window_size, window_size + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                d = depth_frame.get_distance(nx, ny)
                if d > 0:
                    depths.append(d)
    return sum(depths)/len(depths) if depths else None

def detect_markers(color_image, depth_frame, cam_matrix, dist_coeffs):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)
    markers = []

    if ids is not None:
        aruco.drawDetectedMarkers(color_image, corners, ids)
        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        for i, marker_id in enumerate(ids):
            c = corners[i][0]
            cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
            depth = get_average_depth(depth_frame, cx, cy)
            if depth is None:
                continue
            coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
            x, y, z = coords
            label = f"ID:{marker_id[0]} X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
            cv2.putText(color_image, label, (cx - 60, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            markers.append({
                "id": int(marker_id[0]),
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "timestamp": datetime.now().isoformat()
            })
    return markers

def write_json(data, path):
    try:
        with open(path, "r") as f:
            existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}


    for marker in data:
        mid = str(marker["id"])
        if mid not in existing:
            existing[mid] = []
        existing[mid].append({
            "x": marker["x"],
            "y": marker["y"],
            "z": marker["z"],
            "timestamp": marker["timestamp"]
        })

    with open(path, "w") as f:
        json.dump(existing, f, indent=4)
    print("[JSON LOG] Dodato u", path)


def main():
    pipeline, align, cam_matrix, dist_coeffs, colorizer = setup_camera()
    json_path = "aruco_markers.json"
    last_write = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            markers = detect_markers(color_image, depth_frame, cam_matrix, dist_coeffs)

            now = time.time()
            if now - last_write >= 2.0 and markers:
                write_json(markers, json_path)
                last_write = now

            cv2.imshow("RGB + ArUco", color_image)
            cv2.imshow("DEPTH", depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
