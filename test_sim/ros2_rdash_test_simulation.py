#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Author: Vamsi Karnam
# Description: Dummy robot ROS2 node simulator for smoke test 
# -----------------------------------------------------------------------------
import math, random, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, UInt8MultiArray, String
from sensor_msgs.msg import Image, CompressedImage, Range, LaserScan
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time as RosTime

import numpy as np
import cv2

BEST = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                  durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)

def now():
    t = time.time()
    sec = int(t)
    nsec = int((t-sec)*1e9)
    rt = RosTime()
    rt.sec = sec; rt.nanosec = nsec
    return rt

class DroneNode(Node):
    """
    Node 1: camera (compressed), speed, altitude (drone profile)
    Altitude: climb to 10m at 8 km/h (~2.22 m/s), then hover.
    Speed: noisy around commanded (2.22 m/s during climb, then ~0).
    """
    def __init__(self):
        super().__init__('node1_drone')
        self.pub_cam = self.create_publisher(CompressedImage, '/drone/camera/image/compressed', BEST)
        self.pub_speed = self.create_publisher(Float32, '/drone/speed_mps', RELIABLE)
        self.pub_alt = self.create_publisher(Float32, '/drone/altitude_m', RELIABLE)
        # NEW: text log publisher
        self.pub_log = self.create_publisher(String, '/drone/log', BEST)
        # NEW: battery voltage (mV)
        self.pub_batt = self.create_publisher(Float32, '/power/battery_mv', RELIABLE)

        self.timer = self.create_timer(1.0/15.0, self._tick)  # 15 FPS camera
        # NEW: log timer (~2 Hz)
        self.log_timer = self.create_timer(0.5, self._tick_log)
        # NEW: battery publish timer (1 Hz; value changes every 10 s)
        self.batt_timer = self.create_timer(1.0, self._tick_battery)

        self.start = time.time()
        self.alt = 0.0
        self.width, self.height = 640, 360
        self._last_speed = 0.0

        # --- Battery model params (mV) ---
        self._batt_max_mv = 25.0
        self._batt_min_mv = 1.0
        self._batt_step_sec = 10.0  # change by 1 mV every 10 s
        # current value derived from elapsed time; keep a base for reproducibility
        self._batt_start_time = self.start

    def _tick(self):
        t = time.time() - self.start
        climb_rate = 8.0/3.6  # 8 km/h in m/s
        if self.alt < 10.0:
            self.alt = min(10.0, self.alt + climb_rate * (1.0/15.0))
            speed = climb_rate + random.gauss(0, 0.1)
        else:
            speed = random.gauss(0.0, 0.05)  # hover wiggle
        self._last_speed = float(speed)

        # Publish altitude/speed
        self.pub_alt.publish(Float32(data=float(self.alt)))
        self.pub_speed.publish(Float32(data=float(speed)))

        # Synthetic camera frame with overlay
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(img, f"ALT {self.alt:4.1f} m  SPD {speed:3.2f} m/s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # moving dot
        x = int((math.sin(t*0.8)*0.5+0.5) * (self.width-20)) + 10
        y = int((math.cos(t*0.6)*0.5+0.5) * (self.height-20)) + 10
        cv2.circle(img, (x,y), 10, (0,255,0), -1)

        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok: return
        msg = CompressedImage()
        msg.header.stamp = now()
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.pub_cam.publish(msg)

    # NEW: emit textual status lines
    def _tick_log(self):
        msg = String()
        msg.data = f"[DRONE] alt={self.alt:4.1f} m, speed={self._last_speed:3.2f} m/s"
        self.pub_log.publish(msg)

    # NEW: battery triangular waveform between 25 mV and 1 mV, step 1 mV every 10 s
    def _tick_battery(self):
        elapsed = time.time() - self._batt_start_time
        steps_down = int(self._batt_max_mv - self._batt_min_mv)  # 24 steps from 25 -> 1
        period_steps = steps_down * 2                            # down+up -> 48 steps
        step = int(elapsed // self._batt_step_sec) % max(1, period_steps)

        if step < steps_down:
            # descending: 25,24,...,1
            val_mv = self._batt_max_mv - step
        else:
            # ascending: 1,2,...,25
            val_mv = self._batt_min_mv + (step - steps_down)

        self.pub_batt.publish(Float32(data=float(val_mv)))

class AudioDepthSonarNode(Node):
    """
    Node 2: audio, depth image, sonar
    """
    def __init__(self):
        super().__init__('node2_audio_depth_sonar')
        self.pub_audio = self.create_publisher(UInt8MultiArray, '/node2/audio/raw', BEST)
        self.pub_depth = self.create_publisher(Image, '/node2/depth/image', RELIABLE)
        self.pub_sonar = self.create_publisher(Range, '/node2/sonar/range', RELIABLE)
        # NEW: text log publisher
        self.pub_log = self.create_publisher(String, '/node2/log', BEST)

        self.audio_timer = self.create_timer(1.0/20.0, self._tick_audio)  # 20 Hz audio chunks
        self.depth_timer = self.create_timer(1.0/5.0, self._tick_depth)   # 5 Hz depth frames
        self.sonar_timer = self.create_timer(1.0/10.0, self._tick_sonar)  # 10 Hz sonar
        # NEW: log timer (~2 Hz)
        self.log_timer = self.create_timer(0.5, self._tick_log)

        self.depth_w, self.depth_h = 320, 240
        self.sonar_phase = 0.0
        self._last_sonar = 0.0

    def _tick_audio(self):
        # 20 Hz of ~20ms mono PCM-ish bytes (just simulated noise)
        n_samples = 320  # 16 kHz * 0.02 sec
        noise = (np.random.randn(n_samples) * 8000).astype(np.int16).tobytes()
        arr = UInt8MultiArray()
        arr.data = list(noise)  # ByteMultiArray/UInt8MultiArray both supported
        self.pub_audio.publish(arr)

    def _tick_depth(self):
        # 16UC1 gradient with noise
        z = np.linspace(500, 2000, self.depth_w, dtype=np.uint16)  # 0.5m..2.0m mm units
        frame = np.tile(z, (self.depth_h,1))
        frame = frame + (np.random.randn(self.depth_h, self.depth_w)*10).astype(np.int16)
        frame = np.clip(frame, 0, 4095).astype(np.uint16)

        msg = Image()
        msg.header.stamp = now()
        msg.height = self.depth_h
        msg.width = self.depth_w
        msg.encoding = '16UC1'
        msg.is_bigendian = 0
        msg.step = self.depth_w * 2
        msg.data = frame.tobytes()
        self.pub_depth.publish(msg)

    def _tick_sonar(self):
        self.sonar_phase += 0.1
        rng = 2.0 + 0.5*math.sin(self.sonar_phase) + random.uniform(-0.05, 0.05)  # 2.0±~0.55 m
        msg = Range()
        msg.header.stamp = now()
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = math.radians(20.0)
        msg.min_range = 0.2
        msg.max_range = 5.0
        msg.range = float(max(msg.min_range, min(msg.max_range, rng)))
        self._last_sonar = float(msg.range)
        self.pub_sonar.publish(msg)

    # NEW: emit textual status lines
    def _tick_log(self):
        msg = String()
        msg.data = f"[NODE2] sonar={self._last_sonar:0.2f} m; depth=16UC1 {self.depth_w}x{self.depth_h}"
        self.pub_log.publish(msg)

class LidarCameraNode(Node):
    """
    Node 3: lidar (LaserScan), raw camera (Image)
    """
    def __init__(self):
        super().__init__('node3_lidar_camera')
        self.pub_scan = self.create_publisher(LaserScan, '/robot/lidar/scan', RELIABLE)
        self.pub_cam_raw = self.create_publisher(Image, '/robot/camera/image', BEST)
        # NEW: text log publisher
        self.pub_log = self.create_publisher(String, '/robot/log', BEST)

        self.timer_scan = self.create_timer(1.0/12.0, self._tick_scan)  # 12 Hz
        self.timer_cam = self.create_timer(1.0/10.0, self._tick_cam)    # 10 FPS
        # NEW: log timer (~3 Hz to vary)
        self.log_timer = self.create_timer(1.0/3.0, self._tick_log)

        self.w, self.h = 320, 240
        self.phase = 0.0
        self._last_scan_base = 0.0

    def _tick_scan(self):
        self.phase += 0.15
        n = 360
        scan = LaserScan()
        scan.header.stamp = now()
        scan.angle_min = -math.pi
        scan.angle_max =  math.pi
        scan.angle_increment = (scan.angle_max - scan.angle_min)/n
        scan.range_min = 0.2
        scan.range_max = 15.0
        base = 6.0 + 1.0*math.sin(self.phase)
        self._last_scan_base = float(base)
        ranges = [max(scan.range_min,
                      min(scan.range_max,
                          base + 0.3*math.sin(self.phase + i*0.05) + random.uniform(-0.05,0.05)))
                  for i in range(n)]
        scan.ranges = ranges
        self.pub_scan.publish(scan)

    def _tick_cam(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        t = time.time()
        cv2.putText(img, "RAW CAM", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        x = int((math.sin(t)*0.5+0.5)*(self.w-1))
        img[:, x:x+2, :] = 255
        msg = Image()
        msg.header.stamp = now()
        msg.height, msg.width = self.h, self.w
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = self.w*3
        msg.data = img.tobytes()
        self.pub_cam_raw.publish(msg)

    # NEW: emit textual status lines
    def _tick_log(self):
        msg = String()
        msg.data = f"[LIDAR] base_range~{self._last_scan_base:0.2f} m; raw_cam={self.w}x{self.h}"
        self.pub_log.publish(msg)

class TfNode(Node):
    """
    Publishes a tiny TF tree so the dashboard can show edges.
    base_link -> {camera_link, lidar_link, sonar_link}
    """
    def __init__(self):
        super().__init__('tf_broadcaster')
        self.pub_tf = self.create_publisher(TFMessage, '/tf', RELIABLE)
        self.timer = self.create_timer(0.5, self._tick)

    def _tick(self):
        tfs = []
        def mk(parent, child, x, y, z):
            ts = TransformStamped()
            ts.header.stamp = now()
            ts.header.frame_id = parent
            ts.child_frame_id = child
            ts.transform.translation.x = float(x)
            ts.transform.translation.y = float(y)
            ts.transform.translation.z = float(z)
            ts.transform.rotation.w = 1.0
            return ts
        tfs.append(mk('base_link', 'camera_link', 0.2, 0.0, 0.3))
        tfs.append(mk('base_link', 'lidar_link',  0.0, 0.0, 0.6))
        tfs.append(mk('base_link', 'sonar_link',  0.1, 0.0, -0.1))
        self.pub_tf.publish(TFMessage(transforms=tfs))

def main():
    rclpy.init()
    execu = rclpy.executors.MultiThreadedExecutor()
    nodes = [DroneNode(), AudioDepthSonarNode(), LidarCameraNode(), TfNode()]
    for n in nodes:
        execu.add_node(n)
    try:
        execu.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
