import cv2
import numpy as np
import time
import threading
import onnxruntime as ort

# ────────────────────────────────────────────────────────────────
# USER-ADJUSTABLE OPTIONS:
CONF_THRESHOLD = 0.65             # Confidence threshold for detection
MODEL_INPUT_SIZE = (640, 640)     # Model input size for the ONNX model
DISPLAY_RESOLUTION = (1280, 720)   # Camera/display resolution (e.g., 720p)
FPS = 30                          # Desired frame rate
ROBOT_STOP_SLEEP_TIME = 3         # Seconds to wait after issuing a stop command
# Nominal durations for trajectories (in seconds)
NOMINAL_DURATION = 5.0            # For "normal" motion (robot takes 5 sec)
SLOW_DURATION = 10.0              # For "slow" motion (robot takes 10 sec)
# Control loop update period
CONTROL_LOOP_DT = 0.05
# Detection buffer time (in seconds): how long to keep a slow or stop flag active
DETECTION_BUFFER_TIME = 2
# ────────────────────────────────────────────────────────────────

# Global shared variables (protected by locks)
# detection_command can be "normal", "slow", or "stop"
detection_command = "normal"
detection_lock = threading.Lock()

# Global variables for detection time stamps
last_stop_detection_time = None
last_slow_detection_time = None

# ──────────────── YOLOv8n Inference Functions ────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(frame, target_size=MODEL_INPUT_SIZE):
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(transposed, axis=0)
    return input_tensor

def postprocess(output, conf_threshold=CONF_THRESHOLD):
    output = np.squeeze(output, axis=0).transpose(1, 0)  # Shape: (8400, 84)
    boxes_center = output[:, :4]
    cls_logits = output[:, 4:]
    cls_probs = sigmoid(cls_logits)
    scores = np.max(cls_probs, axis=1)
    cls_ids = np.argmax(cls_probs, axis=1)
    mask = (scores > conf_threshold) & (cls_ids == 0)
    filtered_boxes = boxes_center[mask]
    filtered_scores = scores[mask]
    return filtered_boxes, filtered_scores

def convert_to_xyxy(boxes_center):
    boxes_xyxy = np.empty_like(boxes_center)
    boxes_xyxy[:, 0] = boxes_center[:, 0] - boxes_center[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_center[:, 1] - boxes_center[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_center[:, 0] + boxes_center[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_center[:, 1] + boxes_center[:, 3] / 2  # y2
    return boxes_xyxy

# ──────────────── ROS Robot Control Code ────────────────
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState  # For subscribing to joint states

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')
        self.joint_publisher = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        self.joint_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_positions = None  # Latest joint positions from the robot.
        self.robot_stopped = False
        self.current_duration = NOMINAL_DURATION

    def joint_state_callback(self, msg: JointState):
        self.current_joint_positions = list(msg.position)

    def stop_robot(self):
        """
        Immediately stop the robot by publishing a JointTrajectory with zero positions.
        """
        traj = JointTrajectory()
        traj.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        point = JointTrajectoryPoint()
        point.positions = [0.0] * 6
        point.time_from_start.sec = 0  # Immediate effect
        traj.points.append(point)
        self.joint_publisher.publish(traj)
        self.robot_stopped = True
        self.current_duration = float('inf')
        self.get_logger().info("Detection switched to STOP; stopping robot immediately.")

    def move_to_position(self, positions, duration):
        """
        Publish a trajectory to move the robot to the given joint positions over the specified duration.
        """
        traj = JointTrajectory()
        traj.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        traj.points.append(point)
        self.joint_publisher.publish(traj)
        self.robot_stopped = False
        self.current_duration = duration
        self.get_logger().info(f"Moving to {positions} over {duration:.1f} sec.")

    def get_joint_positions(self):
        if self.current_joint_positions is not None:
            return self.current_joint_positions
        return [0.0] * 6

def move_to_position_with_interrupt(robot_control_node, positions, duration, initial_mode):
    """
    Issue a movement command and monitor for a change in detection mode.
    If the mode changes:
      - to STOP, interrupt the motion and stop the robot.
      - between NORMAL and SLOW, immediately cancel the current command and
        let the control loop reissue a new command from the current state.
    In either case, the function returns False to signal an interruption.
    If the move completes without a mode change, return True.
    """
    robot_control_node.move_to_position(positions, duration)
    start_time = time.time()
    while time.time() - start_time < duration:
        with detection_lock:
            if detection_command != initial_mode:
                if detection_command == "stop":
                    robot_control_node.get_logger().info("Detection switched to STOP during motion; stopping robot.")
                    robot_control_node.stop_robot()
                    return False
                else:
                    robot_control_node.get_logger().info(
                        "Detection mode changed (normal <-> slow) during motion; reissuing command immediately without stopping."
                    )
                    return False
        time.sleep(0.1)
    return True

# ──────────────── Robot Control Loop (Zone Based) ────────────────
def robot_control_loop():
    global detection_command, robot_control_node

    # Predefined trajectories.
    home_position = [-1.932, -1.858, -2.363, -0.411, 1.440, 0.030]
    target_position = [-0.698, -1.975, -1.571, -0.413, 1.457, 0.030]
    current_target = home_position

    while rclpy.ok():
        with detection_lock:
            current_cmd = detection_command

        if current_cmd == "stop":
            if not robot_control_node.robot_stopped:
                robot_control_node.stop_robot()
            time.sleep(ROBOT_STOP_SLEEP_TIME)
            continue

        # Use fixed durations based on the detection command.
        duration = SLOW_DURATION if current_cmd == "slow" else NOMINAL_DURATION

        move_success = move_to_position_with_interrupt(robot_control_node, current_target, duration, current_cmd)
        if move_success:
            # Toggle between predefined targets after a successful move.
            current_target = target_position if current_target == home_position else home_position
        time.sleep(CONTROL_LOOP_DT)

# ──────────────── Main Function: Detection + ROS 2 ────────────────
def main():
    global detection_command, robot_control_node, last_stop_detection_time, last_slow_detection_time

    # Set up the ONNX Runtime session.
    onnx_model_path = "yolov8n.onnx"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_model_path,
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Open the camera.
    cap = cv2.VideoCapture('/dev/video0')
    if not cap.isOpened():
        print("Error: Could not open camera /dev/video0.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return
    orig_h, orig_w = frame.shape[:2]
    left_zone = orig_w // 4
    right_zone = 3 * orig_w // 4


    print("Starting live feed. Press 'q' to quit.")

    import rclpy
    rclpy.init(args=None)
    global robot_control_node
    robot_control_node = RobotControlNode()
    threading.Thread(target=rclpy.spin, args=(robot_control_node,), daemon=True).start()
    threading.Thread(target=robot_control_loop, daemon=True).start()

    # Main loop: perform detection and display live feed.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame, target_size=MODEL_INPUT_SIZE)
        start_inf = time.time()
        output = session.run(None, {input_name: input_tensor})[0]
        inf_time = (time.time() - start_inf) * 1000  # Inference time in ms

        boxes_center, scores = postprocess(output, conf_threshold=CONF_THRESHOLD)
        boxes_xyxy = None
        local_det_cmd = "normal"  # Default command.
        current_time = time.time()

        if boxes_center.shape[0] > 0:
            boxes_xyxy = convert_to_xyxy(boxes_center)
            scale_x = orig_w / MODEL_INPUT_SIZE[0]
            scale_y = orig_h / MODEL_INPUT_SIZE[1]
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y

            for box in boxes_xyxy:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x_center = (x1 + x2) / 2
                if left_zone <= x_center <= right_zone:
                    local_det_cmd = "stop"
                    last_stop_detection_time = current_time
                    break
                else:
                    local_det_cmd = "slow"
                    last_slow_detection_time = current_time


        if last_stop_detection_time is not None and (current_time - last_stop_detection_time) < DETECTION_BUFFER_TIME:
            local_det_cmd = "stop"
        elif last_slow_detection_time is not None and (current_time - last_slow_detection_time) < DETECTION_BUFFER_TIME:
            if local_det_cmd != "stop":
                local_det_cmd = "slow"
        else:
            local_det_cmd = "normal"

        with detection_lock:
            detection_command = local_det_cmd

        print(f"Detection Command: {detection_command}")

        cv2.putText(frame, f"Inference: {inf_time:.1f} ms", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.line(frame, (left_zone, 0), (left_zone, orig_h), (255, 0, 0), 2)
        cv2.line(frame, (right_zone, 0), (right_zone, orig_h), (255, 0, 0), 2)
        cv2.imshow("Live Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
