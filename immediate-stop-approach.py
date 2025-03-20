import cv2
import numpy as np
import time
import threading
import onnxruntime as ort

# ────────────────────────────────────────────────────────────────
# USER-ADJUSTABLE OPTIONS:
CONF_THRESHOLD = 0.65           # Confidence threshold for detection
MODEL_INPUT_SIZE = (640, 640)   # Model input size (width, height)
DISPLAY_RESOLUTION = (1280, 720)  # Display/capture resolution (e.g., 720p)
FPS = 30                        # Desired frame rate
ROBOT_STOP_SLEEP_TIME = 3      # Time (in seconds) the robot will halt when a person is detected
# ────────────────────────────────────────────────────────────────

# Global variables to share detection status with the robot control thread.
human_detected_flag = False
human_detected_lock = threading.Lock()
robot_control_node = None  # This will be initialized later

# ──────────────── YOLOv8n Inference Functions ────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(frame, target_size=MODEL_INPUT_SIZE):
    """
    Resize the frame to the target size, convert BGR to RGB,
    normalize pixel values to [0,1], and reformat to (1, C, H, W).
    """
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(transposed, axis=0)
    return input_tensor

def postprocess(output, conf_threshold=CONF_THRESHOLD):
    """
    Process the raw model output:
      - Assumes output shape (1, 84, 8400):
          • The first 4 values per candidate are the box parameters in center format (cx, cy, w, h)
          • The next 80 values are the class logits.
      - Applies a sigmoid to the class logits.
      - Filters detections for the person class (assumed class 0) with confidence above threshold.
    Returns:
      - filtered boxes (in center format)
      - filtered scores.
    """
    output = np.squeeze(output, axis=0).transpose(1, 0)  # Shape becomes (8400, 84)
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
    """
    Convert boxes from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2].
    """
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

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')
        self.joint_publisher = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        self.robot_stopped = False

    def stop_robot(self):
        """
        Send a zero-position command to stop the robot immediately.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        point = JointTrajectoryPoint()
        point.positions = [0.0] * 6
        point.time_from_start.sec = 0  # Immediate stop
        trajectory.points.append(point)
        self.joint_publisher.publish(trajectory)
        self.robot_stopped = True
        self.get_logger().info("Human detected! Stopping robot immediately.")

    def move_to_position(self, positions, duration):
        """
        Command the robot to move to the given joint positions over the specified duration.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = duration
        trajectory.points.append(point)
        self.joint_publisher.publish(trajectory)
        self.robot_stopped = False
        self.get_logger().info(f"Resuming motion: moving to {positions} over {duration} sec.")

def move_to_position_with_interrupt(robot_control_node, positions, duration):
    """
    Issue a movement command and monitor for human detection during movement.
    If a human is detected during the move, the robot is stopped immediately.
    Returns True if the movement completes successfully, or False if interrupted.
    """
    robot_control_node.move_to_position(positions, duration)
    start_time = time.time()
    while time.time() - start_time < duration:
        with human_detected_lock:
            if human_detected_flag:
                robot_control_node.get_logger().info("Human detected during movement, stopping robot immediately.")
                robot_control_node.stop_robot()
                return False
        time.sleep(0.1)
    return True

def robot_control_loop():
    """
    Continuously control the robot: if a human is detected, stop immediately and wait
    for ROBOT_STOP_SLEEP_TIME seconds. Otherwise, continue the usual movement pattern.
    """
    global human_detected_flag, robot_control_node
    # Define two example positions.
    home_position = [-1.932, -1.858, -2.363, -0.411, 1.440, 0.030]
    target_position = [-0.698, -1.975, -1.571, -0.413, 1.457, 0.030]
    current_target = home_position

    while rclpy.ok():
        with human_detected_lock:
            detected = human_detected_flag

        if detected:
            # If detected, ensure the robot is stopped and sleep.
            if not robot_control_node.robot_stopped:
                robot_control_node.get_logger().info("Human detected! Initiating immediate stop.")
                robot_control_node.stop_robot()
            time.sleep(ROBOT_STOP_SLEEP_TIME)
            continue  # Re-check detection after sleep

        # No human detected – attempt movement with interrupt monitoring.
        success = move_to_position_with_interrupt(robot_control_node, current_target, duration=5)
        if success:
            # Toggle between two positions.
            current_target = target_position if current_target == home_position else home_position
        else:
            # If movement was interrupted, wait and then try again.
            time.sleep(ROBOT_STOP_SLEEP_TIME)

# ──────────────── Main Function (Detection + ROS Control) ────────────────
def main():
    global human_detected_flag  # Declare global to update the shared variable
    # Set up the ONNX Runtime session.
    onnx_model_path = "yolov8n.onnx"  # Path to your ONNX model
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

    # Read one frame to obtain original resolution.
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return
    orig_h, orig_w = frame.shape[:2]
    # Compute scaling factors from model input to display resolution.
    scale_x = orig_w / MODEL_INPUT_SIZE[0]
    scale_y = orig_h / MODEL_INPUT_SIZE[1]

    print("Starting live feed (720p @30fps). Press 'q' to quit.")

    # ── Initialize ROS and start the robot control thread ──
    rclpy.init(args=None)
    global robot_control_node
    robot_control_node = RobotControlNode()
    # Spin the ROS node in a separate thread.
    threading.Thread(target=rclpy.spin, args=(robot_control_node,), daemon=True).start()
    # Start the robot control loop in its own thread.
    threading.Thread(target=robot_control_loop, daemon=True).start()

    # ── Main loop: perform detection and display live feed ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and run inference.
        input_tensor = preprocess(frame, target_size=MODEL_INPUT_SIZE)
        start_time = time.time()
        output = session.run(None, {input_name: input_tensor})[0]
        inference_time = (time.time() - start_time) * 1000  # in milliseconds

        # Process detections.
        boxes_center, scores = postprocess(output, conf_threshold=CONF_THRESHOLD)
        human_detected = False
        if boxes_center.shape[0] > 0:
            human_detected = True
            boxes_xyxy = convert_to_xyxy(boxes_center)
            # Scale boxes back to the display resolution.
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y
            for box, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update the shared detection flag.
        with human_detected_lock:
            human_detected_flag = human_detected

        # Overlay inference time.
        cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Live Person Detection (720p @30fps)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up.
    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

