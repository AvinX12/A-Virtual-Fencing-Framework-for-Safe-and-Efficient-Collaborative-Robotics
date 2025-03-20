import cv2
import numpy as np
import time
import onnxruntime as ort

# ────────────────────────────────────────────────────────────────
# USER-ADJUSTABLE OPTIONS:
CONF_THRESHOLD = 0.65          # Confidence threshold for detection
MODEL_INPUT_SIZE = (640, 640)  # Model input size (width, height)
DISPLAY_RESOLUTION = (1280, 720)  # Display/capture resolution (720p)
FPS = 30                       # Desired frame rate
# ────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(frame, target_size=MODEL_INPUT_SIZE):
    """
    Resize the frame to the target size, convert BGR to RGB,
    normalize pixel values to [0,1], and reformat to (1, C, H, W).
    """
    # Resize for inference (this may distort the image)
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
      - filtered boxes (still in center format)
      - filtered scores.
    """
    # Remove the batch dimension and transpose to shape (8400, 84)
    output = np.squeeze(output, axis=0).transpose(1, 0)
    boxes_center = output[:, :4]
    cls_logits = output[:, 4:]
    cls_probs = sigmoid(cls_logits)
    scores = np.max(cls_probs, axis=1)
    cls_ids = np.argmax(cls_probs, axis=1)
    # Filter: only keep detections with score > threshold and class 0 (person)
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

def main():
    # Set up the ONNX Runtime session (with GPU if available).
    onnx_model_path = "yolov8n.onnx"  # Path to your ONNX model
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_model_path,
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Open the camera using a simple method.
    cap = cv2.VideoCapture('/dev/video0')
    if not cap.isOpened():
        print("Error: Could not open camera /dev/video0.")
        return

    # Set capture resolution and frame rate.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Read one frame to determine original resolution.
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return
    orig_h, orig_w = frame.shape[:2]
    # Compute scale factors from model input to original resolution.
    scale_x = orig_w / MODEL_INPUT_SIZE[0]  # 1280 / 640 = 2.0
    scale_y = orig_h / MODEL_INPUT_SIZE[1]  # 720 / 640 = 1.125

    print("Starting live feed (720p @30fps). Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For inference, resize a copy to 640x640.
        input_tensor = preprocess(frame, target_size=MODEL_INPUT_SIZE)
        start_time = time.time()
        output = session.run(None, {input_name: input_tensor})[0]
        inference_time = (time.time() - start_time) * 1000  # in ms

        # Process detections.
        boxes_center, scores = postprocess(output, conf_threshold=CONF_THRESHOLD)
        if boxes_center.shape[0] > 0:
            boxes_xyxy = convert_to_xyxy(boxes_center)
            # Scale detection boxes from 640x640 input back to 1280x720 display.
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y
            for box, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Overlay inference time.
        cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Live Person Detection (720p @30fps)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

