import rclpy
import cv2 as cv
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from detection_interface.msg import Detection, DetectionArray
from sensor_msgs.msg import CameraInfo

from ultralytics import YOLO

class YOLO_detector(Node):
    def __init__(self):
        super().__init__('YOLO_detector')
        self.get_logger().info(f"YOLO detector node initiated.'")
        self.load_node_parameters()

        self.color_img_sub = self.create_subscription(
            Image,
            self.color_img_input_topic,
            self.color_image_callback,
            10)

        # Publishers
        self.detection_pub = self.create_publisher(
            DetectionArray,
            self.detection_output_topic,
            10)
        
        self.processed_pub = self.create_publisher(
            Image,
            self.processed_img_output_topic,
            10)

    def load_node_parameters(self):
        # Declare and get topic name parameters
        self.declare_parameter('color_img_input_topic', 'camera/down/color/image_raw')
        self.declare_parameter('processed_img_output_topic', '/shape_detection/processed_image')
        self.declare_parameter('detection_output_topic', '/shape_detection/detections')
        self.color_img_input_topic = self.get_parameter('color_img_input_topic').get_parameter_value().string_value
        self.processed_img_output_topic = self.get_parameter('processed_img_output_topic').get_parameter_value().string_value
        self.detection_output_topic = self.get_parameter('detection_output_topic').get_parameter_value().string_value

        # Log the topics being used
        self.get_logger().info(f"Subscribing to: '{self.color_img_input_topic}'")
        self.get_logger().info(f"Publishing to: '{self.processed_img_output_topic}'")
        self.get_logger().info(f"Publishing to: '{self.detection_output_topic}'")

        # Load algorithm parameters
        self.declare_parameter('model_path', '/root/harpia_ws/src/yolo_detector/models/best8.pt')
        self.declare_parameter('confidence_threshold', 0.90)
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

    def process_frame(self, frame):
        if frame is None:
            self.get_logger().error("Failed to load frame.")
            return []

        # Run inference
        results = self.model.predict(source=frame, conf=self.confidence_threshold)


        self.get_logger().info(f"Classes do modelo: {self.model.names}")
        self.get_logger().info(f"Shape da imagem: {frame.shape}")

        # Process and draw detections
        platforms = []
        for result in results:
            if len(result.boxes) == 0:
                self.get_logger().debug("No object detected.")
            else:
                for box in result.boxes:
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    #self.get_logger().info(f"Object detected: {class_name}, : {confidence:.2f}")

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    label = f"{class_name} {confidence:.2f}"
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
                    width = x2 - x1
                    height = y2 - y1
                    cx = x1 + width / 2
                    cy = y1 + height / 2

                    platforms.append({
                        "name": class_name,
                        "confidence": confidence,
                        "width": width,
                        "height": height,
                        "cx": cx,
                        "cy": cy
                    })
        return frame, platforms

    def color_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        pframe, platforms = self.process_frame(cv_image)


        #Publish detection results
        detection_array_msg = DetectionArray()
        for platform in platforms:
            detection_msg = Detection()
            detection_msg.name = platform['name']
            detection_msg.height = float(platform['height'])
            detection_msg.width = float(platform['width'])
            detection_msg.x = float(platform['cx'])
            detection_msg.y = float(platform['cy'])
            detection_array_msg.detections.append(detection_msg) #type: ignore (msg field is a list, not recognized by interpreter)

        self.detection_pub.publish(detection_array_msg)

        # Publish processed image
        processed_img_msg = self.bridge.cv2_to_imgmsg(pframe, encoding='bgr8')
        self.processed_pub.publish(processed_img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLO_detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()