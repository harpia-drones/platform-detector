import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

# Importa a biblioteca YOLO
from ultralytics import YOLO

# Ferramentas para sincronizar os topicos da camera de cor e de profundidade
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

class YoloDetectorNode(Node):
    def _init_(self):
        super()._init_('yolo_detector_node')

        # --- Parametros Configuraveis via Launch File ---
        self.declare_parameter('model_path', 'models/best.pt')
        self.declare_parameter('target_class_id', 0)
        self.declare_parameter('confidence_threshold', 0.6)
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.target_class_id = self.get_parameter('target_class_id').get_parameter_value().integer_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.get_logger().info("YOLOv8 Detector Node has started.")

        # --- Variaveis de Estado ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.bridge = CvBridge()
        
        # --- Carregamento do Modelo YOLO ---
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f"YOLO model '{model_path}' loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise e

        # --- Publishers ---
        self.target_position_pub = self.create_publisher(PointStamped, '/target_tracker/relative_position', 10)
        self.debug_image_pub = self.create_publisher(Image, '/yolo_detector/debug_image', 10)

        # --- Subscribers ---
        # Subscriber para CameraInfo (pega a informacao uma vez)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/down/camera_info', self.camera_info_callback, 10)

        # Subscribers Sincronizados para Imagem RGB e de Profundidade
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/down/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/down/depth/image_rect_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.detection_callback)
        self.get_logger().info("Ready and waiting for synchronized images...")

    def camera_info_callback(self, msg):
        """Callback para receber os parametros de calibracao da camera apenas uma vez."""
        if self.camera_matrix is None:
            self.get_logger().info('Camera calibration parameters received.')
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.destroy_subscription(self.camera_info_sub)

    def detection_callback(self, rgb_msg, depth_msg):
        """Callback principal, executado com um par sincronizado de imagens RGB e de profundidade."""
        if self.camera_matrix is None:
            self.get_logger().warn('Waiting for camera calibration data...', throttle_duration_sec=5)
            return

        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # --- Executa a Deteccao com YOLOv8 ---
        results = self.model(rgb_frame, verbose=False)
        result = results[0]

        best_detection = None
        highest_confidence = 0.0

        # Itera sobre os resultados para encontrar o melhor alvo
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == self.target_class_id and confidence > highest_confidence:
                highest_confidence = confidence
                best_detection = box

        found_target = False
        if best_detection is not None and highest_confidence > self.confidence_threshold:
            found_target = True
            
            # Extrai o centro do alvo
            xyxy = best_detection.xyxy[0]
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            
            # Remove a distorcao do pixel do centro para precisao no calculo 3D
            distorted_pixel = np.array([[[cx, cy]]], dtype=np.float32)
            undistorted_pixel = cv2.undistortPoints(distorted_pixel, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            ucx, ucy = int(undistorted_pixel[0,0,0]), int(undistorted_pixel[0,0,1])
            
            # Mede a distancia REAL ate a plataforma usando a imagem de profundidade
            if 0 <= ucy < depth_frame.shape[0] and 0 <= ucx < depth_frame.shape[1]:
                true_distance_mm = depth_frame[ucy, ucx]
                if true_distance_mm > 0:
                    true_distance_m = true_distance_mm / 1000.0
                    
                    # --- CaLCULO DA POSIcaO RELATIVA 3D ---
                    fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                    cam_cx, cam_cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

                    X_cam = (ucx - cam_cx) * true_distance_m / fx
                    Y_cam = (ucy - cam_cy) * true_distance_m / fy
                    
                    # Publica a mensagem de posicao
                    position_msg = PointStamped()
                    position_msg.header = rgb_msg.header
                    position_msg.header.frame_id = "drone_base_link"
                    position_msg.point.x, position_msg.point.y, position_msg.point.z = Y_cam, -X_cam, -true_distance_m
                    self.target_position_pub.publish(position_msg)
                else:
                    found_target = False
            else:
                found_target = False

        if not found_target:
            # Se nenhum alvo valido foi encontrado, publica NaN
            position_msg = PointStamped()
            position_msg.header = rgb_msg.header
            position_msg.header.frame_id = "drone_base_link"
            position_msg.point.x = position_msg.point.y = position_msg.point.z = float('nan')
            self.target_position_pub.publish(position_msg)
        
        # Publica a imagem de debug com as deteccoes desenhadas
        debug_frame = result.plot() # O metodo .plot() ja desenha tudo para nos!
        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_frame, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except Exception as e:
        rclpy.logging.get_logger('yolo_detector_node').error(f'Unhandled exception: {e}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
