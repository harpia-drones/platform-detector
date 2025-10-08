import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Encontra o diretorio de instalacao do pacote
    package_dir = get_package_share_directory('platform_detector')
    
    # Monta o caminho completo para o arquivo do modelo
    model_path = os.path.join(package_dir, 'models', 'best.pt')

    yolo_detector_node = Node(
        package='platform_detector',
        executable='yolo_detector',
        name='yolo_detector_node',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'confidence_threshold': 0.6,
            'target_class_id': 0
        }]
    )

    return LaunchDescription([
        yolo_detector_node
    ])
