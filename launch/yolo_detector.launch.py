from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():

    
    ros_setup_path = '/opt/ros/jazzy/setup.bash'
    venv_path = '/root/venv_yolo/lib/python3.12/site-packages'

    # Abre a pasta do nó(por algum motivo isso foi preciso), faz o terminal olhar para os pacotes instalado dentro da venv local, da um source no setup.bahs do ros e roda o nó comom sendo um CLI
    full_command = f"cd /root/harpia_ws/src/yolo_detector && \
                    export PYTHONPATH={venv_path}:$PYTHONPATH && \
                    source {ros_setup_path} && \
                    ros2 run yolo_detector detection"
    
    # Cria a Ação para Executar o Comando ---
    run_yolo_node_process = ExecuteProcess(
        cmd=['bash', '-c', full_command],
        output='screen'
    )

    return LaunchDescription([
        run_yolo_node_process
    ])