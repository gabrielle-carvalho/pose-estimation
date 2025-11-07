import cv2
import numpy as np
from openni import openni2

# === [1] Inicializar OpenNI ===
openni2.initialize()  # Caminho do driver se necessário
dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

depth_stream.start()
color_stream.start()

# === [2] Parâmetros da câmera (exemplo, ajuste para sua Xtion) ===
intrinsics = {
    'fx': 525.0,  # focal length x
    'fy': 525.0,  # focal length y
    'cx': 319.5,  # principal point x
    'cy': 239.5   # principal point y
}

# === [3] Função para converter depth map para point cloud ===
def depth_to_pointcloud(depth_map, intrinsics):
    H, W = depth_map.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map / 1000.0  # converter mm para metros
    x = (i - intrinsics['cx']) * z / intrinsics['fx']
    y = (j - intrinsics['cy']) * z / intrinsics['fy']
    return np.stack((x, y, z), axis=-1)  # shape: [H, W, 3]

# === [4] Loop principal (exemplo com 1 frame) ===
frame = None
keypoints = None  # Substitua com sua saída de pose estimation

while True:
    # --- Captura RGB e DEPTH ---
    frame_depth = depth_stream.read_frame()
    frame_color = color_stream.read_frame()
    frame_depth_data = np.frombuffer(frame_depth.get_buffer_as_uint16(), dtype=np.uint16).reshape((480, 640))
    frame_color_data = np.frombuffer(frame_color.get_buffer_as_uint8(), dtype=np.uint8).reshape((480, 640, 3))

    # Converter para BGR para exibir com OpenCV
    frame_bgr = cv2.cvtColor(frame_color_data, cv2.COLOR_RGB2BGR)

    # --- Pose estimation aqui (você já tem isso) ---
    # Suponha que você já tenha o dict: keypoints = {'left_shoulder': (x, y), ...}

    if keypoints:
        # 1. Desenhar keypoints
        for key, (x, y) in keypoints.items():
            if x is not None and y is not None:
                x_int = int(x * 640)
                y_int = int(y * 480)
                cv2.circle(frame_bgr, (x_int, y_int), 4, (0, 0, 255), -1)

        # 2. Calcular bounding box do tronco
        trunk_keys = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        trunk_points = []
        for key in trunk_keys:
            if key in keypoints:
                x, y = keypoints[key]
                if x is not None and y is not None:
                    x_int = int(x * 640)
                    y_int = int(y * 480)
                    trunk_points.append((x_int, y_int))

        if trunk_points:
            x_coords, y_coords = zip(*trunk_points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Adicionar margem
            margin_x = int((x_max - x_min) * 0.2)
            margin_y = int((y_max - y_min) * 0.3)

            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(639, x_max + margin_x)
            y_max = min(479, y_max + margin_y)

            # Desenhar bounding box
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 3. Converter depth map para point cloud
            point_cloud = depth_to_pointcloud(frame_depth_data, intrinsics)

            # 4. Extrair região do tronco na point cloud
            region = point_cloud[y_min:y_max, x_min:x_max]
            points = region.reshape(-1, 3)

            # 5. Filtrar pontos válidos
            valid_points = points[
                (np.isfinite(points[:, 0])) &
                (np.isfinite(points[:, 1])) &
                (np.isfinite(points[:, 2])) &
                (points[:, 2] > 0.1)
            ]

            # 6. Calcular posição 3D média
            if len(valid_points) > 0:
                center_3d = np.mean(valid_points, axis=0)
                print("Posição 3D do tronco:", center_3d)
            else:
                print("Sem pontos válidos para o tronco.")

    # Mostrar imagem
    cv2.imshow("RGB", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
depth_stream.stop()
color_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
