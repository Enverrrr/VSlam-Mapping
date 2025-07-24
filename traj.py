import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory):
    # Trajectory numpy array'e dönüştür
    traj = np.array(trajectory)
    
    # 2D X-Z düzlemini çiz (Y eksenini yükseklik olarak kabul edip ihmal ediyoruz)
    plt.figure(figsize=(10, 8))
    plt.plot(traj[:, 0], traj[:, 2], 'b-', linewidth=2, label='Robot Yolu')
    plt.scatter(traj[0, 0], traj[0, 2], c='green', s=100, label='Başlangıç')
    plt.scatter(traj[-1, 0], traj[-1, 2], c='red', s=100, label='Bitiş')
    
    plt.title('Robot Trajektorisi (X-Z Düzlemi)')
    plt.xlabel('X Pozisyonu (metre)')
    plt.ylabel('Z Pozisyonu (metre)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# Kullanımı:
# trajectory = slam.trajectory veya np.load("trajectory.npy")
trajectory= np.load("point_cloud_map.npy")
plot_trajectory(trajectory)