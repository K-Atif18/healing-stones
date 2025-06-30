import open3d as o3d
import os

# === CONFIG ===
#FOLDER_PATH = "Ground_Truth/reconstructed/artifact_1"  # CHANGE THIS to the folder with your .ply files
FOLDER_PATH = "Ground_Truth/artifact_1"
SHOW_COORDINATE_FRAME = True  # Toggle on/off coordinate frame display

def visualize_all_ply(folder_path):
    ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    
    if not ply_files:
        print("❌ No .ply files found in the folder.")
        return

    geometries = []
    
    for ply in ply_files:
        ply_path = os.path.join(folder_path, ply)
        mesh = o3d.io.read_triangle_mesh(ply_path)
        
        if not mesh.has_vertices():
            print(f"⚠️ Empty mesh skipped: {ply_path}")
            continue
        
        geometries.append(mesh)
        print(f"✅ Loaded: {ply}")

    if SHOW_COORDINATE_FRAME:
        cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        geometries.append(cframe)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    visualize_all_ply(FOLDER_PATH)
