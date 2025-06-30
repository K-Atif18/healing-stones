import open3d as o3d
import os

# === CONFIG: change this ===
ply_folder = "Ground_Truth/artifact_1"

# === Collect all .ply files ===
ply_files = [f for f in os.listdir(ply_folder) if f.lower().endswith('.ply')]

if not ply_files:
    print("No .ply files found in folder:", ply_folder)
    exit()

print(f"Found {len(ply_files)} PLY files.\n")

# === Loop through and visualize each one ===
for file in sorted(ply_files):
    file_path = os.path.join(ply_folder, file)
    print(f"Displaying: {file}")

    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()

    # Show the file in a window → close it → next file will show after
    o3d.visualization.draw_geometries([mesh], window_name=file, width=1280, height=720)

    input("Press ENTER to view the next .ply...\n")
