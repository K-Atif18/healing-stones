import open3d as o3d

# === CONFIG: Hardcode your .ply path here ===
ply_file = "/home/kira/Downloads/NAR_ST_43B_FR_ALL_ver_03_10M.ply"

# === Load mesh ===
mesh = o3d.io.read_triangle_mesh(ply_file)
mesh.compute_vertex_normals()

# === Visualize ===
o3d.visualization.draw_geometries([mesh], window_name=ply_file, width=1280, height=720)
