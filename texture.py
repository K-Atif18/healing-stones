import trimesh
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

def fit_plane(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    normal = np.cross(pca.components_[0], pca.components_[1])
    centroid = np.mean(points, axis=0)
    dists = np.dot(points - centroid, normal)
    return normal, centroid, dists

def uv_area(uv_coords):
    # Shoelace formula for triangle area in 2D
    a, b, c = uv_coords
    return 0.5 * abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])))

def extract_carved_faces(mesh, n_surfaces=2, planarity_thresh=0.01):
    uvs = mesh.visual.uv
    faces = mesh.faces
    face_uv_areas = []
    face_planarity = []
    for face in faces:
        verts = mesh.vertices[face]
        uv_coords = uvs[face]
        area = uv_area(uv_coords)
        normal, centroid, dists = fit_plane(verts)
        planarity = np.mean(np.abs(dists))
        face_uv_areas.append(area)
        face_planarity.append(planarity)
    face_uv_areas = np.array(face_uv_areas)
    face_planarity = np.array(face_planarity)
    # Select faces with large UV area and low planarity (flat)
    idx = np.lexsort((face_planarity, -face_uv_areas))  # prioritize area, then planarity
    selected = []
    for i in idx:
        if face_planarity[i] < planarity_thresh:
            selected.append(i)
        if len(selected) >= n_surfaces:
            break
    return selected

from scipy.interpolate import griddata

def save_height_map(mesh, face_idx, out_path, img_size=256):
    face = mesh.faces[face_idx]
    verts = mesh.vertices[face]
    # Fit plane
    pca = PCA(n_components=2)
    pca.fit(verts)
    plane = pca.components_
    centroid = np.mean(verts, axis=0)
    # Project to 2D
    proj_2d = (verts - centroid) @ plane.T
    # Normalize to [0, 1]
    min_xy = proj_2d.min(axis=0)
    max_xy = proj_2d.max(axis=0)
    norm_xy = (proj_2d - min_xy) / (max_xy - min_xy + 1e-8)
    # Interpolate Z (height) onto a grid using griddata
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    z = verts[:, 2]  # Use Z as height (or use (verts @ normal) for generality)
    points = norm_xy
    values = z
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.min(z))
    # Normalize to 0-255
    grid_z = (grid_z - np.min(grid_z)) / (np.max(grid_z) - np.min(grid_z) + 1e-8)
    z_img = (grid_z * 255).astype(np.uint8)
    cv2.imwrite(out_path, z_img)
    print(f"Saved height map to {out_path}")

def process_fragment(obj_path, out_dir, n_surfaces=2):
    mesh = trimesh.load(obj_path, process=False)
    selected_faces = extract_carved_faces(mesh, n_surfaces=n_surfaces)
    for i, face_idx in enumerate(selected_faces):
        out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(obj_path))[0]}_carved_{i+1}.png")
        save_height_map(mesh, face_idx, out_path)

# Example usage:
if __name__ == "__main__":
    obj_dir = "uv"
    out_dir = "uv/heightmaps"
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(obj_dir):
        if fname.endswith(".obj"):
            process_fragment(os.path.join(obj_dir, fname), out_dir, n_surfaces=2)