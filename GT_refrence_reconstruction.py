import open3d as o3d
import json, os
import numpy as np

# Base paths
GROUND_TRUTH_FOLDER = 'Ground_Truth'
JSON_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, 'json')
RECONSTRUCTED_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, 'reconstructed')

def main():
    # List available artifacts
    artifacts = [d for d in os.listdir(GROUND_TRUTH_FOLDER) if d.startswith('artifact_') and os.path.isdir(os.path.join(GROUND_TRUTH_FOLDER, d))]
    
    if not artifacts:
        print("No artifacts found in Ground_Truth/")
        return

    print("Available Artifacts:")
    for idx, artifact in enumerate(artifacts):
        print(f"{idx + 1}. {artifact}")

    selection = input("\nEnter the number of the artifact you want to reconstruct: ")

    try:
        selected_idx = int(selection) - 1
        artifact_name = artifacts[selected_idx]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    artifact_folder = os.path.join(GROUND_TRUTH_FOLDER, artifact_name)
    json_path = os.path.join(JSON_FOLDER, f"{artifact_name}.json")
    output_folder = os.path.join(RECONSTRUCTED_FOLDER, artifact_name)

    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, 'r') as jp:
        gt = json.load(jp)

    meshes = []
    meshes_names = []
    all_pts = np.array([])

    for gtk in gt.keys():
        ply_path = os.path.join(artifact_folder, f"{gtk}.ply")
        meshes_names.append(f"{gtk}.ply")

        if not os.path.exists(ply_path):
            print(f"Missing fragment file: {ply_path}")
            continue

        mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=True)

        if len(np.asarray(mesh.vertices)) == 0:
            print(f"Empty mesh: {ply_path}")
            continue

        # Place at origin before applying transform
        mesh.translate(-mesh.get_center())

        # Apply rotations (Blender style)
        rot_angles = gt[gtk]['rotation_euler']
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([rot_angles[0], 0, 0]), center=mesh.get_center())
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, rot_angles[1], 0]), center=mesh.get_center())
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, 0, rot_angles[2]]), center=mesh.get_center())

        # Apply translation
        mesh.translate(gt[gtk]['location'])
        print(f"{gtk}: placed at {mesh.get_center()}")

        meshes.append(mesh)

        if all_pts.size == 0:
            all_pts = np.asarray(mesh.vertices)
        else:
            all_pts = np.concatenate((all_pts, np.asarray(mesh.vertices)))

    if all_pts.size == 0:
        print("No mesh data reconstructed. Exiting.")
        return

    # Visualization before normalization
    cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    o3d.visualization.draw_geometries(meshes + [cframe])

    # Create point cloud of combined meshes
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(all_pts))
    translation = -pcd.get_center()

    print("Center translation to apply:", translation)

    # Translate all meshes so that center is at origin
    for mesh, mesh_name in zip(meshes, meshes_names):
        mesh.translate(translation)
        out_path = os.path.join(output_folder, mesh_name)
        o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=True)
        print(f"Saved reconstructed mesh: {out_path}")

    # Show reconstructed normalized mesh cloud
    pcd.translate(translation)
    o3d.visualization.draw_geometries([pcd, cframe])

if __name__ == '__main__':
    main()
