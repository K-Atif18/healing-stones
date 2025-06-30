import bpy
import os
import json

# Paths setup
root_path = 'Ground_Truth/blender/done'
solved_puzzles = os.path.join(root_path)
output_json_folder = 'Ground_Truth/json'

os.makedirs(output_json_folder, exist_ok=True)

# Get list of .blend files
list_of_solved_puzzles = [sp for sp in os.listdir(solved_puzzles) if sp.endswith('.blend')]

if not list_of_solved_puzzles:
    print("No .blend files found in DONE folder.")
else:
    print("\nFound the following artifacts:")
    for idx, name in enumerate(list_of_solved_puzzles):
        print(f"{idx+1}. {name}")

    # Ask user which one to select
    selection = input("\nEnter the number of the artifact you want to process: ")

    try:
        selected_index = int(selection) - 1
        selected_file = list_of_solved_puzzles[selected_index]
        base_name = os.path.splitext(selected_file)[0]
        print(f"Processing: {selected_file}")

        # Load the selected blend file
        bpy.ops.wm.open_mainfile(filepath=os.path.join(solved_puzzles, selected_file))
        bpy.data.use_autopack = True
        bpy.context.scene.unit_settings.scale_length = 0.001
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

        # Extract ground truth info
        gt_dict = {}
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                loc = obj.location
                rot_euler = obj.rotation_euler
                rot_quat = rot_euler.to_quaternion()
                gt_piece = {
                    'location': [loc.x, loc.y, loc.z],
                    'rotation_euler': [rot_euler.x, rot_euler.y, rot_euler.z],
                    'rotation_quaternion': [rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z]
                }
                gt_dict[obj.name] = gt_piece

        # File path (only one now!)
        output_json_path = os.path.join(output_json_folder, f"{base_name}.json")

        with open(output_json_path, 'w') as jtp:
            json.dump(gt_dict, jtp, indent=3)
        print(f"✔️ Saved JSON: {output_json_path}")

    except (ValueError, IndexError):
        print("❌ Invalid selection. Please enter a number corresponding to the listed artifacts.")
