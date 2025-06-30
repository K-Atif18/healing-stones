import bpy
import os
import numpy as np
import sys
from pathlib import Path

# Add the healing-stones module to path
sys.path.append('/home/kira/Desktop/gsoc/healing-stones')
from align import get_rotation_angles


def setup_blender_environment():
    """Configure Blender scene settings for proper units and packing."""
    bpy.data.use_autopack = True
    bpy.context.scene.unit_settings.scale_length = 0.001
    bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'
    print("Blender environment configured.")


def setup_directories(root_path):
    """Create necessary directory structure."""
    blender_folder = Path(root_path) / 'blender'
    todo_folder = blender_folder / 'todo'
    done_folder = blender_folder / 'done'
    
    # Create directories if they don't exist
    todo_folder.mkdir(parents=True, exist_ok=True)
    done_folder.mkdir(parents=True, exist_ok=True)
    
    return todo_folder, done_folder


def find_artifact_folders(root_path):
    """Find all artifact folders in the root directory."""
    root = Path(root_path)
    artifact_folders = [f for f in root.iterdir() 
                       if f.is_dir() and f.name.startswith("artifact_")]
    return sorted(artifact_folders, key=lambda x: x.name)


def select_artifact_folder(artifact_folders):
    """Interactive selection of artifact folder."""
    if not artifact_folders:
        print(f"No artifact folders found!")
        return None
    
    print(f"\nFound {len(artifact_folders)} artifact folder(s):")
    for idx, folder in enumerate(artifact_folders):
        ply_count = len(list(folder.glob("*.ply")))
        print(f"  [{idx + 1}] {folder.name} ({ply_count} PLY files)")
    
    while True:
        try:
            selection = input(f"\nEnter number (1-{len(artifact_folders)}) to select artifact folder: ")
            selected_idx = int(selection) - 1
            
            if 0 <= selected_idx < len(artifact_folders):
                return artifact_folders[selected_idx]
            else:
                print(f"Please enter a number between 1 and {len(artifact_folders)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None


def find_ply_files(folder_path):
    """Find all PLY files in the given folder."""
    folder = Path(folder_path)
    ply_files = list(folder.glob("*.ply"))
    return sorted(ply_files, key=lambda x: x.name.lower())


def calculate_grid_layout(num_pieces, grid_size=250):
    """Calculate grid positions for placing objects."""
    if num_pieces == 0:
        return [], []
    
    on_axis = max(1, int(np.ceil(np.sqrt(num_pieces))))
    x_positions = np.linspace(-grid_size, grid_size, on_axis)
    y_positions = np.linspace(-grid_size, grid_size, on_axis)
    
    return x_positions, y_positions


def clear_existing_meshes():
    """Clear existing mesh objects from the scene."""
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select and delete all mesh objects
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
        print(f"Cleared {len(bpy.context.selected_objects)} existing mesh objects.")


def import_and_position_ply(ply_file, position, rotation_angles):
    """Import a PLY file and position it in the scene."""
    try:
        # Import PLY file
        bpy.ops.import_mesh.ply(filepath=str(ply_file))
        
        # Get the imported object (should be the active object)
        imported_obj = bpy.context.active_object
        
        if imported_obj:
            # Set origin to geometry center
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
            
            # Set position and rotation
            imported_obj.location = position
            imported_obj.rotation_euler = rotation_angles
            
            # Rename object to match file name
            imported_obj.name = ply_file.stem
            
            print(f"  ✓ Imported and positioned: {ply_file.name}")
            return True
        else:
            print(f"  ✗ Failed to import: {ply_file.name}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error importing {ply_file.name}: {str(e)}")
        return False


def load_or_create_scene(blend_file_path):
    """Load existing blend file or create new scene."""
    if blend_file_path.exists():
        print(f"Loading existing scene: {blend_file_path.name}")
        bpy.ops.wm.open_mainfile(filepath=str(blend_file_path))
        clear_existing_meshes()  # Clear existing meshes for fresh import
    else:
        print("Creating new empty scene...")
        bpy.ops.wm.read_factory_settings(use_empty=True)


def main():
    """Main execution function."""
    # Configuration
    ROOT_PATH = 'Ground_Truth'
    GRID_SIZE = 250
    
    print("=== Blender PLY Artifact Loader ===")
    
    # Setup directories
    try:
        todo_folder, done_folder = setup_directories(ROOT_PATH)
        print(f"Directory structure ready:")
        print(f"  Todo folder: {todo_folder}")
        print(f"  Done folder: {done_folder}")
    except Exception as e:
        print(f"Error setting up directories: {e}")
        return
    
    # Find artifact folders
    artifact_folders = find_artifact_folders(ROOT_PATH)
    selected_folder = select_artifact_folder(artifact_folders)
    
    if not selected_folder:
        print("No artifact folder selected. Exiting.")
        return
    
    print(f"Selected artifact folder: {selected_folder.name}")
    
    # Find PLY files
    ply_files = find_ply_files(selected_folder)
    
    if not ply_files:
        print(f"No PLY files found in '{selected_folder}'!")
        return
    
    print(f"Found {len(ply_files)} PLY files to import")
    
    # Setup blend file path
    blend_file_path = todo_folder / f"{selected_folder.name}_todo.blend"
    
    # Load or create Blender scene
    load_or_create_scene(blend_file_path)
    
    # Setup Blender environment
    setup_blender_environment()
    
    # Calculate grid layout
    x_positions, y_positions = calculate_grid_layout(len(ply_files), GRID_SIZE)
    
    # Import and position PLY files
    print(f"\nImporting PLY files...")
    successful_imports = 0
    
    for i, ply_file in enumerate(ply_files):
        try:
            # Get rotation angles
            a_x, a_y, a_z = get_rotation_angles(str(ply_file))
            
            # Calculate grid position
            row = i // len(x_positions)
            col = i % len(x_positions)
            
            if row < len(y_positions):
                position = (x_positions[col], y_positions[row], 0)
                rotation = (a_x, a_y, a_z)
                
                if import_and_position_ply(ply_file, position, rotation):
                    successful_imports += 1
            else:
                print(f"  ! Skipping {ply_file.name} - grid overflow")
                
        except Exception as e:
            print(f"  ✗ Error processing {ply_file.name}: {str(e)}")
    
    # Save the blend file
    try:
        bpy.ops.wm.save_mainfile(filepath=str(blend_file_path))
        print(f"\n✓ Scene saved successfully!")
        print(f"  File: {blend_file_path}")
        print(f"  Objects imported: {successful_imports}/{len(ply_files)}")
        
        # Summary
        print(f"\n=== Import Summary ===")
        print(f"Artifact folder: {selected_folder.name}")
        print(f"PLY files found: {len(ply_files)}")
        print(f"Successfully imported: {successful_imports}")
        print(f"Blend file: {blend_file_path}")
        
    except Exception as e:
        print(f"✗ Error saving blend file: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()