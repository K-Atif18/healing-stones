import os
import numpy as np
from pathlib import Path

def read_ply_header(file_path):
    """Read PLY header to understand the file structure"""
    with open(file_path, 'rb') as f:
        header_lines = []
        line = f.readline().decode('ascii').strip()
        header_lines.append(line)
        
        if line != 'ply':
            raise ValueError("Not a valid PLY file")
        
        vertex_count = 0
        face_count = 0
        properties = []
        
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif line.startswith('property'):
                properties.append(line)
            elif line == 'end_header':
                break
        
        header_size = f.tell()
    
    return {
        'header_lines': header_lines,
        'header_size': header_size,
        'vertex_count': vertex_count,
        'face_count': face_count,
        'properties': properties
    }

def read_ply_data(file_path, header_info):
    """Read vertex and face data from PLY file"""
    with open(file_path, 'rb') as f:
        f.seek(header_info['header_size'])
        
        # Read vertices (assuming format: x y z [other properties])
        vertex_data = []
        for _ in range(header_info['vertex_count']):
            line = f.readline().decode('ascii').strip()
            values = list(map(float, line.split()))
            vertex_data.append(values)
        
        # Read faces if they exist
        face_data = []
        for _ in range(header_info['face_count']):
            line = f.readline().decode('ascii').strip()
            values = list(map(int, line.split()))
            face_data.append(values)
    
    return np.array(vertex_data), face_data

def write_ply_fragment(output_path, header_info, vertices, faces, fragment_index):
    """Write a fragment of the PLY file"""
    with open(output_path, 'w') as f:
        # Write header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'comment Fragment {fragment_index + 1}\n')
        f.write(f'element vertex {len(vertices)}\n')
        
        # Write property lines (assuming x, y, z coordinates)
        for prop in header_info['properties']:
            if 'vertex' in prop:
                continue  # Skip element vertex line
            f.write(prop + '\n')
        
        if len(faces) > 0:
            f.write(f'element face {len(faces)}\n')
            f.write('property list uchar int vertex_indices\n')
        
        f.write('end_header\n')
        
        # Write vertex data
        for vertex in vertices:
            f.write(' '.join(map(str, vertex)) + '\n')
        
        # Write face data (adjust indices for the fragment)
        vertex_offset = 0  # This would need to be calculated based on fragment
        for face in faces:
            f.write(' '.join(map(str, face)) + '\n')

def split_ply_file(input_path, output_folder, num_fragments=10):
    """Split PLY file into specified number of fragments"""
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Reading PLY file: {input_path}")
    
    # Read header information
    header_info = read_ply_header(input_path)
    print(f"Found {header_info['vertex_count']} vertices and {header_info['face_count']} faces")
    
    # Read all data
    vertices, faces = read_ply_data(input_path, header_info)
    
    # Calculate fragment size
    vertices_per_fragment = len(vertices) // num_fragments
    remainder = len(vertices) % num_fragments
    
    print(f"Splitting into {num_fragments} fragments of ~{vertices_per_fragment} vertices each")
    
    # Split vertices into fragments
    start_idx = 0
    for i in range(num_fragments):
        # Add one extra vertex to first 'remainder' fragments
        fragment_size = vertices_per_fragment + (1 if i < remainder else 0)
        end_idx = start_idx + fragment_size
        
        # Get vertices for this fragment
        fragment_vertices = vertices[start_idx:end_idx]
        
        # For simplicity, we'll skip faces in fragments as they would need index remapping
        # In a full implementation, you'd need to:
        # 1. Find faces that reference vertices in this fragment
        # 2. Remap face indices to the new vertex indices
        # 3. Handle faces that span multiple fragments
        fragment_faces = []
        
        # Create output filename
        input_name = Path(input_path).stem
        output_path = Path(output_folder) / f"{input_name}_fragment_{i+1:02d}.ply"
        
        # Write fragment
        write_ply_fragment(output_path, header_info, fragment_vertices, fragment_faces, i)
        
        print(f"Created fragment {i+1}: {output_path} ({len(fragment_vertices)} vertices)")
        
        start_idx = end_idx
    
    print(f"\nSuccessfully split PLY file into {num_fragments} fragments in folder: {output_folder}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = "/home/kira/Desktop/gsoc/healing-stones/Ground_Truth/reconstructed/artifact_1/frag_3.ply"
    output_folder = "ply_fragments"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please update the 'input_file' variable with the correct path to your PLY file.")
    else:
        try:
            split_ply_file(input_file, output_folder, num_fragments=10)
        except Exception as e:
            print(f"Error processing PLY file: {e}")
            print("Make sure your PLY file is in ASCII format and properly formatted.")


            