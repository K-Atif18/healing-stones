#!/usr/bin/env python3
"""
Blender script to extract ground truth assembly data.
Run this inside Blender with the assembled artifact loaded.

Usage in Blender:
1. Open Blender with your assembled artifact
2. Open Scripting tab
3. Load and run this script
4. Output will be saved to 'ground_truth_assembly.json'
"""

import bpy
import bmesh
import numpy as np
import json
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import os
from pathlib import Path

class GroundTruthExtractor:
    def __init__(self, output_dir="Ground_Truth/blender"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Fragment naming pattern - adjust based on your naming
        self.fragment_prefix = "frag_"
        
        # Contact detection threshold (in Blender units)
        self.contact_threshold = 3.0  # 8mm assuming mm units
        
    def extract_assembly_data(self):
        """Extract all assembly ground truth data."""
        print("Starting ground truth extraction...")
        
        # Get all fragment objects
        fragments = self._get_fragment_objects()
        print(f"Found {len(fragments)} fragments")
        
        # Extract fragment data
        fragment_data = {}
        for frag in fragments:
            frag_info = self._extract_fragment_info(frag)
            fragment_data[frag.name] = frag_info
        
        # Find fragment contacts
        contact_pairs = self._find_contact_pairs(fragments)
        print(f"Found {len(contact_pairs)} contact pairs")
        
        # Extract detailed contact regions
        contact_details = []
        for frag1, frag2 in contact_pairs:
            details = self._analyze_contact_region(frag1, frag2)
            if details:
                contact_details.append(details)
        
        # Compile results
        ground_truth = {
            'fragments': fragment_data,
            'contact_pairs': [(f1.name, f2.name) for f1, f2 in contact_pairs],
            'contact_details': contact_details,
            'world_settings': {
                'unit_scale': bpy.context.scene.unit_settings.scale_length,
                'units': bpy.context.scene.unit_settings.length_unit
            }
        }
        
        # Save results
        output_file = self.output_dir / "ground_truth_assembly.json"
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2, cls=NumpyEncoder)
        
        print(f"Saved ground truth to: {output_file}")
        
        # Also export contact visualization
        self._export_contact_visualization(contact_pairs)
        
        return ground_truth
    
    def _get_fragment_objects(self):
        """Get all fragment objects from the scene."""
        fragments = []
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and self.fragment_prefix in obj.name:
                fragments.append(obj)
        return sorted(fragments, key=lambda x: x.name)
    
    def _extract_fragment_info(self, fragment):
        """Extract pose and geometry info for a fragment."""
        # Get world matrix (4x4 transformation)
        world_matrix = fragment.matrix_world
        
        # Decompose into location, rotation, scale
        loc, rot, scale = world_matrix.decompose()
        
        # Get mesh statistics
        mesh = fragment.data
        
        # Ensure mesh is in world space for analysis
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = fragment.evaluated_get(depsgraph)
        mesh_world = eval_obj.to_mesh()
        
        # Transform mesh to world coordinates
        mesh_world.transform(world_matrix)
        
        # Calculate bounding box in world space
        bbox_min = Vector((float('inf'), float('inf'), float('inf')))
        bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for vert in mesh_world.vertices:
            bbox_min.x = min(bbox_min.x, vert.co.x)
            bbox_min.y = min(bbox_min.y, vert.co.y)
            bbox_min.z = min(bbox_min.z, vert.co.z)
            bbox_max.x = max(bbox_max.x, vert.co.x)
            bbox_max.y = max(bbox_max.y, vert.co.y)
            bbox_max.z = max(bbox_max.z, vert.co.z)
        
        # Clean up temporary mesh
        eval_obj.to_mesh_clear()
        
        return {
            'transform_matrix': [list(row) for row in world_matrix],
            'location': list(loc),
            'rotation_euler': list(rot.to_euler()),
            'rotation_quaternion': list(rot),
            'scale': list(scale),
            'bbox_world': {
                'min': list(bbox_min),
                'max': list(bbox_max)
            },
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.polygons)
        }
    
    def _find_contact_pairs(self, fragments):
        """Find which fragment pairs are in contact."""
        contact_pairs = []
        
        for i, frag1 in enumerate(fragments):
            for j in range(i + 1, len(fragments)):
                frag2 = fragments[j]
                
                # Quick bbox check first
                if self._bboxes_nearby(frag1, frag2):
                    # Detailed proximity check
                    if self._fragments_in_contact(frag1, frag2):
                        contact_pairs.append((frag1, frag2))
        
        return contact_pairs
    
    def _bboxes_nearby(self, obj1, obj2):
        """Quick check if bounding boxes are close enough."""
        # Get world space bounding boxes
        bbox1 = self._get_world_bbox(obj1)
        bbox2 = self._get_world_bbox(obj2)
        
        # Check if bboxes are within threshold distance
        for i in range(3):
            if bbox1['max'][i] + self.contact_threshold < bbox2['min'][i]:
                return False
            if bbox2['max'][i] + self.contact_threshold < bbox1['min'][i]:
                return False
        
        return True
    
    def _get_world_bbox(self, obj):
        """Get bounding box in world coordinates."""
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        
        bbox_min = Vector((float('inf'), float('inf'), float('inf')))
        bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for corner in bbox_corners:
            for i in range(3):
                bbox_min[i] = min(bbox_min[i], corner[i])
                bbox_max[i] = max(bbox_max[i], corner[i])
        
        return {'min': bbox_min, 'max': bbox_max}
    
    def _fragments_in_contact(self, frag1, frag2):
        """Detailed check if fragments are actually in contact."""
        # Get evaluated meshes in world space
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # Sample points from frag1
        eval_obj1 = frag1.evaluated_get(depsgraph)
        mesh1 = eval_obj1.to_mesh()
        mesh1.transform(frag1.matrix_world)
        
        # Create BVH tree for frag2 for efficient distance queries
        eval_obj2 = frag2.evaluated_get(depsgraph)
        mesh2 = eval_obj2.to_mesh()
        mesh2.transform(frag2.matrix_world)
        
        # Use Blender's BVH tree for proximity queries
        bm2 = bmesh.new()
        bm2.from_mesh(mesh2)
        bm2.faces.ensure_lookup_table()
        tree2 = BVHTree.FromBMesh(bm2)
        
        # Sample vertices from mesh1
        contact_found = False
        sample_rate = max(1, len(mesh1.vertices) // 1000)  # Sample up to 1000 points
        
        for i in range(0, len(mesh1.vertices), sample_rate):
            vert = mesh1.vertices[i]
            location, normal, index, distance = tree2.find_nearest(vert.co)
            
            if distance is not None and distance < self.contact_threshold:
                contact_found = True
                break
        
        # Clean up
        bm2.free()
        eval_obj1.to_mesh_clear()
        eval_obj2.to_mesh_clear()
        
        return contact_found
    
    def _analyze_contact_region(self, frag1, frag2):
        """Analyze the contact region between two fragments."""
        print(f"Analyzing contact between {frag1.name} and {frag2.name}")
        
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # Get meshes in world space
        eval_obj1 = frag1.evaluated_get(depsgraph)
        mesh1 = eval_obj1.to_mesh()
        mesh1.transform(frag1.matrix_world)
        
        eval_obj2 = frag2.evaluated_get(depsgraph)
        mesh2 = eval_obj2.to_mesh()
        mesh2.transform(frag2.matrix_world)
        
        # Build BVH trees
        bm1 = bmesh.new()
        bm1.from_mesh(mesh1)
        tree1 = BVHTree.FromBMesh(bm1)

        
        bm2 = bmesh.new()
        bm2.from_mesh(mesh2)
        tree2 = BVHTree.FromBMesh(bm2)
        
        # Find contact points
        contact_points_1 = []
        contact_points_2 = []
        contact_distances = []
        
        # Sample mesh1 vertices
        for vert in mesh1.vertices:
            location, normal, index, distance = tree2.find_nearest(vert.co)
            if distance is not None and distance < self.contact_threshold:
                contact_points_1.append({
                    'position': list(vert.co),
                    'normal': list(vert.normal),
                    'index': vert.index
                })
                contact_points_2.append({
                    'position': list(location),
                    'normal': list(normal) if normal else [0, 0, 0],
                    'face_index': index
                })
                contact_distances.append(distance)
        
        # Calculate contact region statistics
        if contact_points_1:
            contact_center_1 = np.mean([p['position'] for p in contact_points_1], axis=0)
            contact_center_2 = np.mean([p['position'] for p in contact_points_2], axis=0)
            
            # Estimate contact area (simplified)
            contact_positions = np.array([p['position'] for p in contact_points_1])
            if len(contact_positions) > 3:
                # Use PCA to find contact plane
                centered = contact_positions - contact_center_1
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Contact area approximation (ellipse)
                contact_area = np.pi * np.sqrt(eigenvalues[1]) * np.sqrt(eigenvalues[2])
            else:
                contact_area = 0.0
            
            result = {
                'fragment_1': frag1.name,
                'fragment_2': frag2.name,
                'contact_point_count': len(contact_points_1),
                'contact_center_1': list(contact_center_1),
                'contact_center_2': list(contact_center_2),
                'mean_gap': float(np.mean(contact_distances)),
                'max_gap': float(np.max(contact_distances)),
                'estimated_contact_area': float(contact_area),
                'contact_indices_1': [p['index'] for p in contact_points_1][:100],  # Limit for file size
                'contact_indices_2': [p['face_index'] for p in contact_points_2][:100]
            }
        else:
            result = None
        
        # Clean up
        bm1.free()
        bm2.free()
        eval_obj1.to_mesh_clear()
        eval_obj2.to_mesh_clear()
        
        return result
    
    def _export_contact_visualization(self, contact_pairs):
        """Create visualization objects for contact regions."""
        # Create a collection for contact visualizations
        viz_collection = bpy.data.collections.new("Contact_Regions")
        bpy.context.scene.collection.children.link(viz_collection)
        
        for i, (frag1, frag2) in enumerate(contact_pairs):
            # Create a small sphere at contact center
            details = self._analyze_contact_region(frag1, frag2)
            if details and 'contact_center_1' in details:
                # Create sphere
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=2.0,
                    location=details['contact_center_1']
                )
                sphere = bpy.context.active_object
                sphere.name = f"Contact_{frag1.name}_{frag2.name}"
                
                # Move to visualization collection
                for coll in sphere.users_collection:
                    coll.objects.unlink(sphere)
                viz_collection.objects.link(sphere)
                
                # Add custom properties
                sphere["fragment_1"] = frag1.name
                sphere["fragment_2"] = frag2.name
                sphere["mean_gap"] = details['mean_gap']
                sphere["contact_area"] = details['estimated_contact_area']


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# Run the extraction
if __name__ == "__main__":
    extractor = GroundTruthExtractor()
    ground_truth = extractor.extract_assembly_data()
    print("Extraction complete!")



    