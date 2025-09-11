import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
import json
from pathlib import Path

class ReconstructionVisualizer:
    """
    Comprehensive visualization tools for the reconstruction process
    """
    
    def __init__(self):
        self.colors = {
            'fragment_colors': [
                [0.8, 0.2, 0.2],  # Red
                [0.2, 0.8, 0.2],  # Green
                [0.2, 0.2, 0.8],  # Blue
                [0.8, 0.8, 0.2],  # Yellow
                [0.8, 0.2, 0.8],  # Magenta
                [0.2, 0.8, 0.8],  # Cyan
                [0.8, 0.5, 0.2],  # Orange
                [0.5, 0.2, 0.8],  # Purple
            ],
            'break_surface_colors': {
                'blue': [0.0, 0.0, 1.0],
                'green': [0.0, 1.0, 0.0],
                'red': [1.0, 0.0, 0.0]
            }
        }
    
    def visualize_original_fragments(self, fragments, show_break_surfaces=True):
        """Visualize original fragments with optional break surface highlighting"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Original Fragments")
        
        for i, fragment in enumerate(fragments):
            if 'mesh' not in fragment:
                continue
            
            mesh = copy.deepcopy(fragment['mesh'])
            
            # Color the main fragment
            fragment_color = self.colors['fragment_colors'][i % len(self.colors['fragment_colors'])]
            mesh.paint_uniform_color(fragment_color)
            
            # Offset fragments for better visualization
            offset = [i * 0.3, 0, 0]
            mesh.translate(offset)
            
            vis.add_geometry(mesh)
            
            # Add break surface visualizations
            if show_break_surfaces:
                for color, surfaces in fragment['break_surfaces'].items():
                    for j, surface in enumerate(surfaces):
                        # Create point cloud for break surface
                        pcd = o3d.geometry.PointCloud()
                        points = surface['points'] + offset
                        pcd.points = o3d.utility.Vector3dVector(points)
                        pcd.paint_uniform_color(self.colors['break_surface_colors'][color])
                        
                        # Make break surface points larger
                        pcd = pcd.uniform_down_sample(every_k_points=2)
                        
                        vis.add_geometry(pcd)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coordinate_frame)
        
        vis.run()
        vis.destroy_window()
    
    def visualize_surface_matches(self, fragment1, fragment2, matches, 
                                fragment1_offset=[0, 0, 0], fragment2_offset=[0.5, 0, 0]):
        """Visualize matched break surfaces between two fragments"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Surface Matches")
        
        # Add fragments
        if 'mesh' in fragment1:
            mesh1 = copy.deepcopy(fragment1['mesh'])
            mesh1.paint_uniform_color([0.8, 0.8, 0.8])
            mesh1.translate(fragment1_offset)
            vis.add_geometry(mesh1)
        
        if 'mesh' in fragment2:
            mesh2 = copy.deepcopy(fragment2['mesh'])
            mesh2.paint_uniform_color([0.7, 0.7, 0.7])
            mesh2.translate(fragment2_offset)
            vis.add_geometry(mesh2)
        
        # Visualize matches
        for i, match in enumerate(matches):
            color = match['color']
            idx1 = match['fragment1_idx']
            idx2 = match['fragment2_idx']
            
            # Get surface points
            points1 = fragment1['break_surfaces'][color][idx1]['points'] + fragment1_offset
            points2 = fragment2['break_surfaces'][color][idx2]['points'] + fragment2_offset
            
            # Create point clouds
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1)
            pcd1.paint_uniform_color(self.colors['break_surface_colors'][color])
            
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points2)
            pcd2.paint_uniform_color(self.colors['break_surface_colors'][color])
            
            vis.add_geometry(pcd1)
            vis.add_geometry(pcd2)
            
            # Draw line connecting centroids
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)
            
            line_points = [centroid1, centroid2]
            line_indices = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.paint_uniform_color([0, 0, 0])  # Black lines
            
            vis.add_geometry(line_set)
            
            # Add text labels for similarity scores
            text_pos = (centroid1 + centroid2) / 2
            # Note: Open3D doesn't have built-in text rendering, 
            # so we'll use spheres with different sizes to indicate quality
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.005 * match['similarity']
            )
            sphere.translate(text_pos)
            sphere.paint_uniform_color([1, 1, 0])  # Yellow for match indicator
            vis.add_geometry(sphere)
        
        vis.run()
        vis.destroy_window()
    
    def visualize_reconstruction_progress(self, fragments, transformations, 
                                        step_by_step=True):
        """Visualize the reconstruction process step by step"""
        if step_by_step:
            # Show reconstruction steps
            for step in range(len(transformations)):
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=f"Reconstruction Step {step + 1}")
                
                for i, fragment in enumerate(fragments):
                    if 'mesh' not in fragment:
                        continue
                    
                    mesh = copy.deepcopy(fragment['mesh'])
                    
                    # Apply transformation if available and within current step
                    if i in transformations and i <= step:
                        mesh.transform(transformations[i])
                    else:
                        # Offset unaligned fragments
                        mesh.translate([i * 0.5, -0.5, 0])
                    
                    # Color by fragment
                    color = self.colors['fragment_colors'][i % len(self.colors['fragment_colors'])]
                    mesh.paint_uniform_color(color)
                    
                    vis.add_geometry(mesh)
                
                # Add coordinate frame
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                vis.add_geometry(coordinate_frame)
                
                vis.run()
                vis.destroy_window()
        else:
            # Show final reconstruction
            self.visualize_final_reconstruction(fragments, transformations)
    
    def visualize_final_reconstruction(self, fragments, transformations):
        """Visualize the final reconstructed artifact"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Final Reconstruction")
        
        combined_mesh = o3d.geometry.TriangleMesh()
        
        for i, fragment in enumerate(fragments):
            if 'mesh' not in fragment:
                continue
            
            mesh = copy.deepcopy(fragment['mesh'])
            
            # Apply transformation if available
            if i in transformations:
                mesh.transform(transformations[i])
            
            # Color by fragment for identification
            color = self.colors['fragment_colors'][i % len(self.colors['fragment_colors'])]
            mesh.paint_uniform_color(color)
            
            # Add to combined mesh
            combined_mesh += mesh
            vis.add_geometry(mesh)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coordinate_frame)
        
        # Enable mesh smoothing
        combined_mesh.compute_vertex_normals()
        
        vis.run()
        vis.destroy_window()
        
        return combined_mesh
    
    def create_match_quality_report(self, all_matches, output_path=None):
        """Create a visual report of match qualities"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Surface Matching Quality Report', fontsize=16)
        
        # Collect all similarity scores
        all_similarities = []
        color_similarities = {'blue': [], 'green': [], 'red': []}
        pair_counts = {}
        
        for pair_key, color_matches in all_matches.items():
            pair_counts[pair_key] = 0
            for color, matches in color_matches.items():
                for match in matches:
                    similarity = match['similarity']
                    all_similarities.append(similarity)
                    color_similarities[color].append(similarity)
                    pair_counts[pair_key] += 1
        
        # Plot 1: Overall similarity distribution
        axes[0, 0].hist(all_similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Match Similarities')
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(all_similarities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_similarities):.3f}')
        axes[0, 0].legend()
        
        # Plot 2: Similarity by color
        colors = ['blue', 'green', 'red']
        positions = [1, 2, 3]
        box_data = [color_similarities[color] for color in colors if color_similarities[color]]
        
        if box_data:
            bp = axes[0, 1].boxplot(box_data, positions=positions[:len(box_data)], 
                                   patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[0, 1].set_title('Match Quality by Break Surface Color')
        axes[0, 1].set_xlabel('Break Surface Color')
        axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].set_xticks(positions[:len(box_data)])
        axes[0, 1].set_xticklabels(colors[:len(box_data)])
        
        # Plot 3: Number of matches per fragment pair
        pairs = list(pair_counts.keys())
        counts = list(pair_counts.values())
        
        axes[1, 0].bar(range(len(pairs)), counts, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Number of Matches per Fragment Pair')
        axes[1, 0].set_xlabel('Fragment Pair')
        axes[1, 0].set_ylabel('Number of Matches')
        axes[1, 0].set_xticks(range(len(pairs)))
        axes[1, 0].set_xticklabels([p.replace('fragment_', '').replace('_to_', '→') for p in pairs], 
                                  rotation=45)
        
        # Plot 4: Match quality heatmap-style visualization
        axes[1, 1].axis('off')
        
        # Create summary statistics table
        summary_text = f"""
        Summary Statistics:
        
        Total Matches Found: {len(all_similarities)}
        Mean Similarity: {np.mean(all_similarities):.3f}
        Std Similarity: {np.std(all_similarities):.3f}
        Best Match: {np.max(all_similarities) if all_similarities else 0:.3f}
        
        Matches by Color:
        Blue: {len(color_similarities['blue'])}
        Green: {len(color_similarities['green'])}
        Red: {len(color_similarities['red'])}
        
        Fragment Pairs with Matches: {len([p for p, c in pair_counts.items() if c > 0])}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                        verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Match quality report saved to {output_path}")
        
        plt.show()
    
    def create_reconstruction_report(self, fragments, transformations, quality_metrics, 
                                   output_path=None):
        """Create a comprehensive reconstruction report"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Reconstruction Quality Report', fontsize=16)
        
        # Plot 1: Fragment positions (before)
        ax = axes[0, 0]
        for i, fragment in enumerate(fragments):
            if 'mesh' in fragment:
                mesh = fragment['mesh']
                vertices = np.asarray(mesh.vertices)
                centroid = np.mean(vertices, axis=0)
                
                color = self.colors['fragment_colors'][i % len(self.colors['fragment_colors'])]
                ax.scatter(centroid[0], centroid[1], s=100, c=[color], 
                          label=f'Fragment {i}', alpha=0.7)
        
        ax.set_title('Original Fragment Positions')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Fragment positions (after)
        ax = axes[0, 1]
        for i, fragment in enumerate(fragments):
            if 'mesh' in fragment:
                mesh = copy.deepcopy(fragment['mesh'])
                if i in transformations:
                    mesh.transform(transformations[i])
                
                vertices = np.asarray(mesh.vertices)
                centroid = np.mean(vertices, axis=0)
                
                color = self.colors['fragment_colors'][i % len(self.colors['fragment_colors'])]
                ax.scatter(centroid[0], centroid[1], s=100, c=[color], 
                          label=f'Fragment {i}', alpha=0.7)
        
        ax.set_title('Aligned Fragment Positions')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Alignment quality metrics
        if quality_metrics:
            metrics_names = list(quality_metrics.keys())
            metrics_values = list(quality_metrics.values())
            
            ax = axes[0, 2]
            bars = ax.bar(range(len(metrics_names)), metrics_values, alpha=0.7)
            ax.set_title('Alignment Quality Metrics')
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_xticks(range(len(metrics_names)))
            ax.set_xticklabels(metrics_names, rotation=45, ha='right')
            
            # Color bars based on values (green for good, red for bad)
            for bar, value in zip(bars, metrics_values):
                if 'distance' in metrics_names[bars.index(bar)].lower():
                    # Lower is better for distance metrics
                    color_intensity = min(1.0, value / 0.05)  # Normalize to 5cm max
                    bar.set_color((color_intensity, 1 - color_intensity, 0))
                else:
                    # Higher is better for other metrics
                    color_intensity = min(1.0, value)
                    bar.set_color((1 - color_intensity, color_intensity, 0))
        
        # Plot 4: Transformation analysis
        ax = axes[1, 0]
        if transformations:
            translations = []
            rotations = []
            
            for i, transform in transformations.items():
                translation = np.linalg.norm(transform[:3, 3])
                rotation_matrix = transform[:3, :3]
                rotation_angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
                rotation_angle = np.degrees(rotation_angle)
                
                translations.append(translation)
                rotations.append(rotation_angle)
            
            ax.scatter(translations, rotations, s=100, alpha=0.7)
            ax.set_title('Transformation Magnitudes')
            ax.set_xlabel('Translation Distance')
            ax.set_ylabel('Rotation Angle (degrees)')
            ax.grid(True, alpha=0.3)
            
            for i, (t, r) in enumerate(zip(translations, rotations)):
                ax.annotate(f'F{i}', (t, r), xytext=(5, 5), 
                           textcoords='offset points')
        
        # Plot 5: Coverage analysis
        ax = axes[1, 1]
        
        # Calculate break surface coverage
        coverage_data = {'blue': 0, 'green': 0, 'red': 0}
        total_surfaces = {'blue': 0, 'green': 0, 'red': 0}
        
        for fragment in fragments:
            for color, surfaces in fragment['break_surfaces'].items():
                total_surfaces[color] += len(surfaces)
        
        # This would need to be calculated from actual matching results
        colors = list(coverage_data.keys())
        coverage_values = [coverage_data[color] / total_surfaces[color] 
                          if total_surfaces[color] > 0 else 0 for color in colors]
        
        bars = ax.bar(colors, coverage_values, alpha=0.7)
        ax.set_title('Break Surface Match Coverage')
        ax.set_xlabel('Surface Color')
        ax.set_ylabel('Coverage Ratio')
        ax.set_ylim(0, 1)
        
        for bar, color in zip(bars, colors):
            bar.set_color(self.colors['break_surface_colors'][color])
            bar.set_alpha(0.7)
        
        # Plot 6: Summary statistics
        axes[1, 2].axis('off')
        
        summary_text = f"""
        Reconstruction Summary:
        
        Total Fragments: {len(fragments)}
        Aligned Fragments: {len(transformations)}
        
        Quality Metrics:
        """
        
        if quality_metrics:
            for key, value in quality_metrics.items():
                if isinstance(value, float):
                    summary_text += f"{key}: {value:.4f}\n"
                else:
                    summary_text += f"{key}: {value}\n"
        
        summary_text += f"""
        
        Break Surface Summary:
        Total Blue Surfaces: {total_surfaces['blue']}
        Total Green Surfaces: {total_surfaces['green']}
        Total Red Surfaces: {total_surfaces['red']}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                        verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Reconstruction report saved to {output_path}")
        
        plt.show()
    
    def save_reconstruction(self, fragments, transformations, output_dir):
        """Save the reconstruction results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save individual aligned fragments
        for i, fragment in enumerate(fragments):
            if 'mesh' not in fragment or i not in transformations:
                continue
            
            mesh = copy.deepcopy(fragment['mesh'])
            mesh.transform(transformations[i])
            
            output_file = output_dir / f"aligned_fragment_{i}.ply"
            o3d.io.write_triangle_mesh(str(output_file), mesh)
        
        # Save combined reconstruction
        combined_mesh = o3d.geometry.TriangleMesh()
        for i, fragment in enumerate(fragments):
            if 'mesh' not in fragment or i not in transformations:
                continue
            
            mesh = copy.deepcopy(fragment['mesh'])
            mesh.transform(transformations[i])
            combined_mesh += mesh
        
        combined_output = output_dir / "reconstructed_artifact.ply"
        o3d.io.write_triangle_mesh(str(combined_output), combined_mesh)
        
        # Save transformation data
        transform_data = {}
        for i, transform in transformations.items():
            transform_data[f"fragment_{i}"] = transform.tolist()
        
        with open(output_dir / "transformations.json", 'w') as f:
            json.dump(transform_data, f, indent=2)
        
        print(f"Reconstruction saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Assuming you have the reconstruction data
    visualizer = ReconstructionVisualizer()
    
    # Create various visualizations
    # visualizer.visualize_original_fragments(fragments)
    # visualizer.create_match_quality_report(all_matches, "match_report.png")
    # visualizer.visualize_final_reconstruction(fragments, transformations)
    # visualizer.save_reconstruction(fragments, transformations, "reconstruction_output")
    
    print("Visualization tools ready!")