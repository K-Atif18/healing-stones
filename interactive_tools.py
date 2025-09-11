import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import our reconstruction modules
from main_pipeline import ReconstructionPipeline
from config_manager import ConfigurationManager
from surface_matcher import SurfaceMatcher
from fragment_aligner import FragmentAligner

class InteractiveReconstructionTool:
    """
    Interactive GUI tool for manual reconstruction refinement
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Interactive Stele Reconstruction Tool")
        self.root.geometry("1200x800")
        
        # Data
        self.fragments = []
        self.current_matches = {}
        self.transformations = {}
        self.pipeline = None
        self.config_manager = ConfigurationManager()
        
        # Open3D visualizer
        self.vis = None
        self.visualization_thread = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Main menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load PLY Directory", command=self.load_ply_directory)
        file_menu.add_command(label="Load Reconstruction", command=self.load_reconstruction)
        file_menu.add_command(label="Save Reconstruction", command=self.save_reconstruction)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show 3D Viewer", command=self.show_3d_viewer)
        view_menu.add_command(label="Show Fragment List", command=self.show_fragment_list)
        view_menu.add_command(label="Show Match Details", command=self.show_match_details)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Auto-Align All", command=self.auto_align_all)
        tools_menu.add_command(label="Manual Alignment", command=self.manual_alignment_mode)
        tools_menu.add_command(label="Validate Matches", command=self.validate_matches)
        tools_menu.add_command(label="Reset Transformations", command=self.reset_transformations)
        
        # Create main layout
        self.setup_main_layout()
        
    def setup_main_layout(self):
        """Setup the main layout with panels"""
        
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Visualization and details
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.setup_control_panel(left_frame)
        self.setup_visualization_panel(right_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        
        # Project section
        project_frame = ttk.LabelFrame(parent, text="Project", padding=10)
        project_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(project_frame, text="Load PLY Directory", 
                  command=self.load_ply_directory).pack(fill=tk.X, pady=2)
        
        self.status_label = ttk.Label(project_frame, text="No project loaded")
        self.status_label.pack(fill=tk.X, pady=2)
        
        # Fragment list
        fragment_frame = ttk.LabelFrame(parent, text="Fragments", padding=10)
        fragment_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Fragment listbox with scrollbar
        list_frame = ttk.Frame(fragment_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fragment_listbox = tk.Listbox(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.fragment_listbox.yview)
        self.fragment_listbox.config(yscrollcommand=scrollbar.set)
        
        self.fragment_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.fragment_listbox.bind('<<ListboxSelect>>', self.on_fragment_select)
        
        # Fragment controls
        frag_control_frame = ttk.Frame(fragment_frame)
        frag_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frag_control_frame, text="Show", 
                  command=self.show_selected_fragment).pack(side=tk.LEFT, padx=2)
        ttk.Button(frag_control_frame, text="Hide", 
                  command=self.hide_selected_fragment).pack(side=tk.LEFT, padx=2)
        ttk.Button(frag_control_frame, text="Reset", 
                  command=self.reset_selected_fragment).pack(side=tk.LEFT, padx=2)
        
        # Reconstruction controls
        recon_frame = ttk.LabelFrame(parent, text="Reconstruction", padding=10)
        recon_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(recon_frame, text="Find Matches", 
                  command=self.find_matches).pack(fill=tk.X, pady=2)
        ttk.Button(recon_frame, text="Auto Align", 
                  command=self.auto_align_all).pack(fill=tk.X, pady=2)
        ttk.Button(recon_frame, text="Manual Align", 
                  command=self.manual_alignment_mode).pack(fill=tk.X, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(recon_frame, variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
    def setup_visualization_panel(self, parent):
        """Setup the visualization panel"""
        
        # Create notebook for different views
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 3D View tab
        view_3d_frame = ttk.Frame(notebook)
        notebook.add(view_3d_frame, text="3D View")
        
        view_control_frame = ttk.Frame(view_3d_frame)
        view_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(view_control_frame, text="Update View", 
                  command=self.update_3d_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(view_control_frame, text="Reset View", 
                  command=self.reset_3d_view).pack(side=tk.LEFT, padx=5)
        
        # 3D view placeholder
        self.view_3d_label = ttk.Label(view_3d_frame, 
                                      text="3D visualization will appear here\\nClick 'Show 3D Viewer' to open separate window")
        self.view_3d_label.pack(expand=True)
        
        # Matches tab
        matches_frame = ttk.Frame(notebook)
        notebook.add(matches_frame, text="Matches")
        
        # Match list
        self.match_tree = ttk.Treeview(matches_frame, columns=('Fragments', 'Color', 'Similarity'), show='headings')
        self.match_tree.heading('Fragments', text='Fragments')
        self.match_tree.heading('Color', text='Color')
        self.match_tree.heading('Similarity', text='Similarity')
        self.match_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        match_control_frame = ttk.Frame(matches_frame)
        match_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(match_control_frame, text="Accept Match", 
                  command=self.accept_match).pack(side=tk.LEFT, padx=5)
        ttk.Button(match_control_frame, text="Reject Match", 
                  command=self.reject_match).pack(side=tk.LEFT, padx=5)
        ttk.Button(match_control_frame, text="Show Match", 
                  command=self.show_match_details).pack(side=tk.LEFT, padx=5)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def load_ply_directory(self):
        """Load PLY files from directory"""
        directory = filedialog.askdirectory(title="Select PLY Directory")
        if not directory:
            return
        
        self.status_label.config(text="Loading PLY files...")
        self.root.update()
        
        try:
            # Run pipeline extraction in separate thread
            def load_thread():
                self.pipeline = ReconstructionPipeline()
                self.fragments = self.pipeline.load_and_process_fragments(directory)
                self.fragments = self.pipeline.extract_features()
                
                # Update GUI in main thread
                self.root.after(0, self.on_fragments_loaded)
            
            threading.Thread(target=load_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PLY files: {e}")
            self.status_label.config(text="Error loading files")
    
    def on_fragments_loaded(self):
        """Called when fragments are loaded"""
        self.update_fragment_list()
        self.update_statistics()
        self.status_label.config(text=f"Loaded {len(self.fragments)} fragments")
        
    def update_fragment_list(self):
        """Update the fragment list widget"""
        self.fragment_listbox.delete(0, tk.END)
        
        for i, fragment in enumerate(self.fragments):
            name = Path(fragment['filepath']).name
            
            # Count break surfaces
            total_surfaces = sum(len(surfaces) for surfaces in fragment['break_surfaces'].values())
            
            self.fragment_listbox.insert(tk.END, f"{i}: {name} ({total_surfaces} surfaces)")
    
    def on_fragment_select(self, event):
        """Handle fragment selection"""
        selection = self.fragment_listbox.curselection()
        if selection:
            fragment_idx = selection[0]
            self.show_fragment_details(fragment_idx)
    
    def show_fragment_details(self, fragment_idx):
        """Show details for selected fragment"""
        if fragment_idx >= len(self.fragments):
            return
        
        fragment = self.fragments[fragment_idx]
        
        details = f"Fragment {fragment_idx}:\\n"
        details += f"File: {Path(fragment['filepath']).name}\\n"
        details += f"Vertices: {len(fragment['mesh'].vertices)}\\n"
        details += f"Triangles: {len(fragment['mesh'].triangles)}\\n\\n"
        
        details += "Break Surfaces:\\n"
        for color, surfaces in fragment['break_surfaces'].items():
            details += f"  {color}: {len(surfaces)} surfaces\\n"
            for i, surface in enumerate(surfaces):
                details += f"    Surface {i}: {surface['size']} points\\n"
        
        # Update statistics text
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, details)
    
    def find_matches(self):
        """Find surface matches between fragments"""
        if not self.fragments:
            messagebox.showwarning("Warning", "No fragments loaded")
            return
        
        self.status_label.config(text="Finding matches...")
        self.progress_var.set(0)
        self.root.update()
        
        def match_thread():
            try:
                self.current_matches = self.pipeline.find_surface_matches()
                self.root.after(0, self.on_matches_found)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Match finding failed: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="Match finding failed"))
        
        threading.Thread(target=match_thread, daemon=True).start()
    
    def on_matches_found(self):
        """Called when matches are found"""
        self.update_match_list()
        self.status_label.config(text="Matches found")
        self.progress_var.set(100)
    
    def update_match_list(self):
        """Update the match list widget"""
        # Clear existing items
        for item in self.match_tree.get_children():
            self.match_tree.delete(item)
        
        # Add matches
        for pair_key, color_matches in self.current_matches.items():
            for color, matches in color_matches.items():
                for match in matches:
                    fragments_text = pair_key.replace('fragment_', '').replace('_to_', ' ↔ ')
                    similarity_text = f"{match['similarity']:.3f}"
                    
                    self.match_tree.insert('', tk.END, values=(fragments_text, color, similarity_text))
    
    def auto_align_all(self):
        """Automatically align all fragments"""
        if not self.current_matches:
            messagebox.showwarning("Warning", "No matches found. Run 'Find Matches' first.")
            return
        
        self.status_label.config(text="Auto-aligning fragments...")
        self.progress_var.set(0)
        self.root.update()
        
        def align_thread():
            try:
                self.transformations = self.pipeline.align_fragments()
                self.root.after(0, self.on_alignment_complete)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Alignment failed: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="Alignment failed"))
        
        threading.Thread(target=align_thread, daemon=True).start()
    
    def on_alignment_complete(self):
        """Called when alignment is complete"""
        self.status_label.config(text=f"Aligned {len(self.transformations)} fragments")
        self.progress_var.set(100)
        self.update_statistics()
        
        # Update 3D view if open
        if self.vis is not None:
            self.update_3d_view()
    
    def show_3d_viewer(self):
        """Show 3D viewer in separate window"""
        if not self.fragments:
            messagebox.showwarning("Warning", "No fragments loaded")
            return
        
        def visualization_thread():
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Interactive 3D Viewer", width=800, height=600)
            
            # Add fragments to visualization
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
            
            for i, fragment in enumerate(self.fragments):
                if 'mesh' in fragment:
                    mesh = fragment['mesh'].copy()
                    
                    # Apply transformation if available
                    if i in self.transformations:
                        mesh.transform(self.transformations[i])
                    
                    # Color the fragment
                    color = colors[i % len(colors)]
                    mesh.paint_uniform_color(color)
                    
                    self.vis.add_geometry(mesh)
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(coordinate_frame)
            
            self.vis.run()
            self.vis.destroy_window()
            self.vis = None
        
        if self.visualization_thread is None or not self.visualization_thread.is_alive():
            self.visualization_thread = threading.Thread(target=visualization_thread, daemon=True)
            self.visualization_thread.start()
    
    def update_3d_view(self):
        """Update the 3D view with current transformations"""
        # This would update the existing 3D view if implemented
        pass
    
    def reset_3d_view(self):
        """Reset the 3D view to default"""
        # This would reset the 3D view if implemented
        pass
    
    def accept_match(self):
        """Accept the selected match"""
        selection = self.match_tree.selection()
        if selection:
            messagebox.showinfo("Info", "Match accepted")
    
    def reject_match(self):
        """Reject the selected match"""
        selection = self.match_tree.selection()
        if selection:
            item = self.match_tree.item(selection[0])
            self.match_tree.delete(selection[0])
            messagebox.showinfo("Info", "Match rejected")
    
    def show_match_details(self):
        """Show detailed information about selected match"""
        selection = self.match_tree.selection()
        if selection:
            messagebox.showinfo("Info", "Match details would be shown here")
    
    def manual_alignment_mode(self):
        """Enter manual alignment mode"""
        messagebox.showinfo("Info", "Manual alignment mode would be activated here")
    
    def validate_matches(self):
        """Validate all current matches"""
        if self.current_matches:
            total_matches = sum(len(matches) for color_matches in self.current_matches.values() 
                              for matches in color_matches.values())
            messagebox.showinfo("Validation", f"Found {total_matches} total matches")
        else:
            messagebox.showwarning("Warning", "No matches to validate")
    
    def reset_transformations(self):
        """Reset all transformations"""
        self.transformations.clear()
        self.status_label.config(text="Transformations reset")
        self.update_statistics()
    
    def show_selected_fragment(self):
        """Show the selected fragment in 3D view"""
        selection = self.fragment_listbox.curselection()
        if selection:
            messagebox.showinfo("Info", f"Showing fragment {selection[0]}")
    
    def hide_selected_fragment(self):
        """Hide the selected fragment from 3D view"""
        selection = self.fragment_listbox.curselection()
        if selection:
            messagebox.showinfo("Info", f"Hiding fragment {selection[0]}")
    
    def reset_selected_fragment(self):
        """Reset the selected fragment's transformation"""
        selection = self.fragment_listbox.curselection()
        if selection:
            fragment_idx = selection[0]
            if fragment_idx in self.transformations:
                del self.transformations[fragment_idx]
                messagebox.showinfo("Info", f"Reset fragment {fragment_idx}")
                self.update_statistics()
    
    def update_statistics(self):
        """Update the statistics display"""
        if not self.fragments:
            return
        
        stats = "RECONSTRUCTION STATISTICS\\n"
        stats += "=" * 30 + "\\n\\n"
        
        stats += f"Total fragments: {len(self.fragments)}\\n"
        stats += f"Aligned fragments: {len(self.transformations)}\\n"
        
        if self.current_matches:
            total_matches = sum(len(matches) for color_matches in self.current_matches.values() 
                              for matches in color_matches.values())
            stats += f"Total matches: {total_matches}\\n"
        
        # Break surface statistics
        total_surfaces = {'blue': 0, 'green': 0, 'red': 0}
        for fragment in self.fragments:
            for color, surfaces in fragment['break_surfaces'].items():
                total_surfaces[color] += len(surfaces)
        
        stats += "\\nBreak surfaces:\\n"
        for color, count in total_surfaces.items():
            stats += f"  {color}: {count}\\n"
        
        # Quality metrics
        if hasattr(self.pipeline, 'quality_metrics') and self.pipeline.quality_metrics:
            stats += "\\nQuality metrics:\\n"
            for key, value in self.pipeline.quality_metrics.items():
                if isinstance(value, float):
                    stats += f"  {key}: {value:.4f}\\n"
                else:
                    stats += f"  {key}: {value}\\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def save_reconstruction(self):
        """Save the current reconstruction"""
        if not self.transformations:
            messagebox.showwarning("Warning", "No reconstruction to save")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if output_dir:
            try:
                # Save using the pipeline's save functionality
                if hasattr(self.pipeline, 'visualizer'):
                    self.pipeline.visualizer.save_reconstruction(
                        self.fragments, self.transformations, output_dir
                    )
                    messagebox.showinfo("Success", f"Reconstruction saved to {output_dir}")
                else:
                    messagebox.showerror("Error", "Cannot save reconstruction")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save reconstruction: {e}")
    
    def load_reconstruction(self):
        """Load a saved reconstruction"""
        file_path = filedialog.askopenfilename(
            title="Select Reconstruction File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Load transformations
                if 'transformations' in data:
                    self.transformations = {}
                    for key, transform_list in data['transformations'].items():
                        idx = int(key.split('_')[1])  # Extract fragment index
                        self.transformations[idx] = np.array(transform_list)
                
                self.update_statistics()
                messagebox.showinfo("Success", "Reconstruction loaded")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load reconstruction: {e}")
    
    def run(self):
        """Run the interactive tool"""
        self.root.mainloop()

# Command line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Stele Reconstruction Tool")
    parser.add_argument('--input', help='Initial PLY directory to load')
    
    args = parser.parse_args()
    
    tool = InteractiveReconstructionTool()
    
    if args.input:
        # Auto-load directory if specified
        tool.root.after(100, lambda: tool.load_ply_directory() if not args.input else None)
    
    tool.run()

if __name__ == "__main__":
    main()