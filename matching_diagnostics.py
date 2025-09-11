#!/usr/bin/env python3
"""
Diagnostic script to debug surface matching issues
"""

import numpy as np
import json
from pathlib import Path
from ply_loader import PLYColorExtractor
from feature_extractor import BreakSurfaceFeatureExtractor
from surface_matcher import SurfaceMatcher

def diagnose_matching_pipeline(input_directory):
    """Diagnose each step of the matching pipeline"""
    
    print("SURFACE MATCHING DIAGNOSTICS")
    print("=" * 50)
    
    # Step 1: Load fragments
    print("\n1. Loading fragments...")
    extractor = PLYColorExtractor()
    fragments = extractor.process_all_fragments(input_directory)
    
    if not fragments:
        print("❌ No fragments loaded!")
        return
    
    print(f"✓ Loaded {len(fragments)} fragments")
    
    # Check break surfaces
    total_surfaces = 0
    for i, fragment in enumerate(fragments):
        surfaces_count = sum(len(surfaces) for surfaces in fragment['break_surfaces'].values())
        total_surfaces += surfaces_count
        print(f"  Fragment {i}: {surfaces_count} break surfaces")
    
    print(f"  Total: {total_surfaces} break surfaces")
    
    if total_surfaces == 0:
        print("❌ No break surfaces found!")
        return
    
    # Step 2: Extract features
    print("\n2. Extracting features...")
    feature_extractor = BreakSurfaceFeatureExtractor()
    enhanced_fragments = feature_extractor.extract_all_features(fragments)
    
    # Check features
    total_features = 0
    for i, fragment in enumerate(enhanced_fragments):
        fragment_features = 0
        for color, features in fragment['features'].items():
            fragment_features += len(features)
        total_features += fragment_features
        print(f"  Fragment {i}: {fragment_features} feature sets")
    
    print(f"  Total: {total_features} feature sets")
    
    if total_features == 0:
        print("❌ No features extracted!")
        return
    
    # Step 3: Examine features in detail
    print("\n3. Examining features in detail...")
    
    for i, fragment in enumerate(enhanced_fragments[:2]):  # Check first 2 fragments
        print(f"\n  Fragment {i} features:")
        for color, features_list in fragment['features'].items():
            if features_list:
                features = features_list[0]  # First surface of this color
                print(f"    {color} surface:")
                print(f"      Size: {features.get('size', 'missing')}")
                print(f"      Normal: {features.get('normal', 'missing')}")
                print(f"      Area: {features.get('area', 'missing')}")
                print(f"      Planarity: {features.get('planarity', 'missing')}")
                
                # Check for NaN or invalid values
                if 'normal' in features:
                    normal = np.array(features['normal'])
                    if np.isnan(normal).any():
                        print(f"      ❌ Normal contains NaN!")
                    elif np.linalg.norm(normal) == 0:
                        print(f"      ❌ Normal is zero vector!")
                
                if 'area' in features:
                    if np.isnan(features['area']) or features['area'] <= 0:
                        print(f"      ❌ Invalid area: {features['area']}")
    
    # Step 4: Test similarity computation
    print("\n4. Testing similarity computation...")
    
    matcher = SurfaceMatcher()
    
    # Find two fragments with features
    frag1, frag2 = None, None
    for fragment in enhanced_fragments:
        total_feats = sum(len(features) for features in fragment['features'].values())
        if total_feats > 0:
            if frag1 is None:
                frag1 = fragment
            elif frag2 is None:
                frag2 = fragment
                break
    
    if frag1 is None or frag2 is None:
        print("❌ Not enough fragments with features for testing!")
        return
    
    print(f"  Testing similarity between fragments...")
    
    # Get all surfaces from both fragments
    surf1_list = []
    surf2_list = []
    
    for color in ['blue', 'green', 'red']:
        if color in frag1['features'] and frag1['features'][color]:
            surf1_list.extend([(color, feat) for feat in frag1['features'][color]])
        if color in frag2['features'] and frag2['features'][color]:
            surf2_list.extend([(color, feat) for feat in frag2['features'][color]])
    
    print(f"  Fragment 1 surfaces: {len(surf1_list)}")
    print(f"  Fragment 2 surfaces: {len(surf2_list)}")
    
    if not surf1_list or not surf2_list:
        print("❌ No surfaces with features to compare!")
        return
    
    # Test all pairwise similarities
    similarities = []
    
    for i, (color1, feat1) in enumerate(surf1_list):
        for j, (color2, feat2) in enumerate(surf2_list):
            try:
                similarity, detailed = matcher.compute_overall_similarity(feat1, feat2)
                similarities.append({
                    'colors': f"{color1}↔{color2}",
                    'similarity': similarity,
                    'detailed': detailed
                })
                print(f"    {color1} ↔ {color2}: {similarity:.4f}")
                
                # Show detailed breakdown
                for key, value in detailed.items():
                    if np.isnan(value):
                        print(f"      ❌ {key}: NaN")
                    else:
                        print(f"      {key}: {value:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error computing {color1} ↔ {color2} similarity: {e}")
    
    if similarities:
        max_sim = max(s['similarity'] for s in similarities)
        avg_sim = np.mean([s['similarity'] for s in similarities])
        print(f"\n  Similarity statistics:")
        print(f"    Maximum: {max_sim:.4f}")
        print(f"    Average: {avg_sim:.4f}")
        print(f"    Count: {len(similarities)}")
        
        # Check thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            count = sum(1 for s in similarities if s['similarity'] >= threshold)
            print(f"    Matches at {threshold:.1f}: {count}")
    
    # Step 5: Test full matching
    print("\n5. Testing full matching pipeline...")
    
    try:
        all_matches = matcher.find_all_matches(enhanced_fragments, min_similarity=0.3)
        
        total_matches = 0
        for pair_key, matches in all_matches.items():
            if isinstance(matches, list):  # New structure
                total_matches += len(matches)
                print(f"  {pair_key}: {len(matches)} matches")
                if matches:
                    best_match = max(matches, key=lambda x: x['similarity'])
                    print(f"    Best: {best_match['fragment1_color']} ↔ {best_match['fragment2_color']} ({best_match['similarity']:.4f})")
            else:  # Old structure
                for color, color_matches in matches.items():
                    total_matches += len(color_matches)
                    print(f"  {pair_key} {color}: {len(color_matches)} matches")
        
        print(f"  Total matches at 0.3 threshold: {total_matches}")
        
    except Exception as e:
        print(f"❌ Error in full matching: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose surface matching issues")
    parser.add_argument('input_dir', help='Directory containing PLY files')
    
    args = parser.parse_args()
    
    diagnose_matching_pipeline(args.input_dir)

if __name__ == "__main__":
    main()