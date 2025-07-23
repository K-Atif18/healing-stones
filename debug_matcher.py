#!/usr/bin/env python3
"""
Debug script to understand why the matcher isn't predicting matches
"""

import numpy as np
import torch
import h5py
import pickle
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_predictions():
    """Analyze what's happening with the predictions."""
    
    # Import the matcher
    from forward_search_matcher import ForwardSearchMatcher
    
    # Initialize matcher
    matcher = ForwardSearchMatcher(
        fragment_embeddings_file="output_2/fragment_embeddings.h5",
        cluster_assembly_file="output/cluster_assembly_with_gt.h5",
        clusters_file="output/feature_clusters_fixed.pkl"
    )
    
    # Load trained models
    matcher.load_models("final")
    matcher.siamese_matcher.eval()
    
    logger.info("Analyzing predictions...")
    
    # Get a test pair with known GT matches
    test_pairs = []
    for match in matcher.dataset.positive_samples[:5]:  # Look at first 5 GT matches
        pair = (match['fragment_1'], match['fragment_2'])
        if pair not in test_pairs:
            test_pairs.append(pair)
    
    logger.info(f"Testing on {len(test_pairs)} fragment pairs with GT matches")
    
    for frag1, frag2 in test_pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing pair: {frag1} <-> {frag2}")
        logger.info(f"{'='*60}")
        
        # Get GT matches for this pair
        gt_matches = []
        for match in matcher.dataset.positive_samples:
            if (match['fragment_1'] == frag1 and match['fragment_2'] == frag2) or \
               (match['fragment_1'] == frag2 and match['fragment_2'] == frag1):
                gt_matches.append(match)
        
        logger.info(f"Ground truth matches: {len(gt_matches)}")
        
        # Get fragment embeddings
        frag_emb_1 = torch.FloatTensor(matcher.dataset.fragment_embeddings[frag1]).unsqueeze(0)
        frag_emb_2 = torch.FloatTensor(matcher.dataset.fragment_embeddings[frag2]).unsqueeze(0)
        
        # Test on actual GT matches
        for i, gt_match in enumerate(gt_matches[:3]):  # Test first 3 GT matches
            # Get cluster features
            if gt_match['fragment_1'] == frag1:
                c1, c2 = gt_match['cluster_id_1'], gt_match['cluster_id_2']
            else:
                c1, c2 = gt_match['cluster_id_2'], gt_match['cluster_id_1']
            
            feat1 = matcher.dataset.cluster_features.get((frag1, c1))
            feat2 = matcher.dataset.cluster_features.get((frag2, c2))
            
            if feat1 is None or feat2 is None:
                logger.warning(f"  Missing features for clusters {c1}, {c2}")
                continue
            
            cluster_feat_1 = torch.FloatTensor(feat1).unsqueeze(0)
            cluster_feat_2 = torch.FloatTensor(feat2).unsqueeze(0)
            
            # Use actual distance and normal similarity from GT
            distance_tensor = torch.FloatTensor([[gt_match['distance']]])
            normal_sim_tensor = torch.FloatTensor([[gt_match['normal_similarity']]])
            
            with torch.no_grad():
                match_score, ransac_residual = matcher.siamese_matcher(
                    frag_emb_1, frag_emb_2,
                    cluster_feat_1, cluster_feat_2,
                    distance_tensor, normal_sim_tensor
                )
            
            logger.info(f"\n  GT Match {i+1}: C{c1} <-> C{c2}")
            logger.info(f"    Distance: {gt_match['distance']:.2f}")
            logger.info(f"    Normal similarity: {gt_match['normal_similarity']:.3f}")
            logger.info(f"    GT confidence: {gt_match['gt_confidence']:.3f}")
            logger.info(f"    Predicted score: {match_score.item():.3f}")
            logger.info(f"    RANSAC residual: {ransac_residual.item():.3f}")
            
        # Also check why predict_matches fails
        logger.info("\nDebugging predict_matches...")
        
        # Check if clusters exist for these fragments
        clusters_1 = [k[1] for k in matcher.dataset.cluster_features.keys() if k[0] == frag1]
        clusters_2 = [k[1] for k in matcher.dataset.cluster_features.keys() if k[0] == frag2]
        
        logger.info(f"  Clusters for {frag1}: {len(clusters_1)} - samples: {clusters_1[:5]}")
        logger.info(f"  Clusters for {frag2}: {len(clusters_2)} - samples: {clusters_2[:5]}")
        
        # Check if the GT cluster IDs exist
        for gt in gt_matches[:3]:
            if gt['fragment_1'] == frag1:
                c1, c2 = gt['cluster_id_1'], gt['cluster_id_2']
            else:
                c1, c2 = gt['cluster_id_2'], gt['cluster_id_1']
            
            logger.info(f"  GT cluster {c1} in {frag1} list: {c1 in clusters_1}")
            logger.info(f"  GT cluster {c2} in {frag2} list: {c2 in clusters_2}")
        
        # Try to manually check a single pair
        if clusters_1 and clusters_2:
            logger.info("\n  Testing manual prediction on first cluster pair...")
            c1, c2 = clusters_1[0], clusters_2[0]
            
            feat1 = matcher.dataset.cluster_features.get((frag1, c1))
            feat2 = matcher.dataset.cluster_features.get((frag2, c2))
            
            if feat1 is not None and feat2 is not None:
                logger.info(f"    Features found for C{c1} and C{c2}")
                
                cluster_feat_1 = torch.FloatTensor(feat1).unsqueeze(0)
                cluster_feat_2 = torch.FloatTensor(feat2).unsqueeze(0)
                
                distance = np.linalg.norm(feat1[:3] - feat2[:3])
                distance_tensor = torch.FloatTensor([[distance]])
                normal_sim_tensor = torch.FloatTensor([[0.5]])
                
                with torch.no_grad():
                    match_score, ransac_residual = matcher.siamese_matcher(
                        frag_emb_1, frag_emb_2,
                        cluster_feat_1, cluster_feat_2,
                        distance_tensor, normal_sim_tensor
                    )
                
                logger.info(f"    Distance: {distance:.2f}")
                logger.info(f"    Predicted score: {match_score.item():.3f}")
                
                # Check shape outlier detector with batch fix
                shape_features = torch.cat([
                    cluster_feat_1[:, 3:5],
                    cluster_feat_2[:, 3:5],
                    torch.abs(cluster_feat_1[:, 3:5] - cluster_feat_2[:, 3:5])
                ], dim=1)
                
                # Add dummy batch to avoid batch norm issues
                if shape_features.shape[0] == 1:
                    shape_features = torch.cat([shape_features, shape_features], dim=0)
                    outlier_score = matcher.shape_outlier_detector(shape_features)[0:1]
                else:
                    outlier_score = matcher.shape_outlier_detector(shape_features)
                
                logger.info(f"    Outlier score: {outlier_score.item():.3f}")
                logger.info(f"    Would be filtered: {outlier_score.item() >= 0.7}")
        
        # Let's check the validation step specifically
        logger.info("\nChecking validation step...")
        
        # Try without validation first
        logger.info("Testing predict_matches without validation...")
        
        # Temporarily disable validation
        matches_no_validation = []
        
        frag_emb_1 = torch.FloatTensor(matcher.dataset.fragment_embeddings[frag1]).unsqueeze(0)
        frag_emb_2 = torch.FloatTensor(matcher.dataset.fragment_embeddings[frag2]).unsqueeze(0)
        
        clusters_1 = [k[1] for k in matcher.dataset.cluster_features.keys() if k[0] == frag1]
        clusters_2 = [k[1] for k in matcher.dataset.cluster_features.keys() if k[0] == frag2]
        
        # Just check first 50 cluster pairs to speed up
        count = 0
        for c1 in clusters_1[:50]:
            for c2 in clusters_2[:50]:
                feat1 = matcher.dataset.cluster_features.get((frag1, c1))
                feat2 = matcher.dataset.cluster_features.get((frag2, c2))
                
                if feat1 is None or feat2 is None:
                    continue
                
                cluster_feat_1 = torch.FloatTensor(feat1).unsqueeze(0)
                cluster_feat_2 = torch.FloatTensor(feat2).unsqueeze(0)
                
                distance = np.linalg.norm(feat1[:3] - feat2[:3])
                
                # Only process if distance < 100mm
                if distance > 100:
                    continue
                
                distance_tensor = torch.FloatTensor([[distance]])
                normal_sim_tensor = torch.FloatTensor([[0.5]])
                
                with torch.no_grad():
                    match_score, ransac_residual = matcher.siamese_matcher(
                        frag_emb_1, frag_emb_2,
                        cluster_feat_1, cluster_feat_2,
                        distance_tensor, normal_sim_tensor
                    )
                
                # Check shape outlier with batch fix
                shape_features = torch.cat([
                    cluster_feat_1[:, 3:5],
                    cluster_feat_2[:, 3:5],
                    torch.abs(cluster_feat_1[:, 3:5] - cluster_feat_2[:, 3:5])
                ], dim=1)
                
                if shape_features.shape[0] == 1:
                    shape_features = torch.cat([shape_features, shape_features], dim=0)
                    outlier_score = matcher.shape_outlier_detector(shape_features)[0:1]
                else:
                    outlier_score = matcher.shape_outlier_detector(shape_features)
                
                matches_no_validation.append({
                    'c1': c1,
                    'c2': c2,
                    'distance': distance,
                    'confidence': match_score.item(),
                    'outlier_score': outlier_score.item(),
                    'passes_outlier': outlier_score.item() < 0.7,
                    'passes_confidence': match_score.item() > 0.5
                })
                
                count += 1
                if count >= 20:  # Just check first 20 that pass distance filter
                    break
            if count >= 20:
                break
        
        # Sort by confidence
        matches_no_validation.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"\nWithout validation, found {len(matches_no_validation)} candidate matches")
        logger.info(f"Matches passing confidence > 0.5: {sum(1 for m in matches_no_validation if m['passes_confidence'])}")
        logger.info(f"Matches passing outlier < 0.7: {sum(1 for m in matches_no_validation if m['passes_outlier'])}")
        logger.info(f"Matches passing both: {sum(1 for m in matches_no_validation if m['passes_confidence'] and m['passes_outlier'])}")
        
        if matches_no_validation:
            logger.info("\nTop 5 matches (no validation):")
            for i, m in enumerate(matches_no_validation[:5]):
                logger.info(f"  {i+1}. C{m['c1']} <-> C{m['c2']}: conf={m['confidence']:.3f}, "
                          f"dist={m['distance']:.1f}, outlier={m['outlier_score']:.3f}")
        
        # Check if any GT matches are in the top candidates
        gt_in_candidates = 0
        for m in matches_no_validation:
            for gt in gt_matches:
                if gt['fragment_1'] == frag1:
                    gt_c1, gt_c2 = gt['cluster_id_1'], gt['cluster_id_2']
                else:
                    gt_c1, gt_c2 = gt['cluster_id_2'], gt['cluster_id_1']
                
                if m['c1'] == gt_c1 and m['c2'] == gt_c2:
                    gt_in_candidates += 1
                    logger.info(f"\nFound GT match C{gt_c1} <-> C{gt_c2} at position "
                              f"{matches_no_validation.index(m)+1} with confidence {m['confidence']:.3f}")
                    break
        
        logger.info(f"\nGT matches in candidates: {gt_in_candidates}/{len(gt_matches)}")
        
        # Now check the validation threshold
        logger.info("\nChecking validation threshold...")
        test_match = {
            'fragment_1': frag1,
            'fragment_2': frag2,
            'distance': 10.0,
            'normal_sim': 0.5
        }
        is_valid, score = matcher.validator.validate_cluster_match(test_match, threshold=5.0)
        logger.info(f"Test match (dist=10, normal=0.5): valid={is_valid}, score={score:.3f}")
        
        if all_predictions:
            logger.info(f"  Total predictions: {len(all_predictions)}")
            logger.info(f"  Top 5 confidence scores: {[p['confidence'] for p in all_predictions[:5]]}")
            
            # Check if any GT matches are in predictions
            gt_found = 0
            for pred in all_predictions:
                for gt in gt_matches:
                    if gt['fragment_1'] == frag1:
                        gt_c1, gt_c2 = gt['cluster_id_1'], gt['cluster_id_2']
                    else:
                        gt_c1, gt_c2 = gt['cluster_id_2'], gt['cluster_id_1']
                    
                    if pred['cluster_id_1'] == gt_c1 and pred['cluster_id_2'] == gt_c2:
                        gt_found += 1
                        logger.info(f"  Found GT match at position {all_predictions.index(pred)+1} with confidence {pred['confidence']:.3f}")
                        break
            
            logger.info(f"  GT matches found in predictions: {gt_found}/{len(gt_matches)}")
        else:
            logger.info("  No predictions returned!")
    
    # Check the distribution of scores on training data
    logger.info(f"\n{'='*60}")
    logger.info("Checking score distribution on training data...")
    logger.info(f"{'='*60}")
    
    positive_scores = []
    negative_scores = []
    
    # Sample some data
    for i, sample in enumerate(matcher.dataset.all_samples[:100]):
        frag_emb_1 = torch.FloatTensor(matcher.dataset.fragment_embeddings[sample['fragment_1']]).unsqueeze(0)
        frag_emb_2 = torch.FloatTensor(matcher.dataset.fragment_embeddings[sample['fragment_2']]).unsqueeze(0)
        
        feat1 = matcher.dataset.cluster_features.get((sample['fragment_1'], sample['cluster_id_1']))
        feat2 = matcher.dataset.cluster_features.get((sample['fragment_2'], sample['cluster_id_2']))
        
        if feat1 is None or feat2 is None:
            continue
        
        cluster_feat_1 = torch.FloatTensor(feat1).unsqueeze(0)
        cluster_feat_2 = torch.FloatTensor(feat2).unsqueeze(0)
        
        distance_tensor = torch.FloatTensor([[sample['distance']]])
        normal_sim_tensor = torch.FloatTensor([[sample['normal_similarity']]])
        
        with torch.no_grad():
            match_score, _ = matcher.siamese_matcher(
                frag_emb_1, frag_emb_2,
                cluster_feat_1, cluster_feat_2,
                distance_tensor, normal_sim_tensor
            )
        
        if sample['is_ground_truth']:
            positive_scores.append(match_score.item())
        else:
            negative_scores.append(match_score.item())
    
    logger.info(f"\nPositive samples: {len(positive_scores)}")
    if positive_scores:
        logger.info(f"  Mean score: {np.mean(positive_scores):.3f}")
        logger.info(f"  Min/Max: {np.min(positive_scores):.3f} / {np.max(positive_scores):.3f}")
        logger.info(f"  > 0.5: {sum(s > 0.5 for s in positive_scores)}")
    
    logger.info(f"\nNegative samples: {len(negative_scores)}")
    if negative_scores:
        logger.info(f"  Mean score: {np.mean(negative_scores):.3f}")
        logger.info(f"  Min/Max: {np.min(negative_scores):.3f} / {np.max(negative_scores):.3f}")
        logger.info(f"  > 0.5: {sum(s > 0.5 for s in negative_scores)}")


if __name__ == "__main__":
    analyze_predictions()