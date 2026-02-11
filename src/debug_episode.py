#!/usr/bin/env python3
"""Debug script to trace evader behavior in detail."""
import numpy as np
import sys
import os

# Add parent dir to path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_environment import MultiAgentPursuitEnv

def main():
    env = MultiAgentPursuitEnv(train_pursuers=False, train_evaders=False)
    obs, _ = env.reset()
    
    print("=== INITIAL STATE ===")
    print(f"Arena radius: {env.arena_cfg.radius}")
    print(f"Evader speed: {env.evader_cfg.max_speed}")
    print(f"Pursuer speed: {env.pursuer_cfg.max_speed}")
    
    print("\n=== STATION POSITIONS ===")
    for s in env.stations:
        dist = np.linalg.norm(s.position)
        print(f"  Station {s.station_id}: {s.position} (dist from center: {dist:.1f})")
    
    print("\n=== PURSUER POSITIONS ===")
    for p in env.pursuers:
        dist = np.linalg.norm(p.position)
        print(f"  Pursuer {p.pursuer_id} @ station {p.assigned_station}: {p.position} (dist: {dist:.1f})")
    
    print("\n=== EVADER POSITIONS ===")
    for e in env.evaders:
        dist_to_goal = np.linalg.norm(e.goal - e.position)
        dist_from_center = np.linalg.norm(e.position)
        print(f"  Evader {e.evader_id}: pos={e.position}, goal={e.goal}")
        print(f"    dist to goal: {dist_to_goal:.1f}, dist from center: {dist_from_center:.1f}")
    
    print("\n=== DISTANCES EVADER->PURSUER ===")
    for e in env.evaders:
        print(f"  Evader {e.evader_id}:")
        for p in env.pursuers:
            dist = np.linalg.norm(p.position - e.position)
            print(f"    -> Pursuer {p.pursuer_id}: {dist:.1f}m")
    
    # Run more steps
    print("\n=== RUNNING STEPS ===")
    for step in range(200):
        # Scan first (before step clears)
        for e in env.evaders:
            e.scan_for_pursuers(env.pursuers)
        
        print(f"\nStep {step+1} (BEFORE move):")
        active_evaders = [e for e in env.evaders if not e.is_captured and not e.is_at_goal]
        for e in active_evaders:
            print(f"  Evader {e.evader_id}: detected_pursuers={len(e.detected_pursuers)}, range={e.detection_range}")
        
        # Dummy actions (heuristics will run)
        dummy_action = {
            'pursuers': np.zeros((env.num_pursuers, 2)),
            'evaders': np.zeros((env.num_evaders, 3)),
        }
        obs, rewards, terminated, truncated, info = env.step(dummy_action)
        
        active_evaders = [e for e in env.evaders if not e.is_captured and not e.is_at_goal]
        chased = [e for e in active_evaders if e.is_being_chased]
        free = [e for e in active_evaders if not e.is_being_chased]
        
        print(f"\nStep {step+1}:")
        print(f"  Active evaders: {len(active_evaders)}, Chased: {len(chased)}, Free: {len(free)}")
        
        for e in active_evaders:
            dist_to_goal = np.linalg.norm(e.goal - e.position)
            min_pursuer_dist = min(np.linalg.norm(p.position - e.position) for p in env.pursuers)
            detected = len(e.detected_pursuers)
            print(f"  Evader {e.evader_id}: goal_dist={dist_to_goal:.1f}, pursuer_dist={min_pursuer_dist:.1f}, detected={detected}, chased={e.is_being_chased}")
        
        for p in env.pursuers:
            if p.target_evader_id is not None:
                target_evader = next((e for e in env.evaders if e.evader_id == p.target_evader_id), None)
                if target_evader:
                    dist = np.linalg.norm(p.position - target_evader.position)
                    print(f"  Pursuer {p.pursuer_id} -> Evader {p.target_evader_id}: {dist:.1f}m")
        
        if terminated or truncated:
            print(f"\n=== EPISODE ENDED at step {step+1} ===")
            print(f"  Captures: {sum(1 for e in env.evaders if e.is_captured)}")
            print(f"  Goals: {sum(1 for e in env.evaders if e.is_at_goal)}")
            break
        
        # Check if any evader captured this step
        for e in env.evaders:
            if e.is_captured and not hasattr(e, '_capture_logged'):
                print(f"\n  *** EVADER {e.evader_id} CAPTURED at step {step+1}! ***")
                e._capture_logged = True

if __name__ == '__main__':
    main()
