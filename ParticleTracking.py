import os
import numpy as np
import pandas as pd

def read_particle_data(frame_path):
    if os.path.exists(frame_path):
        return pd.read_csv(frame_path)
    return pd.DataFrame()

def find_nearby_particles(particle, next_frame_particles, search_radius=50):
    distances = np.sqrt((next_frame_particles['X'] - particle['X'])**2 + (next_frame_particles['Y'] - particle['Y'])**2)
    return next_frame_particles[distances <= search_radius]

def match_particles(particle, candidates):
    if candidates.empty:
        return None
    
    # Calculate the normalized difference for ConcaveHullDiameter
    diameter_diff = np.abs(candidates['ConcaveHullDiameter'] - particle['ConcaveHullDiameter'])
    
    # Calculate the normalized difference for LaplacianVariance
    laplacian_diff = np.abs(candidates['LaplacianVariance'] - particle['LaplacianVariance'])
    
    # Normalize the differences by dividing by the maximum difference to scale between 0 and 1
    if diameter_diff.max() != 0:
        diameter_diff /= diameter_diff.max()
    if laplacian_diff.max() != 0:
        laplacian_diff /= laplacian_diff.max()

    # Compute the combined score with equal weight
    combined_score = 0.5 * diameter_diff + 0.5 * laplacian_diff
    
    # Select the candidate with the minimum combined score
    min_index = np.argmin(combined_score.values)
    if min_index < len(candidates):
        return candidates.iloc[min_index]
    
    return None

def update_csv_with_tracking_info(csv_path, tracking_data):
    df = pd.read_csv(csv_path)
    tracked_particle_nums = {data['ParticleNum'] for data in tracking_data}

    # Clear the fields for particles that have no tracking data
    df.loc[~df['ParticleNum'].isin(tracked_particle_nums), ['NextX', 'NextY', 'Velocity', 'Direction']] = np.nan

    # Update the CSV with the tracking data
    for data in tracking_data:
        df.loc[df['ParticleNum'] == data['ParticleNum'], ['NextX', 'NextY', 'Velocity', 'Direction']] = [
            data['NextX'], data['NextY'], data['Velocity'], data['Direction']]
    
    df.to_csv(csv_path, index=False)

def track_particles(base_save_dir, frame_nums):
    for frame_num in frame_nums:
        current_frame_csv = os.path.join(base_save_dir, f'frame_{frame_num}', 'contour_log.csv')
        next_frame_csv = os.path.join(base_save_dir, f'frame_{frame_num + 1}', 'contour_log.csv')

        if not os.path.exists(next_frame_csv):
            continue

        current_particles = read_particle_data(current_frame_csv)
        next_particles = read_particle_data(next_frame_csv)

        if current_particles.empty or next_particles.empty:
            continue

        tracking_data = []

        for index, particle in current_particles.iterrows():
            nearby_particles = find_nearby_particles(particle, next_particles)
            best_candidate = match_particles(particle, nearby_particles)
            if best_candidate is not None:
                displacement = np.array([best_candidate['X'] - particle['X'], best_candidate['Y'] - particle['Y']])
                distance_microns = np.linalg.norm(displacement) 
                velocity = distance_microns 
                direction = np.arctan2(displacement[1], displacement[0]) * (180 / np.pi)  # Direction in degrees
                tracking_data.append({'ParticleNum': particle['ParticleNum'], 'NextX': best_candidate['X'], 'NextY': best_candidate['Y'], 'Velocity': velocity, 'Direction': direction})
        update_csv_with_tracking_info(current_frame_csv, tracking_data)

    print("Tracking complete for specified frames.")