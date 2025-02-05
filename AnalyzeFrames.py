import pandas as pd
import numpy as np
import os

def load_frame_data(base_dir, frame_number):
    # Update the file path to point to the new contour_log.csv files
    file_path = os.path.join(base_dir, f'frame_{frame_number}', 'contour_log.csv')
    
    # Define the expected columns based on the new CSV format
    expected_columns = [
        'ParticleNum', 'LaplacianVariance', 'MeanIntensity', 'StddevIntensity', 'X', 'Y',
        'ConcaveHullDiameter', 'IndividualDiameters', 'ContourPerimeter', 'ContourArea',
        'IntensityFractal', 'PerimeterFractal', 'BoundingRectX', 'BoundingRectY',
        'BoundingRectW', 'BoundingRectH', 'NextX', 'NextY', 'Velocity', 'Direction',
        'AdjustedVelocity', 'AdjustedDirection'
    ]
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df
    return pd.DataFrame(columns=expected_columns)

def merge_and_filter_continuous_lines(all_lines, min_connections):
    connections = {}
    
    # Build the connections dictionary
    for start_frame, start_particle, end_frame, end_particle in all_lines:
        if (start_frame, start_particle) not in connections:
            connections[(start_frame, start_particle)] = []
        connections[(start_frame, start_particle)].append((end_frame, end_particle))
    
    # Function to recursively trace the connection
    def trace_path(start):
        path = [start]
        while start in connections and connections[start]:
            next_start = connections[start].pop(0)
            path.append(next_start)
            start = next_start
        return path
    
    # Tracing all paths
    all_paths = []
    for key in list(connections.keys()):
        while connections[key]:
            path = trace_path(key)
            if len(path) > min_connections:  # Only keep paths with more than `min_connections` connections
                all_paths.append(path)
    
    return all_paths

def analyze_frames(base_dir, frame_range, min_connections):
    # Initialize lists to store all particles and lines for plotting
    all_particles = {}
    all_lines = []

    # Load data for all frames and store in a dictionary
    for frame in frame_range:
        all_particles[frame] = load_frame_data(base_dir, frame)

    # Extract necessary columns for each frame
    for frame in all_particles:
        all_particles[frame] = all_particles[frame][[
            'ParticleNum', 'X', 'Y', 'NextX', 'NextY', 'ConcaveHullDiameter',
            'IndividualDiameters', 'ContourPerimeter', 'AdjustedVelocity'
        ]]

    # Find matches and prepare lines for plotting
    for frame in frame_range[:-1]:
        if frame not in all_particles or (frame + 1) not in all_particles:
            continue
        frame_data = all_particles[frame]
        next_frame_data = all_particles[frame + 1]
        
        if frame_data.empty or next_frame_data.empty:
            continue
        
        next_coords = {tuple(row[['NextX', 'NextY']]): row['ParticleNum'] for idx, row in frame_data.iterrows() if not np.isnan(row['NextX']) and not np.isnan(row['NextY'])}
        
        for idx, row in next_frame_data.iterrows():
            coord = tuple(row[['X', 'Y']])
            if coord in next_coords:
                particle_current = next_coords[coord]
                particle_next = row['ParticleNum']
                all_lines.append((frame, particle_current, frame + 1, particle_next))

    # Generate continuous lines and filter them
    filtered_continuous_lines_paths = merge_and_filter_continuous_lines(all_lines, min_connections)

    # Calculate averages and save as NumPy array
    averages_list = []

    for path in filtered_continuous_lines_paths:
        concave_hull_diameters = []
        individual_diameters = []
        contour_perimeters = []
        adjusted_velocities = []
        
        for frame, particle in path:
            particle_data = all_particles[frame][all_particles[frame]['ParticleNum'] == particle]
            
            concave_hull_diameters.append(particle_data['ConcaveHullDiameter'].values[0])
            
            # Parse the IndividualDiameters string and flatten it
            individual_diameters_value = particle_data['IndividualDiameters'].values[0]
            if isinstance(individual_diameters_value, str):
                try:
                    individual_diameters.extend([float(d) for d in individual_diameters_value.split(',')])  # Split and convert to floats
                except ValueError:
                    pass  # Skip if parsing fails

            contour_perimeters.append(particle_data['ContourPerimeter'].values[0])
            adjusted_velocities.append(particle_data['AdjustedVelocity'].values[0])
        
        # Calculate averages for the path
        average_concave_hull_diameter = np.mean(concave_hull_diameters)
        average_individual_diameters = np.mean(individual_diameters) if individual_diameters else np.nan
        average_contour_perimeter = np.mean(contour_perimeters)
        average_adjusted_velocity = np.mean(adjusted_velocities)
        
        averages_list.append([
            average_concave_hull_diameter, average_individual_diameters,
            average_contour_perimeter, average_adjusted_velocity
        ])

    averages_array = np.array(averages_list)

    # Display the filtered continuous lines
    filtered_continuous_lines_data = []
    for path in filtered_continuous_lines_paths:
        filtered_continuous_lines_data.append(" -> ".join([f"Particle {p} in Frame {f}" for f, p in path]))

    return averages_array, filtered_continuous_lines_data

