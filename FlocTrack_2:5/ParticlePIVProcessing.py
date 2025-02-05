import pandas as pd
import numpy as np
import cv2
import os
from openpiv import tools, scaling, validation, filters, pyprocess
import matplotlib.pyplot as plt
import pathlib

def get_average_image_path(frame_num, ranges, base_path):
    for frame_range, file_name in ranges.items():
        if frame_range[0] <= frame_num < frame_range[1]:
            return os.path.join(base_path, file_name)
    raise ValueError(f"No average image found for frame number {frame_num}")

def process_frames(base_save_dir, image_dir, average_image_ranges, frame_range):
    os.makedirs(base_save_dir, exist_ok=True)

    for frame_num in frame_range:
        csv_file_path = os.path.join(base_save_dir, f'frame_{frame_num}', 'contour_log.csv')
        flocs_dir = os.path.join(base_save_dir, f'frame_{frame_num}', 'flocs')
        os.makedirs(flocs_dir, exist_ok=True)

        if not os.path.exists(csv_file_path):
            print(f"CSV file not found: {csv_file_path}")
            continue

        df = pd.read_csv(csv_file_path)

        if 'Velocity' not in df.columns or 'Direction' not in df.columns:
            print(f"Columns 'Velocity' or 'Direction' not found in CSV for frame {frame_num}")
            continue

        particles_with_velocity = df.dropna(subset=['Velocity', 'Direction'])

        image_path_a = os.path.join(image_dir, f'frame_{frame_num}.png')
        image_path_b = os.path.join(image_dir, f'frame_{frame_num + 1}.png')

        frame_a = cv2.imread(image_path_a)
        frame_b = cv2.imread(image_path_b)
        if frame_a is None or frame_b is None:
            print(f"Images not found at {image_path_a} or {image_path_b}")
            continue

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Get the appropriate average image path
        average_image_path = get_average_image_path(frame_num, average_image_ranges, image_dir)
        average_image = cv2.imread(average_image_path, cv2.IMREAD_GRAYSCALE)

        graysubtracted_a = cv2.absdiff(gray_a, average_image)
        graysubtracted_b = cv2.absdiff(gray_b, average_image)

        normalized_subtracted_a = cv2.normalize(graysubtracted_a, None, 0, 255, cv2.NORM_MINMAX)
        normalized_subtracted_b = cv2.normalize(graysubtracted_b, None, 0, 255, cv2.NORM_MINMAX)

        # Apply Otsu's thresholding to the normalized images
        _, thresh_a = cv2.threshold(normalized_subtracted_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_b = cv2.threshold(normalized_subtracted_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours_a, _ = cv2.findContours(thresh_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_b, _ = cv2.findContours(thresh_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        filtered_contours_a = [cnt for cnt in contours_a if cv2.contourArea(cnt) < 500]
        filtered_contours_b = [cnt for cnt in contours_b if cv2.contourArea(cnt) < 500]

        # Create empty masks
        mask_a = np.zeros_like(gray_a)
        mask_b = np.zeros_like(gray_b)
        
        # Draw filtered contours into the masks
        cv2.drawContours(mask_a, filtered_contours_a, -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_b, filtered_contours_b, -1, 255, thickness=cv2.FILLED)

        # Apply masks to the original grayscale images
        masked_a = cv2.bitwise_and(gray_a, gray_a, mask=mask_a)
        masked_b = cv2.bitwise_and(gray_b, gray_b, mask=mask_b)

        contour_image_a = frame_a.copy()
        contour_image_b = frame_b.copy()
        cv2.drawContours(contour_image_a, filtered_contours_a, -1, (0, 255, 0), 2)
        cv2.drawContours(contour_image_b, filtered_contours_b, -1, (0, 255, 0), 2)

        multi = 3
        winsize = 32 * multi
        searchsize = 36 * multi
        overlap = 16 * multi
        dt = 1

        u0, v0, sig2noise = pyprocess.extended_search_area_piv(
            masked_a.astype(np.int32),
            masked_b.astype(np.int32),
            window_size=winsize,
            overlap=overlap,
            dt=dt,
            search_area_size=searchsize,
            sig2noise_method='peak2peak',
        )

        x, y = pyprocess.get_coordinates(
            image_size=masked_a.shape,
            search_area_size=searchsize,
            overlap=overlap,
        )

        invalid_mask = validation.sig2noise_val(
            sig2noise,
            threshold=1.05,
        )

        u2, v2 = filters.replace_outliers(
            u0, v0,
            invalid_mask,
            method='localmean',
            max_iter=3,
            kernel_size=3,
        )

        scaling_factor = 1
        x, y, u3, v3 = scaling.uniform(
            x, y, u2, v2,
            scaling_factor=scaling_factor,
        )

        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

        vector_file_path = os.path.join(flocs_dir, 'vector_field_masked.txt')
        tools.save(vector_file_path, x, y, u3, v3, invalid_mask)

        adjusted_velocities = []
        adjusted_directions = []

        x_flat = x.flatten()
        y_flat = y.flatten()
        u3_flat = u3.flatten()
        v3_flat = v3.flatten()

        for index, particle in particles_with_velocity.iterrows():
            px, py = int(particle['X']), int(particle['Y'])
            next_x, next_y = int(particle['NextX']), int(particle['NextY'])
            particle_velocity_x = (next_x - px)
            particle_velocity_y = (next_y - py)
            
            distances = np.sqrt((x_flat - px)**2 + (y_flat - py)**2)
            nearest_index = np.argmin(distances)
            nearest_piv_u = u3_flat[nearest_index]
            nearest_piv_v = v3_flat[nearest_index]
            
            adjusted_velocity_x = particle_velocity_x - nearest_piv_u
            adjusted_velocity_y = particle_velocity_y - nearest_piv_v
            
            adjusted_velocity_magnitude = np.sqrt(adjusted_velocity_x**2 + adjusted_velocity_y**2)
            adjusted_direction = np.degrees(np.arctan2(adjusted_velocity_y, adjusted_velocity_x))
            
            adjusted_velocities.append(adjusted_velocity_magnitude)
            adjusted_directions.append(adjusted_direction)

        df.loc[particles_with_velocity.index, 'AdjustedVelocity'] = adjusted_velocities
        df.loc[particles_with_velocity.index, 'AdjustedDirection'] = adjusted_directions
        df.to_csv(csv_file_path, index=False)
