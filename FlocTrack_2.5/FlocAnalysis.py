import pandas as pd
import numpy as np
import cv2
import os
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from skimage.io import imread
from skimage.transform import resize
from skimage.measure import perimeter
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import interp1d
from sympy import symbols, solve
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 10

def detect_individual_particles(normalized_img, original_contour):
    _, binary_img = cv2.threshold(normalized_img, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    particle_diameters = []
    original_area = cv2.contourArea(original_contour)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1 and area <= original_area:
            diameter = np.sqrt(4 * area / np.pi)
            particle_diameters.append(diameter)
    
    if not particle_diameters:  # If no individual diameters are found
        original_diameter = np.sqrt(4 * original_area / np.pi)
        particle_diameters.append(original_diameter)
    
    return particle_diameters

def capacity_dimension_3d(filename, ell):
    image01 = imread(filename, as_gray=True)

    if image01.max() > 1:
        _, image01 = cv2.threshold(image01, 128, 255, cv2.THRESH_BINARY)

    image02 = binary_fill_holes(image01).astype(int)

    if min(image01.shape) > ell:
        print(f"Error-- Desired resolution ell has to be smaller than the smallest dimension of the original image size = {min(image01.shape)}.")
        return

    image03 = resize(image02, (ell, ell), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    N_A = np.sum(image03 == 1)
    N_P = perimeter(image03)
    dP_0 = 2 * np.log10(N_P) / np.log10(N_A)
    Z = 2 * np.log10(4*ell - 4) / np.log10(ell**2)
    alpha = (4 - 2) * dP_0 / (Z - 2) + 2 * (Z - 4) / (Z - 2)
    A = -1 * alpha * (Z - dP_0) / (2 - alpha)
    B = Z - dP_0 - A
    C = dP_0

    I_max = -1 * B / (2 * A)
    dP_I_max = A * I_max**2 + B * I_max + C

    if dP_I_max > 2:
        tmpA = Z - dP_0
        tmpB = 2 * (dP_0 - 2)
        tmpC = 2 - dP_0
        tmpI = np.roots([tmpA, tmpB, tmpC])
        I_max = tmpI[tmpI > 0][0] if tmpI[tmpI > 0].size > 0 else I_max

        A = (dP_0 - Z) / (2 * I_max - 1)
        B = Z - dP_0 - A

    ell_given = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    beta1_given = np.array([3.1239, 4.1186, 5.0458, 6.1108, 7.5930, 10.4402, 13.7207, 16.4458])
    beta2_given = np.array([-9.7720, -12.4070, -14.8611, -17.6664, -21.6212, -29.3393, -38.0654, -44.7104])
    beta3_given = np.array([8.3318, 10.0342, 11.5941, 13.3657, 15.9176, 21.0255, 26.6887, 30.6337])

    interp1 = interp1d(ell_given, beta1_given, kind='linear', fill_value='extrapolate')
    interp2 = interp1d(ell_given, beta2_given, kind='linear', fill_value='extrapolate')
    interp3 = interp1d(ell_given, beta3_given, kind='linear', fill_value='extrapolate')

    beta1 = interp1(ell)
    beta2 = interp2(ell)
    beta3 = interp3(ell)

    dp_I_opt_sym = symbols('dp_I_opt')
    eq = A * (beta1 * dp_I_opt_sym**2 + beta2 * dp_I_opt_sym + beta3)**2 + B * (beta1 * dp_I_opt_sym**2 + beta2 * dp_I_opt_sym + beta3) + C - dp_I_opt_sym
    dp_I_opt_solutions = solve(eq, dp_I_opt_sym)
    dp_I_opt_real = [float(sol.evalf()) for sol in dp_I_opt_solutions if sol.is_real]

    if len(dp_I_opt_real) == 0:
        print("Error: No real solution found for dp_I_opt.")
        return

    dp_I_opt = max(dp_I_opt_real)

    k = Z * (Z - 1) + 1
    a = 9 * (Z - ((2 * k**2 - 9 * Z) / (k**2 - 9)))
    b = (2 * k**2 - 9 * Z) / (k**2 - 9)

    d0_3D = np.sqrt(a / (dp_I_opt - b))

    return d0_3D

# Function to calculate distance between two points
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate the concave hull area
def calculate_concave_hull_area(points, alpha):
    concave_hull = alphashape.alphashape(points, alpha)
    if isinstance(concave_hull, Polygon):
        return concave_hull.area
    elif isinstance(concave_hull, MultiPolygon):
        return sum(polygon.area for polygon in concave_hull.geoms)
    else:
        return 0

def process_frame(frame_num, csv_file_path_template, image_path_template, save_dir_template):
    try:
        # Load the CSV file
        csv_file_path = csv_file_path_template.format(frame_num)
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found for frame {frame_num}, skipping.")
            return

        df = pd.read_csv(csv_file_path)

        # Check if necessary columns exist
        if 'AdjustedVelocity' not in df.columns or 'AdjustedDirection' not in df.columns:
            print(f"Necessary columns not found in CSV for frame {frame_num}, skipping.")
            return

        # Load the original image
        image_path = image_path_template.format(frame_num)
        if not os.path.exists(image_path):
            print(f"Image file not found for frame {frame_num}, skipping.")
            return

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the appropriate background image
        background_image_path = image_path_template.format((frame_num // 1000) * 1000)
        if not os.path.exists(background_image_path):
            print(f"Background image not found for frame {frame_num}, skipping.")
            return

        background_image = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)

        # Create directory to save flocs CSV and images
        save_dir = save_dir_template.format(frame_num)
        os.makedirs(save_dir, exist_ok=True)

        # Function to calculate velocity difference
        def velocity_diff(v1, v2):
            return abs(v1 - v2)

        # Function to calculate direction difference
        def direction_diff(d1, d2):
            return abs(d1 - d2)

        # Function to merge bounding boxes
        def merge_bounding_boxes(bb1, bb2):
            x1, y1, w1, h1 = bb1
            x2, y2, w2, h2 = bb2
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            return (x, y, w, h)

        # Initialize bounding boxes list and floc information
        bounding_boxes = [(row['BoundingRectX'], row['BoundingRectY'], row['BoundingRectW'], row['BoundingRectH']) for _, row in df.iterrows()]
        flocs = []

        # Initialize a list to keep track of merged bounding boxes
        merged_bounding_boxes = []

        # Initialize a list to keep track of which particles have been processed
        processed = [False] * len(df)

        # Initialize ParticleNum counter
        particle_counter = 1

        # Process particles and connect them based on criteria
        for i in range(len(df)):
            if processed[i]:
                continue
            bb = bounding_boxes[i]
            velocities = [df.iloc[i]['AdjustedVelocity']]
            directions = [df.iloc[i]['AdjustedDirection']]
            connected_particles = [i]
            is_aggregate = 0
            for j in range(i + 1, len(df)):
                if (distance(df.iloc[i]['X'], df.iloc[i]['Y'], df.iloc[j]['X'], df.iloc[j]['Y']) < 50 and
                    velocity_diff(df.iloc[i]['AdjustedVelocity'], df.iloc[j]['AdjustedVelocity']) < 5 and
                    direction_diff(df.iloc[i]['AdjustedDirection'], df.iloc[j]['AdjustedDirection']) < 5):
                    # Merge bounding boxes for connected particles
                    bb = merge_bounding_boxes(bb, bounding_boxes[j])
                    velocities.append(df.iloc[j]['AdjustedVelocity'])
                    directions.append(df.iloc[j]['AdjustedDirection'])
                    connected_particles.append(j)
                    processed[j] = True
                    is_aggregate = 1

            merged_bounding_boxes.append(bb)
            avg_velocity = np.mean(velocities)
            avg_direction = np.mean(directions)
            cx = bb[0] + bb[2] / 2
            cy = bb[1] + bb[3] / 2

            x, y, w, h = bb
            particle_img = gray[int(y):int(y + h), int(x):int(x + w)]
            inverted_img = 255 - particle_img
            normalized_img = cv2.normalize(inverted_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # Subtract the background
            background_subtracted_img = cv2.absdiff(particle_img, background_image[int(y):int(y + h), int(x):int(x + w)])
            
            filename = f"particle_{particle_counter}.bmp"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, background_subtracted_img)

            d0_3D_value = capacity_dimension_3d(filepath, 1024)

            # Calculate concave hull area
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_points = []
            for contour in contours:
                for point in contour:
                    all_points.append(point[0])
            points = np.array(all_points)
            alpha = 0.05
            concave_hull = alphashape.alphashape(points, alpha)
            if isinstance(concave_hull, Polygon):
                concave_hull_area = concave_hull.area
            elif isinstance(concave_hull, MultiPolygon):
                concave_hull_area = sum(polygon.area for polygon in concave_hull.geoms)
            else:
                concave_hull_area = 0

            # Detect individual particles
            original_contour = contours[0]
            particle_diameters = detect_individual_particles(normalized_img, original_contour)
            num_particles = len(particle_diameters)
            if num_particles > 1:
                is_aggregate = 1
            particle_diameters_str = ','.join(map(str, particle_diameters))

            # Calculate Laplacian variance, mean intensity, and standard deviation intensity from BMP file
            bmp_filepath = os.path.join(save_dir, filename)
            bmp_image = cv2.imread(bmp_filepath, cv2.IMREAD_GRAYSCALE)
            laplacian = cv2.Laplacian(bmp_image, cv2.CV_64F)
            laplacian_variance = laplacian.var()
            mean_intensity, stddev_intensity = cv2.meanStdDev(bmp_image)
            mean_intensity = mean_intensity[0][0]
            stddev_intensity = stddev_intensity[0][0]

            flocs.append([particle_counter, cx, cy, avg_velocity, avg_direction, is_aggregate, d0_3D_value, concave_hull_area, num_particles, particle_diameters_str, laplacian_variance, mean_intensity, stddev_intensity])
            processed[i] = True

            particle_counter += 1

        flocs_df = pd.DataFrame(flocs, columns=["ParticleNum", "BoundingBoxCenterX", "BoundingBoxCenterY", "AverageVelocity", "AverageDirection", "IsAggregate", "FractalDimension", "ConcaveHullArea", "NumParticles", "ParticleDiameters", "LaplacianVariance", "MeanIntensity", "StdDevIntensity"])
        flocs_df.to_csv(os.path.join(save_dir, "flocs.csv"), index=False)

    except Exception as e:
        print(f"An error occurred while processing frame {frame_num}: {e}")

def process_single_frame(args):
    frame_num, csv_file_path_template, image_path_template, save_dir_template = args
    process_frame(frame_num, csv_file_path_template, image_path_template, save_dir_template)
