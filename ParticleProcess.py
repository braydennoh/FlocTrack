import os
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes
from skimage.measure import perimeter
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import math
from scipy.optimize import fsolve
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from pyefd import elliptic_fourier_descriptors  # Ensure you have installed pyefd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_average_image(filepath):
    """Load the average image once."""
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) if os.path.exists(filepath) else None


def process_frame_wrapper(task):
    """Wrapper function for multiprocessing."""
    try:
        return process_frame(*task)  # Ensure this correctly unpacks (frame_num, average_image, base_dir, base_save_dir)
    except Exception as e:
        logging.error(f"Error processing frame {task[0]}: {e}")
        return None


def capacity_dimension_3d(filename, cnt_perimeter, cnt_area):
    try:
        if cnt_area <= 0 or cnt_area > 1e9 or cnt_perimeter <= 0 or cnt_perimeter > 1e9:
            logging.warning("Contour perimeter and area must be greater than 0 and less than 1e9.")
            return None

        # Step 1: Calculate N_f2 (Equation A1)
        N_f2 = 2 * (math.log(cnt_perimeter) / math.log(cnt_area))

        # Step 2: Calculate the side length 'l' of the equivalent area of the rectangular box surrounding the floc (Equation A2)
        image = imread(filename, as_gray=True)
        height, width = image.shape[:2]
        ell = math.sqrt(height * width)

        # Calculate area and perimeter
        N_A = cnt_area
        N_P = cnt_perimeter
        dP_0 = 2 * np.log10(N_P) / np.log10(N_A)

        # Calculate other geometric parameters
        Z = 2 * np.log10(4 * ell - 4) / np.log10(ell ** 2)
        denominator = (Z - 2)
        if denominator == 0:
            logging.warning("Denominator in alpha calculation is zero.")
            return None
        alpha = (4 - 2) * dP_0 / denominator + 2 * (Z - 4) / denominator
        A = -1 * alpha * (Z - dP_0) / (2 - alpha)
        B = Z - dP_0 - A
        C = dP_0

        # Calculate the maximum value for I and adjust based on conditions
        I_max = -1 * B / (2 * A)
        dP_I_max = A * I_max ** 2 + B * I_max + C

        if dP_I_max > 2:
            tmpA = Z - dP_0
            tmpB = 2 * (dP_0 - 2)
            tmpC = 2 - dP_0
            tmp_roots = np.roots([tmpA, tmpB, tmpC])
            tmp_roots_real = tmp_roots[np.isreal(tmp_roots)].real
            I_max_candidates = tmp_roots_real[tmp_roots_real > 0]
            if I_max_candidates.size > 0:
                I_max = I_max_candidates[0]
            else:
                logging.warning("No valid I_max found.")
                return None

            A = (dP_0 - Z) / (2 * I_max - 1)
            B = Z - dP_0 - A

        # Interpolation for given parameters
        ell_given = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=float)
        beta1_given = np.array([3.1239, 4.1186, 5.0458, 6.1108, 7.5930, 10.4402, 13.7207, 16.4458], dtype=float)
        beta2_given = np.array([-9.7720, -12.4070, -14.8611, -17.6664, -21.6212, -29.3393, -38.0654, -44.7104],
                               dtype=float)
        beta3_given = np.array([8.3318, 10.0342, 11.5941, 13.3657, 15.9176, 21.0255, 26.6887, 30.6337], dtype=float)

        interp1 = interp1d(ell_given, beta1_given, kind='linear', fill_value='extrapolate')
        interp2 = interp1d(ell_given, beta2_given, kind='linear', fill_value='extrapolate')
        interp3 = interp1d(ell_given, beta3_given, kind='linear', fill_value='extrapolate')

        # Interpolate beta values based on ell
        beta1 = float(interp1(ell))
        beta2 = float(interp2(ell))
        beta3 = float(interp3(ell))

        # Define the equation for numerical solving
        def eq_numeric(dp_I_opt):
            return A * (beta1 * dp_I_opt ** 2 + beta2 * dp_I_opt + beta3) ** 2 + \
                   B * (beta1 * dp_I_opt ** 2 + beta2 * dp_I_opt + beta3) + C - dp_I_opt

        # Provide an initial guess for dp_I_opt
        initial_guess = 1.0  # You might need to adjust this based on expected values

        # Use fsolve to find the root
        dp_I_opt_solution, info, ier, mesg = fsolve(eq_numeric, initial_guess, full_output=True)

        if ier != 1:
            logging.warning(f"fsolve did not converge: {mesg}")
            return None

        dp_I_opt = dp_I_opt_solution[0]

        # Final calculation for d0_3D
        k = Z * (Z - 1) + 1
        denominator = (k ** 2 - 9)
        if denominator == 0:
            logging.warning("Denominator in d0_3D calculation is zero.")
            return None
        a = 9 * (Z - ((2 * k ** 2 - 9 * Z) / denominator))
        b = (2 * k ** 2 - 9 * Z) / denominator

        if dp_I_opt - b <= 0:
            logging.warning("Invalid dp_I_opt leading to negative value under square root.")
            return None

        d0_3D = np.sqrt(a / (dp_I_opt - b))

        return d0_3D
    except Exception as e:
        logging.error(f"Error in capacity_dimension_3d: {e}")
        return None


def calculate_fractal_dimension_d0_S3(filename, cnt_perimeter, cnt_area):
    try:
        if cnt_area <= 0 or cnt_perimeter <= 0:
            logging.warning("Contour perimeter and area must be greater than 0.")
            return None

        # Step 1: Calculate N_f2
        N_f2 = 2 * (math.log(cnt_perimeter) / math.log(cnt_area))

        # Step 2: Calculate side length 'l'
        image = imread(filename, as_gray=True)
        height, width = image.shape[:2]
        l = math.sqrt(height * width)

        if l <= 4:
            logging.warning(f"Skipping particle due to invalid l value: {l}")
            return None

        # Step 3: Calculate z(l)
        z_l = math.log(4 * l - 4) / math.log(l)

        # Step 4: Define k(l)
        k_l = z_l * (z_l - 1) + 1

        # Step 5: Calculate coefficients
        denominator = (k_l ** 2) - 9
        if denominator == 0:
            logging.warning("Invalid denominator value. Skipping calculation.")
            return None

        a_l = 9 * (z_l - ((2 * (k_l ** 2) - 9 * z_l) / denominator))
        b_l = (2 * (k_l ** 2) - 9 * z_l) / denominator

        # Step 6: Calculate N_f3
        if N_f2 - b_l <= 0:
            logging.warning("Invalid N_f2 - b_l value. Skipping calculation.")
            return None

        N_f3 = math.sqrt(a_l / (N_f2 - b_l))

        return N_f3
    except Exception as e:
        logging.error(f"Error in calculate_fractal_dimension_d0_S3: {e}")
        return None


def detect_individual_particles(normalized_img, original_contour):
    try:
        _, binary_img = cv2.threshold(normalized_img, 128, 255, cv2.THRESH_BINARY)
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
    except Exception as e:
        logging.error(f"Error in detect_individual_particles: {e}")
        return []


def process_frame(frame_num, average_image, base_dir, base_save_dir):
    try:
        frame_str = f"frame_{frame_num}"
        save_dir = os.path.join(base_save_dir, frame_str)
        os.makedirs(save_dir, exist_ok=True)
        csv_file_path = os.path.join(save_dir, "contour_log.csv")

        image_path = os.path.join(base_dir, f"{frame_str}.png")
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image at {image_path}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        contrast_enhanced_image = clahe.apply(gray)
        contrast_enhanced_image2 = clahe.apply(average_image)

        graysubtracted = contrast_enhanced_image - contrast_enhanced_image2
        graysubtractednorm = (graysubtracted - np.min(graysubtracted)) / (np.max(graysubtracted) - np.min(graysubtracted))
        equalized_gray = (graysubtractednorm * 255).astype(np.uint8)

        ret, binary_img = cv2.threshold(equalized_gray, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated_binary_img = cv2.dilate(binary_img, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particle_num = 0
        data = []

        for cnt in contours:
            try:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                masked_img = cv2.bitwise_and(equalized_gray, equalized_gray, mask=mask)
                laplacian = cv2.Laplacian(masked_img, cv2.CV_64F)
                laplacian_variance = laplacian.var()

                if laplacian_variance < 4:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0 or M["m00"] > 1e9:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                mean_intensity, stddev_intensity = cv2.meanStdDev(gray, mask=mask)
                mean_intensity = mean_intensity[0][0]
                stddev_intensity = stddev_intensity[0][0]

                if stddev_intensity < 2:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                particle_img = gray[y:y + h, x:x + w]
                inverted_img = 255 - particle_img
                normalized_img = cv2.normalize(inverted_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                particle_num += 1
                filename = f"particle_{particle_num}.bmp"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, normalized_img)

                image = cv2.imread(filepath)

                if image is None:
                    logging.error(f"Failed to load image: {filepath}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pixels = image_rgb.reshape((-1, 3))
                if pixels.ndim != 2 or pixels.shape[0] == 0 or pixels.shape[1] != 3:
                    logging.warning("Invalid pixels data for KMeans. Defaulting to empty mask.")
                    binary_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                    continue
                try:
                    num_clusters = 2  # Assuming 2 clusters for background and particles
                    kmeans = KMeans(n_clusters=2, random_state=42)
                    kmeans.fit(pixels)
                    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
                    clustered_image = clustered_pixels.reshape(image_rgb.shape).astype(np.uint8)

                    cluster_averages = np.mean(kmeans.cluster_centers_, axis=1)
                    darkest_cluster = np.argmin(cluster_averages)

                    binary_mask = (kmeans.labels_.reshape(image_rgb.shape[:2]) != darkest_cluster).astype(np.uint8) * 255
                except Exception as e:
                    logging.error(f"KMeans fitting failed for particle {particle_num} in frame {frame_num}: {e}")
                    binary_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

                # Find contours of the binary mask
                contours_mask, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours_mask:
                    logging.warning(f"No contours found in binary mask for particle {particle_num}")
                    continue

                # Get centroids of the contours
                centroids = []
                for contour in contours_mask:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:  # Avoid divide by zero for degenerate cases
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centroids.append((cX, cY))

                centroids = np.array(centroids)

                if len(centroids) == 0:
                    logging.warning(f"No centroids found for particle {particle_num}")
                    continue

                # Compute pairwise distances
                dist_matrix = distance.cdist(centroids, centroids, 'euclidean')

                # Compute the Minimum Spanning Tree (MST)
                mst = minimum_spanning_tree(dist_matrix).toarray()

                # Draw the MST lines on the binary mask
                connected_mask = binary_mask.copy()
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        if mst[i, j] > 0:
                            # Draw a line between the centroids
                            cv2.line(connected_mask, tuple(centroids[i]), tuple(centroids[j]), color=255, thickness=1)

                # Ensure connected_mask is binary
                connected_mask_binary = (connected_mask > 0).astype(np.uint8)

                # Find contours
                contours_connected, _ = cv2.findContours(connected_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours_connected:
                    logging.warning(f"No contours found in connected mask for particle {particle_num}")
                    continue

                # Find the largest contour by area
                largest_contour = max(contours_connected, key=cv2.contourArea)

                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                kclusterarea = cv2.contourArea(largest_contour)
                kclusterperimeter = cv2.arcLength(largest_contour, True)

                # Fractal dimension calculation
                fractal_dimension = 2 * (np.log(perimeter) / np.log(area)) if area > 0 and perimeter > 0 else 0

                # Aspect ratio of the bounding box of the largest contour
                x_largest, y_largest, w_largest, h_largest = cv2.boundingRect(largest_contour)
                aspect_ratio = w_largest / h_largest if h_largest != 0 else 0

                # Circularity calculation
                circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

                # Solidity calculation
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Complexity score using Elliptic Fourier Descriptors (EFD)
                efd_coeffs = elliptic_fourier_descriptors(largest_contour.reshape(-1, 2), order=10, normalize=True)
                harmonic_magnitudes = np.sum(efd_coeffs ** 2, axis=1)
                complexity_score = np.sum(harmonic_magnitudes[1:])

                blackandwhiteimage = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                _, otsu_thresh = cv2.threshold(blackandwhiteimage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours_otsu, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_points = []
                for contour in contours_otsu:
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

                cnt_perimeter = cv2.arcLength(cnt, True)
                cnt_area = cv2.contourArea(cnt)

                hull_cnt = cv2.convexHull(cnt)
                hull_area_cnt = cv2.contourArea(hull_cnt)
                convexhulldiameter = np.sqrt(4 * hull_area_cnt / np.pi)

                d0_S3 = calculate_fractal_dimension_d0_S3(filepath, cnt_perimeter, cnt_area)

                individual_particle_diameters = detect_individual_particles(normalized_img, cnt)
                individual_particle_diameters_str = ','.join(map(str, individual_particle_diameters))
                concavehulldiameter = np.sqrt(4 * concave_hull_area / np.pi) if concave_hull_area > 0 else 0

                kclusterdiameter = np.sqrt(4 * kclusterarea / np.pi) if kclusterarea > 0 else 0

                data.append([
                    particle_num,
                    laplacian_variance,
                    mean_intensity,
                    stddev_intensity,
                    cx,
                    cy,
                    concavehulldiameter,
                    kclusterdiameter,
                    convexhulldiameter,
                    individual_particle_diameters_str,
                    cnt_perimeter,
                    cnt_area,
                    d0_S3,
                    x,
                    y,
                    w,
                    h,
                    aspect_ratio,
                    circularity,
                    solidity,
                    complexity_score
                ])
            except Exception as e:
                logging.error(f"Error processing particle {particle_num} in frame {frame_num}: {e}")
                continue  # Skip to next contour

        if data:
            df = pd.DataFrame(data, columns=[
                "ParticleNum",
                "LaplacianVariance",
                "MeanIntensity",
                "StddevIntensity",
                "X",
                "Y",
                "ConcaveHullDiameter",
                "KClusterDiameter",
                "ConvexDiameter",
                "IndividualDiameters",
                "ContourPerimeter",
                "ContourArea",
                "PerimeterFractal",
                "BoundingRectX",
                "BoundingRectY",
                "BoundingRectW",
                "BoundingRectH",
                "AspectRatio",
                "Circularity",
                "Solidity",
                "ComplexityScore"
            ])

            df.to_csv(csv_file_path, index=False)
        else:
            logging.info(f"No data to save for frame {frame_num}")
    except Exception as e:
        logging.error(f"Error processing frame {frame_num}: {e}")
