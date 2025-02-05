
import os
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes
from skimage.measure import perimeter
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import math
from sympy import symbols, solve

def get_average_image_path(frame_num, base_dir):
    if 0 <= frame_num <= 53000:
        base_number = ((frame_num // 1000) + 1) * 1000
        average_image_filename = f'average_image{base_number}.png'
        return os.path.join(base_dir, average_image_filename)
    return None


def load_average_image(filepath):
    if filepath:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return None

def process_frame_wrapper(frame_data):
    return process_frame(*frame_data)


def capacity_dimension_3d(filename, cnt_perimeter, cnt_area):

    if cnt_area <= 0 or cnt_perimeter <= 0:
        raise ValueError("Contour perimeter and area must be greater than 0.")
    
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
    Z = 2 * np.log10(4*ell - 4) / np.log10(ell**2)
    alpha = (4 - 2) * dP_0 / (Z - 2) + 2 * (Z - 4) / (Z - 2)
    A = -1 * alpha * (Z - dP_0) / (2 - alpha)
    B = Z - dP_0 - A
    C = dP_0

    # Calculate the maximum value for I and adjust based on conditions
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

    # Interpolation for given parameters
    ell_given = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    beta1_given = np.array([3.1239, 4.1186, 5.0458, 6.1108, 7.5930, 10.4402, 13.7207, 16.4458])
    beta2_given = np.array([-9.7720, -12.4070, -14.8611, -17.6664, -21.6212, -29.3393, -38.0654, -44.7104])
    beta3_given = np.array([8.3318, 10.0342, 11.5941, 13.3657, 15.9176, 21.0255, 26.6887, 30.6337])

    interp1 = interp1d(ell_given, beta1_given, kind='linear', fill_value='extrapolate')
    interp2 = interp1d(ell_given, beta2_given, kind='linear', fill_value='extrapolate')
    interp3 = interp1d(ell_given, beta3_given, kind='linear', fill_value='extrapolate')

    # Interpolate beta values based on ell
    beta1 = interp1(ell)
    beta2 = interp2(ell)
    beta3 = interp3(ell)

    # Solve for dp_I_opt
    dp_I_opt_sym = symbols('dp_I_opt')
    eq = A * (beta1 * dp_I_opt_sym**2 + beta2 * dp_I_opt_sym + beta3)**2 + B * (beta1 * dp_I_opt_sym**2 + beta2 * dp_I_opt_sym + beta3) + C - dp_I_opt_sym
    dp_I_opt_solutions = solve(eq, dp_I_opt_sym)
    dp_I_opt_real = [float(sol.evalf()) for sol in dp_I_opt_solutions if sol.is_real]

    if len(dp_I_opt_real) == 0:
        print("Error: No real solution found for dp_I_opt.")
        return

    # Select the maximum real solution
    dp_I_opt = max(dp_I_opt_real)

    # Final calculation for d0_3D
    k = Z * (Z - 1) + 1
    a = 9 * (Z - ((2 * k**2 - 9 * Z) / (k**2 - 9)))
    b = (2 * k**2 - 9 * Z) / (k**2 - 9)

    d0_3D = np.sqrt(a / (dp_I_opt - b))

    return d0_3D

import math
from skimage.io import imread

def calculate_fractal_dimension_d0_S3(filename, cnt_perimeter, cnt_area):
    """
    This function calculates the 3D fractal dimension (N_f3) of an individual floc based on its perimeter and area.

    Parameters:
    - filename: The path to the image file containing the floc.
    - cnt_perimeter: The perimeter of the floc (in pixels).
    - cnt_area: The area of the floc (in pixels^2).

    Returns:
    - N_f3: The 3D fractal dimension of the floc if N_f2 < 2, otherwise None.
    """
    if cnt_area <= 0 or cnt_perimeter <= 0:
        raise ValueError("Contour perimeter and area must be greater than 0.")
    
    # Step 1: Calculate N_f2 (Equation A1)
    N_f2 = 2 * (math.log(cnt_perimeter) / math.log(cnt_area))

    # Step 2: Calculate the side length 'l' of the equivalent area of the rectangular box surrounding the floc (Equation A2)
    image = imread(filename, as_gray=True)
    height, width = image.shape[:2]
    l = math.sqrt(height * width)

    # Step 3: Calculate z(l) (Equation A3)
    if l <= 4:
        raise ValueError("Invalid value for l. Ensure that l > 4 for valid z(l) calculation.")
    
    z_l = math.log(4 * l - 4) / math.log(l)

    # Step 4: Define k(l) based on boundary condition at N_f2 = 2 (Equation A4)
    k_l = z_l * (z_l - 1) + 1

    # Step 5: Calculate coefficients a(l) and b(l) (Equations A6 and A7)
    denominator = (k_l ** 2) - 9
    if denominator == 0:
        raise ValueError("Invalid denominator value. Ensure k(l) is such that the denominator is non-zero.")
    
    # Correct formula for a(l)
    a_l = 9 * (z_l - ((2 * (k_l ** 2) - 9 * z_l) / denominator))
    
    # Correct formula for b(l)
    b_l = (2 * (k_l ** 2) - 9 * z_l) / denominator
    
    # Step 6: Calculate N_f3 if N_f2 < 2 (Equation A8)
    if N_f2 < 2:
        N_f3 = math.sqrt(a_l / (N_f2 - b_l))
    else:
        N_f3 = None  # N_f3 is not defined if N_f2 >= 2

    return N_f3
        

def detect_individual_particles(normalized_img, original_contour):
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
    
def process_frame(frame_num, average_image, base_dir, base_save_dir):
    frame_str = f"frame_{frame_num}"
    save_dir = os.path.join(base_save_dir, frame_str)
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, "contour_log.csv")

    image_path = os.path.join(base_dir, f"{frame_str}.png")
    image = cv2.imread(image_path)
    if image is None:
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
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        masked_img = cv2.bitwise_and(equalized_gray, equalized_gray, mask=mask)
        laplacian = cv2.Laplacian(masked_img, cv2.CV_64F)
        laplacian_variance = laplacian.var()

        if laplacian_variance < 1:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        mean_intensity, stddev_intensity = cv2.meanStdDev(gray, mask=mask)
        mean_intensity = mean_intensity[0][0]
        stddev_intensity = stddev_intensity[0][0]

        if stddev_intensity < 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        particle_img = gray[y:y + h, x:x + w]
        inverted_img = 255 - particle_img
        normalized_img = cv2.normalize(inverted_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        particle_num += 1
        filename = f"particle_{particle_num}.bmp"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, normalized_img)

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
            
        cnt_perimeter = cv2.arcLength(cnt, True)
        cnt_area = cv2.contourArea(cnt)

        d0_S3 = calculate_fractal_dimension_d0_S3(filepath, cnt_perimeter, cnt_area)

        d0_3D_value = capacity_dimension_3d(filepath, cnt_perimeter, cnt_area)

        individual_particle_diameters = detect_individual_particles(normalized_img, cnt)
        individual_particle_diameters_str = ','.join(map(str, individual_particle_diameters))
        

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexhulldiameter = np.sqrt(4 * hull_area / np.pi)

        hull_perimeter = cv2.arcLength(hull, True)
        cnt_perimeter = cv2.arcLength(cnt, True)
        
        
    
        concavehullarea = concave_hull_area
        concavehulldiameter = np.sqrt(4 * concavehullarea / np.pi)

        hull_area = cv2.contourArea(hull)
        hull_diameter = np.sqrt(4 * hull_area / np.pi)

        data.append([particle_num, laplacian_variance, mean_intensity, stddev_intensity, cx, cy, concavehulldiameter, convexhulldiameter, individual_particle_diameters_str,
                     cnt_perimeter, cnt_area, d0_3D_value, d0_S3, x, y, w, h])


    df = pd.DataFrame(data, columns=["ParticleNum", "LaplacianVariance", "MeanIntensity", "StddevIntensity",
                                     "X", "Y", "ConcaveHullDiameter","ConvexHullDiameter","IndividualDiameters", "ContourPerimeter", "ContourArea", "IntensityFractal",
                                     "PerimeterFractal", "BoundingRectX", "BoundingRectY",
                                     "BoundingRectW", "BoundingRectH"])
    df.to_csv(csv_file_path, index=False)
