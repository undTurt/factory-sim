import cv2
import numpy as np
import json
import base64
import pytesseract # type: ignore
from pytesseract import Output # type: ignore
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def validate_image(image_path):
    """Validate image file before processing"""
    try:
        if not image_path:
            return False, "Image path is empty"
            
        if not os.path.exists(image_path):
            return False, f"Image file not found: {image_path}"
            
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Failed to load image: {image_path}"
            
        if len(image.shape) != 3:
            return False, "Image must be a color image"
            
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            return False, f"Image too small: {width}x{height}"
            
        return True, "Image validation successful"
        
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"

def get_corner_label(x, y, max_x, max_y):
    """Determine corner label based on coordinates"""
    if y < max_y / 2:  # Top half
        return "top_left" if x < max_x / 2 else "top_right"
    else:  # Bottom half
        return "bottom_left" if x < max_x / 2 else "bottom_right"

def detect_red_markers(image):
    """Detect non-black markers in the image"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Save grayscale debug
        cv2.imwrite('debug_grayscale.png', gray)
        
        # Try multiple threshold approaches
        markers = []
        
        # Approach 1: Adaptive thresholding
        mask1 = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # C constant
        )
        
        # Approach 2: Simple thresholding
        _, mask2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Approach 3: Color-based detection for red markers
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask3 = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine masks
        masks = [cv2.bitwise_not(mask1), cv2.bitwise_not(mask2), mask3]
        
        # Save debug masks
        for idx, mask in enumerate(masks):
            cv2.imwrite(f'corners_mask_{idx}_debug.png', mask)
        
        for mask in masks:
            # Apply morphological operations
            kernel = np.ones((5,5), np.uint8)  # Increased kernel size
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20 or area > 5000:  # More permissive area thresholds
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # More permissive aspect ratio
                    continue
                
                center_x = x + w//2
                center_y = y + h//2
                
                # Check if point is near corners
                margin_percent = 0.25  # Increased margin
                margin_x = int(image.shape[1] * margin_percent)
                margin_y = int(image.shape[0] * margin_percent)
                
                is_left = x < margin_x
                is_right = x + w > image.shape[1] - margin_x
                is_top = y < margin_y
                is_bottom = y + h > image.shape[0] - margin_y
                
                if (is_left or is_right) and (is_top or is_bottom):
                    corner_type = get_corner_label(center_x, center_y, image.shape[1], image.shape[0])
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in markers:
                        if abs(existing['x'] - center_x) < 20 and abs(existing['y'] - center_y) < 20:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        markers.append({
                            "x": center_x,
                            "y": center_y,
                            "type": corner_type
                        })
        
        # Create debug visualization
        debug_image = image.copy()
        for marker in markers:
            cv2.circle(debug_image, (marker['x'], marker['y']), 10, (0, 255, 0), -1)
            cv2.putText(debug_image, marker['type'],
                       (marker['x'] - 20, marker['y'] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imwrite('corners_detection_debug.png', debug_image)
        
        logger.debug(f"Found {len(markers)} markers")
        
        if len(markers) == 0:
            logger.warning("No corners detected - check the debug images")
        elif len(markers) != 4:
            logger.warning(f"Expected 4 corners, found {len(markers)}")
        
        return markers
        
    except Exception as e:
        logger.error(f"Error in detect_red_markers: {str(e)}")
        logger.exception("Corner detection failed:")
        raise

def save_debug_visualizations(image, mask, corners):
    """Save debug visualizations"""
    cv2.imwrite('mask_debug.png', mask)
    debug_image = image.copy()
    for corner in corners:
        cv2.circle(debug_image, (corner['x'], corner['y']), 10, (0, 255, 0), -1)
    cv2.imwrite('corners_debug.png', debug_image)

def draw_grid_debug(image, grid_cells, num_cols, num_rows):
    """Draw grid and detected elements for debugging"""
    debug_image = image.copy()
    height, width = image.shape[:2]
    
    # Calculate cell dimensions
    cell_width = width / num_cols
    cell_height = height / num_rows
    
    # Draw grid lines
    for i in range(num_cols + 1):
        x = int(i * cell_width)
        cv2.line(debug_image, (x, 0), (x, height), (0, 255, 0), 2)
    
    for i in range(num_rows + 1):
        y = int(i * cell_height)
        cv2.line(debug_image, (0, y), (width, y), (0, 255, 0), 2)
    
    # Draw detected elements
    for cell in grid_cells:
        if cell['element'] != 'none':
            cell_x = int((cell['x'] - 1) * cell_width)
            cell_y = int((num_rows - cell['y']) * cell_height)
            
            # Draw cell highlight
            cv2.rectangle(debug_image, 
                         (cell_x, cell_y), 
                         (int(cell_x + cell_width), int(cell_y + cell_height)),
                         (0, 255, 255), 2)
            
            # Draw element text
            text_x = cell_x + int(cell_width/2) - 10
            text_y = cell_y + int(cell_height/2) + 10
            cv2.putText(debug_image, 
                       cell['element'],
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite('grid_debug.png', debug_image)

# ...existing imports and helper functions remain unchanged until detect_text_elements...

def detect_text_elements(image, grid_cells):
    """Detect black text elements and assign them to grid cells"""
    try:
# Replace from line 213 (after try block) with:

        # Initialize single debug image at the start
        debug_image = image.copy()
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply bilateral filter to preserve edges while reducing noise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE with lower clip limit for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # More lenient thresholds with broader range
        thresholds = [
            (0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU),  # Otsu's method
            (40, cv2.THRESH_BINARY_INV),                    # Light text
            (80, cv2.THRESH_BINARY_INV),                    # Medium text
            (120, cv2.THRESH_BINARY_INV),                   # Dark text
            (160, cv2.THRESH_BINARY_INV)                    # Very dark text
        ]
        
        # Save preprocessing debug image
        cv2.imwrite('preprocess_debug.png', gray)
        
        # Initialize elements list
        all_elements = []
        
        # Process each threshold
        for thresh, thresh_type in thresholds:
            if thresh == 0:
                _, black_mask = cv2.threshold(gray, 0, 255, thresh_type)
            else:
                _, black_mask = cv2.threshold(gray, thresh, 255, thresh_type)
            
            # Enhanced morphological operations
            kernel_open = np.ones((3,3), np.uint8)
            kernel_close = np.ones((3,3), np.uint8)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Save threshold debug image
            cv2.imwrite(f'threshold_{thresh}_debug.png', black_mask)
            
            contours_all = []
            for mode in [cv2.RETR_EXTERNAL, cv2.RETR_LIST]:
                contours, _ = cv2.findContours(black_mask, mode, cv2.CHAIN_APPROX_SIMPLE)
                contours_all.extend(contours)
            
            for contour in contours_all:
                area = cv2.contourArea(contour)
                if area < 15 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                    continue
                
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < 0.1:
                    continue
                
                center_x = x + w//2
                center_y = y + h//2
                
                pad = 30
                roi = gray[max(0, y-pad):min(y+h+pad, image.shape[0]), 
                          max(0, x-pad):min(x+w+pad, image.shape[1])]
                
                if roi.size == 0:
                    continue
                
                roi = cv2.resize(roi, (0,0), fx=10, fy=10)
                roi = clahe.apply(roi)
                roi = cv2.GaussianBlur(roi, (3,3), 0)
                
                try:
                    configs = [
                        '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCD',
                        '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCD',
                        '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCD',
                        '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCD'
                    ]
                    
                    for config in configs:
                        text = pytesseract.image_to_string(roi, config=config).strip()
                        if text and len(text) == 1 and text in 'ABCD':
                            if not any(abs(e['x'] - center_x) < 40 and 
                                     abs(e['y'] - center_y) < 40 and
                                     e['text'] == text
                                     for e in all_elements):
                                
                                all_elements.append({
                                    'text': text,
                                    'x': center_x,
                                    'y': center_y,
                                    'area': area,
                                    'confidence': 1.0,
                                    'solidity': solidity
                                })
                                
                                # Draw character detection
                                cv2.circle(debug_image, (center_x, center_y), 10, (0, 255, 0), -1)
                                cv2.putText(debug_image, text,
                                          (center_x - 15, center_y - 15),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                                break
                except Exception as e:
                    logger.warning(f"OCR failed for element at ({center_x}, {center_y}): {str(e)}")
                    continue
        
        # Grid creation section
        markers = detect_red_markers(image)
        if len(markers) == 4:
            corners = {m['type']: m for m in markers}
            
            # Create and draw outer bounding box exactly on markers
            outer_box = np.array([
                [corners['top_left']['x'], corners['top_left']['y']],
                [corners['top_right']['x'], corners['top_right']['y']],
                [corners['bottom_right']['x'], corners['bottom_right']['y']],
                [corners['bottom_left']['x'], corners['bottom_left']['y']]
            ], np.int32)
            cv2.polylines(debug_image, [outer_box], True, (0, 255, 255), 2)
            
            # Calculate grid dimensions from markers
            grid_width = abs(corners['top_right']['x'] - corners['top_left']['x'])
            grid_height = abs(corners['bottom_left']['y'] - corners['top_left']['y'])
            
            num_rows = len(set(c['y'] for c in grid_cells))
            num_cols = len(set(c['x'] for c in grid_cells))
            
            cell_width = grid_width / num_cols
            cell_height = grid_height / num_rows
            
            # Map elements to cells
            updated_cells = []
            processed_elements = set()
            
            for row in range(num_rows):
                for col in range(num_cols):
                    cell = next((c for c in grid_cells 
                               if int(c['x']) == col + 1 and 
                               int(c['y']) == num_rows - row), None)
                    
                    if cell:
                        cell_data = cell.copy()
                        cell_data['element'] = 'none'
                        
                        # Calculate exact cell boundaries from markers
                        cell_left = corners['top_left']['x'] + (col * cell_width)
                        cell_right = cell_left + cell_width
                        cell_top = corners['top_left']['y'] + (row * cell_height)
                        cell_bottom = cell_top + cell_height
                        
                        # Find elements in this cell
                        cell_elements = [
                            element for element in all_elements
                            if (element['text'] not in processed_elements and
                                cell_left <= element['x'] < cell_right and
                                cell_top <= element['y'] <= cell_bottom)
                        ]
                        
                        if cell_elements:
                            cell_center_x = cell_left + cell_width/2
                            cell_center_y = cell_top + cell_height/2
                            
                            best_element = min(cell_elements,
                                key=lambda e: ((e['x'] - cell_center_x)**2 + 
                                             (e['y'] - cell_center_y)**2))
                            
                            cell_data['element'] = best_element['text']
                            processed_elements.add(best_element['text'])
                        
                        # Always draw cell boundaries
                        cv2.rectangle(debug_image, 
                                    (int(cell_left), int(cell_top)),
                                    (int(cell_right), int(cell_bottom)),
                                    (0, 255, 0) if cell_data['element'] != 'none' else (0, 0, 255),
                                    2)
                        
                        if cell_data['element'] != 'none':
                            cv2.putText(debug_image,
                                      cell_data['element'],
                                      (int(cell_left + 5), int(cell_bottom - 5)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7,
                                      (255, 0, 0),
                                      2)
                        
                        updated_cells.append(cell_data)
            
            # Save single debug image with all information
            cv2.imwrite('grid_detection_debug.png', debug_image)
            return updated_cells
        
        return [{**cell, 'element': 'none'} for cell in grid_cells]
        
    except Exception as e:
        logger.error(f"Error in detect_text_elements: {str(e)}")
        logger.exception("Full traceback:")
        return [{**cell, 'element': 'none'} for cell in grid_cells]

# ...rest of your existing code remains unchanged...

def normalize_rectangle_coordinates(cells, num_cols, num_rows):
    """Normalize coordinates to create a perfect rectangle"""
    return [
        {"type": "top_left", "x": 0, "y": num_rows},
        {"type": "top_right", "x": num_cols, "y": num_rows},
        {"type": "bottom_left", "x": 0, "y": 0},
        {"type": "bottom_right", "x": num_cols, "y": 0}
    ]

def calculate_grid_cells(num_cols, num_rows):
    """Calculate grid cell center coordinates"""
    cells = []
    cell_id = 1
    
    for row in range(num_rows):
        for col in range(num_cols):
            center_x = col + 0.5
            center_y = (num_rows - row) - 0.5
            
            cells.append({
                "id": cell_id,
                "x": center_x,
                "y": center_y,
                "element": "none"
            })
            cell_id += 1
    
    return cells

def decode_base64_image(base64_string):
    """Convert base64 image data to OpenCV format"""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def process_grid_image(image_path, num_cols=None, num_rows=None):
    """Process an image to detect grid cells and identify symbols"""
    try:
        # Validate image first
        is_valid, message = validate_image(image_path)
        if not is_valid:
            logger.error(message)
            raise ValueError(message)
            
        if num_cols is None or num_rows is None:
            raise ValueError("Number of columns and rows must be specified")

        logger.info(f"Processing image: {image_path}")
        logger.info(f"Grid dimensions: {num_cols}x{num_rows}")

        # Load image
        image = cv2.imread(image_path)
        logger.debug(f"Loaded image shape: {image.shape}")
        cv2.imwrite('debug_original.png', image)

        # Validate inputs
        if not image_path:
            raise ValueError("Image path cannot be empty")
        
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Grid dimensions: {num_cols}x{num_rows}")
        
        # Check if file exists
        import os
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if num_cols is None or num_rows is None:
            raise ValueError("Number of columns and rows must be specified")

        # Load image with error checking
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Failed to load image from {image_path}")
            
        logger.debug(f"Loaded image shape: {image.shape}")
        cv2.imwrite('debug_original.png', image)

        try:
            corners = detect_red_markers(image)
            logger.debug(f"Detected corners: {corners}")
            
            if len(corners) != 4:
                raise ValueError(f"Expected 4 corners, but found {len(corners)}")

            normalized_corners = normalize_rectangle_coordinates(corners, num_cols, num_rows)
            logger.debug(f"Normalized corners: {normalized_corners}")
            
            grid_cells = calculate_grid_cells(num_cols, num_rows)
            logger.debug(f"Created {len(grid_cells)} grid cells")
            
            try:
                grid_cells = detect_text_elements(image, grid_cells)
                logger.debug("Text detection completed")
            except Exception as text_err:
                logger.error(f"Text detection failed: {str(text_err)}")
                raise
            
            result = {
                "corners": normalized_corners,
                "cells": grid_cells
            }
            
            logger.info("Image processing completed successfully")
            return result
            
        except Exception as inner_err:
            logger.error(f"Error during image processing: {str(inner_err)}")
            logger.exception("Inner exception details:")
            raise
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.exception("Full exception traceback:")
        raise