import cv2
import numpy as np
import json
import base64
import logging
import os
import pytesseract

# Add at top of file with other imports
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug')
os.makedirs(DEBUG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        cv2.imwrite(os.path.join(DEBUG_DIR, 'debug_grayscale.png'), gray)
        
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
            cv2.imwrite(os.path.join(DEBUG_DIR, f'corners_mask_{idx}_debug.png'), mask)
        
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
        
        cv2.imwrite(os.path.join(DEBUG_DIR, 'corners_detection_debug.png'), debug_image)
        
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
    cv2.imwrite(os.path.join(DEBUG_DIR, 'mask_debug.png'), mask)
    debug_image = image.copy()
    for corner in corners:
        cv2.circle(debug_image, (corner['x'], corner['y']), 10, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(DEBUG_DIR, 'corners_debug.png'), debug_image)

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
    
    cv2.imwrite(os.path.join(DEBUG_DIR, 'grid_debug.png'), debug_image)

# ...existing imports and helper functions remain unchanged until detect_text_elements...



def detect_text_elements(image, grid_cells):
    """Detect any letter or number in grid cells using Tesseract OCR"""
    try:
        debug_image = image.copy()
        height, width = image.shape[:2]
        
        # Calculate cell dimensions
        num_cols = len(set(c['x'] for c in grid_cells))
        num_rows = len(set(c['y'] for c in grid_cells))
        cell_width = width / num_cols
        cell_height = height / num_rows
        
        # Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray)
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        processed_cells = []
        for cell in grid_cells:
            # Calculate cell boundaries with margin
            margin = 0.1  # 10% margin
            cell_x = int((cell['x'] - 1) * cell_width)
            cell_y = int((num_rows - cell['y']) * cell_height)
            
            # Add margin to ROI
            roi_x = max(0, cell_x + int(cell_width * margin))
            roi_y = max(0, cell_y + int(cell_height * margin))
            roi_w = min(int(cell_width * (1 - 2*margin)), width - roi_x)
            roi_h = min(int(cell_height * (1 - 2*margin)), height - roi_y)
            
            # Extract cell ROI
            roi = thresh[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            if roi.size == 0:
                continue
            
            # Scale up ROI for better OCR
            roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # OCR configuration
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            
            # Perform OCR
            text = pytesseract.image_to_string(roi, config=custom_config).strip()
            
            cell_info = {
                'x': cell['x'],
                'y': cell['y'],
                'element': text.upper() if text else 'none'
            }
            
            # Draw debug visualization
            if cell_info['element'] != 'none':
                cv2.rectangle(debug_image, 
                            (roi_x, roi_y), 
                            (roi_x + roi_w, roi_y + roi_h), 
                            (0, 255, 0), 2)
                cv2.putText(debug_image, 
                          cell_info['element'],
                          (roi_x, roi_y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 0, 255), 2)
            
            # Save individual cell ROI for debugging
            cv2.imwrite(
                os.path.join(DEBUG_DIR, f'cell_{cell["x"]}_{cell["y"]}.png'), 
                roi
            )
            
            processed_cells.append(cell_info)
        
        # Save debug visualization
        cv2.imwrite(os.path.join(DEBUG_DIR, 'text_detection_debug.png'), debug_image)
        return processed_cells
        
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
        if image is None:
            raise IOError(f"Failed to load image from {image_path}")
            
        logger.debug(f"Loaded image shape: {image.shape}")
        cv2.imwrite(os.path.join(DEBUG_DIR, 'debug_original.png'), image)

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