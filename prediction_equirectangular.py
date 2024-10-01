#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:42:13 2024

@author: federicocandela
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

# Camera calibration parameters
FOV = 90  # Field of View for perspective projection

class Equirectangular:
    def __init__(self, img_name):
        # Load the image using OpenCV
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if self._img is None:
            raise ValueError(f"Unable to load image '{img_name}'. Check if the file is a valid image.")
        # Get the dimensions of the image
        self._height, self._width, _ = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        # Compute the intrinsic camera matrix for perspective projection
        f = 0.5 * width / np.tan(0.5 * FOV * np.pi / 180)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)

        # Generate a grid of coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = np.ones_like(x)
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3) @ K_inv.T

        # Rotation based on THETA and PHI angles
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        R1, _ = cv2.Rodrigues(np.radians(THETA) * y_axis)
        R2, _ = cv2.Rodrigues(np.radians(PHI) * x_axis)
        R = R1 @ R2  # Apply rotation around Y axis (THETA) and then X axis (PHI)

        # Apply the rotation to the coordinates
        xyz = xyz @ R.T

        # Convert 3D coordinates to longitude and latitude
        lonlat = xyz2lonlat(xyz)
        # Convert longitude and latitude to 2D image coordinates
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)

        # Map the equirectangular image to the perspective view
        persp = cv2.remap(self._img, XY[:, 0].reshape(height, width), XY[:, 1].reshape(height, width),
                          interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp

def xyz2lonlat(xyz):
    # Normalize the 3D coordinates
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz_norm = xyz / np.where(norm == 0, 1e-6, norm)
    # Convert to longitude and latitude
    lon = np.arctan2(xyz_norm[:, 0], xyz_norm[:, 2])
    lat = np.arcsin(xyz_norm[:, 1])
    return np.stack([lon, lat], axis=1)

def lonlat2XY(lonlat, shape):
    # Convert longitude and latitude to 2D image coordinates
    width = shape[1]
    height = shape[0]
    X = (lonlat[:, 0] / (2 * np.pi) + 0.5) * (width - 1)
    Y = (lonlat[:, 1] / np.pi + 0.5) * (height - 1)
    return np.stack([X, Y], axis=1)

def perspective_bbox_to_equirectangular(bbox, THETA, PHI, FOV, equirect_width, equirect_height, perspective_shape):
    # Convert bounding box from perspective image to equirectangular image coordinates
    x_min, y_min, x_max, y_max = bbox
    width_p = perspective_shape[1]
    height_p = perspective_shape[0]

    # Compute the intrinsic matrix for perspective projection
    f = 0.5 * width_p / np.tan(0.5 * FOV * np.pi / 180)
    cx = (width_p - 1) / 2.0
    cy = (height_p - 1) / 2.0
    K_inv = np.linalg.inv(np.array([[f, 0, cx],
                                    [0, f, cy],
                                    [0, 0, 1]], dtype=np.float32))

    # Define the bounding box points
    points = np.array([
        [x_min, y_min, 1],
        [x_max, y_min, 1],
        [x_max, y_max, 1],
        [x_min, y_max, 1]
    ]).T

    # Convert bounding box points to 3D coordinates
    xyz = K_inv @ points

    # Apply rotation based on THETA and PHI
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    R1, _ = cv2.Rodrigues(np.radians(THETA) * y_axis)
    R2, _ = cv2.Rodrigues(np.radians(PHI) * x_axis)
    R = R1 @ R2

    # Apply rotation to the 3D coordinates
    xyz_sphere = R @ xyz

    # Normalize the 3D coordinates
    norm = np.linalg.norm(xyz_sphere, axis=0, keepdims=True)
    xyz_sphere /= np.where(norm == 0, 1e-6, norm)

    # Convert to longitude and latitude
    lon = np.arctan2(xyz_sphere[0, :], xyz_sphere[2, :])
    lat = np.arcsin(xyz_sphere[1, :])

    # Convert longitude and latitude to equirectangular coordinates
    x_equirect = (lon / (2 * np.pi) + 0.5) * (equirect_width - 1)
    y_equirect = (lat / np.pi + 0.5) * (equirect_height - 1)

    # Clip the bounding box to the image dimensions
    x_min_e = np.clip(np.min(x_equirect), 0, equirect_width - 1)
    y_min_e = np.clip(np.min(y_equirect), 0, equirect_height - 1)
    x_max_e = np.clip(np.max(x_equirect), 0, equirect_width - 1)
    y_max_e = np.clip(np.max(y_equirect), 0, equirect_height - 1)

    return [int(x_min_e), int(y_min_e), int(x_max_e), int(y_max_e)]

def predict_on_perspective_image(model, perspective_img, THETA, PHI, FOV, equirect_width, equirect_height):
    # Predict bounding boxes, scores, and classes on the perspective image using the given model
    results = model.predict(perspective_img)
    bboxes_mapped = []
    scores_mapped = []
    classes_mapped = []

    # Get the shape of the perspective image
    perspective_shape = perspective_img.shape

    # Map bounding boxes from perspective to equirectangular coordinates
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            equirect_bbox = perspective_bbox_to_equirectangular(
                box, THETA, PHI, FOV, equirect_width, equirect_height, perspective_shape)
            bboxes_mapped.append(equirect_bbox)
            scores_mapped.append(scores[i])
            classes_mapped.append(classes[i])

    return bboxes_mapped, scores_mapped, classes_mapped
'''
# Alternative model prediction function (uncomment if using the alternative model)
def predict_on_perspective_image_alt(model, lb, perspective_img, THETA, PHI, FOV, equirect_width, equirect_height):
    # Preprocess the image as required by your model
    input_image = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (224, 224))  # Adjust size as per model input
    input_image = input_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    preds = model.predict(input_image)
    class_idx = np.argmax(preds[0])
    class_label = lb.classes_[class_idx]
    confidence = preds[0][class_idx]

    # Assuming the model outputs one prediction per image (classification)
    # Create a bounding box covering the entire image
    bbox = [0, 0, perspective_img.shape[1], perspective_img.shape[0]]
    equirect_bbox = perspective_bbox_to_equirectangular(
        bbox, THETA, PHI, FOV, equirect_width, equirect_height, perspective_img.shape)
    
    # Map class label to an integer ID (need to define mapping)
    class_id = class_idx  # Modify as needed

    bboxes_mapped = [equirect_bbox]
    scores_mapped = [confidence]
    classes_mapped = [class_id]

    return bboxes_mapped, scores_mapped, classes_mapped
'''


def non_max_suppression_fast(boxes, scores, classes, overlapThresh=0.5):
    # Apply Non-Maximum Suppression to eliminate overlapping bounding boxes
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    pick = []  # List of indices of selected bounding boxes

    unique_classes = np.unique(classes)

    for cls in unique_classes:
        # Filter boxes by class
        idxs = np.where(classes == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]
        x1 = cls_boxes[:, 0]
        y1 = cls_boxes[:, 1]
        x2 = cls_boxes[:, 2]
        y2 = cls_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)  # Compute area of each bounding box
        order = cls_scores.argsort()[::-1]  # Sort by confidence score (descending order)
        while order.size > 0:
            i = order[0]
            pick.append(idxs[i])
            # Find the intersection between the current box and remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Compute width and height of the intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            inter = w * h  # Intersection area
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  # Compute IoU (Intersection over Union)
            # Select boxes with IoU less than the threshold
            inds = np.where(ovr <= overlapThresh)[0]
            order = order[inds + 1]

    return boxes[pick], scores[pick], classes[pick]

def filter_overlapping_same_class_advanced(bboxes, classes, scores=None, iou_threshold=0.05, distance_threshold=30):
    """
    Removes bounding boxes of the same class that are overlapping or very close to each other.
    Args:
    - bboxes: List of bounding boxes (x_min, y_min, x_max, y_max).
    - classes: List of classes corresponding to each bounding box.
    - scores: List of confidence scores (optional).
    - iou_threshold: IoU threshold to consider two boxes as overlapping (value between 0 and 1).
    - distance_threshold: Minimum distance between centers of two bounding boxes of the same class.

    Returns:
    - bboxes_filtrati: List of filtered bounding boxes.
    """
    if scores is None:
        scores = [1.0] * len(bboxes)  # Assign default score of 1.0 if not available

    bboxes_filtrati = []
    used_indices = set()  # Track indices of boxes that are filtered

    def calculate_iou(box1, box2):
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection coordinates
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calculate intersection area
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calculate area of both bounding boxes
        area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
        area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate union area
        union_area = area_box1 + area_box2 - inter_area

        # Calculate IoU
        return inter_area / union_area if union_area > 0 else 0

    for i in range(len(bboxes)):
        if i in used_indices:
            continue

        box1 = bboxes[i]
        class1 = classes[i]
        score1 = scores[i]
        center_x1 = (box1[0] + box1[2]) / 2
        center_y1 = (box1[1] + box1[3]) / 2

        keep_box = True

        for j in range(i + 1, len(bboxes)):
            if j in used_indices:
                continue

            box2 = bboxes[j]
            class2 = classes[j]
            score2 = scores[j]
            center_x2 = (box2[0] + box2[2]) / 2
            center_y2 = (box2[1] + box2[3]) / 2

            # Calculate IoU between boxes
            iou = calculate_iou(box1, box2)

            # Check if they belong to the same class and have significant overlap
            if class1 == class2 and iou > iou_threshold:
                # Keep the box with the highest confidence score
                if score1 >= score2:
                    used_indices.add(j)  # Remove box2
                else:
                    used_indices.add(i)  # Remove box1
                    keep_box = False
                    break

            # Check if the boxes are too close to each other
            distance = np.sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
            if class1 == class2 and distance < distance_threshold:
                # Keep the box with the largest area
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                if area1 >= area2:
                    used_indices.add(j)
                else:
                    used_indices.add(i)
                    keep_box = False
                    break

        if keep_box:
            bboxes_filtrati.append(box1)

    return bboxes_filtrati

# Function to filter out excessively large bounding boxes
def filter_large_bboxes(bboxes, equirect_width, equirect_height, max_width_ratio=0.5, max_height_ratio=1.0):
    """
    Removes bounding boxes that cover more than 50% of the width or the entire height of the image.
    Args:
    - bboxes: List of bounding boxes.
    - equirect_width: Width of the equirectangular image.
    - equirect_height: Height of the equirectangular image.
    - max_width_ratio: Maximum allowed ratio of bounding box width to image width.
    - max_height_ratio: Maximum allowed ratio of bounding box height to image height.

    Returns:
    - bboxes_filtrati: List of bounding boxes that satisfy the size constraints.
    """
    bboxes_filtrati = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width_bbox = x_max - x_min
        height_bbox = y_max - y_min

        # Calculate width and height ratios with respect to the image
        width_ratio = width_bbox / equirect_width
        height_ratio = height_bbox / equirect_height

        # Filter: remove bounding boxes that exceed size limits
        if width_ratio <= max_width_ratio and height_ratio <= max_height_ratio:
            bboxes_filtrati.append(bbox)
        else:
            print(f"[INFO] Removed bounding box that is too large: {bbox}")

    return bboxes_filtrati

# Load YOLOv8 model
model = YOLO('yolov8/yolov8n.pt')

'''
# Alternative model loading (uncomment if using the alternative model)
# Load the alternative model and label binarizer
model = load_model('output/detector.h5')
with open('output/lb.pickle', 'rb') as f:
    lb = pickle.load(f)
'''




# Paths and configurations
equirectangular_img_path = 'input/person-yard1030.jpg'
output_directory = 'prediction'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"[INFO] Created output directory: {output_directory}")

# Load the original equirectangular image
equ = Equirectangular(equirectangular_img_path)
equirect_height, equirect_width = equ._height, equ._width

all_boxes = []
all_scores = []
all_classes = []




# Generate a grid of THETA values (viewing angles) with fixed PHI
def generate_theta_angles(start=0, end=360, step=30):
    theta_values = np.arange(start, end, step)
    phi_value = 0  # PHI fixed at 0 degrees (equator)
    grid = np.array([[theta, phi_value] for theta in theta_values])
    return grid

# Create a grid of viewing angles for generating perspective images
view_angles_grid = generate_theta_angles(step=30)

# Define the height and width of the perspective images
height_persp, width_persp = 500, 500

# Generate perspective images and run object detection
for idx, (THETA, PHI) in enumerate(view_angles_grid):
    print(f"[INFO] Generating perspective image for THETA={THETA}, PHI={PHI}")
    perspective_img = equ.GetPerspective(FOV, THETA, PHI, height_persp, width_persp)

    # Predict bounding boxes, scores, and classes using YOLOv8 model
    bboxes_mapped, scores_mapped, classes_mapped = predict_on_perspective_image(
        model, perspective_img, THETA, PHI, FOV, equirect_width, equirect_height)

    all_boxes.extend(bboxes_mapped)
    all_scores.extend(scores_mapped)
    all_classes.extend(classes_mapped)

# Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
nms_boxes, nms_scores, nms_classes = non_max_suppression_fast(
    all_boxes, all_scores, all_classes, overlapThresh=0.005)

# Apply filter to remove excessively large bounding boxes
nms_boxes = filter_large_bboxes(nms_boxes, equirect_width, equirect_height, max_width_ratio=0.8)

# Apply filter to remove overlapping bounding boxes of the same class
nms_boxes = filter_overlapping_same_class_advanced(nms_boxes, nms_classes, distance_threshold=30)

# Draw the filtered bounding boxes on the image and save the result
equ_img_high_res = equ._img.copy()
for bbox, cls_id in zip(nms_boxes, nms_classes):
    x_min_e, y_min_e, x_max_e, y_max_e = bbox
    cv2.rectangle(equ_img_high_res, (x_min_e, y_min_e), (x_max_e, y_max_e), (0, 0, 255), 2)

def save_annotations_to_txt(bboxes, classes, output_path):
    """
    Save bounding boxes and class annotations to a .txt file in YOLO format.
    
    Args:
    - bboxes: List of bounding boxes in the format [x_min, y_min, x_max, y_max].
    - classes: List of classes corresponding to each bounding box.
    - output_path: Path to save the .txt file.
    """
    with open(output_path, 'w') as f:
        for bbox, cls in zip(bboxes, classes):
            # Calculate the center, width, and height in normalized format (YOLO format)
            x_center = (bbox[0] + bbox[2]) / 2 / equirect_width
            y_center = (bbox[1] + bbox[3]) / 2 / equirect_height
            width = (bbox[2] - bbox[0]) / equirect_width
            height = (bbox[3] - bbox[1]) / equirect_height
            # Write the class and bounding box to the file
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Define the output path for the .txt annotations file
output_txt_path = os.path.join(output_directory, 'equirectangular_with_bboxes_final.txt')

# Save the bounding box annotations to a .txt file
save_annotations_to_txt(nms_boxes, nms_classes, output_txt_path)
print(f"[INFO] Bounding box annotations saved at {output_txt_path}")


# Save the final image with the bounding boxes
output_equirect_img_path = os.path.join(output_directory, 'equirectangular_with_bboxes_final.jpg')
cv2.imwrite(output_equirect_img_path, equ_img_high_res)
print(f"[INFO] Final image with bounding boxes saved at {output_equirect_img_path}")



