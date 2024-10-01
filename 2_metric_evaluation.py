import os
import pandas as pd
import numpy as np
from prediction_equirectangular import Equirectangular, predict_on_perspective_image, non_max_suppression_fast, filter_large_bboxes, filter_overlapping_same_class_advanced, YOLO

# Define folder paths and parameters
image_folder = 'test_equirectangular_yolo/images'  # Folder containing equirectangular images
output_directory = 'output'  # Folder to save the metrics
annotations_output_folder = os.path.join(output_directory, 'annotations_elaborated')  # Folder to save predicted bounding boxes
ground_truth_folder = 'test_equirectangular_yolo/annotations'  # Folder containing ground-truth annotation files

# Create output folders if they do not exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(annotations_output_folder, exist_ok=True)

# Load the YOLO model
model_path = 'yolov8/yolov8n.pt'  # Path to the pre-trained YOLO model
model = YOLO(model_path)

# Define camera parameters and the dimensions of perspective images
FOV = 90
height_persp, width_persp = 500, 500  # Height and width of perspective images
view_angles_grid = np.arange(0, 360, 30)  # Generate a grid of THETA angles (step = 30 degrees)

# List to store the results of metrics for each image
metrics_results = []

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
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

    # Calculate the area of both bounding boxes
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union area
    union_area = area_box1 + area_box2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Function to calculate metrics (Precision, Recall, F1-Score, and Mean IoU)
def calculate_metrics(true_boxes, pred_boxes):
    true_positives = 0
    iou_sum = 0

    # Count true positives and calculate the sum of IoU values
    for true_box in true_boxes:
        for pred_box in pred_boxes:
            iou = calculate_iou(true_box, pred_box)
            # Check if IoU is above the threshold (0.5) to consider a match
            if iou > 0.5:
                true_positives += 1
                iou_sum += iou

    # Calculate metrics
    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = true_positives / len(true_boxes) if len(true_boxes) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = iou_sum / true_positives if true_positives > 0 else 0

    return precision, recall, f1_score, mean_iou

# Function to save predicted bounding boxes in .txt format
def save_bounding_boxes_to_txt(bboxes, classes, image_name, output_folder, equirect_width, equirect_height):
    base_filename = os.path.splitext(image_name)[0]
    txt_output_path = os.path.join(output_folder, f"{base_filename}.txt")

    with open(txt_output_path, 'w') as f:
        for bbox, cls in zip(bboxes, classes):
            # Calculate the center and dimensions normalized for the bounding box
            x_center = (bbox[0] + bbox[2]) / 2 / equirect_width
            y_center = (bbox[1] + bbox[3]) / 2 / equirect_height
            width = (bbox[2] - bbox[0]) / equirect_width
            height = (bbox[3] - bbox[1]) / equirect_height

            # Write the bounding box in YOLO format
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Function to read ground-truth bounding boxes from annotation files
def read_ground_truth_boxes(file_path, img_width, img_height):
    true_boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            # Read the annotations in YOLO format
            class_id, x_center, y_center, width, height = [float(x) for x in line.strip().split()]
            # Convert YOLO coordinates to Pascal VOC format (xmin, ymin, xmax, ymax)
            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)
            true_boxes.append([xmin, ymin, xmax, ymax])
    return true_boxes

# Process all images in the specified folder
for image_file in os.listdir(image_folder):
    if not image_file.endswith(('.jpg', '.png')):
        continue  # Ignore non-image files

    image_path = os.path.join(image_folder, image_file)
    equ = Equirectangular(image_path)  # Load the equirectangular image
    equirect_height, equirect_width = equ._height, equ._width

    # Lists to store bounding boxes, scores, and classes for each image
    all_boxes, all_scores, all_classes = [], [], []

    # Generate perspective images for each THETA angle and perform prediction
    for THETA in view_angles_grid:
        print(f"[INFO] Processing image {image_file} with THETA={THETA}, PHI=0")
        perspective_img = equ.GetPerspective(FOV, THETA, 0, height_persp, width_persp)

        # Perform prediction on the perspective image
        bboxes_mapped, scores_mapped, classes_mapped = predict_on_perspective_image(
            model, perspective_img, THETA, 0, FOV, equirect_width, equirect_height)

        # Store prediction results
        all_boxes.extend(bboxes_mapped)
        all_scores.extend(scores_mapped)
        all_classes.extend(classes_mapped)

    # Apply Non-Maximum Suppression (NMS) and filter overlapping or excessively large bounding boxes
    nms_boxes, nms_scores, nms_classes = non_max_suppression_fast(all_boxes, all_scores, all_classes, overlapThresh=0.5)
    nms_boxes = filter_large_bboxes(nms_boxes, equirect_width, equirect_height, max_width_ratio=0.8)
    nms_boxes = filter_overlapping_same_class_advanced(nms_boxes, nms_classes, distance_threshold=30)

    # Save the predicted bounding boxes in .txt format
    save_bounding_boxes_to_txt(nms_boxes, nms_classes, image_file, annotations_output_folder, equirect_width, equirect_height)

    # Load ground-truth bounding boxes from the annotations folder
    ground_truth_file = os.path.join(ground_truth_folder, f"{os.path.splitext(image_file)[0]}.txt")
    if os.path.exists(ground_truth_file):
        true_boxes = read_ground_truth_boxes(ground_truth_file, equirect_width, equirect_height)
    else:
        print(f"[WARNING] Ground-truth annotations not found for {image_file}.")
        true_boxes = []

    # Calculate metrics for the current image
    precision, recall, f1_score, mean_iou = calculate_metrics(true_boxes, nms_boxes)
    print(f"[INFO] Image {image_file}: Precision={precision}, Recall={recall}, F1-Score={f1_score}, Mean IoU={mean_iou}")

    # Add the image path and metrics to the results list
    metrics_results.append([image_path, precision, recall, f1_score, mean_iou])

# Save the metrics results to a CSV file
metrics_df = pd.DataFrame(metrics_results, columns=["Image", "Precision", "Recall", "F1-Score", "Mean IoU"])
output_csv_path = os.path.join(output_directory, 'test_evaluation_metrics.csv')
metrics_df.to_csv(output_csv_path, index=False)
print(f"[INFO] Metrics saved at {output_csv_path}")
