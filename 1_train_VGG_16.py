"""
Created on Tue Jun 25 15:42:13 2024

@author: federicocandela
[INFO] Number of training samples: 1764
[INFO] Number of test samples: 442
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc, precision_recall_fscore_support)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config  # Make sure you have a config.py with necessary configurations
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2


# Define the number of epochs
NUM_EPOCHS = 50  # Set your desired number of epochs

# Define the number of classes
num_classes = len(config.CLASSES)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("[INFO] GPUs found:", len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[ERROR] No GPUs found. Ensure that GPU drivers and CUDA are properly installed.")

# Configure multi-GPU distribution strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    print("[INFO] Loading dataset...")

    # Function to preprocess images and bounding boxes
    def preprocess_image_and_bbox(image, bbox, target_size):
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = target_size

        # Denormalize bounding box coordinates with respect to the original image
        center_x = bbox[0] * orig_w
        center_y = bbox[1] * orig_h
        width = bbox[2] * orig_w
        height = bbox[3] * orig_h

        # Debug: Print denormalized bounding box with respect to the original image
        print(f"[DEBUG] Denormalized bounding box (original size): "
              f"(center_x: {center_x}, center_y: {center_y}, width: {width}, height: {height})")

        # Calculate scale factor to maintain aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Debug: Print scale factor and new dimensions
        print(f"[DEBUG] Scale: {scale}, new_w: {new_w}, new_h: {new_h}")

        # Resize the image while maintaining aspect ratio
        resized_image = cv2.resize(image, (new_w, new_h))

        # Create a blank image with target dimensions and fill it with black pixels
        new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate offsets to center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Debug: Print offsets
        print(f"[DEBUG] x_offset: {x_offset}, y_offset: {y_offset}")

        # Insert the resized image into the blank image
        new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

        # Update bounding box coordinates
        center_x = center_x * scale + x_offset
        center_y = center_y * scale + y_offset
        width = width * scale
        height = height * scale

        # Normalize the bounding box with respect to the 224x224 image
        print(f"[DEBUG] Before normalization - center_x: {center_x}, center_y: {center_y}, "
              f"width: {width}, height: {height}")
        center_x /= target_w
        center_y /= target_h
        width /= target_w
        height /= target_h
        print(f"[DEBUG] After normalization - center_x: {center_x}, center_y: {center_y}, "
              f"width: {width}, height: {height}")

        new_bbox = (center_x, center_y, width, height)

        # Debug: Print normalized bounding box with respect to 224x224 image
        print(f"[DEBUG] Final normalized bounding box (224x224): {new_bbox}")

        return new_image, new_bbox

    # Function to load images and annotations
    def load_data_and_labels(base_path, classes):
        data, labels, bboxes, imagePaths = [], [], [], []

        for class_id, class_label in enumerate(classes):
            image_folder = os.path.join(base_path, class_label, "images")
            annotation_folder = os.path.join(base_path, class_label, "labels")

            if not os.path.isdir(image_folder) or not os.path.isdir(annotation_folder):
                print(f"[WARNING] {image_folder} or {annotation_folder} is not a directory")
                continue

            for image_filename in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image_filename)
                
                # Correzione del nome del file di annotazione
                if image_filename.endswith('.jpg'):
                    annotation_filename = image_filename.replace('.jpg', '.txt')
                elif image_filename.endswith('.png'):
                    annotation_filename = image_filename.replace('.png', '.txt')
                else:
                    print(f"[WARNING] Unsupported file format: {image_filename}")
                    continue  # Salta l'iterazione se il formato del file non è supportato
                                
                annotation_path = os.path.join(annotation_folder, annotation_filename)

                if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                    print(f"[WARNING] Image or annotation not found: {image_path}, {annotation_path}")
                    continue

                # Load the original image
                orig_image = cv2.imread(image_path)
                if orig_image is None:
                    print(f"[WARNING] Unable to read image: {image_path}")
                    continue

                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        values = line.strip().split()
                        bbox = [float(x) for x in values[1:5]]  # Normalized values

                        # Debug: Print original bounding boxes (normalized)
                        print(f"[DEBUG] Original normalized bounding box: {bbox} for image: {image_filename}")

                        # Preprocess the image and bounding box
                        image, new_bbox = preprocess_image_and_bbox(orig_image, bbox, (224, 224))

                        # Add the image and labels to their respective sets
                        data.append(image)
                        labels.append(class_id)
                        bboxes.append(new_bbox)
                        imagePaths.append(image_path)

        return np.array(data), np.array(labels), np.array(bboxes), np.array(imagePaths)

    # Load the data
    data, labels, bboxes, imagePaths = load_data_and_labels(
        base_path=config.BASE_PATH, classes=config.CLASSES
    )

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels, train_bboxes, test_bboxes, train_imagePaths, test_imagePaths = train_test_split(
        data, labels, bboxes, imagePaths, test_size=0.20, random_state=42
    )

    # Normalize the images
    train_data = train_data.astype("float32") / 255.0
    test_data = test_data.astype("float32") / 255.0

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    print(f"[INFO] Number of training samples: {len(train_data)}")
    print(f"[INFO] Number of test samples: {len(test_data)}")

    # Create a mapping from class indices to class labels
    class_indices = {i: label for i, label in enumerate(config.CLASSES)}

    # Load the VGG16 network, ensuring the head FC layers are left off
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    vgg.trainable = False  # Freeze the base model
    # Sbloccare gli ultimi due livelli
    #for layer in vgg.layers[-2:]:
        #layer.trainable = True
    
    
    flatten = vgg.output
    flatten = Flatten()(flatten)
    
    #AGGIUNTI
    # Aggiunta di nuovi livelli convoluzionali
    x = vgg.output
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Aggiunta di nuovi livelli fully connected
    flatten = Flatten()(x)
    flatten = Dense(512, activation="relu")(flatten)
    flatten = BatchNormalization()(flatten)
    flatten = Dropout(0.2)(flatten)
    flatten = Dense(512, activation="relu")(flatten)
    flatten = BatchNormalization()(flatten)
    flatten = Dropout(0.2)(flatten)
   #AGGIUNTI 
    
    
    # Bounding box head
    bboxHead = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(flatten)
    bboxHead = BatchNormalization()(bboxHead)
    bboxHead = Dense(128, activation="relu", kernel_regularizer=l2(0.0001))(bboxHead)
    bboxHead = BatchNormalization()(bboxHead)
    bboxHead = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(bboxHead)
    bboxHead = BatchNormalization()(bboxHead)
    bboxHead = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(bboxHead)
    bboxHead = BatchNormalization()(bboxHead)
    bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
    
    # Classification head with L2 regularization and Batch Normalization
    softmaxHead = Dense(1024, activation="relu", kernel_regularizer=l1(0.0001))(flatten)
    softmaxHead = BatchNormalization()(softmaxHead)
    softmaxHead = Dropout(0.5)(softmaxHead)
    softmaxHead = Dense(512, activation="relu", kernel_regularizer=l1(0.0001))(softmaxHead)
    softmaxHead = BatchNormalization()(softmaxHead)
    softmaxHead = Dropout(0.5)(softmaxHead)
    softmaxHead = Dense(num_classes, activation="softmax", name="class_label")(softmaxHead)

    # Construct the model
    model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))
    '''
    def giou_loss(y_true, y_pred):
        """
        Calcola la GIoU (Generalized Intersection over Union) loss.
        
        Args:
        - y_true: Tensor di forma (batch_size, 4) con le vere coordinate dei bounding box [center_x, center_y, width, height].
        - y_pred: Tensor di forma (batch_size, 4) con le coordinate predette dei bounding box [center_x, center_y, width, height].
        
        Returns:
        - loss: valore della perdita GIoU.
        """
        # Dividi y_true e y_pred in coordinate x, y, width e height
        x_true, y_true, w_true, h_true = tf.split(y_true, 4, axis=-1)
        x_pred, y_pred, w_pred, h_pred = tf.split(y_pred, 4, axis=-1)
        
        # Calcolo delle coordinate dei box predetti e veri
        x_min_true = x_true - w_true / 2
        y_min_true = y_true - h_true / 2
        x_max_true = x_true + w_true / 2
        y_max_true = y_true + h_true / 2
        
        x_min_pred = x_pred - w_pred / 2
        y_min_pred = y_pred - h_pred / 2
        x_max_pred = x_pred + w_pred / 2
        y_max_pred = y_pred + h_pred / 2
    
        # Calcolo delle coordinate dell'intersezione
        inter_x_min = tf.maximum(x_min_true, x_min_pred)
        inter_y_min = tf.maximum(y_min_true, y_min_pred)
        inter_x_max = tf.minimum(x_max_true, x_max_pred)
        inter_y_max = tf.minimum(y_max_true, y_max_pred)
        inter_area = tf.maximum(inter_x_max - inter_x_min, 0) * tf.maximum(inter_y_max - inter_y_min, 0)
        
        # Calcolo dell'area dei box predetti e veri
        true_area = (x_max_true - x_min_true) * (y_max_true - y_min_true)
        pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
        
        # Calcolo dell'unione dei box
        union_area = true_area + pred_area - inter_area
        
        # Calcolo dell'IoU
        iou = inter_area / (union_area + 1e-10)  # Evita la divisione per zero
    
        # Calcolo del bounding box minimo che racchiude sia il predetto che il vero box (enclosing box)
        enclose_x_min = tf.minimum(x_min_true, x_min_pred)
        enclose_y_min = tf.minimum(y_min_true, y_min_pred)
        enclose_x_max = tf.maximum(x_max_true, x_max_pred)
        enclose_y_max = tf.maximum(y_max_true, y_max_pred)
        enclose_area = (enclose_x_max - enclose_x_min) * (enclose_y_max - enclose_y_min)
        
        # Calcolo della GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-10)
        
        # Calcolo della perdita GIoU (valore tra -1 e 1, quindi somma 1 per ottenere una perdita positiva)
        giou_loss_value = 1 - giou
    
        # Restituisci la media della perdita GIoU su tutto il batch
        return tf.reduce_mean(giou_loss_value)
    '''
    #
    
    '''
    # Funzione di perdita combinata: IoU + MSE
    def combined_bbox_loss(y_true, y_pred):
        # Calcolo dell'IoU (Intersection over Union)
        x_true, y_true, w_true, h_true = tf.split(y_true, 4, axis=-1)
        x_pred, y_pred, w_pred, h_pred = tf.split(y_pred, 4, axis=-1)
    
        # Calcolo delle coordinate x_min, y_min, x_max, y_max
        x_min_true = x_true - w_true / 2.0
        y_min_true = y_true - h_true / 2.0
        x_max_true = x_true + w_true / 2.0
        y_max_true = y_true + h_true / 2.0
    
        x_min_pred = x_pred - w_pred / 2.0
        y_min_pred = y_pred - h_pred / 2.0
        x_max_pred = x_pred + w_pred / 2.0
        y_max_pred = y_pred + h_pred / 2.0
    
        # Calcolo area di intersezione e unione
        inter_x_min = tf.maximum(x_min_true, x_min_pred)
        inter_y_min = tf.maximum(y_min_true, y_min_pred)
        inter_x_max = tf.minimum(x_max_true, x_max_pred)
        inter_y_max = tf.minimum(y_max_true, y_max_pred)
        inter_area = tf.maximum(inter_x_max - inter_x_min, 0) * tf.maximum(inter_y_max - inter_y_min, 0)
    
        true_area = (x_max_true - x_min_true) * (y_max_true - y_min_true)
        pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    
        union_area = true_area + pred_area - inter_area
        iou = inter_area / (union_area + 1e-10)  # Per evitare divisione per zero
    
        # Calcolo della perdita IoU
        iou_loss = 1 - iou
    
        # Mean Squared Error (MSE) per bounding box
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    
        # Somma pesata di IoU e MSE
        total_loss = 0.5 * iou_loss + 0.5 * mse_loss
        return total_loss
    '''
    
    # Define targets for training
    trainTargets = {
        "class_label": train_labels,
        "bounding_box": train_bboxes
    }
    testTargets = {
        "class_label": test_labels,
        "bounding_box": test_bboxes
    }

    # Compile the model
    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",   
        #"bounding_box": Huber(delta=1.0)
        #"bounding_box": combined_bbox_loss  # Usa la perdita combinata per i bounding box
        #"bounding_box": giou_loss
    }
    lossWeights = {
        "class_label": 1.0,
        "bounding_box": 20.0
    }
    opt = Adam(learning_rate=config.INIT_LR)
    model.compile(
        loss=losses,
        optimizer=opt,
        metrics={"class_label": "accuracy", "bounding_box": "mse"},
        loss_weights=lossWeights
    )

    # Configure callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    print("[INFO] Training model...")
    H = model.fit(
        train_data, trainTargets,
        validation_data=(test_data, testTargets),
        batch_size=config.BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    # Save the model
    print("[INFO] Saving object detector model...")
    model.save(config.MODEL_PATH)
    def calculate_map(true_bboxes, pred_bboxes, iou_thresholds=[0.5, 0.75, 0.9]):
        """
        Calcola la Mean Average Precision (mAP) su più soglie IoU.
        
        Args:
        - true_bboxes: Lista di array con le coordinate dei bounding box reali [center_x, center_y, width, height].
        - pred_bboxes: Lista di array con le coordinate dei bounding box predetti [center_x, center_y, width, height].
        - iou_thresholds: Lista delle soglie di IoU da utilizzare per il calcolo della mAP.
        
        Returns:
        - map_score: Valore della mAP calcolata.
        """
        # Inizializza contatori per true positives, false positives e false negatives
        tp = {iou: 0 for iou in iou_thresholds}
        fp = {iou: 0 for iou in iou_thresholds}
        fn = {iou: 0 for iou in iou_thresholds}
    
        # Itera su ogni soglia IoU
        for iou_threshold in iou_thresholds:
            for true_bbox, pred_bbox in zip(true_bboxes, pred_bboxes):
                iou = calculate_iou(true_bbox, pred_bbox)
                if iou >= iou_threshold:
                    tp[iou_threshold] += 1
                else:
                    fp[iou_threshold] += 1
    
            # Calcola false negatives
            fn[iou_threshold] = len(true_bboxes) - tp[iou_threshold]
    
        # Calcola precisione e recall per ogni soglia di IoU
        precision = {iou: tp[iou] / (tp[iou] + fp[iou] + 1e-10) for iou in iou_thresholds}
        recall = {iou: tp[iou] / (tp[iou] + fn[iou] + 1e-10) for iou in iou_thresholds}
    
        # Calcola l'Average Precision (AP) per ogni soglia di IoU
        ap = {iou: (precision[iou] * recall[iou]) / (precision[iou] + recall[iou] + 1e-10) for iou in iou_thresholds}
    
        # Calcola la mAP come media delle AP su tutte le soglie di IoU
        map_score = np.mean(list(ap.values()))
    
        return map_score
    
    def calculate_iou(true_bbox, pred_bbox):
        """
        Calcola l'Intersection over Union (IoU) tra un bounding box reale e uno predetto.
        
        Args:
        - true_bbox: Lista o array con coordinate [center_x, center_y, width, height] del bounding box reale.
        - pred_bbox: Lista o array con coordinate [center_x, center_y, width, height] del bounding box predetto.
        
        Returns:
        - iou: Valore dell'IoU (tra 0 e 1).
        """
        # Convertire da [center_x, center_y, width, height] a [x_min, y_min, x_max, y_max]
        true_x_min = true_bbox[0] - true_bbox[2] / 2
        true_y_min = true_bbox[1] - true_bbox[3] / 2
        true_x_max = true_bbox[0] + true_bbox[2] / 2
        true_y_max = true_bbox[1] + true_bbox[3] / 2
        
        pred_x_min = pred_bbox[0] - pred_bbox[2] / 2
        pred_y_min = pred_bbox[1] - pred_bbox[3] / 2
        pred_x_max = pred_bbox[0] + pred_bbox[2] / 2
        pred_y_max = pred_bbox[1] + pred_bbox[3] / 2
    
        # Calcolo delle coordinate dell'area di intersezione
        inter_x_min = max(true_x_min, pred_x_min)
        inter_y_min = max(true_y_min, pred_y_min)
        inter_x_max = min(true_x_max, pred_x_max)
        inter_y_max = min(true_y_max, pred_y_max)
    
        # Calcolo dell'area di intersezione
        inter_area = max(inter_x_max - inter_x_min, 0) * max(inter_y_max - inter_y_min, 0)
    
        # Calcolo delle aree del bounding box reale e predetto
        true_area = (true_x_max - true_x_min) * (true_y_max - true_y_min)
        pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    
        # Calcolo dell'area di unione
        union_area = true_area + pred_area - inter_area
    
        # Calcolo dell'IoU
        iou = inter_area / union_area if union_area > 0 else 0
    
        return iou
    
    
    # Function to compare bounding boxes
    def compare_bounding_boxes(image, original_bbox, predicted_bbox):
        """Displays and compares original and predicted bounding boxes"""
        print(f"[DEBUG] Image dimensions: {image.shape}")

        if len(original_bbox) != 4:
            print(f"[ERROR] Original bounding box has {len(original_bbox)} values, expected 4: {original_bbox}")
            return
        if len(predicted_bbox) != 4:
            print(f"[ERROR] Predicted bounding box has {len(predicted_bbox)} values, expected 4: {predicted_bbox}")
            return

        # Original bounding box
        print(f"[DEBUG] Original bounding box (normalized): {original_bbox}")
        orig_x, orig_y, orig_w, orig_h = original_bbox
        orig_x = orig_x * image.shape[1]
        orig_y = orig_y * image.shape[0]
        orig_w = orig_w * image.shape[1]
        orig_h = orig_h * image.shape[0]
        print(f"[DEBUG] Original bounding box (denormalized): (x: {orig_x}, y: {orig_y}, w: {orig_w}, h: {orig_h})")

        plt.imshow(image)
        plt.gca().add_patch(plt.Rectangle(
            (orig_x - orig_w / 2, orig_y - orig_h / 2), orig_w, orig_h,
            fill=False, edgecolor='green', linewidth=2))

        # Predicted bounding box
        print(f"[DEBUG] Predicted bounding box (normalized): {predicted_bbox}")
        pred_x, pred_y, pred_w, pred_h = predicted_bbox
        pred_x = pred_x * image.shape[1]
        pred_y = pred_y * image.shape[0]
        pred_w = pred_w * image.shape[1]
        pred_h = pred_h * image.shape[0]
        print(f"[DEBUG] Predicted bounding box (denormalized): (x: {pred_x}, y: {pred_y}, w: {pred_w}, h: {pred_h})")

        plt.gca().add_patch(plt.Rectangle(
            (pred_x - pred_w / 2, pred_y - pred_h / 2), pred_w, pred_h,
            fill=False, edgecolor='red', linewidth=2))

        plt.legend(["Original", "Predicted"])
        plt.show()

    # Visualization during training
    for i, (image, bbox) in enumerate(zip(train_data, train_bboxes)):
        if len(bbox) != 4:
            print(f"[ERROR] Bounding box has dimension {len(bbox)}: {bbox}")
            continue

        # Model prediction for the current image
        predicted_bbox = model.predict(np.expand_dims(image, axis=0))[0][0]

        # Debug predictions
        print(f"[DEBUG] Original bounding box: {bbox}, Predicted bounding box: {predicted_bbox}")

        # Display comparison
        compare_bounding_boxes(image, bbox, predicted_bbox)

        if i > 5:  # Only display the first 5 comparisons
            break

    # Ensure the plots directory exists
    if not os.path.exists(config.PLOTS_PATH):
        os.makedirs(config.PLOTS_PATH)

    # Handle metrics and plots
    lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
    history = H.history
    N = np.arange(0, len(history["loss"]))
    plt.style.use("ggplot")
    fig, ax = plt.subplots(3, 1, figsize=(13, 13))

    for i, l in enumerate(lossNames):
        if l in history:
            title = f"Loss for {l}" if l != "loss" else "Total loss"
            ax[i].set_title(title)
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel("Loss")
            ax[i].plot(N, history[l], label=l)
            ax[i].plot(N, history["val_" + l], label="val_" + l)
            ax[i].legend()

    plt.tight_layout()
    plotPath = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
    plt.savefig(plotPath)
    plt.close()

    # Classification accuracy
    plt.figure()
    if "class_label_accuracy" in history and "val_class_label_accuracy" in history:
        plt.plot(N, history["class_label_accuracy"], label="class_label_train_acc")
        plt.plot(N, history["val_class_label_accuracy"], label="val_class_label_acc")
    plt.title("Class Label Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    plotPath = os.path.sep.join([config.PLOTS_PATH, "accuracy.png"])
    plt.savefig(plotPath)
    plt.close()

    # Final model evaluation
    print("[INFO] Evaluating model...")
    predictions = model.predict(test_data)
    class_predictions = np.argmax(predictions[1], axis=1)
    test_labels_argmax = np.argmax(test_labels, axis=1)

    report = classification_report(
        test_labels_argmax,
        class_predictions,
        target_names=config.CLASSES
    )
    print(report)
    cm = confusion_matrix(test_labels_argmax, class_predictions)
    print(cm)
    
    # Calcolo delle metriche sui bounding box
    iou_values = []
    for true_bbox, pred_bbox in zip(test_bboxes, predictions[0]):
        iou = calculate_iou(true_bbox, pred_bbox)
        iou_values.append(iou)
    
    mean_iou = np.mean(iou_values)
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Calcolo della mAP
    map_score = calculate_map(test_bboxes, predictions[0], iou_thresholds=[0.5, 0.75, 0.9])
    print(f"Mean Average Precision (mAP) a soglie [0.5, 0.75, 0.9]: {map_score:.4f}")
        
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_argmax, class_predictions, average='macro')
    mse = np.mean(np.square(predictions[0] - test_bboxes))
    
    metrics_file = os.path.sep.join([config.PLOTS_PATH, "metrics.txt"])
    with open(metrics_file, "w") as f:
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\nPrecision: {:.4f}\nRecall: {:.4f}\nF1 Score: {:.4f}\nMSE: {:.4f}\n".format(
            precision, recall, f1, mse))
    # Aggiornamento del report finale con le metriche dei bounding box
    with open(metrics_file, "a") as f:
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Average Precision (mAP): {map_score:.4f}\n")

    # Confusion Matrix plot
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, config.CLASSES, rotation=45)
    plt.yticks(tick_marks, config.CLASSES)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plotPath = os.path.sep.join([config.PLOTS_PATH, "confusion_matrix.png"])
    plt.savefig(plotPath)
    plt.close()

    # ROC Curve
    # For multi-class ROC, we need to binarize the labels
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(test_labels_argmax, classes=range(num_classes))
    y_pred_scores = predictions[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(config.CLASSES[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plotPath = os.path.sep.join([config.PLOTS_PATH, "roc_curve.png"])
    plt.savefig(plotPath)
    plt.close()
