import cv2
import os
import json
import random
import shutil

frames_folder = "frames/"
output_dir = "YOLOX_dataset"
train_ratio = 0.8  # 80% frames go to train, 20% to val
category_name = "player"
category_id = 1

os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

train_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": category_id, "name": category_name}]
}

val_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": category_id, "name": category_name}]
}

# Prepare frame list and shuffle
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".png", ".jpg"))])
random.shuffle(frame_files)

train_count = int(len(frame_files) * train_ratio)
train_files = frame_files[:train_count]
val_files = frame_files[train_count:]

boxes = []
current_box = None
drawing = False
ann_id = 1

def draw_box(event, x, y, flags, param):
    global boxes, drawing, current_box
    img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_box = [(x, y), (x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box[1] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box[1] = (x, y)
        boxes.append((current_box[0][0], current_box[0][1],
                      current_box[1][0], current_box[1][1]))
        current_box = None

def annotate_frames(frame_list, split_name, data_dict):
    global boxes, current_box, ann_id

    for i, fname in enumerate(frame_list):
        img_path = os.path.join(frames_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        h, w = img.shape[:2]

        # Copy image to train/val folder
        dest_path = os.path.join(output_dir, f"images/{split_name}", fname)
        shutil.copy(img_path, dest_path)

        boxes = []
        current_box = None

        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", draw_box, param=img)

        while True:
            display_img = img.copy()

            # Draw all completed boxes
            for box in boxes:
                cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Draw the box currently being drawn
            if current_box:
                cv2.rectangle(display_img, current_box[0], current_box[1], (0, 0, 255), 2)

            cv2.imshow("Annotate", display_img)
            key = cv2.waitKey(50) & 0xFF

            if key == ord("n"):  # next frame
                break
            elif key == ord("r"):  # reset all boxes
                boxes = []
            elif key == ord("u"):  # undo last box
                if boxes:
                    boxes.pop()
            elif key == ord("q"):  # quit annotation
                cv2.destroyAllWindows()
                print("Annotation canceled")
                exit()

        # If boxes exist, save image info and annotations
        if boxes:
            data_dict["images"].append({"id": i+1, "file_name": fname, "width": w, "height": h})
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                w_box, h_box = x_max - x_min, y_max - y_min
                data_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": i+1,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0
                })
                ann_id += 1

cv2.destroyAllWindows()

# Annotate train frames
print("Annotate TRAIN frames:")
annotate_frames(train_files, "train", train_data)

# Annotate val frames
print("Annotate VAL frames:")
annotate_frames(val_files, "val", val_data)

# Save JSONs
with open(os.path.join(output_dir, "annotations/instances_train.json"), "w") as f:
    json.dump(train_data, f, indent=4)
with open(os.path.join(output_dir, "annotations/instances_val.json"), "w") as f:
    json.dump(val_data, f, indent=4)

print("Annotations saved!")
