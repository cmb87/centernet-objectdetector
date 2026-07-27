import os
import glob
import csv

def convert_yolo_to_csv(dataset_dir, output_dir):
    """
    Dependency-free converter script to process YOLO datasets and generate
    CenterNet CSV formats without pandas, sklearn, or tqdm.
    - YOLO Format: [class_id, x_center, y_center, width, height] (normalized)
    - CenterNet Albumentations Format: [xmin, ymin, xmax, ymax] (normalized)
    """
    for split in ["train", "valid"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        label_dir = os.path.join(dataset_dir, "labels", split)
        
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        csv_filename = "merged_drone_train.csv" if split == "train" else "merged_drone_test.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        print(f"Processing '{split}' split ({len(image_paths)} images)...")
        
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # CenterNet CsvDataset select_cols=[1,2,3] -> maps index 1: imagePath, index 2: bboxes, index 3: labels
            # Header line is expected since header=True is passed
            writer.writerow(["id", "imagePath", "bboxes", "labels", "size"])
            
            for ctr, img_path in enumerate(image_paths):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, base_name + ".txt")
                
                bboxes = []
                labels = []
                
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            
                            class_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:])
                            
                            # Convert [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
                            xmin = max(0.0, min(1.0, x_center - w / 2.0))
                            ymin = max(0.0, min(1.0, y_center - h / 2.0))
                            xmax = max(0.0, min(1.0, x_center + w / 2.0))
                            ymax = max(0.0, min(1.0, y_center + h / 2.0))
                            
                            bboxes.append([xmin, ymin, xmax, ymax])
                            labels.append(class_id)
                
                # Write matching row
                writer.writerow([
                    ctr,
                    img_path,
                    str(bboxes),
                    str(labels),
                    "[256, 256, 3]"
                ])
                
                if (ctr + 1) % 1000 == 0:
                    print(f"  Processed {ctr + 1}/{len(image_paths)} images...")
                    
        print(f"Successfully saved records to {csv_path}")

if __name__ == "__main__":
    dataset_directory = "/home/cpeeren/Downloads/uavDet/merged_drone_dataset"
    project_directory = "/home/cpeeren/Downloads/uavDet/centernet-objectdetector"
    convert_yolo_to_csv(dataset_directory, project_directory)
