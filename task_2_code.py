import cv2
import numpy as np
import os
import glob
from pathlib import Path

def load_image_pair(before_path, after_path):
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)
    
    if before_img is None or after_img is None:
        raise ValueError(f"Could not load images: {before_path} or {after_path}")
    
    return before_img, after_img

def detect_changes(before_img, after_img, threshold=30):
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    
    
    diff = cv2.absdiff(before_gray, after_gray)
    
    
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

def find_contours_and_draw_boxes(change_mask, after_img, min_area=100):
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_img = after_img.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
           
            x, y, w, h = cv2.boundingRect(contour)
            
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            
            cv2.putText(annotated_img, f"Change: {area:.0f}px", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated_img

def process_image_pair(before_path, after_path, output_path, threshold=30, min_area=100):
    try:
       
        before_img, after_img = load_image_pair(before_path, after_path)
        
        
        change_mask = detect_changes(before_img, after_img, threshold)
        
        
        annotated_img = find_contours_and_draw_boxes(change_mask, after_img, min_area)
        
        
        cv2.imwrite(output_path, annotated_img)
        
        print(f"Processed: {os.path.basename(before_path)} -> {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"Error processing {before_path}: {str(e)}")

def main():
    input_dir = Path("input-images")
    output_dir = Path("task_2_output")
    output_dir.mkdir(exist_ok=True)
    if not input_dir.exists():
        print(f"Input directory '{input_dir}' not found.")
        return
    
   
    before_images = list(input_dir.glob("*.jpg"))
    before_images = [img for img in before_images if not img.name.endswith("~2.jpg") and not img.name.endswith("~3.jpg")]
    
    if not before_images:
        print(f"No before images (X.jpg) found in {input_dir} directory.")
        return
    
    print(f"Found {len(before_images)} before images to process.")
    

    for before_img_path in before_images:
        
        base_name = before_img_path.stem
        after_img_path = input_dir / f"{base_name}~2.jpg"
        output_img_name = f"{base_name}~3.jpg"
        output_path = output_dir / output_img_name
        if not after_img_path.exists():
            print(f"Warning: After image {after_img_path.name} not found, skipping {before_img_path.name}")
            continue
        process_image_pair(str(before_img_path), str(after_img_path), str(output_path))
    
    print(f"\nProcessing complete! Annotated images saved in {output_dir}/")

if __name__ == "__main__":
    main() 