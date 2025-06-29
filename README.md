# Change Detection Algorithm

This project implements a pixel-wise change detection algorithm using OpenCV to compare "before" and "after" images and highlight the regions where changes have occurred.



## Input Format

Place paired images in the `input-images/` directory:

- `X.jpg` → **Before** image  
- `X~2.jpg` → **After** image  
- Both images should have the same resolution.



##  How It Works

1. **Image Loading**  
   Reads image pairs for processing.

2. **Grayscale Conversion**  
   Converts both images to grayscale to simplify computation.

3. **Difference Calculation**  
   Computes absolute difference using `cv2.absdiff()`.

4. **Thresholding**  
   Applies a threshold (default: 30) to generate a binary mask.

5. **Morphological Operations**  
   Uses `cv2.morphologyEx()` to clean up the mask:
   - **Closing**: Fills small gaps  
   - **Opening**: Removes noise

6. **Contour Detection**  
   Detects external contours using `cv2.findContours()`.

7. **Change Highlighting**  
   - Filters small regions (<100 px)  
   - Draws red bounding boxes  
   - Labels each region with its pixel area



## 🛠️ Tech Stack

- Python 3  
- OpenCV  
- NumPy



## Output

- Red rectangles highlight detected changes  
- Each rectangle is annotated with the pixel area of the changed region  
- Output images are saved or displayed (as configured)


##  Key Concepts

- Image Preprocessing  
- Morphological Operations  
- Contour Analysis  
- Real-World Change Detection with OpenCV


Sample images 


![1](https://github.com/user-attachments/assets/e1693cff-ee81-4428-a492-07bcbca580ad)                                                                            ![1~3](https://github.com/user-attachments/assets/613b0e3f-1402-47d1-9741-663587c53408)














StreamLit Output



![Screenshot 2025-06-29 201842](https://github.com/user-attachments/assets/1e891527-3c61-4c73-b564-5dce362cb682)









                                  





























