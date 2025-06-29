# Change Detection Algorithm Implementation Summary
Compares before and after images using absolute difference
Converts images to grayscale, applies thresholding, and uses morphological operations
Finds contours in the difference mask to identify changed regions
Draws red rectangles around detected changes
Adds labels showing the pixel area of each change

Algorithm Details
The change detection algorithm works as follows:
Image Loading**: Loads paired before (X.jpg) and after (X~2.jpg) images from `input-images/`
Grayscale Conversion**: Converts both images to grayscale for comparison
Difference Calculation**: Computes absolute difference between images using `cv2.absdiff()`
Thresholding**: Applies threshold (default: 30) to create binary mask of changes
Morphological Operations: 
   - Closing operation to fill small gaps
   - Opening operation to remove noise
Contour Detection: Finds external contours in the change mask
Bounding Box Drawing: Draws red rectangles around significant changes (>100 pixels)
Text Annotation: Adds labels showing the area of each detected change
 

