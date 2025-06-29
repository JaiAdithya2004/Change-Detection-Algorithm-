import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
import io
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Change Detection Algorithm",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format."""
    # Convert PIL to numpy array
    numpy_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    if len(numpy_image.shape) == 3:
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    else:
        opencv_image = numpy_image
    
    return opencv_image

def cv2_to_pil(opencv_image):
    """Convert OpenCV image to PIL format."""
    # Convert BGR to RGB
    if len(opencv_image.shape) == 3:
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = opencv_image
    
    return Image.fromarray(rgb_image)

def detect_changes(before_img, after_img, threshold=30):
    """Detect changes between before and after images."""
    # Convert images to grayscale for comparison
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(before_gray, after_gray)
    
    # Apply threshold to get binary mask
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

def find_contours_and_draw_boxes(change_mask, after_img, min_area=100):
    """Find contours and draw bounding boxes around changes."""
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_img = after_img.copy()
    change_info = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw rectangle
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Add text label
            cv2.putText(annotated_img, f"Change {i+1}: {area:.0f}px", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            change_info.append({
                'id': i + 1,
                'area': int(area),
                'position': (x, y),
                'size': (w, h)
            })
    
    return annotated_img, change_info

def process_image_pair(before_img, after_img, threshold=30, min_area=100):
    """Process a single pair of before and after images."""
    try:
        # Convert PIL images to OpenCV format
        before_cv = pil_to_cv2(before_img)
        after_cv = pil_to_cv2(after_img)
        
        # Detect changes
        change_mask = detect_changes(before_cv, after_cv, threshold)
        
        # Find contours and draw bounding boxes
        annotated_img, change_info = find_contours_and_draw_boxes(change_mask, after_cv, min_area)
        
        # Convert back to PIL for display
        annotated_pil = cv2_to_pil(annotated_img)
        change_mask_pil = Image.fromarray(change_mask)
        
        return annotated_pil, change_mask_pil, change_info
        
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
        return None, None, []

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Change Detection Algorithm</h1>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Parameters</h2>', unsafe_allow_html=True)
    
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=1,
        max_value=100,
        value=30,
        help="Lower values detect more subtle changes"
    )
    
    min_area = st.sidebar.slider(
        "Minimum Change Area (pixels)",
        min_value=10,
        max_value=1000,
        value=100,
        help="Minimum area of a change to be highlighted"
    )
    
    # Information box
    st.sidebar.markdown("""
    <div class="info-box">
        <h4>💡 Tips:</h4>
        <ul>
            <li>Images should be perfectly aligned</li>
            <li>Lower threshold = more sensitive</li>
            <li>Higher min area = fewer small changes</li>
            <li>Red boxes show detected changes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Images</h2>', unsafe_allow_html=True)
        
        before_img = st.file_uploader(
            "Upload Before Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload the 'before' image"
        )
        
        after_img = st.file_uploader(
            "Upload After Image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload the 'after' image"
        )
        
        if st.button("🔍 Detect Changes", type="primary", use_container_width=True):
            if before_img is not None and after_img is not None:
                with st.spinner("Processing images..."):
                    # Load images using PIL
                    before_pil = Image.open(before_img)
                    after_pil = Image.open(after_img)
                    
                    annotated_img, change_mask, change_info = process_image_pair(
                        before_pil, after_pil, threshold, min_area
                    )
                    
                    if annotated_img is not None:
                        st.session_state.annotated_img = annotated_img
                        st.session_state.change_mask = change_mask
                        st.session_state.change_info = change_info
                        st.session_state.processed = True
                        st.success("✅ Processing complete!")
            else:
                st.error("Please upload both before and after images.")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>', unsafe_allow_html=True)
        
        if 'processed' in st.session_state and st.session_state.processed:
            tab1, tab2, tab3 = st.tabs(["🎯 Annotated Image", "🔍 Change Mask", "📈 Statistics"])
            
            with tab1:
                st.image(st.session_state.annotated_img, caption="Annotated After Image", use_column_width=True)
                
                if st.button("💾 Download Annotated Image", use_container_width=True):
                    img_buffer = io.BytesIO()
                    st.session_state.annotated_img.save(img_buffer, format='JPEG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download",
                        data=img_buffer.getvalue(),
                        file_name="annotated_image.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            with tab2:
                st.image(st.session_state.change_mask, caption="Change Detection Mask", use_column_width=True)
            
            with tab3:
                if st.session_state.change_info:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown(f"**Found {len(st.session_state.change_info)} changes:**")
                    
                    for change in st.session_state.change_info:
                        st.write(f"• Change {change['id']}: {change['area']} pixels at position {change['position']}")
                    
                    total_area = sum(change['area'] for change in st.session_state.change_info)
                    st.write(f"**Total changed area: {total_area} pixels**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No significant changes detected with current parameters.")
        
        else:
            st.info("👆 Upload images and click 'Detect Changes' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Change Detection Algorithm | Built with Streamlit & OpenCV</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 