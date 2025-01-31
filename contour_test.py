import cv2
import numpy as np

def find_contours(image_path, blur_kernel=(17, 17), threshold_method="simple", threshold_value=127,  
                  min_area=100, max_area=None):
    """
    Finds contours in an image.

    Args:
        image_path: Path to the input image.
        blur_kernel: Tuple representing the kernel size for Gaussian blur.  (e.g., (5,5), (3,3)).  Helps reduce noise.
        threshold_method:  "adaptive" or "simple".  Adaptive is generally better for varying lighting.
        threshold_value:  Only used if threshold_method is "simple". The threshold value.
        min_area: Minimum area of a contour to be considered.  Helps filter out small noise.
        max_area: Maximum area of a contour to be considered. If None, no maximum area filter is applied.

    Returns:
        A tuple containing:
            - A copy of the original image with contours drawn on it.
            - A list of the contours found (as numpy arrays of points).  Can be empty.
            - A grayscale thresholded image (useful for debugging).
            - The original image (useful for debugging).

    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # reverse color
        gray = cv2.bitwise_not(gray)

        # Blur the image to reduce noise
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

        # Threshold the image
        if threshold_method == "adaptive":
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # Adjust block size (11) and C (2) as needed
        elif threshold_method == "simple":
            _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError("Invalid threshold_method. Choose 'adaptive' or 'simple'.")


        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Use RETR_EXTERNAL for outer contours

        # Filter contours based on area (optional, but highly recommended)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area and (max_area is None or area <= max_area): # Check min and max area
                filtered_contours.append(contour)


        # Draw contours on a copy of the original image
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 0, 255), 5)  # Red contours

        return img_with_contours, filtered_contours, thresh, img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


# Example usage:
image_path = "assets/IMG_2849.jpg"  # Replace with your image path
img_with_contours, contours, thresh, original_image = find_contours(image_path, min_area=10000) # Example min area. Adjust as needed.

if img_with_contours is not None:
    print(f"Found {len(contours)} contours.")

    w, h = img_with_contours.shape[:2]
    if w > 1000 or h > 1000:
        scale_percent = 1000 / max(w, h)
        img_with_contours = cv2.resize(img_with_contours, (0, 0), fx=scale_percent, fy=scale_percent)
        thresh = cv2.resize(thresh, (0, 0), fx=scale_percent, fy=scale_percent)
        original_image = cv2.resize(original_image, (0, 0), fx=scale_percent, fy=scale_percent)
        contours = [c * scale_percent for c in contours]
        contours = [c.astype(np.int32) for c in contours]

    print("BE CAREFUL: RESIZED!")

    cv2.imshow("Image with Contours", img_with_contours)
    cv2.imshow("Thresholded Image", thresh) # Useful for debugging
    cv2.imshow("Original Image", original_image) # Useful for debugging

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # You can now work with the 'contours' list.  Each element is a numpy array of points.
    # Example:  To get the bounding box of the largest contour:
    if contours:
        largest_contour = max(contours, key=cv2.contourArea) # Find the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2 draw largest_contour
        cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 255, 255), 5)
        print(f"Bounding box of largest contour: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(img_with_contours, (x,y), (x+w, y+h), (0,255,0), 5) # Draw bounding box
        cv2.imshow("Image with Bounding Box", img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



else:
    print("Failed to process the image.")