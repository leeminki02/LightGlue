from pathlib import Path

from lightglue.superpoint import SuperPoint
from lightglue.disk import DISK

from lightglue import LightGlue
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import cv2
import torch
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
        cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 0, 255), 10)  # Red contours
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest_contour)
        cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 255, 0), 10)  # Green largest contour
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (255, 0, 0), 10)  # Blue bounding box
        cv2.imwrite("results/"+image_path[8:], img_with_contours)
        return img_with_contours, filtered_contours, thresh, img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

img0 = "IMG_2849.jpg"
img1 = "IMG_2850.jpg"

image0 = load_image(images / "IMG_2849.jpg")
image1 = load_image(images / "IMG_2850.jpg")

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))

img_with_contours0, contours0, thresh0, original_image0 = find_contours("assets/"+img0, min_area=10000) # Example min area. Adjust as needed.
img_with_contours1, contours1, thresh1, original_image1 = find_contours("assets/"+img1, min_area=10000) # Example min area. Adjust as needed.

use_contour = True

if use_contour:

    contour0 = torch.tensor(max(contours0, key=cv2.contourArea)).to(device)
    contour1 = torch.tensor(max(contours1, key=cv2.contourArea)).to(device)

    feat0_exp = feats0['keypoints'][0].unsqueeze(1)
    feat1_exp = feats1['keypoints'][0].unsqueeze(1)
    cont0_exp = contour0.unsqueeze(0).reshape(1, -1, 2)
    cont1_exp = contour1.unsqueeze(0).reshape(1, -1, 2)

    dist0 = torch.sqrt(torch.sum((feat0_exp - cont0_exp) ** 2, dim=2))
    dist1 = torch.sqrt(torch.sum((feat1_exp - cont1_exp) ** 2, dim=2))

    min_d0, idx0 = torch.min(dist0, dim=1)
    min_d1, idx1 = torch.min(dist1, dim=1)

    close_inds0 = torch.nonzero(min_d0 < 13).squeeze(1)
    close_inds1 = torch.nonzero(min_d1 < 13).squeeze(1)

    selected_f0 = {"keypoints": [[]], "descriptors": [[]], "keypoint_scores": [], "image_size": feats0["image_size"]}
    selected_f1 = {"keypoints": [[]], "descriptors": [[]], "keypoint_scores": [], "image_size": feats1["image_size"]}


    selected_f0['keypoints'] = torch.index_select(feats0['keypoints'][0], 0, close_inds0).reshape(1,-1,2)
    selected_f0['descriptors'] = torch.index_select(feats0['descriptors'][0], 0, close_inds0).reshape(1,-1,256)
    selected_f0['keypoint_scores'] = torch.index_select(feats0['keypoint_scores'][0], 0, close_inds0).reshape(1,-1)

    selected_f1['keypoints'] = torch.index_select(feats1['keypoints'][0], 0, close_inds1).reshape(1,-1,2)
    selected_f1['descriptors'] = torch.index_select(feats1['descriptors'][0], 0, close_inds1).reshape(1,-1,256)
    selected_f1['keypoint_scores'] = torch.index_select(feats1['keypoint_scores'][0], 0, close_inds1).reshape(1,-1)

    matches01 = matcher({"image0": selected_f0, "image1": selected_f1})
    selected_f0, selected_f1, matches01 = [
        rbd(x) for x in [selected_f0, selected_f1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = selected_f0["keypoints"], selected_f1["keypoints"], matches01["matches"]
else:
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
viz2d.save_plot("results/"+img0[:-4]+"_"+img1[:-4]+"_matches_"+str(use_contour)+".jpg")

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([okpt0, okpt1], colors=["red", "red"], ps=6)
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)

viz2d.save_plot("results/"+img0[:-4]+"_"+img1[:-4]+"_keypoints_"+str(use_contour)+".jpg")

viz2d.show()
cv2.waitKey(0)