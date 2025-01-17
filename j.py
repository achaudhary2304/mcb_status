import cv2
import numpy as np
#also works fine and with normal images of real 

def find_largest_color_area(image_path):
    #Loading the image 
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return None, None, None
    
    # converting to HSV as they are practical for real life application 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # defining the range of white and gray so that masks can be created
    #white and greyt would be removed from the image to get the mcb only 

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_grey = np.array([0, 0, 50])
    upper_grey = np.array([180, 25, 200])

    # Create masks for white and grey
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    mask_grey = cv2.inRange(hsv_image, lower_grey, upper_grey)

    
    mask_combined = cv2.bitwise_or(mask_white, mask_grey)

    #to get all colors except white and grey
    mask_inverted = cv2.bitwise_not(mask_combined)

    # Find contours of the remaining colors
    contours, _ = cv2.findContours(mask_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    largest_area = 0
    largest_color = None

    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_color = contour

    if largest_color is None:
        return None, None, None

    # creating a mask for the largest color
    largest_color_mask = np.zeros_like(mask_inverted)
    cv2.drawContours(largest_color_mask, [largest_color], -1, (255), thickness=cv2.FILLED)

    # Calculate the mean color within the largest contour
    mean_color = cv2.mean(hsv_image, mask=largest_color_mask)[:3]  
    # we just want the mean values of the all the HSV values not the mask itself


    # Define color ranges based on the mean color
    lower_bound = np.array([max(mean_color[0] - 10, 0), max(mean_color[1] - 40, 0), max(mean_color[2] - 40, 0)])
    upper_bound = np.array([min(mean_color[0] + 10, 180), min(mean_color[1] + 40, 255), min(mean_color[2] + 40, 255)])

    return lower_bound, upper_bound


def detect_breaker_state(image_path):
    # Read the image
    img = cv2.imread(image_path)
    resized_image = cv2.resize(img, (500, 500))
    img = resized_image
    original = img.copy()
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color,upper_color  = find_largest_color_area(image_path)
    
    
    
    # Create mask for color regions
    color_mask = cv2.inRange(hsv, lower_color,upper_color )
    
    # Find contours of the color switch
    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not color_contours:
        return "Unable to detect switch"
    
    # Get the largest color contour (the switch)
    color_switch = max(color_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(color_switch)
    
    # Create debug images
    contour_img = original.copy()
    color_mask_colored = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    
    # Draw the color contour
    cv2.drawContours(contour_img, [color_switch], -1, (255, 0, 0), 2)
    
    # Analyze color distribution within the bounding box
    switch_region = color_mask[y:y+h, x:x+w]
    
    # Split the region into upper and lower half
    mid_y = h //2
    upper_half = switch_region[:mid_y, :]
    lower_half = switch_region[mid_y:, :]
    
    # Count color pixels in each half
    upper_color_pixels = cv2.countNonZero(upper_half)
    lower_color_pixels = cv2.countNonZero(lower_half)
    
    # Calculate percentages for visualization
    total_pixels = upper_color_pixels + lower_color_pixels
    if total_pixels > 0:
        upper_percentage = (upper_color_pixels / total_pixels) * 100
        lower_percentage = (lower_color_pixels / total_pixels) * 100
    else:
        upper_percentage = lower_percentage = 0
    
    # Determine state based on color distribution
    if total_pixels > 0:
        state = "ON" if upper_color_pixels > lower_color_pixels else "OFF"
    else:
        state = "Unable to determine state"
    
    # Create visualization
    rect_img = original.copy()
    # Draw bounding box
    cv2.rectangle(rect_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Draw middle line
    cv2.line(rect_img, (x, y + mid_y), (x + w, y + mid_y), (0, 255, 0), 2)
    
    # Add distribution visualization
    distribution_img = np.zeros((500, 200, 3), dtype=np.uint8)
    # Draw upper percentage bar
    upper_height = int(180 * (upper_color_pixels / total_pixels)) if total_pixels > 0 else 0
    cv2.rectangle(distribution_img, (50, 190-upper_height), (150, 190), (255, 0, 0), -1)
    # Draw lower percentage bar
    lower_height = int(180 * (lower_color_pixels / total_pixels)) if total_pixels > 0 else 0
    cv2.rectangle(distribution_img, (50, 10), (150, 10+lower_height), (0, 0, 255), -1)
    
    # Add text to distribution visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(distribution_img, f"Upper: {upper_percentage:.1f}%", (10, 30), font, 0.5, (255, 255, 255), 1)
    cv2.putText(distribution_img, f"Lower: {lower_percentage:.1f}%", (10, 180), font, 0.5, (255, 255, 255), 1)
    
    # Create composite debug image
    # debug_h, debug_w = img.shape[:2]
    # debug_images = np.zeros((debug_h * 2, debug_w * 2, 3), dtype=np.uint8)
    
    # # Place images in grid
    # debug_images[:debug_h, :debug_w] = original  # Original
    # debug_images[:debug_h, debug_w:debug_w+200] = distribution_img  # Distribution
    # debug_images[debug_h:, :debug_w] = color_mask_colored  # color mask
    # debug_images[debug_h:, debug_w:] = rect_img  # Bounding box visualization
    # Create composite debug image
    debug_h, debug_w = img.shape[:2]
    debug_images = np.zeros((debug_h * 2, debug_w * 2, 3), dtype=np.uint8)

    # Place images in grid
    debug_images[:debug_h, :debug_w] = original  # Original
    debug_images[:debug_h, debug_w:debug_w + 200] = distribution_img[:debug_h, :200]  # Adjusted distribution
    debug_images[debug_h:, :debug_w] = color_mask_colored  # Color mask
    debug_images[debug_h:, debug_w:] = rect_img  # Bounding box visualization


    # Add labels
    labels = [
        (10, 30, "Original"), 
        (debug_w + 10, 30, "color Distribution"),
        (10, debug_h + 30, "color Mask"), 
        (debug_w + 10, debug_h + 30, "Detection")
    ]
    
    for x, y, text in labels:
        cv2.putText(debug_images, text, (x, y), font, 1, (255, 255, 255), 2)
    
    return {
        "state": state,
        "debug_image": debug_images,
        "upper_percentage": upper_percentage,
        "lower_percentage": lower_percentage,
        "upper_pixels": upper_color_pixels,
        "lower_pixels": lower_color_pixels
    }

def main(image_path):
    result = detect_breaker_state(image_path)
    
    if isinstance(result, dict):
        print(f"Circuit Breaker State: {result['state']}")
        print(f"Upper color Percentage: {result['upper_percentage']:.1f}%")
        print(f"Lower color Percentage: {result['lower_percentage']:.1f}%")
        
        # Display debug image
        cv2.imshow("Circuit Breaker Analysis", result['debug_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(result)

if __name__ == "__main__":
    image_path = r"/home/aryan/code_qr/images/blac.jpeg"
    main(image_path)