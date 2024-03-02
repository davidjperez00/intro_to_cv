# Author David Perez
# Date: 3/1/2024
# Python 1 Assignment
# Description: Perform canny edge detection on an image using the following steps:
# 1. Apply a Gaussian filter to the image to reduce noise
# 2. Apply a Sobel kernel to the image to detect the edges
# 3. Use the gradient magnitude and direction to perform non-maximum suppression
# 4. Perform dual thresholding by hysteresis to detect the correct edges in the image

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Note: function expects a 2D image
#    with pixel values in the range [0, 255]
#    and a 3x3 kernel
def preprocess_image(image):
  # Surround the outside of the image with 1 pixel layer of zeros
  image_padded = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))

  return image_padded

def create_gaussian_kernel(kernel_size, sigma=1):
    kernel_size = int(kernel_size) // 2
    x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel

# For each pixel neighborhood (3x3 set of pixels), multiply the pixel
# values by the kernel values. This is equivalent
# to applying a correlation operation between the image and the kernel.
def apply_kernel_to_image(image, kernel):
  # Create a new image to store the result of the correlation operation
  # This image will have the same size as the original image
  output_image = np.zeros(image.shape, dtype=np.float32)

  # Center for each pixel neighborhood indexes
  # not (0,0) since the image is padded (as for "- 1")
  for curr_row in range(1, image.shape[0] - 1):
    for curr_col in range(1, image.shape[1] - 1):
      neighborhood_total = 0
      row_index = 0
      col_index = 0

      # apply the kernel to each pixel in image neighborhood
      for i in range(curr_row - 1 , curr_row + 2):
        for j in range(curr_col - 1, curr_col + 2):
          neighborhood_total += image[i, j] * kernel[row_index, col_index]

          col_index += 1 # increment kernel column index

        row_index += 1 # increment kernel row index

        # reset colunm index for next row
        col_index = 0

      # Apply result of correlation operation to output image
      output_image[curr_row - 1, curr_col - 1] = neighborhood_total

      # Reset all kernel variables for next image pixel neighborhood
      neighborhood_total = 0
      row_index = 0
      col_index = 0
      
  return output_image

def compute_gradient_magnitude(image, kernel_row, kernel_col):
  # Apply Sobel kernel to the image results in the partial derivative in the x and y direction
  output_image_row = apply_kernel_to_image(image, kernel_row) 
  output_image_col = apply_kernel_to_image(image, kernel_col)

  # Square the partial derivatives pixel values
  output_image_row_sqr = np.square(output_image_row)
  output_image_col_sqr = np.square(output_image_col)

  # Add the squared partial derivatives together
  summed_pds_sqr = output_image_row_sqr + output_image_col_sqr

  # Take the square root of the sum of the squared partial derivatives
  output_mag = np.sqrt(summed_pds_sqr)

  return output_image_row, output_image_col, output_mag

def compute_gradient_direction(gradient_x, gradient_y):
  # Perform the calculation of the gradient direction
  gradient_direction = np.arctan2(gradient_y, gradient_x)

  # Convert gradient direction from radians to degrees
  gradient_direction_degrees = np.degrees(gradient_direction)

  # Display the resulting image
  fig, ax = plt.subplots(1, 1)
  plt.imshow(gradient_direction_degrees, cmap='hsv')
  plt.axis('off')

  # cv2.imwrite("output_images/base_setup/gradient_direction_degrees.png", gradient_direction_degrees)

  return gradient_direction

# @brief: Function to display the kernel correlation for a 
#     given 2D image. 
# @note: This implies that the kernel operation results in an image of the same size.
def display_kernel_correlation(original_image, grad_x, grad_y, grad_mag):
  # Create a figure to display a row of all 4 images
  fig, ax = plt.subplots(1, 4)
  ax[0].imshow(original_image, cmap='gray')
  ax[0].set_title("Original Image")
  ax[0].axis("off")

  ax[1].imshow(grad_x, cmap='gray')
  ax[1].set_title("Gradient X")
  ax[1].axis("off")

  ax[2].imshow(grad_y, cmap='gray')
  ax[2].set_title("Gradient Y")
  ax[2].axis("off")

  ax[3].imshow(grad_mag, cmap='gray')
  ax[3].set_title("Gradient Magnitude")
  ax[3].axis("off")

  # cv2.imwrite("output_images/base_setup/gradient_x.png", grad_x)
  # cv2.imwrite("output_images/base_setup/gradient_y.png", grad_y)
  # cv2.imwrite("output_images/base_setup/gradient_magnitude.png", grad_mag)


# The angles that the gradient direction are based on the cartesian coordinate system
#                     ^ 90 degrees
#                     |
# 180 degrees <-------|--------> 0 degrees 
#                     |
#                     v -90 degrees
def non_max_suppression(gradient_magnitude, gradient_direction):
  # Create a new image to store the result of the non-maximum suppression operation
  output_image = np.zeros(gradient_magnitude.shape, dtype=np.float32)

  # Create a padded image to make the implementation of the non-maximum suppression operation easier
  preproc_image = preprocess_image(gradient_magnitude)

  # Iterate through the image and apply the non-maximum suppression operation
  for curr_row in range(1, preproc_image.shape[0] - 1):
    for curr_col in range(1, preproc_image.shape[1] - 1):
      # Determine the direction of the gradient at the current pixel
      # Note offset if required since gradient direction is not zero padded with 1x1 pixel layer
      angle = gradient_direction[curr_row -1 , curr_col - 1]

      # If the angle is between 0 and 22.5 (directly to the top right of the current loop pixel) 
      if ( 22.5 <= angle < 67.5):
        grad_dir_pix = preproc_image[curr_row - 1, curr_col + 1]
        rev_grad_dir_pix = preproc_image[curr_row + 1, curr_col - 1]
      # If the angle is between 67.5 and 112.5 (directly above current loop pixel)
      elif ( 67.5 <= angle < 112.5):
        grad_dir_pix = preproc_image[curr_row - 1, curr_col]
        rev_grad_dir_pix = preproc_image[curr_row + 1, curr_col]
      elif ( 112.5 <= angle < 157.5):
        grad_dir_pix = preproc_image[curr_row - 1, curr_col - 1]
        rev_grad_dir_pix = preproc_image[curr_row + 1, curr_col + 1]
      # Check the bottom half of the cartesian coordinate system
      # for negative angles
      elif ( -22.5 <= angle < 22.5):
        grad_dir_pix = preproc_image[curr_row, curr_col + 1]
        rev_grad_dir_pix = preproc_image[curr_row, curr_col - 1]
      elif ( -67.5 <= angle < -22.5):
        grad_dir_pix = preproc_image[curr_row + 1, curr_col + 1]
        rev_grad_dir_pix = preproc_image[curr_row - 1, curr_col - 1]
      elif ( -112.5 <= angle < -67.5):
        grad_dir_pix = preproc_image[curr_row + 1, curr_col]
        rev_grad_dir_pix = preproc_image[curr_row - 1, curr_col]
      elif ( -157.5 <= angle < -112.5):
        grad_dir_pix = preproc_image[curr_row + 1, curr_col - 1]
        rev_grad_dir_pix = preproc_image[curr_row - 1, curr_col + 1]
      elif ( 157.5 <= angle < -157.5):
        grad_dir_pix = preproc_image[curr_row - 1, curr_col - 1]
        rev_grad_dir_pix = preproc_image[curr_row + 1, curr_col + 1]

      # Determine if the current pixel is a local maximum
      if (preproc_image[curr_row, curr_col] >= grad_dir_pix) and (preproc_image[curr_row, curr_col] >= rev_grad_dir_pix):
        output_image[curr_row - 1, curr_col - 1] = preproc_image[curr_row, curr_col]
      else:
        output_image[curr_row -1 , curr_col - 1] = 0

  return output_image


def dual_thresholding_by_hysteresis(gradient_mag_image, high_threshold, low_threshold):
  # Create binary edge image from gradient madnitude image where a pixel value of 
  # 255 indicates an edge 0 indicates non-edge pixel
  strong_edges_t1 = np.where(gradient_mag_image >= high_threshold, 255, 0)
  weak_edges_t2 = np.where((gradient_mag_image > low_threshold) & (gradient_mag_image < high_threshold), 255, 0)

  # Pad both images with a 1 pixel layer of zeros to simplify
  # loooking at the 8 neighboring pixels
  strong_edges_t1 = np.pad(strong_edges_t1, (1, 1), 'constant', constant_values=(0, 0))
  weak_edges_t2 = np.pad(weak_edges_t2, (1, 1), 'constant', constant_values=(0, 0))

  # Create output image
  output_image = np.zeros(gradient_mag_image.shape, dtype=np.uint8)

  i = 0
  while ( i < 25):
    # Iterate through the image and apply the dual thresholding by hysteresis operation
    for curr_row in range(1, weak_edges_t2.shape[0] - 1):
      for curr_col in range(1, weak_edges_t2.shape[1] - 1):

        # If the current pixel is a weak edge, check if it is connected to a strong edge
        if weak_edges_t2[curr_row, curr_col] == 255:
          # Check if any of the 8 neighboring pixels are strong edges
          if (strong_edges_t1[curr_row - 1, curr_col - 1] == 255 or
              strong_edges_t1[curr_row - 1, curr_col] == 255 or
              strong_edges_t1[curr_row - 1, curr_col + 1] == 255 or
              strong_edges_t1[curr_row, curr_col - 1] == 255 or
              strong_edges_t1[curr_row, curr_col + 1] == 255 or
              strong_edges_t1[curr_row + 1, curr_col - 1] == 255 or
              strong_edges_t1[curr_row + 1, curr_col] == 255 or
              strong_edges_t1[curr_row + 1, curr_col + 1] == 255):
            
            # If a weak edge is connected to a strong edge, set the pixel value to 255
            output_image[curr_row - 1, curr_col - 1] = 255
          else:
            output_image[curr_row - 1, curr_col - 1] = 0
        else:
          output_image[curr_row - 1, curr_col - 1] = strong_edges_t1[curr_row, curr_col]
    i += 1

  fig, ax = plt.subplots(1, 3)
  ax[0].imshow(strong_edges_t1, cmap='gray')
  ax[1].imshow(weak_edges_t2, cmap='gray')
  ax[2].imshow(output_image, cmap='gray')

  # cv2.imwrite("output_images/base_setup/strong_edge.png", strong_edges_t1)
  # cv2.imwrite("output_images/base_setup/weak_edge.png", weak_edges_t2)
  cv2.imwrite("output_images/fishingboat_thresholds/boat_H6_L0_03.png", output_image)



def main():
  input_image = cv2.imread('input_images/fishingboat.tif', 0)

  preproc_image = preprocess_image(input_image)

  # Apply noise reduction gaussian kernel
  gaussian_kernel = create_gaussian_kernel(3, 2)

  blurred_img = apply_kernel_to_image(preproc_image, gaussian_kernel)

  fix, ax = plt.subplots(1, 1)
  plt.imshow(blurred_img, cmap='gray')
  plt.axis('off')
  # cv2.imwrite("output_images/sig3/noise_reduced.png", blurred_img)

  # Create a Sobel kernel to detect the edges in the image
  soble_kernel_row = (1/8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  soble_kernel_col = (1/8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  # Apply the Sobel kernel to the image
  grad_row, grad_col, grad_mag = compute_gradient_magnitude(blurred_img, soble_kernel_row, soble_kernel_col)

  # Display the effects of the kernel correlation on the image
  display_kernel_correlation(blurred_img, grad_row, grad_col, grad_mag)

  # Using gradient direction in x and y axis
  grad_direction = compute_gradient_direction(grad_row, grad_col)

  # Reduce the number of edge pixels by checking if the pixel is a local maximum
  # via comparing the pixel value to its gradient direction neighbors and opposite direction
  suppressed_img = non_max_suppression(grad_mag, grad_direction)

  # Display the resulting image
  fig, ax = plt.subplots(1, 1)
  plt.imshow(suppressed_img, cmap='gray')
  plt.axis('off')
  # cv2.imwrite("output_images/base_setup/non_max_output.png", suppressed_img)

  # Perform dual thresholding by hysteresis
  dual_thresholding_by_hysteresis(suppressed_img, 6, .03)

  # Display the resulting images
  plt.show()
  # Wait for a user response
  cv2.waitKey(0)

  return 0

if __name__ == "__main__":
  main()

  # closing all open windows 
  cv2.destroyAllWindows() 

