# Author David Perez
# Date: 1/26/2023
# Image Manipulation Assignment 0
# Description: Basic image maniuplation techniques using numpy and opencv

import numpy as np
import cv2
import matplotlib.pyplot as plt

# @brief: Function to create a desired
#     kernel for image processing
def kernel_selector():

  print("")


# note: function expects a 2D image
#    with pixel values in the range [0, 255]
#    and a 3x3 kernel
def preprocess_image(image):
  # Surround the outside of the image with 1 pixel layer of zeros
  image_padded = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))

  return image_padded

# For each pixel neighborhood (3x3 set of pixels), multiply the pixel
# values by the kernel values. This is equivalent
# to applying a correlation operation between the image and the kernel.
def apply_kernel_to_image(image, kernel):
  # Create a new image to store the result of the correlation operation
  # This image will have the same size as the original image
  output_image = np.zeros(image.shape, dtype=np.uint32)

  preproc_image = preprocess_image(image)

  # Center for each pixel neighborhood indexes
  # not (0,0) since the image is padded (as for "- 1")
  for curr_row in range(1, preproc_image.shape[0] - 1):
    for curr_col in range(1, preproc_image.shape[1] - 1):
      neighborhood_total = 0
      row_index = 0
      col_index = 0

      # apply the kernel to each pixel in image neighborhood
      for i in range(curr_row - 1 , curr_row + 2):
        for j in range(curr_col - 1, curr_col + 2):
          neighborhood_total += preproc_image[i, j] * kernel[row_index, col_index]

          col_index += 1 # increment kernel column index

        row_index += 1 # increment kernel row index

        # reset colunm index for next row
        col_index = 0

      # Apply result of correlation operation to output image
      output_image[curr_row - 1, curr_col - 1] = abs(int(neighborhood_total))

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

  # remove decimal points and cast to 8 bit integer
  output_mag = output_mag.astype(np.uint8)

  return output_image_row, output_image_col, output_mag

def compute_gradient_direction(gradient_x, gradient_y):
  # Perform the calculation of the gradient direction
  gradient_direction = np.arctan2(gradient_y, gradient_x)

  # # Convert gradient direction to the range [0, 2*pi)
  # gradient_direction[gradient_direction < 0] += 2 * np.pi

  # # Normalize gradient direction to the range [0, 1] for colormap
  # gradient_direction_normalized = gradient_direction / (2 * np.pi)


  # # Normalize the gradient direction values to the range [0, 1]
  # gradient_direction_normalized = (gradient_direction + np.pi) / (2 * np.pi)

  # Normalize the gradient direction values to the range [0, 1]
  gradient_direction_normalized = gradient_direction * ( 180 + np.pi) % 180

  # # Convert the normalized gradient direction to RGB
  # gradient_direction_color = plt.get_cmap('hsv')(gradient_direction_normalized)

  # # Convert HSV to RGB
  # gradient_direction_color_rgb = plt.get_cmap('hsv')(gradient_direction_color)

  # Display the resulting image
  fig, ax = plt.subplots(1, 1)
  plt.imshow(gradient_direction_normalized)
  plt.axis('off')
  plt.show()

  print("gradient_direction= ", gradient_direction)

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

  # Create figure
  # plt.show()

# Create a main function
def main():

  coin_img = cv2.imread('input_images/cameraman.tif', 0)

  soble_kernel_row = (1/8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  soble_kernel_col = (1/8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  # soble_kernel_row = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  # soble_kernel_col = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  grad_row, grad_col, grad_mag = compute_gradient_magnitude(coin_img, soble_kernel_row, soble_kernel_col)
  
  # Display the effects of the kernel correlation on the image
  display_kernel_correlation(coin_img, grad_row, grad_col, grad_mag)

  grad_direction = compute_gradient_direction(grad_row, grad_col)

  # cv2.imshow('Gradient Direction', grad_direction)
  # cv2.show()

  # Wait for a user response
  cv2.waitKey(0)

  return 0

if __name__ == "__main__":
  main()

  # closing all open windows 
  cv2.destroyAllWindows() 

