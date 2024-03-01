# Author David Perez
# Date: 1/26/2023
# Image Manipulation Assignment 0
# Description: Basic image maniuplation techniques using numpy and opencv

import numpy as np
import cv2

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
  print(output_image.shape)

  preproc_image = preprocess_image(image)

  # # Surround the outside of the image with 1 pixel layer of zeros
  # image_padded = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))

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

# Create a main function
def main():

  coin_img = cv2.imread('input_images/cameraman.tif', 0)

  soble_kernel_row = (1/8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  soble_kernel_col = (1/8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  # soble_kernel_row = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  # soble_kernel_col = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  # print("soble_kernel_row: \r\n", soble_kernel_row)
  # print("soble_kernel_col: \r\n", soble_kernel_col)


  # Apply Sobel kernel to the image results in the partial derivative in the x and y direction
  output_image_row = apply_kernel_to_image(coin_img, soble_kernel_row)
  output_image_col = apply_kernel_to_image(coin_img, soble_kernel_col)

  # cv2.imshow('output Image', output_image_row)
  # cv2.imshow("output_image_col", output_image_col)

  print("output_image_row[0] type = ", type(output_image_row[0]))
  print("output_image_row = ", output_image_row)
  print("output_image_col = ", output_image_col)

  # Square the partial derivatives and add them together
  output_image_row = np.square(output_image_row)
  output_image_col = np.square(output_image_col)

  print("output_image_sqrd = ", output_image_row)
  print("output_image_col_sqrd = ", output_image_col)

  # Add the squared partial derivatives together
  summed_pds = output_image_row + output_image_col
  
  # Take the square root of the sum of the squared partial derivatives
  summed_pds = np.sqrt(summed_pds)

  # remove decimal points and cast to 8 bit integer
  summed_pds = summed_pds.astype(np.uint8)

  print("summed_pds = ", summed_pds)
  
  # Print every row and column value
  # for row in summed_pds:
  #   for col in row:
  #     print(col)

  cv2.imshow("summed_pds", summed_pds)
      
  # print("coin_img_padded = ", coin_img_padded)
  # cv2.imwrite("coin_img_padded.png", coin_img_padded)
  # cv2.imwrite("coins5.png", coin_img_std5)
  cv2.imshow("Original Image", coin_img)

  # print("output_image = ", output_image)
  print("output_image.shape = ", output_image_row.shape)
  print("original image shape = ", coin_img.shape)


  cv2.waitKey(0)

  return 0

if __name__ == "__main__":
  main()

  # closing all open windows 
  cv2.destroyAllWindows() 

