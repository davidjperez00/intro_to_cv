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

# For each pixel neighborhood (3x3 set of pixels), multiply the pixel
# values by the kernel values. This is equivalent
# to applying a correlation operation between the image and the kernel.
def apply_kenerl_to_image(image, kernel):
  # Create a new image to store the result of the correlation operation
  # This image will have the same size as the original image
  output_image = np.zeros(image.shape, dtype=np.uint8)
  print(output_image.shape)

  # Surround the outside of the image with 1 pixel layer of zeros
  image_padded = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))

  # Center for each pixel neighborhood indexes
  # not (0,0) since the image is padded (as for "- 1")
  for curr_row in range(1, image_padded.shape[0] - 1):
    for curr_col in range(1, image_padded.shape[1] - 1):
      neighborhood_total = 0
      row_index = 0
      col_index = 0

      # apply the kernel to each pixel in image neighborhood
      for i in range(curr_row - 1 , curr_row + 2):
        for j in range(curr_col - 1, curr_col + 2):
          neighborhood_total += image_padded[i, j] * kernel[row_index, col_index]

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
  # # Save the image as squares.png
  # cv2.imwrite("squares.png", img)

  '''
    Reading and writing images using OpenCV
  '''

  coin_img = cv2.imread('input_images/cameraman.tif', 0)

  # Surround the outside of the image with 1 pixel layer of zeros
  coin_img_padded = np.pad(coin_img, (1, 1), 'constant', constant_values=(0, 0))

  # Display padded image
  cv2.imshow('Padded Image', coin_img_padded)

  soble_kernel_row = (1/8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  soble_kernel_col = (1/8) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  print("soble_kernel_row: \r\n", soble_kernel_row)
  print("soble_kernel_col: \r\n", soble_kernel_col)


  # For each pixel neighborhood (3x3 set of pixels), multiply the pixel
  # values by the kernel values. This is equivalent
  # to applying a correlation operation between the image and the kernel.

  # Create a new image to store the result of the correlation operation
  # This image will have the same size as the original image
  output_image = np.zeros(coin_img.shape, dtype=np.uint8)
  print(output_image.shape)

  # Center for each pixel neighborhood indexes
  # not (0,0) since the image is padded (as for "- 1")
  for curr_row in range(1, coin_img_padded.shape[0] - 1):
    for curr_col in range(1, coin_img_padded.shape[1] - 1):
      
      neighborhood_total = 0
      row_index = 0
      col_index = 0
      f = 0
      # apply the kernel to each pixel in image neighborhood
      for i in range(curr_row - 1 , curr_row + 2):
        for j in range(curr_col - 1, curr_col + 2):

          # print("i: " + str(i) + " j: " + str(j) + "curr_col = " +str(curr_col) + " curr_row = " + str(curr_row) + " row_index = " + str(row_index) + " col_index = " + str(col_index))
          # print("i = " +str(i) + " j = " + str(j))
          # print("coin_img_padded[i, j] = ", coin_img_padded[i, j])
          # print("soble_kernel_row[row_index, col_index] = ", soble_kernel_row[row_index, col_index])
          neighborhood_total += coin_img_padded[i, j] * soble_kernel_row[row_index, col_index]


          col_index += 1

        row_index += 1
        # reset colunm index for next row
        col_index = 0

      # print("neighborhood_total", neighborhood_total)


      # Apply result of correlation operation to output image
      output_image[curr_row - 1, curr_col - 1] = abs(int(neighborhood_total))
    

      # Reset all kernel variables for next pixel neighborhood
      neighborhood_total = 0
      row_index = 0
      col_index = 0
      




      # neighborhood_total += coin_img_padded[curr_row + i, curr_col + j] * soble_kernel_col[i + 1, j + 1]

      # append applied kernel pixel values to output_image


  # print("shape  ")
  # print(coin_img_padded.shape[0])



      
  print("coin_img_padded = ", coin_img_padded)
  cv2.imwrite("coin_img_padded.png", coin_img_padded)
  # cv2.imwrite("coins5.png", coin_img_std5)

  cv2.waitKey(0)

  print("output_image = ", output_image)
  print("output_image.shape = ", output_image.shape)
  print("original image shape = ", coin_img.shape)
  cv2.imshow('output Image', output_image)

  cv2.waitKey(0)

  return 0

if __name__ == "__main__":
  main()

  # closing all open windows 
  cv2.destroyAllWindows() 

