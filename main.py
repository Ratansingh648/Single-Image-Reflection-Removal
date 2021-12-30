import cv2
import numpy as np

from reflection_suppress_convex_optimization import reflectSuppress
from relative_layer_smoothness import reflection_removal, im2double

if __name__ == "__main__":

    h = 0.033
    e = 1e-10
    img = cv2.imread("test data\\sign_board.jpg")  # Read image here
    out = im2double(img.astype("float"))  # Convert to normalized floating poin

    out = reflectSuppress(out, h, e)

    L1, _ = reflection_removal(out, 1, np.zeros(out.shape), out, out)
    L2 = out - L1
    cv2.imshow("Original Image", img)

    # There could be slight decrease in brightness. Consider mulipying a suitable factor to L1 while display if needed
    # E.g. cv2.imshow("Image without Reflection", L1*1.2)
    cv2.imshow("Image without Reflection", L1)
    cv2.imshow("Reflection of Image", L2)

    # To hold the image windows closing when ran from command line / VSCode. Not necessarily needed when running from Spyder
    cv2.waitKey()
