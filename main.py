import cv2
import numpy as np



# Lectura de la imagen
img_path = r'C:\Users\caroj\Documents\Maestria_AI\ProcesamientoImagen\Proyecto\Y11.jpg'
image = cv2.imread(img_path)
print('width: {} pixels'.format(image.shape[1]))
print('height: {} pixels'.format(image.shape[0]))
print('channels: {}'.format(image.shape[2]))
dim=(500,590)
image=cv2.resize(image, dim)
cv2.imshow("Image BRAIN", image)
cv2.waitKey(0)
gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0.7)

(T, thresh) = cv2.threshold(gray_Image, 155, 255, cv2.THRESH_BINARY )


(T, thresh2) = cv2.threshold(gray_Image, 155, 255, cv2.THRESH_BINARY_INV )
cv2.imshow("Image Gray1", thresh)
cv2.imshow("Image Gray2", thresh2)


# MORPHOLOGICAL OPERATIONS para revisar los parametros.......
#
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# The morphological operators used are Erosion and Dilation..

closed = cv2.erode(closed, None, iterations = 1)
closed = cv2.dilate(closed, None, iterations = 1)

cv2.imshow("Image close", closed)
cv2.waitKey(0)


# CANNY

#edged = cv2.Canny(closed,155,255)
#canny = cv2.autocanny(thresh)

def auto_canny(image, sigma=0.033):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged



canny = auto_canny(closed)
cv2.imshow("Image Canny", canny)
cv2.waitKey(0)

# CONTORNOS

(cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
cv2.imshow("Image CONTORNOS", image)
cv2.waitKey(0)