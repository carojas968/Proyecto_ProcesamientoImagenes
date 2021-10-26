import cv2
import numpy as np
import os
import sys

if __name__ == '__main__':

    # Lectura de la imagen
    img_path = '/Users/christianrafaelmoraparga/PycharmProjects/pythonProject/ProyectoFinal/archive/yes/Y7.jpg'
    image = cv2.imread(img_path)
    print('width: {} pixels'.format(image.shape[1]))
    print('height: {} pixels'.format(image.shape[0]))
    print('channels: {}'.format(image.shape[2]))
    dim = (500, 590)
    image = cv2.resize(image, dim)
    cv2.imshow("Imagen original", image)
    cv2.waitKey(0)
    img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0.7)

    # 155 parece el valor justo para el humbral
    T, thresh = cv2.threshold(img_gris, 155, 255, cv2.THRESH_BINARY)
    T, thresh2 = cv2.threshold(img_gris, 155, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Imagen Gris 1", thresh)
    cv2.imshow("Imagen Gris 2", thresh2)

    # Operaciones Morfológicas para revisar los parametros.......

    # se crea kernel ELLIPSE de 10x5
    dim_kernel = (10, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dim_kernel)
    # Se hace la operación morfológica de hit and miss con el kernel

    ## HACER lógica se si según la figura hallada en thresh, usar MORPH_CLOSE o MORPH_HITMISS, lo mismo tresh y tresh2:

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    hitmiss = cv2.morphologyEx(thresh, cv2.MORPH_HITMISS, kernel)

    # Se realizan las operaciones morfológicas de Erosion y Dilation

    closed = cv2.dilate(cv2.erode(closed, None, iterations=1), None, iterations=1)
    hitmiss = cv2.dilate(cv2.erode(hitmiss, None, iterations=1), None, iterations=1)

    cv2.imshow("Imagen MORPH_CLOSED", closed)
    cv2.imshow("Imagen MORPH_HITMISS", hitmiss)
    cv2.waitKey(0)

    # CANNY
    # edged = cv2.Canny(closed,155,255)
    # canny = cv2.autocanny(thresh)

    def auto_canny(image, s=0.033):
        # calcular la mediana de las intensidades de píxeles de un solo canal
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        # aplicar la detección automática de bordes Canny utilizando la mediana calculada
        inferior = int(max(0, (1.0 - s) * v))
        superior = int(min(255, (1.0 + s) * v))
        edged = cv2.Canny(image, inferior, superior)
        # devolver la imagen con bordes
        return edged

    canny = auto_canny(hitmiss)
    cv2.imshow("Image Canny", canny)
    cv2.waitKey(0)

    # CONTORNOS

    cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    cv2.imshow("Imagen CONTORNOS", image)
    cv2.waitKey(0)