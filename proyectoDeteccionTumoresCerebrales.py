import cv2
import numpy as np
import os
import sys

if __name__ == '__main__':

    # Lectura de la imagen
    img_path = r'C:\Users\caroj\Documents\Maestria_AI\ProcesamientoImagen\Proyecto\Y11.jpg'
    image = cv2.imread(img_path)
    print('width: {} pixels'.format(image.shape[1]))
    print('height: {} pixels'.format(image.shape[0]))
    print('channels: {}'.format(image.shape[2]))
    dim = (500, 590)
    image = cv2.resize(image, dim)
    image2 = image.copy()
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
    canny2 = auto_canny(closed)
    cv2.imshow("Image Canny", canny)
    cv2.waitKey(0)

    # CONTORNOS

    cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(canny2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Bode del Cerebro
    cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    cv2.imshow("Imagen CONTORNOS", image)
    cv2.waitKey(0)

    # ENCONTRAR EL MAYOR CONTORNO
    biggest_contour = max(cnts, key=cv2.contourArea) # Maximo Contorno del Tumor
    biggest_contour2 = max(cnts2, key=cv2.contourArea) # Maximo Contorno del Craneo

    bigContour = int(input('Seleccione 1) Contorno del Craneo 2) Contorno del Tumor_'))
    if bigContour == 1:
        contorno =  biggest_contour2
    elif bigContour == 2:
        contorno = biggest_contour
    # Encontar el centro del contorno
    M = cv2.moments(contorno)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.drawContours(image2, contorno, -1, (0, 255, 0), 2)
        cv2.circle(image2, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(image2, "Centro", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"x: {cx} y: {cy}")
    cv2.imshow("Image contorno mas grande", image2)
    cv2.waitKey(0)