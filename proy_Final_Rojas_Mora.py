import cv2
import numpy as np
import os
import sys


class Brain_class_MRI:
    def __init__(self, image):
        # recibe la imagen y se re dimensiona a 500x590
        dim = (500, 590)
        self.image = cv2.resize(image, dim)
        self.image2, self.image3 = self.image.copy(), self.image.copy()

        self.contour_tum = self.contour_cran = True

    def operaciones_morf(self):
        # Se transforma a grises la imagen para poderla binarizar
        img_gris = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY, 0.7)

        # Se tomaran parámetros particulares separados para reconocer al tumor y al cráneo

        # Binarización de la imagen en grises, se eligen humbrales tanto para reconocer particularmente al cráneo
        # (bordes externos) como para reconocer al tumor.

        T, thresh_craneo = cv2.threshold(img_gris, 100, 255, cv2.THRESH_BINARY)  # Tresh -> [100,250]
        T, thresh_tumor = cv2.threshold(img_gris, 140, 255, cv2.THRESH_BINARY)  # Tresh -> [140,250]

        # se crea kernel en forma de ELLIPSE de 50x40 y 10x5
        # Se hace la operación morfológica de CLOSING para el cráneo y de HIT&MISS para el tumor.

        dim_kernel_cran = (50, 40)
        kernel_cran = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dim_kernel_cran)
        close = cv2.morphologyEx(thresh_craneo, cv2.MORPH_CLOSE, kernel_cran)
        # se dilata para poder cerrar los posibles huecos sobre el contorno
        close = cv2.dilate(close, np.ones((10, 10), np.uint8))

        # se genera un recuadro de ceros sobre la imagen de closed para evitar que excedan los bordes del cráneo, y poder
        # cerrar el borde hallado en la misma imagen.
        closed = cv2.copyMakeBorder(close, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

        dim_kernel_tum = (10, 5)
        kernel_tum = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dim_kernel_tum)
        hitmiss = cv2.morphologyEx(thresh_tumor, cv2.MORPH_HITMISS, kernel_tum)
        hitmiss = cv2.morphologyEx(hitmiss, cv2.MORPH_CLOSE, np.ones((20, 5), np.uint8))
        hitmiss = cv2.dilate(hitmiss, np.ones((5, 5), np.uint8), iterations=2)

        return img_gris, closed, hitmiss

    def contornos(self, closed, hitmiss):
        # Se utiliza el algoritmo de Canny para hallar los bordes del cráneo como del tumor
        def auto_canny(in_image, s=0.033):
            # calcular la mediana de las intensidades de píxeles de un solo canal
            v = np.median(in_image)
            # aplicar la detección automática de bordes Canny utilizando la mediana calculada
            inferior = int(max(0, (1.0 - s) * v))
            superior = int(min(255, (1.0 + s) * v))
            img_bordes = cv2.Canny(in_image, inferior, superior)
            # devolver la imagen con bordes
            return img_bordes

        canny_tum, canny_cran = auto_canny(hitmiss), auto_canny(closed)
        #cv2.imshow("Image Canny tumor", canny_tum)
        #cv2.imshow("Image Canny craneo", canny_cran); cv2.waitKey(0)

        # CONTORNOS de los bordes de Canny para dibujarlos encima de la imagen

        cnts_tum, _ = cv2.findContours(canny_tum, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts_cran, _ = cv2.findContours(canny_cran, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Borde del Cerebro

        # ENCONTRAR EL MAYOR CONTORNO, tanto del tumor como del cráneo
        # En caso no se halle ningún contorno con Canny
        try:
            biggest_contour_cran = max(cnts_cran, key=cv2.contourArea)  # Máximo Contorno del Cráneo
            cv2.drawContours(self.image2, biggest_contour_cran, -1, (0, 255, 0), 2)
            cv2.drawContours(self.image3, biggest_contour_cran, -1, (0, 255, 0), 2)
            #cv2.imshow("CONTORNOS cráneo", self.image2)
        except ValueError as ve:
            biggest_contour_cran = np.array([])#np.zeros((height, width, 1), np.uint8)
            self.contour_cran = False
        try:
            biggest_contour_tum = max(cnts_tum, key=cv2.contourArea)  # Máximo Contorno del Tumor
            cv2.drawContours(self.image, biggest_contour_tum, -1, (0, 0, 255), 2)
            #cv2.imshow("CONTORNOS tumor", self.image); cv2.waitKey(0)
        except ValueError as ve:
            biggest_contour_tum = np.array([])#np.zeros((height, width, 1), np.uint8)
            self.contour_tum = False

        return biggest_contour_tum, biggest_contour_cran

    def posiciones_detect(self, cont1, cont2):
        # Se hallan los centroides (sus cordenadas) tanto el tumor como del cráneo usando los momentos

        def cent_contor(in_img, contorno, color,txt):
            # Encontar el centro del contorno
            M = cv2.moments(contorno)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(in_img, contorno, -1, color, 2)
                cv2.circle(in_img, (cx, cy), 2, color, -1)
                cv2.putText(in_img, "Centro " + txt, (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return cx, cy

        if self.contour_tum:
            #dibujarlo en la imagen
            cent_contor(self.image2, cont1, (0, 0, 255), 'tumor')
        if self.contour_cran:
            cent_contor(self.image2, cont2, (0, 255, 0), 'craneo')
        # print('(x,y) Centro Tumor:', cent_contor(self.image2, cont1, (0, 0, 255) 'tumor'))
        # print('(x,y) Centro Cráneo:', cent_contor(self.image2, cont2, (0, 0, 255) 'craneo'))

        # --- Encontrar la ubicación del lóbulo del tumor
        # - 1 paso ubicar el Centroide del Cráneo
        M_1 = cv2.moments(cont1)
        if M_1['m00'] != 0.0: cx_1 = int(M_1['m10'] / M_1['m00'])

        # - 2 paso ubicar el Centroide sobre el eje horizontal del tumor y cráneo
        M_0 = cv2.moments(cont2)
        if M_0['m00'] != 0.0: cx_0 = int(M_0['m10'] / M_0['m00'])

        # - 3 paso,  Regla de decisión si el centroide del tumor es menor al centroide del cráneo en el eje x
        # se espera que el centro del tumor sea equivalente y muy próximo al centro de la imagen
        Lobulo = 'Ninguno'
        if (self.contour_tum and self.contour_cran):
            Lobulo = 'Izquierdo' if (cx_1 < cx_0) else 'Derecho'

        return Lobulo

    def areas_afectacion(self, cont1, cont2, lobulo):
        # Se calculan las áreas dentro de los contornos hallados del tumor y cráneo
        if self.contour_tum:
            areaTumor = cv2.contourArea(cont1)
            # Obtener los limites del recuadro  contenedor del objeto
            x, y, w, h = cv2.boundingRect(cont1)
        else:
            areaTumor = 0.0
            x = w = int((self.image3.shape[0])/2)
            y = h = int((self.image3.shape[1])/2)

        areaCraneo = cv2.contourArea(cont2) if self.contour_cran else 0.0

        if (self.contour_tum and self.contour_cran):
            A_Afect = round((areaTumor / areaCraneo) * 100, 2)

            # Dibujar el rectangulo alrededor del objeto
            cv2.rectangle(self.image2, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(self.image2, f"El area afectada es {A_Afect} %", (x - 20, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        else:
            A_Afect = 0.0

        #print("áreas(tumor, craneo, tumor/cráneo))", (areaCerebro, areaTumor, A_Afect))

        # Impresión en la imagen3 el tipo de Lobulo:
        cv2.putText(self.image2, "Lobulo " + lobulo, (x - 20, y - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        # Mostrar Resultado
        #cv2.imshow("Imagen Resultado", self.image2); cv2.waitKey(0)

        data = [areaTumor, areaCraneo, A_Afect], lobulo
        return data

    #Sección de prueba
    def deteccion_tumor(self, umbral):

        img_gris, closed, hitmiss = self.operaciones_morf()
        contour_tum, contour_cran = self.contornos(closed, hitmiss)
        lobulo = self.posiciones_detect(contour_tum, contour_cran)
        dato, lobulo = self.areas_afectacion(contour_tum, contour_cran, lobulo)

        print("áreas[tumor, craneo, tumor/cráneo]", dato)

        if dato[2] >= umbral:
            tumor = True
            print('CON TUMOR')
            #cv2.imshow("Imagen Resultado", self.image2); cv2.waitKey(0)

        else:
            tumor = False
            print('SIN TUMOR')
            height, width = self.image3.shape[:2]

            cv2.putText(self.image3, "SIN TUMOR", (int(width/2), int(height/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow("Imagen Resultado", self.image3); cv2.waitKey(0)


if __name__ == '__main__':
    ''' Las entradas de código por sys son en orden:
        [1]: path del folder del set de entrenamiento, primero hacer primero yes o no, y anotar alguno de los 
        umbral_desicion, para que luego al volver a hacer el código haga el promedio con el otro set.
        [2]: path del folder del set de prueba, puede ser cualquiera de los dos
        [3]: path de una imagen MRI individual para probar el sistema
    '''
    # Lectura de del directorio de imágenes

    #dir_images = '/Users/christianrafaelmoraparga/PycharmProjects/pythonProject/ProyectoFinal/archive/brain_tumor_dataset/yes/'
    #dir_images = '/Users/christianrafaelmoraparga/PycharmProjects/pythonProject/ProyectoFinal/archive/yes_train/'
    dir_images = sys.argv[1]
    imagenes = os.listdir(dir_images)
    print(imagenes)
    mat_datos, lobulos = [], []
    pathresult_yes = 'Resultados_yes_train' #'Resultados_no_train'
    
    #Se crea un directorio para las imágenes resultantes del TRAINING
    try:
        os.mkdir(pathresult_yes)
    except FileExistsError:
        print('Ya existe el directorio '+pathresult_yes)

    # Se itera sobre el set de imágenes de TRAINING elegido
    for image in imagenes:
        img = cv2.imread(dir_images+image)
        inst = Brain_class_MRI(img)
        img_gris, closed, hitmiss = inst.operaciones_morf()
        contour_tum, contour_cran = inst.contornos(closed, hitmiss)
        lobulo = inst.posiciones_detect(contour_tum, contour_cran)
        dato, lobulo = inst.areas_afectacion(contour_tum, contour_cran, lobulo)

        mat_datos.append(dato)
        lobulos.append([lobulo, image])
        #cv2.imshow("Imagen Resultado "+image, inst.image2); cv2.waitKey(0)

        cv2.imwrite(os.path.join(pathresult_yes, "Imagen Resultado " + image), inst.image2)

    # PRINT de las áreas calculadas y de la posición del tumor
    # [area tumor, area craneo, area relativa tumor a cerebro, posición tumor]
    #print(dato)

    mat_datos = np.round(np.matrix(mat_datos), decimals=1)
    areas_relativas = mat_datos[:, 2]

    umbral_desicion = np.mean(areas_relativas)

    umbral_desicion_primero = 7.429411764705883
    #umbral_desicion_no = 1.1538461538461537
    umbral_desicion = 0.5*(umbral_desicion_primero + umbral_desicion)

    print('Umbral de desición:', umbral_desicion) #; print(mat_datos); print(lobulos); print(umbral_desicion)

    #dir_test = '/Users/christianrafaelmoraparga/PycharmProjects/pythonProject/ProyectoFinal/archive/yes_test/'
    dir_test = sys.argv[2]
    imagenes = os.listdir(dir_test) #; print(imagenes)
    pathresult_test = 'Resultados_yes_test'

    # Se crea un directorio para las imágenes resultantes del TEST
    try:
        os.mkdir(pathresult_test)
    except FileExistsError:
        print('Ya existe el directorio ' + pathresult_test)

    # Se itera sobre el set de imágenes de TEST elegido
    for image in imagenes:
        img = cv2.imread(dir_test + image)
        print(dir_test + image)
        inst_test = Brain_class_MRI(img)
        img_gris, closed, hitmiss = inst_test.operaciones_morf()
        inst_test.deteccion_tumor(umbral_desicion)

        cv2.imwrite(os.path.join(pathresult_test, "Imagen Resultado " + image),
                    cv2.hconcat([inst_test.image2, inst_test.image3]))

    # =============# =============# =============# =============# =============# =============
    # ============= Hacer pruebas individuales: # =============# =============# =============
    # =============# =============# =============# =============# =============# =============
    prueba_individual = input('¿Desea hacer una prueba individual?, "1" si, "0" no: ')
    if prueba_individual:
        # se ingresa ruta de imagen para probar el sistema:

        #img_prueba = '/Users/christianrafaelmoraparga/PycharmProjects/pythonProject/ProyectoFinal/archive/brain_tumor_dataset/yes/Y67.jpg'
        img_prueba = sys.argv[3]
        image = cv2.imread(img_prueba)
        inst2 = Brain_class_MRI(image)
        inst2.deteccion_tumor(umbral_desicion)