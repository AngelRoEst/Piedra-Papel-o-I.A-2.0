import cv2
import numpy as np
import random
import os
import time


# Accedemos a la carpeta
path = "D:/Image Processing/Piedra Papel o IA/Imagenes PPT"
images = []
clases = []
lista = os.listdir(path)

# Leemos los rostros del DB
for lis in lista:
    # Leemos las imagenes de los rostros
    imgdb = cv2.imread(f'{path}/{lis}')
    # Almacenamos imagen
    images.append(imgdb)
    # Almacenamos nombre
    clases.append(os.path.splitext(lis)[0])

print(clases)

jugadaPC =  random.randint(0,2)


def capturar_foto():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return None

    # Dar un tiempo para que la cámara ajuste la exposición
    ret, frame = cap.read()
    if ret:
        nombre_archivo = "foto_capturada.png"
        cv2.imwrite(nombre_archivo, frame)
        print(f"Foto guardada como {nombre_archivo}")
        cap.release()
        return frame
    else:
        print("Error al capturar la foto")
        cap.release()
        return None

def procesar_imagen(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Hola",thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "No se detecta mano"

    # Suponemos que el contorno más grande es la mano
    max_contour = max(contours, key=cv2.contourArea)

    # Encontrar el convex hull
    hull = cv2.convexHull(max_contour)

    # Encontrar los defectos del hull
    hull_indices = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull_indices)

    if defects is None:
        return "Piedra"

    count_defects = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = max_contour[s][0]
        end = max_contour[e][0]
        far = max_contour[f][0]

        # Convierte las tuplas a arrays de NumPy
        start = np.array(start)
        end = np.array(end)
        far = np.array(far)

        # Encontrar las distancias entre los puntos del defecto
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(far - end)

        # Calcular el ángulo usando la ley de cosenos
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

        # Si el ángulo es menor de 90 grados, lo contamos como un defecto
        if angle <= np.pi / 2:
            count_defects += 1

    if count_defects == 0:
        return "Piedra"
    elif count_defects == 1:
        return "Tijera"
    else:
        return "Papel"

def main():
    frame = capturar_foto()
    if frame is not None:
        resultado =  procesar_imagen(frame)
        print(f"El resultado es: {resultado}")
        cv2.imshow("Foto Capturada", frame)
        cv2.imshow("Jugada de la PC", images[jugadaPC])
        time.sleep(2)
        if (resultado == "Piedra" and jugadaPC == 1) or (resultado == "Papel" and jugadaPC == 0) or (resultado == "Tijera" and jugadaPC == 2):
            cv2.imshow("Empate",images[5])
        elif (resultado == "Piedra" and jugadaPC == 0) or (resultado == "Papel" and jugadaPC == 2) or (resultado == "Tijera" and jugadaPC == 1):
            cv2.imshow("Gana la IA", images[3])
        elif(resultado == "Piedra" and jugadaPC == 2) or (resultado == "Papel" and jugadaPC == 1) or (resultado == "Tijera" and jugadaPC == 0):
            cv2.imshow("Gana el jugador", images[4])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
