import cv2
import torch
from mss import mss
import numpy as np
from ultralytics import YOLO


def main():
    # Cargar el modelo YOLOv5
    model = YOLO('yolov5s.pt')  # Cambia a 'best.pt' si tienes tu propio modelo

    # Configurar mss para capturar la pantalla
    sct = mss()
    monitor = sct.monitors[1]  # Captura el monitor principal

    while True:
        # Capturar la pantalla
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)  # Convertir la captura a un array numpy
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convertir BGRA a BGR

        # Realizar detección con YOLOv5
        results = model(frame)

        # Filtrar detecciones solo para humanoides (clase 0: "person")
        filtered_detections = []
        for detection in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            if int(class_id) == 0:  # Clase 0 corresponde a "person"
                filtered_detections.append(detection)

        # Dibujar solo las detecciones de personas
        for detection in filtered_detections:
            x1, y1, x2, y2, confidence, class_id = map(int, detection)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Mostrar la salida
        cv2.imshow("Captura de Pantalla - Solo Personas", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
