import cv2
import numpy as np

# URL za video stream sa mobilnog uređaja ili IP kamere
URL = 'http://192.168.0.23:4747/video'

# Postavke za ArUco detekciju
# Koristi unapred definisan slovar (dictionary) DICT_5X5_250 za ArUco markere
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Parametri za detekciju markera (koriste se podrazumevane vrednosti)
PARAMS = cv2.aruco.DetectorParameters()

"""
Učitavanje kalibracionih podataka kamere iz fajla.
Fajl treba da sadrži `CAMERA_MATRIX` i `DIST_COEFFS` u NumPy formatu.
`np.save` da sačuvaš podatke i `np.load` da ih učitaš.
    np.save('camera_matrix.npy', CAMERA_MATRIX)#kalibracija
    np.save('dist_coeffs.npy', DIST_COEFFS)#file za prepoznavanje

try:
    CAMERA_MATRIX = np.load('camera_matrix.npy')
    DIST_COEFFS = np.load('dist_coeffs.npy')
    print("Kalibracioni podaci uspešno učitani.")
except FileNotFoundError:
    print("Greška: Kalibracioni podaci nisu pronađeni.")
    exit()
"""

# Kalibraciona matrica kamere (fokalne dužine i centar slike)
CAMERA_MATRIX = np.array([
    [800, 0, 320],  # Fokalna dužina x, nulta komponenta, koordinata centra x
    [0, 800, 240],  # Nulta komponenta, fokalna dužina y, koordinata centra y
    [0, 0, 1]       # Homogeni koeficijenti
], dtype=np.float32)

# Koeficijenti za ispravljanje izobličenja kamere (pretpostavljeno bez izobličenja)
DIST_COEFFS = np.zeros(5, dtype=np.float32)

# Dužina stranice ArUco markera u metrima
MARKER_LENGTH = 0.05  # 5 cm

# Otvaranje video streama sa zadatim URL-om
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    raise RuntimeError("Ne može se otvoriti kamera! Proverite URL i konekciju.")

print("Početak detekcije ArUco markera. Pritisnite 'q' za izlaz.")

while True:
    # Čitamo trenutni frame sa video streama
    ret, frame = cap.read()
    if not ret:
        print("Greška pri čitanju frejma sa kamere.")
        break

    # Konvertujemo frame u sivu sliku za bolju obradu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcija ArUco markera
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)

    # Proveravamo da li su markeri detektovani
    if ids is not None:
        # Crtamo detektovane markere na originalnom frejmu
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimacija pozicije svakog markera
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS
        )

        # Iteriramo kroz svaki detektovani marker
        for i, marker_id in enumerate(ids):
            # Crtamo ose koordinatnog sistema markera
            cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], MARKER_LENGTH * 0.5)

            # Ekstraktujemo translacione vektore (pozicija markera u prostoru)
            x, y, z = tvecs[i].flatten()

            # Tekstualna informacija o markeru (ID i pozicija)
            text = f"ID: {marker_id[0]} X: {x:.2f}m Y: {y:.2f}m Z: {z:.2f}m"

            # Pozicija teksta na slici (blizu gornje leve tačke markera)
            pos = (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10))

            # Crtamo tekst na slici
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Ispisujemo informacije u konzolu
            print(text)

    # Prikazujemo frejm sa detekcijom markera
    cv2.imshow('ArUco Marker Detection', frame)

    # Izlaz iz programa pritiskom na 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Zatvaranje programa.")
        break

# Oslobađanje resursa
cap.release()
cv2.destroyAllWindows()
