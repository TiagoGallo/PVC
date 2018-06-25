import cv2
import face_recognition

def main():
    cam = cv2.VideoCapture(0)

    skipFrame = 3
    counter = 0
    while True:
        counter += 1
        _, img = cam.read()

        if img is None:
            break

        if counter == 1:
            face_locations = face_recognition.face_locations(img, model="hog")
            face_landmarks_list = face_recognition.face_landmarks(img, face_locations=face_locations)
            calc_relative_nose(face_locations, face_landmarks_list)
        
        if counter == skipFrame:
            counter = 0

        for top, right, bottom, left in face_locations:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255,0))

        for face_landmarks in face_landmarks_list:
            #Desenha os pontos do olho direito de azul
            for point in face_landmarks["right_eye"]:
                cv2.circle(img, point, 1, (255,0,0), -1)
            #Desenha os pontos da sobrancelha esquerda de vermelho
            for point in face_landmarks["left_eyebrow"]:
                cv2.circle(img, point, 1, (0,0,255), -1)
            #Desenha os pontos da ponta do nariz de verde
            for point in face_landmarks["nose_tip"]:
                cv2.circle(img, point, 1, (0,255,0), -1)
            #Desenha os pontos do corpo do nariz de ciano
            for point in face_landmarks["nose_bridge"]:
                cv2.circle(img, point, 1, (255,255,0), -1)
            #Desenha os pontos da sobrancelha direita de vermelho
            for point in face_landmarks["right_eyebrow"]:
                cv2.circle(img, point, 1, (0,0,255), -1)
            #Desenha os pontos do olho esquerdo de azul
            for point in face_landmarks["left_eye"]:
                cv2.circle(img, point, 1, (255,0,0), -1)
            #Desenha os pontos do labio de verde + vermelho
            for point in face_landmarks["bottom_lip"]:
                cv2.circle(img, point, 1, (0,255,255), -1)
            #Desenha os pontos do labio de verde + vermelho
            for point in face_landmarks["top_lip"]:
                cv2.circle(img, point, 1, (0,255,255), -1)
            #Desenha os pontos da bochecha de preto
            for point in face_landmarks["chin"]:
                cv2.circle(img, point, 1, (0,0,0), -1)
            

        cv2.imshow("Webcam", img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def calc_relative_nose(face_locations, face_landmarks_list):
    for face_location, landmark in zip(face_locations, face_landmarks_list):
        point_med_nose = [0,0]
        for i, point in enumerate(landmark["nose_tip"]):
            point_med_nose[0] = point_med_nose[0] + point[0]
            point_med_nose[1] = point_med_nose[1] + point[1]

        point_med_nose[0] = point_med_nose[0] // i
        point_med_nose[1] = point_med_nose[1] // i
        print("[DEBUG] Ponto medio do nariz = ", point_med_nose)

if __name__ == "__main__":
    main()