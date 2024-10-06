import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar um filtro Gaussiano para remover ruídos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar detecção de bordas (Canny)
    edged = cv2.Canny(blurred, 50, 150)

    cv2.imshow("edged", edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image, gray, edged


def find_license_plate_contour(edged_image):
    # Encontrar contornos na imagem
    contours, _ = cv2.findContours(edged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos para encontrar a possível placa (geralmente um retângulo)
    license_plate_contour = []
    for contour in contours:
        # Aproximação de contorno em forma de polígono
        approx = cv2.approxPolyDP(contour, 10, True)

        # Se o polígono tiver 4 lados, pode ser a placa
        if len(approx) == 4:
            license_plate_contour.append(approx)

    return license_plate_contour


def extract_license_plate(image, contour):
    if contour is None:
        return None

    # Criar uma máscara da área da placa
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # Extrair apenas a região da placa da imagem original
    plate_region = cv2.bitwise_and(image, mask)

    # Definir o bounding box (retângulo que envolve a placa)
    x, y, w, h = cv2.boundingRect(contour)

    # Cortar a região da placa
    plate_image = plate_region[y : y + h, x : x + w]

    return plate_image


def recognize_characters(plate_image):
    # Converter para escala de cinza e binarizar a imagem
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)

    # Utilizar o pytesseract para reconhecer os caracteres da placa
    custom_config = r"--oem 3 --psm 7"  # Configurações para aumentar a precisão do OCR
    text = pytesseract.image_to_string(binary_plate, config=custom_config)

    return text.strip()


# Função principal para detectar e reconhecer a placa em uma imagem
def detect_license_plate(image_path):
    # Pré-processamento da imagem
    image, gray, edged = preprocess_image(image_path)

    # Encontrar o contorno da placa
    plate_contours = find_license_plate_contour(edged)

    possible_answers = []
    if plate_contours is not None:
        for plate_contour in plate_contours:
            # Extrair a placa da imagem
            plate_image = extract_license_plate(image, plate_contour)

            # Mostrar a imagem da placa extraída (opcional)
            # cv2.imshow("Placa", plate_image)
            # cv2.waitKey(0)

            # Reconhecer os caracteres da placa
            plate_text = recognize_characters(plate_image)
            if len(plate_text) > 5:
                possible_answers.append(plate_text)
            # return plate_text
    else:
        return "Placa não encontrada"

    return possible_answers


if __name__ == "__main__":
    for i in range(1, 6):
        image_path = f"imgs/placa{i}.png"
        plate_text = detect_license_plate(image_path)
        print(f"Placa detectada: {plate_text}")
