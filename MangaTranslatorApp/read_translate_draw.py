import cv2
import easyocr
from googletrans import Translator
import os

def process_text(image_path, boxes, target_language="tr", output_name=None):
    reader = easyocr.Reader(['en'])
    translator = Translator()
    
    image = cv2.imread(image_path)

    for box in boxes['boxes']:
        x1, y1, x2, y2 = map(int, box)

        height, width = image.shape[:2]
        if not (0 <= x1 < width and 0 <= x2 <= width and 0 <= y1 < height and 0 <= y2 <= height):
            print(f"Geçersiz koordinatlar: {x1, y1, x2, y2}")
            continue

        cropped_image = image[y1:y2, x1:x2]
        result = reader.readtext(cropped_image, detail=1)

        if result:
            original_text = " ".join([res[1] for res in result])
            translated_text = translator.translate(original_text, dest=target_language).text

            def wrap_text(text, max_width, font, font_scale):
                words = text.split()
                lines = []
                current_line = words[0]
                for word in words[1:]:
                    if cv2.getTextSize(current_line + ' ' + word, font, font_scale, 1)[0][0] < max_width:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line)
                        current_line = word
                lines.append(current_line)
                return lines

            font_scale = 0.7
            wrapped_text = wrap_text(translated_text, x2 - x1, cv2.FONT_HERSHEY_SIMPLEX, font_scale)
            line_height = min((y2 - y1) // len(wrapped_text), int(30 * font_scale)) if len(wrapped_text) > 0 else int(30 * font_scale)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
            for i, line in enumerate(wrapped_text):
                y_position = y1 + i * line_height 
                if y_position + line_height < y2:
                    cv2.putText(image, line, (x1, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    output_name = output_name or image_path.replace(".jpg", "_processed.jpg")
    processed_image_path = os.path.join(os.path.dirname(image_path), output_name)

    cv2.imwrite(processed_image_path, image)
    return processed_image_path
