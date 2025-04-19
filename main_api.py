import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import re
import logging
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnam License Plate Recognition API")

# Lần 2

class PlateRecognizer:
    def __init__(self, model_path, ocr_languages=['vi', 'en']):
        logger.info("Đang khởi tạo PlateRecognizer...")
        self.model = YOLO(model_path)
        logger.info(f"Model YOLO đã tải từ: {model_path}")

        self.reader = easyocr.Reader(
            ocr_languages,
            gpu=False,
            model_storage_directory='./ocr_models', # Đảm bảo thư mục này tồn tại hoặc EasyOCR có quyền tạo
            download_enabled=True
        )
        logger.info(f"EasyOCR Reader đã khởi tạo với ngôn ngữ: {ocr_languages}, GPU: False")
        self.min_confidence = 0.2

        self.min_plate_width = 30
        self.min_plate_height = 10
        logger.info("Khởi tạo PlateRecognizer hoàn tất!")

    def preprocess_plate(self, roi):
        """
        Tiền xử lý ảnh biển số để tăng chất lượng OCR 
        """
        try:
            if roi is None or roi.shape[0] < 5 or roi.shape[1] < 5: # Kiểm tra roi hợp lệ
                logger.warning("ROI không hợp lệ hoặc quá nhỏ để tiền xử lý.")
                return None

            # Chuyển ảnh xám và cân bằng histogram
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Cân bằng histogram thích ứng tương phản giới hạn (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Làm sắc nét 
            blurred = cv2.GaussianBlur(enhanced, (0,0), 3)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


            # Phân ngưỡng nhị phân 
            binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 5) 


            logger.debug("Tiền xử lý ROI hoàn tất.")
            return binary
        except cv2.error as cv_err:
             logger.error(f"Lỗi OpenCV trong quá trình tiền xử lý: {cv_err}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Lỗi không xác định trong tiền xử lý ảnh: {e}", exc_info=True)
            return None

    def format_vietnam_plate(self, text):
        """
        Chuẩn hóa định dạng biển số Việt Nam theo các quy định mới và cũ.
        Xử lý input là chuỗi ký tự đã được OCR ghép lại.

        Args:
            text (str): Chuỗi ký tự gốc từ OCR.

        Returns:
            str: Chuỗi biển số đã định dạng hoặc chuỗi rỗng nếu không hợp lệ.
        """

        if not text:
            logger.debug("Input text is empty or None.")
            return ""

        cleaned_text = re.sub(r'[\s.-]', '', text).upper()
        logger.debug(f"Input: '{text}' -> Cleaned: '{cleaned_text}'")


        # Định dạng Ô tô mới
        car_new_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{5})$', cleaned_text)
        if car_new_match:
            g1, g2, g3 = car_new_match.groups()
            formatted_plate = f"{g1}{g2}-{g3[:3]}.{g3[3:]}"
            logger.info(f"Matched: Car New Pattern -> '{formatted_plate}'")
            return formatted_plate

        # Định dạng Xe máy mới
        motor_new_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{1})([0-9]{5})$', cleaned_text)
        if motor_new_match:
            g1, g2, g3, g4 = motor_new_match.groups()
            formatted_plate = f"{g1}-{g2}{g3}-{g4[:3]}.{g4[3:]}"
            logger.info(f"Matched: Motorbike New Pattern -> '{formatted_plate}'")
            return formatted_plate

        # Định dạng Xe máy cũ
        motor_old_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{1})([0-9]{4})$', cleaned_text)
        if motor_old_match:
            g1, g2, g3, g4 = motor_old_match.groups()
            formatted_plate = f"{g1}-{g2}{g3}-{g4}"
            logger.info(f"Matched: Motorbike Old Pattern -> '{formatted_plate}'")
            return formatted_plate

        # Định dạng Ô tô cũ
        car_old_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{4,5})$', cleaned_text)
        if car_old_match:
            g1, g2, g3 = car_old_match.groups()
            formatted_plate = f"{g1}{g2}-{g3}"
            logger.info(f"Matched: Car Old Pattern -> '{formatted_plate}'")
            return formatted_plate

        # Biển số vuông ô tô
        car_square_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{4})$', cleaned_text)
        if car_square_match:
             g1, g2, g3 = car_square_match.groups()
             formatted_plate = f"{g1}{g2}-{g3}"
             logger.info(f"Matched: Car Square Pattern (predicted) -> '{formatted_plate}'")
             return formatted_plate

        logger.warning(f"No valid VN plate pattern matched for cleaned text: '{cleaned_text}'")
        return ""

    def process_image(self, image_bytes):
        try:
            # Convert bytes to OpenCV image
            img_pil = Image.open(io.BytesIO(image_bytes))

            # Chuyển sang BGR là định dạng OpenCV thường dùng
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            logger.info(f"Đã chuyển đổi bytes thành ảnh OpenCV kích thước: {frame.shape}")

            # YOLO Detection
            results = self.model.predict(
                frame,
                imgsz=640, # Giữ nguyên hoặc thử tăng nếu ảnh mờ
                conf=self.min_confidence,
                verbose=False # Giữ False để tránh log thừa của YOLO
            )
            logger.info(f"YOLO phát hiện {len(results[0].boxes) if results else 0} đối tượng.")

            plates = []
            processed_count = 0
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy() # Lấy điểm conf

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    confidence = scores[i]

                    # Lọc theo kích thước 
                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    if plate_width < self.min_plate_width or plate_height < self.min_plate_height:
                        logger.debug(f"Bỏ qua ROI quá nhỏ: W={plate_width}, H={plate_height}")
                        continue

                    plate_roi = frame[y1:y2, x1:x2]
                    logger.debug(f"Đã cắt ROI tại [{y1}:{y2}, {x1}:{x2}], Conf: {confidence:.2f}")

                    # === THÊM BƯỚC TIỀN XỬ LÝ ===
                    processed_roi = self.preprocess_plate(plate_roi)
                    if processed_roi is None:
                        logger.warning(f"Tiền xử lý ROI thất bại, bỏ qua.")
                        continue

                    # === OCR Processing trên ảnh đã tiền xử lý ===
                    try:
                        
                        ocr_result = self.reader.readtext(
                            processed_roi, 
                            decoder='beamsearch', 
                            beamWidth=10,
                            batch_size=1, 
                            allowlist='0123456789ABCDEFGHKLMNPSTUVXYZ', # Chỉ nhận diện các ký tự này (bỏ dấu '-' vì đã loại bỏ)
                            detail=0,
                            paragraph=False 
                        )
                        combined = ''.join(ocr_result)
                        logger.info(f"OCR Result (raw): {combined} từ ROI {i}")

                        # === SỬ DỤNG HÀM FORMAT ===
                        formatted = self.format_vietnam_plate(combined)
                        if formatted:
                            logger.info(f"Formatted Plate: {formatted}")
                            plates.append(formatted)
                        else:
                             logger.warning(f"Raw text '{combined}' không định dạng được thành biển số.")

                        processed_count += 1

                    except Exception as ocr_err:
                         logger.error(f"Lỗi trong quá trình OCR cho ROI {i}: {ocr_err}", exc_info=True)

            # Loại bỏ trùng lặp nếu có nhiều bounding box cho cùng 1 biển số
            unique_plates = list(set(plates))
            logger.info(f"Hoàn thành xử lý ảnh. Phát hiện {len(unique_plates)} biển số duy nhất.")
            return unique_plates

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong process_image: {str(e)}", exc_info=True) # Thêm exc_info để xem traceback
            return []


# Khởi tạo recognizer khi server start (chỉ một lần)
recognizer = PlateRecognizer("best.pt")


@app.post("/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    try:
        if not file or not file.content_type:
             logger.warning("Request thiếu file hoặc content_type.")
             raise HTTPException(status_code=400, detail="Thiếu file hoặc content type.")

        if not file.content_type.startswith('image/'):
             logger.warning(f"Loại file không hợp lệ: {file.content_type}. Filename: {file.filename}")
             raise HTTPException(status_code=400, detail=f"Loại file không hợp lệ: {file.content_type}. Yêu cầu file ảnh.")

        logger.info(f"Nhận request cho file: {file.filename}, Content-Type: {file.content_type}")
        image_data = await file.read()

        if not image_data:
             logger.warning(f"File nhận được bị rỗng: {file.filename}")
             raise HTTPException(status_code=400, detail="File ảnh rỗng.")

        # Gọi hàm xử lý ảnh đã cập nhật
        plates = recognizer.process_image(image_data)

        logger.info(f"API trả về kết quả cho {file.filename}: {plates}")
        return JSONResponse(content={
            "success": True, # Luôn là True nếu không có exception, ngay cả khi plates rỗng
            "plates": plates,
            "error": None # Thêm trường error cho nhất quán
        })

    except HTTPException as http_exc:
        # Log lỗi HTTP đã được xử lý
        logger.error(f"HTTP Exception: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc # Ném lại để FastAPI xử lý
    except Exception as e:
        logger.exception(f"Lỗi API không xác định khi xử lý {file.filename if file else 'N/A'}: {str(e)}") # Dùng logger.exception
        # Trả về lỗi 500 với cấu trúc nhất quán
        return JSONResponse(
            status_code=500,
            content={"success": False, "plates": [], "error": "Lỗi xử lý phía server."}
        )


if __name__ == "__main__":
    # Cấu hình host và port phù hợp
    uvicorn.run(app, host="0.0.0.0", port=5000)