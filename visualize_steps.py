import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import re
import logging
import io
from PIL import Image

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# === THÊM HÀM HIỂN THỊ ẢNH TIỆN LỢI ===
def display_image(window_name, image, wait_key=0):
    """Hiển thị ảnh trong cửa sổ OpenCV và chờ người dùng nhấn phím."""
    if image is None:
        logger.warning(f"Không thể hiển thị ảnh rỗng cho cửa sổ: {window_name}")
        return
    
    # Thay đổi kích thước nếu ảnh quá lớn để dễ xem (tùy chọn)
    max_height = 800
    max_width = 1200
    height, width = image.shape[:2]
    if height > max_height or width > max_width:
        scaling_factor = min(max_height / height, max_width / width)
        resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, resized_image)
    else:
        cv2.imshow(window_name, image)
    logger.info(f"Hiển thị cửa sổ: '{window_name}'. Nhấn phím bất kỳ trên cửa sổ đó để tiếp tục...")
    cv2.waitKey(wait_key) # wait_key=0 -> Chờ vô hạn cho đến khi nhấn phím

class PlateRecognizer:
    def __init__(self, model_path, ocr_languages=['vi', 'en']):
        logger.info("Đang khởi tạo PlateRecognizer...")
        self.model = YOLO(model_path)
        logger.info(f"Model YOLO đã tải từ: {model_path}")
        self.reader = easyocr.Reader(
            ocr_languages,
            gpu=False, # Đặt thành True nếu bạn có GPU và cài đặt đúng
            model_storage_directory='./ocr_models',
            download_enabled=True
        )
        logger.info(f"EasyOCR Reader đã khởi tạo với ngôn ngữ: {ocr_languages}, GPU: False")
        self.min_confidence = 0.2
        self.min_plate_width = 30
        self.min_plate_height = 10
        logger.info("Khởi tạo PlateRecognizer hoàn tất!")

    # === HIỂN THỊ CÁC BƯỚC ===
    def preprocess_plate(self, roi, roi_index):
        """Tiền xử lý ảnh biển số và hiển thị các bước trung gian."""
        window_prefix = f"ROI {roi_index} - Preprocess"
        try:
            if roi is None or roi.shape[0] < 5 or roi.shape[1] < 5:
                logger.warning(f"{window_prefix}: ROI không hợp lệ hoặc quá nhỏ.")
                return None

            display_image(f"{window_prefix} - 1. Original ROI", roi)

            # 1. Chuyển ảnh xám
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            display_image(f"{window_prefix} - 2. Grayscale", gray)

            # 2. Cân bằng histogram CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            display_image(f"{window_prefix} - 3. CLAHE Enhanced", enhanced)

            # 3. Làm sắc nét (Unsharp Masking)
            blurred = cv2.GaussianBlur(enhanced, (0,0), 3)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            display_image(f"{window_prefix} - 4. Sharpened", sharpened)

            # 4. Phân ngưỡng thích ứng
            binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 5)
            display_image(f"{window_prefix} - 5. Adaptive Threshold (Binary)", binary)

            logger.debug(f"{window_prefix}: Tiền xử lý hoàn tất.")
            return binary
        
        except cv2.error as cv_err:
             logger.error(f"{window_prefix}: Lỗi OpenCV: {cv_err}", exc_info=True)
             # Đóng các cửa sổ có thể đã mở của bước này nếu có lỗi
             cv2.destroyWindow(f"{window_prefix} - 1. Original ROI")
             cv2.destroyWindow(f"{window_prefix} - 2. Grayscale")
             cv2.destroyWindow(f"{window_prefix} - 3. CLAHE Enhanced")
             cv2.destroyWindow(f"{window_prefix} - 4. Sharpened")
             cv2.destroyWindow(f"{window_prefix} - 5. Adaptive Threshold (Binary)")
             return None
        except Exception as e:
            logger.error(f"{window_prefix}: Lỗi không xác định: {e}", exc_info=True)
            # Đóng các cửa sổ có thể đã mở của bước này nếu có lỗi
            cv2.destroyWindow(f"{window_prefix} - 1. Original ROI")
            cv2.destroyWindow(f"{window_prefix} - 2. Grayscale")
            cv2.destroyWindow(f"{window_prefix} - 3. CLAHE Enhanced")
            cv2.destroyWindow(f"{window_prefix} - 4. Sharpened")
            cv2.destroyWindow(f"{window_prefix} - 5. Adaptive Threshold (Binary)")
            return None

    # format_vietnam_plate
    def format_vietnam_plate(self, text):
        if not text:
            return ""
        cleaned_text = re.sub(r'[\s.-]', '', text).upper()
        # Định dạng Ô tô mới
        car_new_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{5})$', cleaned_text)
        if car_new_match:
            g1, g2, g3 = car_new_match.groups(); return f"{g1}{g2}-{g3[:3]}.{g3[3:]}"
        # Định dạng Xe máy mới
        motor_new_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{1})([0-9]{5})$', cleaned_text)
        if motor_new_match:
            g1, g2, g3, g4 = motor_new_match.groups(); return f"{g1}-{g2}{g3}-{g4[:3]}.{g4[3:]}"
        # Định dạng Xe máy cũ
        motor_old_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{1})([0-9]{4})$', cleaned_text)
        if motor_old_match:
            g1, g2, g3, g4 = motor_old_match.groups(); return f"{g1}-{g2}{g3}-{g4}"
        # Định dạng Ô tô cũ
        car_old_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{4,5})$', cleaned_text)
        if car_old_match:
            g1, g2, g3 = car_old_match.groups(); return f"{g1}{g2}-{g3}"
        # Biển số vuông ô tô
        car_square_match = re.match(r'^([0-9]{2})([A-Z]{1})([0-9]{4})$', cleaned_text)
        if car_square_match:
             g1, g2, g3 = car_square_match.groups(); return f"{g1}{g2}-{g3}"
        return ""

    # === process_image_and_visualize ===
    def process_image_and_visualize(self, frame): # Nhận frame OpenCV thay vì bytes
        """Xử lý ảnh và hiển thị các bước trung gian."""
        try:
            if frame is None:
                logger.error("Input frame is None.")
                return []

            logger.info(f"Bắt đầu xử lý ảnh kích thước: {frame.shape}")
            display_image("1. Original Input Image", frame)

            # === Bước 2: YOLO Detection ===
            results = self.model.predict(frame, imgsz=640, conf=self.min_confidence, verbose=False)
            logger.info(f"YOLO phát hiện {len(results[0].boxes) if results and results[0].boxes else 0} đối tượng.") # Kiểm tra results[0].boxes

            # Chuẩn bị ảnh để vẽ bounding box
            frame_with_boxes = frame.copy() # Tạo bản sao để vẽ lên

            # Chỉ vẽ nếu có kết quả và có boxes
            if results and results[0].boxes and len(results[0].boxes) > 0:
                 boxes_data = results[0].boxes # Truy cập đối tượng Boxes
                 for i in range(len(boxes_data)):
                    # Lấy tọa độ box
                    box = boxes_data.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    # Lấy confidence score
                    conf = boxes_data.conf[i].cpu().numpy()
                    # --- SỬA LỖI LẤY CLASS ID VÀ LABEL ---
                    cls_tensor = boxes_data.cls[i]
                    cls_id = int(cls_tensor.item()) # Chuyển tensor thành int
                    

                    # Lấy label từ class id (thêm kiểm tra an toàn)
                    label = "Unknown" # Giá trị mặc định
                    if hasattr(self.model, 'names') and isinstance(self.model.names, (list, dict)):
                        if isinstance(self.model.names, dict): # Nếu là dictionary
                            if cls_id in self.model.names:
                                label = f'{self.model.names.get(cls_id, "Unknown")} {conf:.2f}'
                            else:
                                logger.warning(f"Class ID {cls_id} không tìm thấy trong dictionary self.model.names")
                        else: # Giả sử là list
                            if 0 <= cls_id < len(self.model.names):
                                label = f'{self.model.names[cls_id]} {conf:.2f}'
                            else:
                                logger.warning(f"Class ID {cls_id} nằm ngoài phạm vi list self.model.names (size {len(self.model.names)})")
                    else:
                         logger.warning(f"Thuộc tính 'names' không hợp lệ hoặc không tồn tại trong self.model")

                    # Vẽ hình chữ nhật
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Viết nhãn (label đã được sửa và kiểm tra)
                    cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                 # Hiển thị ảnh với TẤT CẢ bounding box sau khi vòng lặp vẽ kết thúc
                 display_image("2. YOLO Detection Results", frame_with_boxes)
            else:
                 # Trường hợp không có kết quả hoặc không có box nào
                 logger.warning("YOLO không phát hiện được đối tượng nào hoặc không có bounding box.")
                 display_image("2. YOLO Detection Results (No Detections)", frame_with_boxes) # Vẫn hiển thị ảnh (lúc này là bản sao của ảnh gốc)

            # === Vòng lặp xử lý từng ROI được phát hiện ===
            plates = []
            processed_count = 0
            # Chỉ lặp nếu có kết quả và có boxes
            if results and results[0].boxes and len(results[0].boxes) > 0:
                 boxes = results[0].boxes.xyxy.cpu().numpy()
                 scores = results[0].boxes.conf.cpu().numpy()

                 for i, box in enumerate(boxes):
                    logger.info(f"\n--- Đang xử lý ROI {i} ---")
                    x1, y1, x2, y2 = map(int, box)
                    confidence = scores[i] # Lấy lại confidence score cho ROI hiện tại

                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    if plate_width < self.min_plate_width or plate_height < self.min_plate_height:
                        logger.debug(f"ROI {i}: Bỏ qua vì quá nhỏ (W={plate_width}, H={plate_height})")
                        continue

                    # === Bước 3: Cắt ROI ===
                    plate_roi = frame[y1:y2, x1:x2]
                    logger.debug(f"ROI {i}: Đã cắt tại [{y1}:{y2}, {x1}:{x2}], Conf: {confidence:.2f}")

                    # === Bước 4: Tiền xử lý ROI (Hàm này sẽ hiển thị các bước con) ===
                    processed_roi = self.preprocess_plate(plate_roi, i) # Truyền roi gốc và index
                    if processed_roi is None:
                        logger.warning(f"ROI {i}: Tiền xử lý thất bại, bỏ qua.")
                        continue # Bỏ qua nếu tiền xử lý lỗi

                    # === Bước 5: OCR Processing ===
                    logger.info(f"ROI {i}: Bắt đầu OCR...")
                    try:
                        ocr_result = self.reader.readtext(
                            processed_roi, # Sử dụng ảnh đã tiền xử lý (binary)
                            decoder='beamsearch',
                            beamWidth=10,
                            batch_size=1,
                            allowlist='0123456789ABCDEFGHKLMNPSTUVXYZ',
                            detail=0,
                            paragraph=False
                        )
                        combined = ''.join(ocr_result)
                        logger.info(f"ROI {i}: OCR Result (raw): '{combined}'")
                        print(f"-------> ROI {i} - OCR Raw Text: {combined}") # In ra console

                        # === Bước 6: Format Biển số ===
                        formatted = self.format_vietnam_plate(combined)
                        if formatted:
                            logger.info(f"ROI {i}: Formatted Plate: '{formatted}'")
                            print(f"-------> ROI {i} - Formatted Plate: {formatted}") # In ra console
                            plates.append(formatted)
                        else:
                             logger.warning(f"ROI {i}: Raw text '{combined}' không định dạng được.")
                             print(f"-------> ROI {i} - Formatting Failed") # In ra console

                        processed_count += 1

                    except Exception as ocr_err:
                         logger.error(f"ROI {i}: Lỗi trong quá trình OCR: {ocr_err}", exc_info=True)

                    # Tạm dừng sau khi xử lý xong mỗi ROI
                    logger.info(f"Đã xử lý xong ROI {i}. Nhấn phím bất kỳ trên một cửa sổ ảnh để xử lý ROI tiếp theo (nếu có)...")
                    cv2.waitKey(0)
                    # Đóng các cửa sổ tiền xử lý của ROI này trước khi sang ROI mới
                    # Đảm bảo tên cửa sổ khớp với tên trong hàm preprocess_plate
                    cv2.destroyWindow(f"ROI {i} - Preprocess - 1. Original ROI")
                    cv2.destroyWindow(f"ROI {i} - Preprocess - 2. Grayscale")
                    cv2.destroyWindow(f"ROI {i} - Preprocess - 3. CLAHE Enhanced")
                    cv2.destroyWindow(f"ROI {i} - Preprocess - 4. Sharpened")
                    cv2.destroyWindow(f"ROI {i} - Preprocess - 5. Adaptive Threshold (Binary)")

            # === Bước 7: Tổng hợp kết quả cuối cùng ===
            unique_plates = list(set(plates))
            logger.info(f"\n===================================")
            logger.info(f"Hoàn thành xử lý ảnh. Phát hiện {len(unique_plates)} biển số duy nhất:")
            print(f"\n===================================")
            print(f"FINAL DETECTED PLATES: {unique_plates}")
            print(f"===================================")
            return unique_plates

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong process_image_and_visualize: {str(e)}", exc_info=True)
            # Có thể thêm đóng tất cả cửa sổ ở đây nếu lỗi xảy ra sớm
            cv2.destroyAllWindows()
            return []


# --- KHỐI CHẠY CHÍNH ---
if __name__ == "__main__":
    # --- THAY ĐỔI ĐƯỜNG DẪN NÀY ---
    IMAGE_PATH = "AnhBienXeMay.jpg" # <<< THAY ĐỔI ĐƯỜNG DẪN ĐẾN ẢNH CỦA BẠN

    MODEL_PATH = "best.pt" # Đường dẫn đến model YOLO

    # 1. Khởi tạo Recognizer
    try:
        recognizer = PlateRecognizer(MODEL_PATH)
    except Exception as init_err:
        logger.exception(f"Không thể khởi tạo PlateRecognizer: {init_err}")
        exit() # Thoát nếu không khởi tạo được

    # 2. Đọc ảnh đầu vào
    input_image = cv2.imread(IMAGE_PATH)

    if input_image is None:
        logger.error(f"Không thể đọc file ảnh: {IMAGE_PATH}")
        exit() # Thoát nếu không đọc được ảnh

    # 3. Gọi hàm xử lý và hiển thị từng bước
    final_plates = recognizer.process_image_and_visualize(input_image)

    # 4. Giữ cửa sổ cuối cùng (Kết quả YOLO) và đợi người dùng thoát
    logger.info("Hoàn thành tất cả các bước. Nhấn phím bất kỳ trên một cửa sổ ảnh để thoát.")
    cv2.waitKey(0)
    cv2.destroyAllWindows() # Đóng tất cả cửa sổ khi kết thúc
    logger.info("Đã đóng tất cả cửa sổ.")