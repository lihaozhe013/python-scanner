import cv2
import numpy as np
from pathlib import Path

class DocumentScanner:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def get_image_files(self):
        return sorted([
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in self.valid_extensions
        ])

    def scan_effect_bw(self, image):
        """
        Convert to Black and White with adaptive thresholding
        resulting in a clean 'scanned document' look.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        # blockSize: Size of a pixel neighborhood that is used to calculate a threshold value.
        # C: Constant subtracted from the mean or weighted mean.
        # You can tune these values. 11/15 and 10 are common starting points.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
        )
        
        return thresh

    def scan_effect_color(self, image):
        """
        Enhance brightness and contrast while sharpening the image
        to look like a high-quality color scan.
        """
        # 1. Denoise slightly
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # 2. Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 3. Increase contrast and brightness
        # alpha > 1.0 (contrast), beta is bias (brightness)
        alpha = 1.2
        beta = 10
        enhanced = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
        
        return enhanced

    def scan_effect_magic_color(self, image):
        """
        Simulate a 'Magic Color' effect often found in scanner apps.
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on Lab color space.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Slight boost in saturation
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        final = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        
        return final

    def run(self):
        files = self.get_image_files()
        print(f"Found {len(files)} images to process in {self.input_dir}")
        
        for file_path in files:
            print(f"Processing: {file_path.name}")
            image = cv2.imread(str(file_path))
            
            if image is None:
                print(f"Failed to load {file_path.name}")
                continue

            # Generate different versions
            bw = self.scan_effect_bw(image)
            color = self.scan_effect_color(image)
            magic = self.scan_effect_magic_color(image)
            
            # Save files
            base_name = file_path.stem
            
            cv2.imwrite(str(self.output_dir / f"{base_name}_bw.jpg"), bw)
            cv2.imwrite(str(self.output_dir / f"{base_name}_color.jpg"), color)
            cv2.imwrite(str(self.output_dir / f"{base_name}_magic.jpg"), magic)
            
            print(f"Saved enhanced versions for {base_name}")