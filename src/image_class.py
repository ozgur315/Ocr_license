from utils.logger import log
from utils import constant_parameters
import sys
import cv2
import os
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = constant_parameters.PYTESSERACT_PATH
os.environ['TMPDIR'] = r'C:\Users\gulaoz01\Desktop\Image_OCR_Test\temp'


class ImageClass:
    """
        Image class that is going to read the picture and do the modifications to it.
        Will also read the text.

    """

    def __init__(self, image_path):
        try:
            # Define default parameters
            self.attr_image_path = image_path
            self.attr_first_image = None
            self.attr_orig_image = None
            self.attr_hsv_image = None
            self.attr_image_roi = None
            self.attr_image_rotated = None

        except Exception as e:
            log.error('Critical Error occurred on constructor, Exiting program')
            log.error(f'Error {e}')
            sys.exit()

    def start_process(self):
        # Run the method "read_image"
        self.read_image()
        # self.fix_dpi()
        # self.fix_text_size()
        self.automatic_brightness_and_contrast()
        self.convert_image_to_hsv()
        self.find_driving_license()
        self.rotate_image()
        self.read_text()

    def read_image(self):
        """
            Read the image
        """
        log.info(f'Reading the Image!- Location -{self.attr_image_path}')
        try:
            self.attr_orig_image = cv2.imread(self.attr_image_path)
            self.attr_first_image = self.attr_orig_image.copy()
        except Exception as e:
            log.error('Error occurred when reading image! '
                      'Please make sure image path is correct!')
            log.error(f'Error msg {e}')
            sys.exit()

    def fix_dpi(self, target_dpi=300):
        # Calculate scaling factor
        scaling_factor = target_dpi / 96.0  # 96 DPI is the default
        # Resize the image
        resized_image = cv2.resize(self.attr_orig_image, None, fx=scaling_factor, fy=scaling_factor)
        self.attr_orig_image = resized_image

    @staticmethod
    def fix_illumination(image):
        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(image)
        return equalized_image

    def fix_text_size(self):
        # Resize image to achieve desired text height
        text_height = 33  # Height of capital letters
        target_text_height = 30
        scaling_factor = target_text_height / text_height
        resized_image = cv2.resize(self.attr_orig_image, None, fx=scaling_factor, fy=scaling_factor)
        self.attr_orig_image = resized_image

    def automatic_brightness_and_contrast(self, clip_hist_percent=1):
        image = self.attr_orig_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        self.attr_orig_image = auto_result

    def display_image(self, image=None, text='unidentified', resize=True):
        """
            Display the image
        """
        if image is None:
            image = self.attr_orig_image
            text = 'Original'

        log.info(f'Displaying the {text}')
        try:
            if resize:
                image = cv2.resize(image, (640, 480))
            cv2.imshow(text, image)
            # If waitKey is not defined it will crash.
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
        except Exception as e:
            log.error(f'Error on displaying the image \n{e}')
            log.error('Make sure you defined the default object correctly!')

    @staticmethod
    def print_image(image, text='unnamed'):
        """
            Print image as png.
        """
        cv2.imwrite(f'{text}.png', image)

    def convert_image_to_hsv(self):
        """
            "Hue, Saturation, Value of the image
        """
        self.attr_hsv_image = cv2.cvtColor(self.attr_orig_image, cv2.COLOR_BGR2HSV)

    def find_driving_license(self):
        """
            Find the driving license based on ROI
        """
        log.info('Trying to identify ROI of image.')
        # Threshold the HSV image to get only the desired color range
        mask = cv2.inRange(self.attr_hsv_image, lowerb=constant_parameters.lower_color,
                           upperb=constant_parameters.upper_color)
        # Find contours in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to extract potential ROIs
        rois = []
        approx = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Adjust this threshold based on your specific license size
                x, y, w, h = cv2.boundingRect(contour)
                approx = cv2.approxPolyDP(contour, 10, False)
                roi = self.attr_orig_image[y:y+h, x:x+w]
                rois.append(roi)

        if approx is not None:
            # Draw contours on the original image
            image_copy = self.attr_orig_image.copy()
            # Draw contours on a blank mask
            # Create a blank mask with the same dimensions as the original image
            mask2 = np.zeros_like(self.attr_orig_image[:, :, 0])
            cv2.drawContours(mask2, [approx], 0, 255, -1)  # Draw the contour on the mask

            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(image_copy, image_copy, mask=mask2)

            self.attr_image_roi = masked_image
        else:
            log.error('Process could not locate the ROI, terminating')
            sys.exit()

    def rotate_image(self):
        """
            Rotate the image
        """
        log.info('Trying to rotate the image.')
        # Convert image to grayscale
        roi_image_copy = self.attr_image_roi.copy()
        gray = cv2.cvtColor(roi_image_copy, cv2.COLOR_BGR2GRAY)

        # Apply thresholding or other preprocessing techniques as necessary

        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Get the largest contour
        largest_contour = contours[0]

        # Calculate the angle of rotation
        rect = cv2.minAreaRect(largest_contour)
        angle_deg = rect[2]

        # Check if the angle of rotation is significant
        log.info(f'Angle Deg {angle_deg}')
        if abs(angle_deg - 90) > 20:  # Adjust the threshold as needed
            # Get the rotation matrix
            log.info('Rotating the image')
            rotation_matrix = cv2.getRotationMatrix2D((roi_image_copy.shape[1] / 2,
                                                       roi_image_copy.shape[0] / 2), angle_deg, 1.0)

            # Perform the rotation
            rotated_image = cv2.warpAffine(roi_image_copy, rotation_matrix,
                                           (roi_image_copy.shape[1], roi_image_copy.shape[0]))
        else:
            log.info('No rotation needed.')
            rotated_image = roi_image_copy

        self.attr_image_rotated = rotated_image

    def read_text(self):
        """
        Read text from the ROI image using pytesseract.
        """
        if self.attr_image_rotated is not None:

            image = self.attr_image_rotated.copy()
            # convert img to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # do adaptive threshold on gray image
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)

            # make background of input white where thresh is white
            result = image.copy()
            result[thresh == 255] = (255, 255, 255)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            for i in range(len(result)):
                gray[i] = [255 if x > 180 else 0 for x in gray[i]]

            self.display_image(image=gray, text='kcha', resize=False)
            # Use pytesseract to extract text
            try:
                # Define custom configuration for better accuracy
                # chars = ' .)'
                # numbers = '0123456789'
                # eng_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                # tgk_letters = 'БИЛЕТХАРӢОШЁНҒЧҲКМВЊСУДЌЙФЯГҶПӮЪЬІЉЮЭЇЃЖҚЅнамехоҳдубргяцлишэъткфёӯспғљњвӣчйжїќьўҷқюѕіѓ'

                # custom_config = r'--psm 3 -c tessedit_char_whitelist={}{}{}{}'.format(numbers, eng_letters,
                #                                                                       tgk_letters, chars)
                custom_config = r'--psm 6'
                text = pytesseract.image_to_string(gray, lang='eng+tgk', config=custom_config)

                # Print or return the extracted text
                log.info('Extracted text:')
                log.info(f'\n{text}')
                return text
            except Exception as e:
                log.error(f"Error occurred during text extraction: {e}")
        else:
            log.info("No rotated ROI image available. "
                     "Please ensure that the ROI has been correctly identified and rotated.")

    @staticmethod
    def nothing(x):
        pass

    def find_hue_dynamically(self):
        """
            Find the hue yourself.
            https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
        """
        # Load image
        image = self.attr_orig_image.copy()
        image = cv2.resize(image, (640, 480))
        # Create a window
        cv2.namedWindow('image')

        # Create trackbars for color change
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('HMin', 'image', 0, 179, self.nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, self.nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, self.nothing)

        # Set default value for Max HSV trackbars
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize HSV min/max values
        # hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        while 1:
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Print if there is a change in HSV value
            if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)"
                      % (hMin, sMin, vMin, hMax, sMax, vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
