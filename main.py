from utils.logger import log
from utils import constant_parameters
from src.image_class import ImageClass


def main():
    # Call the logger
    log.info('Starting the process...')

    # TODO: later this will have to read from the ui. pyqt5

    # image_path = "C:/Users/ozgur/Desktop/tasks/Image_OCR_Test/Users/person1/front.jpg"
    image_path = "C:/Users/ozgur/Desktop/tasks/Image_OCR_Test/Users/person3/front3.jpg"
    # image_path = "C:/Users/ozgur/Desktop/tasks/Desktop/Image_OCR_Test/Users/person3/back.jpg"
    # Define the imageClass instance.
    image_cv = ImageClass(image_path)
    image_cv.start_process()
    image_cv.display_image(resize=False)
    image_cv.display_image(image=image_cv.attr_image_rotated, text='Rotated', resize=False)
    # Test the hsv colors.
    # image_cv.find_hue_dynamically()
    log.info('Completed!')


if __name__ == "__main__":
    main()
