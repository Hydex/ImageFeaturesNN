
from PIL import Image
from glob import glob

def get_aspect_ratio(pil_image):
    _, _, width, height = pil_image.getbbox()

    return width / height

def normalize_pixel(value):
    return float(value) / float(255)

def get_greyscale_array(pil_image):
    """Convert the image to a 13x13 square grayscale image, and return a
    list of colour values 0-255.

    I've chosen 13x13 as it's very small but still allows you to
    distinguish the gap between legs on jeans in my testing.

    """
    grayscale_image = pil_image.convert('L')
    small_image = grayscale_image.resize((15, 15), Image.ANTIALIAS)

    pixels = []
    for y in range(15):
        for x in range(15):
            pixels.append(small_image.getpixel((x, y)))

    return pixels


def get_image_features(image_path):
    image = Image.open(open(image_path, 'rb'))

    features = []

    for index, pixel in enumerate(get_greyscale_array(image)):
        features.append(normalize_pixel(pixel)) 

    return features

def generate_training_set(folder, label):
    files = glob(folder)

    training_data = [
        [get_image_features(f), [label]] for f in files
    ]

    return training_data
