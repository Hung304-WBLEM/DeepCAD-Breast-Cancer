import random
import numpy as np

from PIL import Image

class IntensityShift(object):
    def __init__(self, intensity_range):
        assert isinstance(intensity_range, tuple)
        self.intensity_range = intensity_range

    def __call__(self, image):
        shift_value = random.randint(self.intensity_range[0], self.intensity_range[1])

        image = np.array(image)
        image = image + shift_value
        image = np.clip(image, a_min=0, a_max=255)

        image = Image.fromarray(image.astype(np.uint8))

        return image

