import cv2 as cv
import numpy as np

from random import random, choice


class Generator:
    def __init__(self, image_path: str, mask_path: str):
        self.image = cv.imread(image_path)
        self.mask = cv.imread(mask_path)
        self.width, self.height, _ = self.image.shape

    def process(self, image, save_path, **kwargs):
        #  Поворот на случайный угол
        rotation_matrix = cv.getRotationMatrix2D((self.width // 2, self.height // 2),
                                                 kwargs['angle'], 1.0)
        image = cv.warpAffine(image, rotation_matrix, (self.width, self.height))

        #  Отражение
        if kwargs['flip1'] < 0.5:  # Горизонтальное
            image = cv.flip(image, 1)
        if kwargs['flip0'] < 0.5:  # Вертикальное
            image = cv.flip(image, 0)

        #  Обрезка
        ...  # TODO

        cv.imwrite(save_path, cv.resize(image, (self.width, self.height)))
        
    def generate_one(self, index):
        #  Параметры
        kwargs = {
            'angle': random() * 360,
            'flip0': random(),
            'flip1': random(),
        }
        ...  # TODO

        self.process(self.image, f'images/image_{index + 1}.jpg', **kwargs)
        self.process(self.mask, f'masks/mask_{index + 1}.jpg', **kwargs)
    
    def generate(self, n: int):
        [self.generate_one(_ + 1) for _ in range(n)]
    
    
if __name__ == "__main__":
    gen = Generator("input/image.jpg", "input/mask.jpg")
    gen.generate_one(1)
    