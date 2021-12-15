import numpy as np
import torch

from torchvision import transforms
from augmix import augmentations


class AugMix(object):
    def __init__(self, all_ops=False, severity=3, width=3, depth=-1, alpha=1.):
        '''
        Args:
            image: Raw input image as float32 np.ndarray of shape (h, w, c)
            severity: Severity of underlying augmentation operators (between 1 to 10).
            width: Width of augmentation chain
            depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
            from [1, 3]
            alpha: Probability coefficient for Beta and Dirichlet distributions.
        '''
        self.all_ops = all_ops
        self.aug_severity = severity
        self.mixture_width = width
        self.mixture_depth = depth
        self.aug_prob_coeff = alpha
        self.preprocess = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.
        Args:
            image: PIL.Image input image
            preprocess: Preprocessing function which should return a torch tensor.
        Returns:
            mixed: Augmented and mixed image.
        """


        aug_list = augmentations.augmentations
        if self.all_ops:
            aug_list = augmentations.augmentations_all

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = torch.zeros_like(self.preprocess(image))
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, self.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)

        mixed = (1 - m) * self.preprocess(image) + m * mix
        return mixed
