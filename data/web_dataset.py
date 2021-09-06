import os.path
import sys
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
from random import randrange, uniform


from scipy import ndimage as ndi
import webdataset as wds
import numpy as np
from math import log10
import torch


def normalize_image(image):
    image = image - np.amin(image)
    image /= np.amax(image)
    return image


def get_patches(size, nsample=100):
    def loop(src):
        for key, image in src:
            image = normalize_image(image)
            if size is None or size < 1:
                assert False
                yield key, image
            for i in range(nsample):
                h, w = image.shape[:2]
                ph, pw = size, size
                y, x = randrange(0, h - ph), randrange(0, w - pw)
                scale = 10 ** uniform(log10(0.5), log10(2.0))
                assert scale >= 0.3 and scale <= 3.1
                patch = ndi.affine_transform(
                    image,
                    np.diag([scale, scale, 1.0]),
                    (y, x, 0),
                    order=1,
                    mode="mirror",
                    output_shape=(ph, pw, 3),
                )
                # patch = image[y:y+ph, x:x+pw, ...]
                assert np.amin(patch) >= 0.0 and np.amax(patch) <= 1.0
                frac = np.sum(np.mean(patch, 2) < 0.5) * 1.0 / (pw * ph)
                if frac < 0.05 or frac > 0.9:
                    continue
                patch = Image.fromarray((255 * patch).astype(np.uint8))
                yield f"{key}/{i}", patch

    return loop


class WebDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets stored in WebDataset format.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        dataroot = opt.dataroot.split(";")
        self.urls_A, self.urls_B, options = dataroot + [None] * (3 - len(dataroot))
        self.options = eval(f"dict({options})") if options is not None else {}
        extensions = self.options.get("extensions", ["jpg", "jpeg", "png"])
        patchsize = self.options.get("patchsize", 224)
        self.size = self.options.get("size", 1000)
        self.ds_A = (
            wds.WebDataset(self.urls_A, resampled=True)
            .shuffle(100)
            .decode("rgb")
            .to_tuple("__key__", extensions)
            .then(get_patches(patchsize))
            .shuffle(1000)
            .repeat()
        )
        self.src_A = iter(self.ds_A)
        self.ds_B = None
        self.src_B = None
        if self.urls_B is not None:
            self.ds_B = (
                None
                if self.urls_B is None
                else wds.WebDataset(self.urls_B, resampled=True)
                .shuffle(100)
                .decode("rgb")
                .to_tuple("__key__", extensions)
                .then(get_patches(patchsize))
                .shuffle(1000)
                .repeat()
            )
            self.src_B = iter(self.ds_B)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        result = {}
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        result["A_paths"], result["A"] = next(self.src_A)
        result["A"] = transform(result["A"])
        if self.src_B is not None:
            result["B_paths"], result["B"] = next(self.src_B)
            result["B"] = transform(result["B"])
        return result

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size
