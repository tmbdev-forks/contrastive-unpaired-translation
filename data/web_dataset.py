import os.path
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


def get_patches(size, nsample=100, threshold=0.5):
    def loop(src):
        for key, image in src:
            if size is None or size < 1:
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
                frac = np.sum(patch < threshold) * 1.0 / (pw * ph)
                if frac < 0.05 or frac > 0.9:
                    continue
                yield f"{key}/{i}", torch.tensor(patch).permute(2, 0, 1)
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
            wds.WebDataset(self.urls_A)
            .shuffle(1000)
            .decode("rgb")
            .to_tuple("__key__", extensions)
            .then(get_patches(patchsize))
            .repeat()
        )
        self.src_A = iter(self.ds_A)
        #print(self.src_A)
        #print(next(self.src_A))
        self.ds_B = None
        self.src_B = None
        if self.urls_B is not None:
            self.ds_B = (
                None
                if self.urls_B is None
                else wds.WebDataset(self.urls_B)
                .shuffle(1000)
                .decode("rgb")
                .to_tuple("__key__", extensions)
                .then(get_patches(patchsize))
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
        A_path, A_img = next(self.src_A)
        B_path, B_img = next(self.src_B) if self.src_B is not None else (None, None)
        return {"A": A_img, "B": B_img, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size
