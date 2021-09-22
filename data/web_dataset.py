import os.path
import sys

from webdataset import dataset
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
from random import randrange, uniform
from torchvision import transforms
from torch.utils.data import IterableDataset


from scipy import ndimage as ndi
import webdataset as wds
import numpy as np
from math import floor, log10
import torch
import yaml
import io
import braceexpand

verbose_wds = int(os.environ.get("VERBOSE_WDS", 0))


def normalize_image(image):
    image = image - np.amin(image)
    image /= np.amax(image)
    return image


def get_patches(size, nsample=100, threshold=-1.0, scale=(0.5, 2.0)):
    def loop(src):
        for key, image in src:
            #image = normalize_image(image)
            if size is None or size < 1:
                assert False
                yield key, image
            h, w = image.shape[:2]
            ph, pw = size, size
            if h-ph <= 5 or w-pw <= :
                continue
            for i in range(nsample):
                y, x = randrange(0, h - ph), randrange(0, w - pw)
                s = 10 ** uniform(log10(scale[0]), log10(scale[1]))
                assert s >= scale[0] and s <= scale[1]
                patch = ndi.affine_transform(
                    image,
                    np.diag([s, s, 1.0]),
                    (y, x, 0),
                    order=1,
                    mode="mirror",
                    output_shape=(ph, pw, 3),
                )
                # patch = image[y:y+ph, x:x+pw, ...]
                assert np.amin(patch) >= 0.0 and np.amax(patch) <= 1.0
                frac = np.sum(np.mean(patch, 2) < 0.5) * 1.0 / (pw * ph)
                if threshold > 0.0 and (frac < threshold or frac > threshold):
                    continue
                patch = Image.fromarray((255 * patch).astype(np.uint8)).convert("RGB")
                yield f"{key}/{i}", patch

    return loop

def expand(urls):
    if isinstance(urls, str):
        return list(braceexpand.braceexpand(urls))
    else:
        return [x for u in urls for x in expand(u)]

def random_rescale(image, scales):
    assert isinstance(image, np.ndarray)
    image = Image.fromarray((image*255.0).astype(np.uint8)).convert("RGB")
    lo, hi = scales
    h, w = image.height, image.width
    s = 10**random.uniform(log10(lo), log10(hi))
    mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
    zoomed = image.resize((int(w * s), int(h * s)), mode)
    zoomed = np.array(zoomed) / 255.0
    if verbose_wds:
        print("# random_rescale ", scales, s, (h, w), zoomed.shape)
    return zoomed

def random_rotation(image, prob=0.5):
    assert isinstance(image, Image.Image)
    if random.random() < prob:
        return image
    assert isinstance(image, Image.Image)
    a = random.choice([0, 90, 180, 270])
    #print("# random_rotation ", a)
    return image.rotate(a)

def random_down_up_scale(image, prob=0.5):
    assert isinstance(image, Image.Image)
    if random.random() < prob:
        return image
    h, w = image.height, image.width
    s = random.uniform(2.0, 5.0)
    if verbose_wds:
        print("# random_down_up_scale ", s)
    mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
    zoomed = image.resize((int(w * s), int(h * s)), mode)
    unzoomed = image.resize((w, h), mode)
    return unzoomed

def make_dataset(spec, options, comment=""):
    print(comment, spec, options)
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [spec]
    result = wds.RoundRobin()
    for item in spec:   
        if isinstance(item, str):
            item = dict(shards=item) 
        extensions = item.get("extensions", ["jpg", "jpeg", "png"])
        patchsize = item.get("patchsize", 256)
        nsample = item.get("nsample", 10)
        rotprob = float(item.get("rotprob", 0.25))
        scaleprob = float(item.get("scaleprob", 0.0))
        noise = float(item.get("noise", 0.0))
        urls = list(braceexpand.braceexpand(item["shards"]))
        print("# make_dataset ", comment, " ", urls[:2])
        dataset = (
            wds.WebDataset(urls, resampled=True, handler=wds.ignore_and_continue)
            .shuffle(item.get("preshuffle", 100))
            .rsample(item.get("subsample", 1.0))
            .decode("rgb")
            .to_tuple("__key__", item.get("extensions", ["png", "jpg", "jpeg"]), handler=wds.ignore_and_continue)
        )
        gray = item.get("gray", True)
        if gray:
            dataset = dataset.map(lambda s: (s[0], np.mean(s[1], 2, keepdims=True).repeat(3, 2)))
        scale = item.get("scale", [])
        if len(scale) > 0:
            the_scale = scale
            dataset = dataset.map(lambda s: (s[0], random_rescale(s[1], the_scale)))
        dataset = (
            dataset.then(get_patches(patchsize, nsample=nsample))
            .shuffle(item.get("shuffle", 10000))
            .repeat()
        )
        if rotprob > 0.0:
            dataset = dataset.map(lambda s: (s[0], random_rotation(s[1], rotprob)))
        if scaleprob > 0.0:
            dataset = dataset.map(lambda s: (s[0], random_down_up_scale(s[1], scaleprob)))
        result.add_dataset(dataset, probability=item.get("probability", 1.0), comment=str(urls)[:80])

    return result

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

        if not opt.dataroot.endswith(".yaml"):
            self.options = yaml.load(io.StringIO(opt.dataroot), Loader=yaml.FullLoader)
        else:
            self.options = yaml.load(open(opt.dataroot, "r"), Loader=yaml.FullLoader)
        assert isinstance(self.options, dict)
        self.ds_A = make_dataset(self.options["A"], self.options, comment="A")
        print("ds_A", id(self.ds_A), self.ds_A)        
        self.ds_B = make_dataset(self.options.get("B", None), self.options, comment="B")
        self.size = self.options.get("epoch", 10000)
        self.src_A = iter(self.ds_A)
        self.src_B = iter(self.ds_B) if self.ds_B is not None else None
        print("ds_A", id(self.ds_A), self.ds_A)
        print("ds_B", id(self.ds_B), self.ds_B)

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
        if int(os.environ.get("VERBOSE_LOAD", "0")) > 0:
            A = transforms.ToTensor()(result["A"])
            print(f"WebDataset {A.shape} {A.min():6.3f} {A.mean():6.3f} {A.max():6.3f} {result['A']}", file=sys.stderr)
        result["A"] = transform(result["A"])
        assert result["A"].shape[0] == 3
        if self.opt.input_nc == 1:
            result["A"] = result["A"].mean(0, keepdim=True)
        if self.src_B is not None:
            result["B_paths"], result["B"] = next(self.src_B)
            result["B"] = transform(result["B"])
            assert result["B"].shape[0] == 3
            if self.opt.output_nc == 1:
                result["B"] = result["B"].mean(0, keepdim=True)
        if verbose_wds:
            print(">>>", result["A_paths"])
        return result

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size

