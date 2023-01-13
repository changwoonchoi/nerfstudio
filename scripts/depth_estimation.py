# monocular depth estimation with a pretrained MIDAS model.
# python depth_estimation.py

import cv2
import torch
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from typing import Union
from typing_extensions import Annotated, Literal

import tyro
from rich.console import Console

CONSOLE = Console(width=120)

Commands = Union[
    Annotated[]

]

@dataclass
class ProcessDepth:
    """Produce depth maps of input images with a pretrained monocular depth estimation model.

    This script does the following:

    1. Downloads a pretrained MIDAS model.
    2. Estimates depth maps of input images.
    """

    data: Path
    """Path to a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    camera_type: Literal["perspective", "fisheye"] = "perspective"
    """Camera model to use."""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If true, print extra logging."""
    model_type: Literal["DPT_Large", "DPT_Hybrid", "MiDaS_small"] = "DPT_Large"
    """"Model type of Midas."""

    def main(self) -> None:
        """Estimate depth maps of whole input images"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # load model
        if self.camera_type == "fisheye":
            raise NotImplementedError("Fisheye camera model is not supported yet.")
        assert self.camera_type == "perspective", "Currently only perspective camera model is supported."

        midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        if self.gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        for img_file in self.data.glob():
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            output = prediction.cpu().numpy()
            # save



def entrypoint():
    """Entry point for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()

if __name__ == "__main__":
    entrypoint()