import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import math
import torch
import pdb
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import pdb

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        foreground_column: str = "foreground",
        background_column: str = "background",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.foreground_column = foreground_column
        self.background_column = background_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds

class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

class VideoDatasetWithResizingTracking(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.hdr_column = kwargs.pop("hdr_column", None)
        self.tracking_column = kwargs.pop("tracking_column", None)
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path, hdr_path: Path, tracking_path: Path, foreground_path: Path, background_path: Path):
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path)

        video_reader = decord.VideoReader(uri=path.as_posix())
        hdr_reader = decord.VideoReader(uri=hdr_path.as_posix()) if hdr_path is not None else None
        tracking_reader = decord.VideoReader(uri=tracking_path.as_posix()) if tracking_path is not None else None
        foreground_reader = decord.VideoReader(uri=foreground_path.as_posix()) if foreground_path is not None else None
        background_reader = decord.VideoReader(uri=background_path.as_posix()) if background_path is not None else None

        lengths = [len(video_reader)]
        if hdr_reader is not None:
            lengths.append(len(hdr_reader))
        if tracking_reader is not None:
            lengths.append(len(tracking_reader))
        if foreground_reader is not None:
            lengths.append(len(foreground_reader))
        if background_reader is not None:
            lengths.append(len(background_reader))

        common_len = min(lengths)

        # Use one fixed target length for all samples in this run.
        # This keeps the batch stackable when using SimpleBatchSampler.
        target_num_frames = min(self.frame_buckets, key=lambda x: abs(x - self.max_num_frames))

        # Skip clips that are too short for the chosen target length.
        if common_len < target_num_frames:
            raise IndexError(
                f"sample too short: common_len={common_len}, required={target_num_frames}"
            )

        # Uniformly sample exactly target_num_frames frames.
        frame_indices = np.linspace(0, common_len - 1, num=target_num_frames, dtype=int).tolist()

        frames = video_reader.get_batch(frame_indices).float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        hdr_frames = None
        if hdr_reader is not None:
            hdr_frames = hdr_reader.get_batch(frame_indices).float()
            hdr_frames = hdr_frames.permute(0, 3, 1, 2).contiguous()
            hdr_frames_resized = torch.stack([resize(hdr_frame, nearest_res) for hdr_frame in hdr_frames], dim=0)
            hdr_frames = torch.stack([self.video_transforms(hdr_frame) for hdr_frame in hdr_frames_resized], dim=0)

        foreground_frames = None
        if foreground_reader is not None:
            foreground_frames = foreground_reader.get_batch(frame_indices).float()
            foreground_frames = foreground_frames.permute(0, 3, 1, 2).contiguous()
            foreground_frames_resized = torch.stack([resize(foreground_frame, nearest_res) for foreground_frame in foreground_frames], dim=0)
            foreground_frames = torch.stack([self.video_transforms(foreground_frame) for foreground_frame in foreground_frames_resized], dim=0)

        background_frames = None
        if background_reader is not None:
            background_frames = background_reader.get_batch(frame_indices).float()
            background_frames = background_frames.permute(0, 3, 1, 2).contiguous()
            background_frames_resized = torch.stack([resize(background_frame, nearest_res) for background_frame in background_frames], dim=0)
            background_frames = torch.stack([self.video_transforms(background_frame) for background_frame in background_frames_resized], dim=0)

        tracking_frames = None
        if tracking_reader is not None:
            tracking_frames = tracking_reader.get_batch(frame_indices).float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

        return image, frames, hdr_frames, tracking_frames, foreground_frames, background_frames

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        foreground_path = self.data_root.joinpath(self.foreground_column)
        background_path, background_paths = None, None
        if self.background_column is not None:
            background_path = self.data_root.joinpath(self.background_column)
        if self.tracking_column is not None:
            tracking_path = self.data_root.joinpath(self.tracking_column)
        hdr_path = self.data_root.joinpath(self.hdr_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not foreground_path.exists() or not foreground_path.is_file():
            raise ValueError(
                "Expected `--foreground_path` to be path to a file in `--data_root` containing line-separated paths to foreground video data in the same directory."
            )
        if not hdr_path.exists() or not hdr_path.is_file():
            raise ValueError(
                "Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated tracking information."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(foreground_path, "r", encoding="utf-8") as file:
            foreground_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(hdr_path, "r", encoding="utf-8") as file:
            hdr_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        try:
            with open(background_path, "r", encoding="utf-8") as file:
                background_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if
                                    len(line.strip()) > 0]
        except:
            background_paths = None

        try:
            with open(tracking_path, "r", encoding="utf-8") as file:
                tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if
                                    len(line.strip()) > 0]
        except:
            tracking_paths = None

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )
        self.foreground_paths = foreground_paths
        self.hdr_paths = hdr_paths
        self.tracking_paths = tracking_paths
        self.background_paths = background_paths
        return prompts, video_paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        try:
            # Always use raw‐frames path'
            background_map = None
            if self.background_paths is not None:
                image, video, hdr_map, tracking_map, foreground_map, background_map = \
                    self._preprocess_video(
                        self.video_paths[index],
                        self.hdr_paths[index],
                        self.tracking_paths[index],
                        self.foreground_paths[index],
                        self.background_paths[index],
                    )
            else:
                image, video, hdr_map, tracking_map, foreground_map, _ = \
                    self._preprocess_video(
                        self.video_paths[index],
                        self.hdr_paths[index],
                        self.tracking_paths[index],
                        self.foreground_paths[index],
                        None,
                    )

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "foreground_map": foreground_map,
                "background_map": background_map,
                "hdr_map": hdr_map,
                "tracking_map": tracking_map,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

        except IndexError as e:
            # Skip any sample that fails (e.g. Decord OOB)
            print(f"[Warning] sample {index} failed: {e}. Skipping to next sample.")
            next_index = (index + 1) % len(self)
            return self.__getitem__(next_index)

    def _load_preprocessed_latents_and_embeds(self, path: Path, tracking_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        tracking_map_path = path.parent.parent.joinpath("tracking_map")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or not tracking_map_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents`, `prompt_embeds`, and `tracking_map`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        tracking_map_filepath = tracking_map_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file() or not tracking_map_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            tracking_map_filepath = tracking_map_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} or {tracking_map_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        tracking_map = torch.load(tracking_map_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, tracking_map, embeds

class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []



class SimpleBatchSampler(Sampler[list[int]]):
    """
    A Sampler that returns sequential (or shuffled) batches of indices.

    Args:
        data_source (Dataset): any PyTorch Dataset
        batch_size (int): number of samples per batch
        shuffle (bool): shuffle indices each epoch
        drop_last (bool): if True, drop the last batch if it's smaller than batch_size
    """
    def __init__(
        self,
        data_source,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # create a full list of indices
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.shuffle(indices)

        # slice into batches
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch  # yield the final smaller batch

    def __len__(self):
        if self.drop_last:
            # only full batches
            return len(self.data_source) // self.batch_size
        else:
            # include final partial batch
            return math.ceil(len(self.data_source) / self.batch_size)
