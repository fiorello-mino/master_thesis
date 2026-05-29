# <<< import external stuff <<<
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils

import os
import numpy as np

import PIL
from PIL import Image

import time
# --- import external stuff ---

# <<< import my stuff <<<
from src.utils import make_square
# --- import my stuff ---


class TabulatedSeries(torch.utils.data.Dataset):
    """
    Dataset 2D: le righe del file di tabella contengono una sequenza di path,
    eventualmente seguiti da parametri scalari.

    table.txt (esempio):
        /path/frame_0.png /path/frame_1.png ... p1 p2 ... pid

    params_num = numero di parametri finali in ciascuna riga (0 se assenti).
    """

    def __init__(
        self,
        table_path,
        params_num: int = 0,
        transform=None,
        rotation: bool = True,
        reflections=(False, False),
        translation: bool = True,
        rotation_90: bool = False,
        rotation_order: int = 0,
        cropkey: bool = True,
        crop_lim=(0.25, 0.75),
        bootstrap_loader: bool = False,
        twin_image: bool = False,
    ):
        super().__init__()

        self.params_num = params_num
        self.table_path = table_path
        self.transform = transform

        self.rotation = rotation
        self.rotation_90 = rotation_90
        self.rotation_order = rotation_order

        self.reflectionX = reflections[0]
        self.reflectionY = reflections[1]
        self.translation = translation

        self.cropkey = cropkey
        self.crop_lim = crop_lim

        self.bootstrap_loader = bootstrap_loader
        self.twin_image = twin_image

        with open(self.table_path, "r") as table_file:
            self.table = table_file.readlines()
        self.length = len(self.table)

        # Bootstrap: rimescola per ID finale
        if self.bootstrap_loader:
            table_old = self.table
            self.table = []
            id_set = set()
            for line in table_old:
                id_set.add(line.split()[-1])
            id_list = list(id_set)
            del id_set

            for _ in range(len(id_list)):
                index = torch.randint(0, len(id_list), (1,)).item()
                lines2append = [
                    line for line in table_old if line.split()[-1] == id_list[index]
                ]
                for line in lines2append:
                    self.table.append(line)

            print(f"Table length is {len(self.table)}; __len__ is {self.length}")
            self.length = len(self.table)

        # Controllo estensione
        print("Dataloader is checking data extension...", end="")
        first_ext = self.table[0].split()[0][-3:]
        for line in self.table:
            ext = line.split()[0][-3:]
            if ext != first_ext:
                raise NotImplementedError(
                    f"Extension {ext} found which is not consistent with previous {first_ext}... Aborting."
                )

        if first_ext == "png":
            self.extension = "png"
        elif first_ext == "npy":
            self.extension = "npy"
        else:
            raise NotImplementedError(
                f"Data format {first_ext} not implemented... Aborting."
            )

        print("DONE!")

    def __len__(self):
        return self.length

    def _parse_line(self, idx):
        table_line = self.table[idx]
        splitted_line = table_line.split()

        if self.params_num != 0:
            paths = splitted_line[: -self.params_num]
            params = [float(p) for p in splitted_line[-self.params_num :]]
        else:
            paths = splitted_line
            params = []

        return paths, params

    def pick_image(self, idx):
        """
        Carica e trasforma immagini .png.
        Ritorna: tensor (T, 1, H, W), lista di parametri (float)
        """
        paths, params = self._parse_line(idx)

        out_list = []
        for path in paths:
            image = make_square(
                Image.open(path), cropkey=self.cropkey, crop_lim=self.crop_lim
            )
            out_list.append(image)

        # Rotazioni
        if self.rotation:
            angle = 360 * torch.rand(1).item()
            for ii in range(len(out_list)):
                out_list[ii] = out_list[ii].rotate(
                    angle, PIL.Image.NEAREST, fillcolor=(0, 0, 0)
                )
        elif self.rotation_90:
            coin = torch.rand(1).item()
            if 0.0 <= coin < 0.25:
                rotation_key = PIL.Image.ROTATE_90
            elif 0.25 <= coin < 0.5:
                rotation_key = PIL.Image.ROTATE_180
            elif 0.5 <= coin < 0.75:
                rotation_key = PIL.Image.ROTATE_270
            else:
                rotation_key = None

            if rotation_key is not None:
                for ii in range(len(out_list)):
                    out_list[ii] = out_list[ii].transpose(rotation_key)
        elif self.rotation_order != 0:
            if not isinstance(self.rotation_order, int):
                raise ValueError(
                    "A non integer order of rotation was used for dataloaders"
                )
            angle = (
                360 / self.rotation_order * torch.randint(self.rotation_order, ()).item()
            )
            for ii in range(len(out_list)):
                out_list[ii] = out_list[ii].rotate(
                    angle, PIL.Image.NEAREST, fillcolor=(0, 0, 0)
                )

        # Riflessi
        if self.reflectionX:
            hor_flip = torch.rand(1) >= 0.5
            if hor_flip:
                for ii in range(len(out_list)):
                    out_list[ii] = out_list[ii].transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self.reflectionY:
            ver_flip = torch.rand(1) >= 0.5
            if ver_flip:
                for ii in range(len(out_list)):
                    out_list[ii] = out_list[ii].transpose(PIL.Image.FLIP_TOP_BOTTOM)

        # Transform (ToTensor + Resize ecc.)
        if self.transform is not None:
            for ii in range(len(out_list)):
                out_list[ii] = self.transform(out_list[ii])  # C x H x W

        # Traslazioni periodiche
        if self.translation:
            vertical = torch.randint(0, out_list[0].shape[-1], ()).item()
            horizontal = torch.randint(0, out_list[0].shape[-2], ()).item()
            for ii in range(len(out_list)):
                out_list[ii] = torch.roll(
                    out_list[ii], (vertical, horizontal), dims=(-1, -2)
                )

        # Aggiungi dimensione time per ciascun frame: (C,H,W) -> (1,C,H,W)
        for ii in range(len(out_list)):
            out_list[ii] = out_list[ii].unsqueeze(0)

        out_tensor = torch.cat(out_list, dim=0)  # T x C x H x W
        return out_tensor, params

    def pick_npy(self, idx):
        """
        Come pick_image ma per file .npy.
        Assumiamo che il .npy contenga già una singola immagine 2D.
        """
        paths, params = self._parse_line(idx)

        out_list = []
        for path in paths:
            image = torch.from_numpy(np.load(path)).float().unsqueeze(0).unsqueeze(
                0
            )  # 1 x 1 x H x W
            out_list.append(image)

        # Rotazioni
        if self.rotation:
            raise NotImplementedError(
                "Continuous rotation is not implemented yet for .npy datasets... Aborting."
            )
        elif self.rotation_90:
            coin = torch.rand(1).item()
            if 0.0 <= coin < 0.25:
                k = 1
            elif 0.25 <= coin < 0.5:
                k = 2
            elif 0.5 <= coin < 0.75:
                k = 3
            else:
                k = 0

            if k > 0:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.rot90(out_list[ii], k=k, dims=(-1, -2))
        elif self.rotation_order != 0:
            raise NotImplementedError(
                "Rotations other than square are not implemented yet for .npy datasets... Aborting."
            )

        # Riflessi
        if self.reflectionX:
            hor_flip = torch.rand(1) >= 0.5
            if hor_flip:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-2,))
        if self.reflectionY:
            ver_flip = torch.rand(1) >= 0.5
            if ver_flip:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-1,))

        out_tensor = torch.cat(out_list, dim=0)  # T x 1 x H x W
        return out_tensor, params

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.extension == "png":
            image1, params = self.pick_image(idx)
            pick_fun = self.pick_image
        elif self.extension == "npy":
            image1, params = self.pick_npy(idx)
            pick_fun = self.pick_npy
        else:
            raise NotImplementedError(
                "Dataset format is not .png nor .npy... Aborting."
            )

        image = image1

        if self.twin_image:
            good_image = False
            while not good_image:
                image2, _ = pick_fun(torch.randint(self.length, ()).item())

                vertical = torch.randint(image1.shape[-1], ()).item()
                horizontal = torch.randint(image1.shape[-2], ()).item()

                image2 = torch.roll(image2, (vertical, horizontal), dims=(-1, -2))
                trial_image = image1 + image2

                if torch.max(trial_image) - 1 <= 1e-3:
                    image = trial_image
                    good_image = True

        if self.params_num == 0:
            return image
        else:
            return image, params


class TabulatedSeries3D(torch.utils.data.Dataset):
    """
    Dataset 3D: righe con path a .npy 3D e, opzionalmente, parametri scalari.
    """

    def __init__(
        self,
        table_path,
        params_num: int = 0,
        reflections=(False, False, False),
        rotation_90: bool = False,
        bootstrap_loader: bool = False,
        size: int = 64,
    ):
        super().__init__()

        self.params_num = params_num
        self.table_path = table_path

        self.rotation_90 = rotation_90

        self.reflectionX = reflections[0]
        self.reflectionY = reflections[1]
        self.reflectionZ = reflections[2]

        self.bootstrap_loader = bootstrap_loader

        self.size = size

        if self.size == -1:
            self.resizer = lambda x: x
        else:
            self.resizer = lambda x: F.interpolate(
                x, size=(self.size, self.size, self.size)
            )

        with open(self.table_path, "r") as table_file:
            self.table = table_file.readlines()
        self.length = len(self.table)

        if self.bootstrap_loader:
            table_old = self.table
            self.table = []
            id_set = set()
            for line in table_old:
                id_set.add(line.split()[-1])
            id_list = list(id_set)
            del id_set

            for _ in range(len(id_list)):
                index = torch.randint(0, len(id_list), (1,)).item()
                lines2append = [
                    line for line in table_old if line.split()[-1] == id_list[index]
                ]
                for line in lines2append:
                    self.table.append(line)

            print(f"Table length is {len(self.table)}; __len__ is {self.length}")
            self.length = len(self.table)

    def __len__(self):
        return self.length

    def load_to_tensor(self, path):
        """
        Carica .npy 3D -> tensor 1x1xD xH xW, con eventuale resize.
        """
        out = torch.from_numpy(np.load(path)).float().unsqueeze(0).unsqueeze(0)
        out = self.resizer(out)
        return out

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        table_line = self.table[idx]
        splitted_line = table_line.split()

        if self.params_num != 0:
            paths = splitted_line[: -self.params_num]
            params = [float(p) for p in splitted_line[-self.params_num :]]
        else:
            paths, params = splitted_line, []

        out_list = [self.load_to_tensor(path) for path in paths]

        # Rotazioni 90° su tre assi
        if self.rotation_90:
            kx, ky, kz = torch.randint(low=0, high=4, size=(3,))
            for ii in range(len(out_list)):
                out = out_list[ii]
                out = torch.rot90(out, kx.item(), dims=(-1, -2))
                out = torch.rot90(out, ky.item(), dims=(-1, -3))
                out = torch.rot90(out, kz.item(), dims=(-2, -3))
                out_list[ii] = out

        # Riflessi
        if self.reflectionX:
            if torch.rand(1) <= 0.5:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-3,))
        if self.reflectionY:
            if torch.rand(1) <= 0.5:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-2,))
        if self.reflectionZ:
            if torch.rand(1) <= 0.5:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.flip(out_list[ii], dims=(-1,))

        out_tensor = torch.cat(out_list, dim=0)  # T x 1 x D x H x W

        if self.params_num == 0:
            return out_tensor
        else:
            return out_tensor, params


def give_dataloaders(args):
    """
    Restituisce dataloader 2D per train/valid/test (solo quelli presenti in args).
    """
    num_workers = args.nproc
    set_names = ["train_set", "valid_set", "test_set"]

    has_sets = any(hasattr(args, set_name) for set_name in set_names)
    if not has_sets:
        raise RuntimeError(
            "No dataset was detected in args. Check arguments are parsed correctly."
        )

    if args.size == -1:
        resizer = nn.Identity()
    else:
        resizer = transforms.Resize(args.size)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            resizer,
            transforms.ToTensor(),
        ]
    )

    dataloaders = {}
    master_path = args.paths["master"]

    for set_name in set_names:
        if not hasattr(args, set_name):
            continue

        set_path = getattr(args, set_name)

        bootstrap = getattr(args, "bootstrap", False)
        twin_image = getattr(args, "twin_image", False)
        params_num = getattr(args, "num_params", 0)

        dataset = TabulatedSeries(
            table_path=set_path,
            params_num=params_num,
            transform=transform,
            translation=args.translation,
            rotation=args.rotation,
            rotation_90=args.rotation90,
            reflections=(args.reflectionX, args.reflectionY),
            cropkey=args.crop,
            crop_lim=args.croplims,
            bootstrap_loader=bootstrap,
            twin_image=twin_image,
        )

        # Salva tabella effettiva
        if bootstrap:
            out_table_path = f"{master_path}/{set_name}_bootstrap.txt"
        else:
            out_table_path = f"{master_path}/{set_name}.txt"

        with open(out_table_path, "w+") as f:
            for line in dataset.table:
                f.write(line)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False if set_name in ("valid_set", "test_set") else True,
            num_workers=num_workers,
            pin_memory=True,
        )

        dataloaders[set_name] = dataloader

    return dataloaders


def give_3D_dataloaders(args):
    """
    Restituisce dataloader 3D per train/valid/test (solo quelli presenti in args).
    """
    num_workers = args.nproc
    set_names = ["train_set", "valid_set", "test_set"]

    has_sets = any(hasattr(args, set_name) for set_name in set_names)
    if not has_sets:
        raise RuntimeError(
            "No dataset was detected in args. Check arguments are parsed correctly."
        )

    dataloaders = {}
    master_path = args.paths["master"]

    for set_name in set_names:
        if not hasattr(args, set_name):
            continue

        set_path = getattr(args, set_name)

        bootstrap = getattr(args, "bootstrap", False)
        params_num = getattr(args, "num_params", 0)

        dataset = TabulatedSeries3D(
            table_path=set_path,
            params_num=params_num,
            rotation_90=args.rotation90,
            reflections=(args.reflectionX, args.reflectionY, args.reflectionZ),
            bootstrap_loader=bootstrap,
            size=args.size,
        )

        if bootstrap:
            out_table_path = f"{master_path}/{set_name}_bootstrap.txt"
        else:
            out_table_path = f"{master_path}/{set_name}.txt"

        with open(out_table_path, "w+") as f:
            for line in dataset.table:
                f.write(line)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False if set_name in ("valid_set", "test_set") else True,
            num_workers=num_workers,
            pin_memory=True,
        )

        dataloaders[set_name] = dataloader

    return dataloaders