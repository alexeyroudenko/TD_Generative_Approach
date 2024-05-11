import numpy as np
import random
# import dnnlib
import torch
import math
import requests
# from easing_functions import *
from typing import Any, List, Tuple, Union

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def easeInOutCubic(x):
    if x < 0.5:
        return 4 * x * x * x
    else:
        return 1 - math.pow(-2 * x + 2, 3) / 2

def easeInOutQuart(x):
    if x < 0.5:
        return 8 * x * x * x * x
    else:
        return 1 - math.pow(-2 * x + 2, 4) / 2


class SeedsThickenPsp:
    def __init__(self):
        self.seeds = []
        print("[SeedsThicken psp] init")
        for i in range(0, 100000):
            self.seeds.append(i)


class SeedWalkerPsp():
    def __init__(self, count_frames):

        self.use_explore = False
        # self.use_eyes_correction = False
        self.explore = None
        self.seeds_thicken = SeedsThickenPsp()
        #self.explore_util = SeedsThickenPsp()
        #print("load custom eyes normalisation")
        #self.explore_util.retrieve(0, sheet_list = "Eyes")
        # self.seeds_thicken.retrieve()
        self.interpolated = None
        self.weight = None
        self.seed_index = 0
        self.frame = None
        self.count_frames = count_frames
        self.start_frame = 0

        self.latent = EasyDict(x=0, y=0, anim=True, speed=0.25)
        self.latent_def = EasyDict(self.latent)
        self.step_y = 30
        self.gui_times = [float('nan')] * 60
        self.w0_seeds = []
        self.seeds_arr = []
        self.latent.x += 0.5 * 4e-2
        self.latent.y += 0.0 * 4e-2
        # self.eyes_distances = np.empty(1, dtype=float)
        self.easing = 0


    # def loaded(self):
    #     return f"seeds count: {len(self.seeds_thicken.seeds)}"

    def update(self, frame, interpolate=True, easing=False):
        self.frame = frame + 7777 * 0 - self.start_frame
        self.seed_index = math.floor(frame / self.count_frames)

        if interpolate:
            self.weight = 1.0 * (self.frame % self.count_frames) / self.count_frames
            v1 = 1.0 - self.weight

            if self.easing == 1:
                v1 = easeInOutCubic(1.0 - self.weight)
            elif self.easing == 2:
                v1 = easeInOutQuart(1.0 - self.weight)

            v2 = 1.0 - v1
            self.seeds_arr = [[self.seeds_thicken.seeds[self.seed_index], v1],
                              [self.seeds_thicken.seeds[self.seed_index + 1], v2]]
            self.interpolated = self.get_interpolated_random()
            data = self.get_seed_data()
            return
        else:
            self.weight = 0

        self.seeds_arr = [[self.seeds_thicken.seeds[self.seed_index], 1.0 - self.weight],
                          [self.seeds_thicken.seeds[self.seed_index + 1], self.weight]]
        self.interpolated = self.get_interpolated_random()
        data = self.get_seed_data()

    def update_by_seed(self, new_seed):
        # print("[SeedsWalker psp] update_by_seed", new_seed)
        self.frame = -1
        self.seed_index = new_seed
        self.weight = 0
        self.seeds_arr = [[self.seeds_thicken.seeds[self.seed_index], 1.0 - self.weight],
                          [self.seeds_thicken.seeds[(self.seed_index + 1)%len(self.seeds_thicken.seeds)], self.weight]]
        self.interpolated = self.get_interpolated_random()
        data = self.get_seed_data()

    def set_start_frame(self, frame):
        self.start_frame = frame

    def get_seed(self):
        return self.seed_index

    def get_seed_data(self):
        return self.seeds_arr

    def calc_tensor(self, src):
        dtype = src.dtype
        if not src.is_floating_point():
            src = src.to(torch.float32)
        torch.manual_seed(self.get_seed())
        t1 = torch.randn_like(src)
        torch.manual_seed(self.get_seed() + 1)
        t2 = torch.randn_like(src)

        seed_data = self.get_seed_data()
        out = t1 * seed_data[0][1] + t2 * seed_data[1][1]

        # if self.explore:
        #     out[0][self.explore:self.explore+10]=-2

        if out.dtype != dtype:
            out = out.to(dtype)
        return out

    def calc_tensor2(self, src):
        dtype = src.dtype
        if not src.is_floating_point():
            src = src.to(torch.float32)
        torch.manual_seed(self.get_seed())
        t1 = torch.randn_like(src)
        torch.manual_seed(self.get_seed() + 1)
        t2 = torch.randn_like(src)

        seed_data = self.get_seed_data()
        out = t1 * seed_data[0][1] + t2 * seed_data[1][1]
        if out.dtype != dtype:
            out = out.to(dtype)
        return out

    def info(self):
        exp = ""
        if self.explore:
            exp = f"{self.explore}:{self.explore}"
            return "count_frames:{}\neasing:{}\nexp:{}".format(self.count_frames, self.easing, exp)
        else:
            return ""

    def get_seed_string(self):
        return str(self.get_seed_data()[0][0])

    def get_string(self):
        seed_data = self.get_seed_data()
        data = seed_data
        if self.frame != -1:
            return "{}:{}-{}:{}-{}:{}".format(self.frame, self.seed_index, f'{data[0][0]:.0f}', f'{data[0][1]:.2f}',
                                              f'{data[1][0]:.0f}', f'{data[1][1]:.2f}')
        else:
            return "{}: {}\n".format("seed", self.seed_index) + self.info()

    def update_explore(self, i=0):
        self.explore = i

    def calc_numpy(self, n_outputs_to_generate):
        seed_data = self.get_seed_data()
        # data = seed_data
        s1 = seed_data[0][0]
        s2 = seed_data[1][0]
        np.random.seed(s1)
        torch.manual_seed(s1)
        t1 = np.random.randn(n_outputs_to_generate, 512).astype('float32')
        np.random.seed(s2)
        torch.manual_seed(s2)
        t2 = np.random.randn(n_outputs_to_generate, 512).astype('float32')
        out = t1 * seed_data[0][1] + t2 * seed_data[1][1]
        np.random.seed(0)
        torch.manual_seed(0)

        if self.explore and self.use_explore:
            out[0][self.explore:self.explore+1]=out[0][self.explore:self.explore+1]-10.0

        # print(self.get_string())
        return out

    def get_interpolated_random(self):
        seed_data1 = self.seeds_arr[0]
        seed1 = seed_data1[0]
        weight1 = seed_data1[1]
        random.seed(seed1)
        value1 = random.random() * weight1
        seed_data2 = self.seeds_arr[1]
        seed2 = seed_data2[0]
        weight2 = seed_data2[1]
        random.seed(seed2)
        value2 = random.random() * weight2

        interpolated = value1 + value2
        return interpolated

    def add_good_seed(self, seed):
        self.seeds_thicken.good_seeds.append(seed)
