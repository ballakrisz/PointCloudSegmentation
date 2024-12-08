import torch
import numpy as np

class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0.1, 0.1, 0.1],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data
    

class PointCloudJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.noise_std = jitter_sigma
        self.noise_clip = jitter_clip

    def __call__(self, data):
        if hasattr(data, 'keys'):
            noise = torch.randn_like(data['pos']) * self.noise_std
            data['pos'] += noise.clamp_(-self.noise_clip, self.noise_clip)
        else:
            noise = torch.randn_like(data) * self.noise_std
            data += noise.clamp_(-self.noise_clip, self.noise_clip)
        return data