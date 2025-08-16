import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

import ot
import collections
import os
import json
import torch
import math
import random as pyrandom 
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as dset
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.data import Subset
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import time

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def set_seed(seed: int):
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Seed set to {seed} for reproducibility.")

set_seed(12345)

SIDE_LEN = 28 #sidelength of image set to 28 pixels
NUM_ANG = 40 #discretization
DATA_PATH = '/n/home09/annabelma/Datasets/'

#### POLAR CLASSES ####
class Polar:
    def __init__(self, W,H,N):
        self.W = W #width of image
        self.H = H
        self.R = min(W,H) // 2 #radius
        self.N = N #discretization of angles

    def makePolar(self,img):

        polarImg = torch.zeros(self.R, self.N)
        def value(r, theta):
            x = int(r * math.cos(theta))
            y = int(r * math.sin(theta))
            new_x = x + self.W // 2
            new_y = y + self.H // 2

            if new_x < 0 or new_x >= self.W or new_y < 0 or new_y >= self.H:
                return 0  
            return img[new_y, new_x].item()
            
        def values(r):
            return [(r**1) * value(r, i * 2 * math.pi / self.N) for i in range(self.N)]
            
        return torch.tensor([values(r) for r in range(self.R)])
    
class PolarBispec:
    def __init__(self, W = SIDE_LEN,H = SIDE_LEN,N = NUM_ANG):
        self.W = W #width of image
        self.H = H
        self.R = min(W,H) // 2 #radius
        self.N = N #discretization of angles
        self.polar = Polar(W,H,N)
        #now we define the kernel for fourier transform
        real, imag = torch.zeros(N,N), torch.zeros(N,N)
        for k in range(N):
            for g in range(N):
                real[k,g] = math.cos(2*math.pi*(k*g/N))
                imag[k,g] = -math.sin(2*math.pi*(k*g/N))
        self.fou_r = real
        self.fou_i = imag
        #now define the kernel to do the f_{i+j}^T
        self.four = torch.complex(self.fou_r, self.fou_i)
        self.conj = (self.four[None,:,:]*self.four[:,None,:]).conj()
        self.conj_r = self.conj.real
        self.conj_i = self.conj.imag
    
    #calculate bispectrum, (W,H) goes in and returns (R,N,N)
    def bispec(self, img):
        pol_img = self.polar.makePolar(img)
        #fourier transform first
        rel = torch.einsum('rg,kg->rk', pol_img, self.fou_r)
        imag = torch.einsum('rg,kg->rk', pol_img, self.fou_i)
        four_trans = torch.complex(rel,imag)
        #conj term second
        r = torch.einsum('rg, ijg->rij', pol_img, self.conj_r)
        i = torch.einsum('rg, ijg->rij', pol_img, self.conj_i)
        conj_term = torch.complex(r,i)
        #put together the bispectrum
        outer_pro = four_trans[:,:,None] * four_trans[:,None,:]
        bisp = outer_pro * conj_term
        return bisp/torch.norm(bisp)


### PULLING DATA FROM SOURCES ####
def build_transform(tensor=True,
                    normalization=((0.1307,), (0.3081,)),
                    resize=None):
    ops = []
    if resize is not None:
        ops.append(transforms.Resize((resize, resize),
                                     interpolation=InterpolationMode.BILINEAR))
    if tensor:
        ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(*normalization))
    return transforms.Compose(ops)

mnist = torchvision.datasets.MNIST(DATA_PATH, download = True, transform = build_transform())
usps = torchvision.datasets.USPS(DATA_PATH, download = True, transform = build_transform(resize=28))
fashion_mnist = torchvision.datasets.FashionMNIST(DATA_PATH, download=True, transform = build_transform())
kmnist = torchvision.datasets.KMNIST(DATA_PATH, download=True, transform = build_transform())

### TRANSFORM DATA PIPELINE #### 
def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 3 else x.unsqueeze(0)

def rotate_degree(image_tensor: torch.Tensor, angle: float) -> torch.Tensor:
    x = _ensure_chw(image_tensor)
    return TF.rotate(
        x, angle,
        interpolation=InterpolationMode.BILINEAR,
        center=(SIDE_LEN/2, SIDE_LEN/2),
        expand=False
    )

def rotate_random_degree(image_tensor: torch.Tensor) -> torch.Tensor:
    x = _ensure_chw(image_tensor)
    angle = pyrandom.uniform(0.0, 360.0)    
    return TF.rotate(  
        x, angle,
        interpolation=InterpolationMode.BILINEAR,
        center=(SIDE_LEN/2, SIDE_LEN/2),
        expand=False
    )

def rotate_random_degree_discrete(image_tensor: torch.Tensor) -> torch.Tensor:
    ## rotate each image a multiple of 60 degrees that is not 0 !! this helps mitigate the problem of the image that is just not rotated being matched
    x = _ensure_chw(image_tensor)
    angle = pyrandom.choice([60, 120, 180, 240, 300])  
    return TF.rotate(
        x, angle,
        interpolation=InterpolationMode.BILINEAR,
        center=(SIDE_LEN/2, SIDE_LEN/2),
        expand=False
    )

def _label_tensor(ds: torch.utils.data.Dataset) -> torch.Tensor:
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            y = getattr(ds, attr)
            return y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    return torch.as_tensor([int(ds[i][1]) for i in range(len(ds))]) # fallback 

def _class_indices(y: torch.Tensor, exclude=()) -> dict[int, list[int]]:
    exclude = set(exclude or [])
    return {c: (y == c).nonzero(as_tuple=True)[0].tolist() if c not in exclude else []
            for c in range(10)}
    
def _assert_enough_per_class(y: torch.Tensor, N: int, *, exclude_labels=(), need_two_sets=False, num_classes: int = 10):
    exclude = set(exclude_labels or [])
    counts = torch.bincount(y.to(torch.long), minlength=num_classes)
    need = 2 * N if need_two_sets else N

    shortages = {c: int(counts[c].item())
                 for c in range(num_classes)
                 if c not in exclude and counts[c] < need}

    if shortages:
        kind = "2*N (label+test)" if need_two_sets else "N"
        raise ValueError(
            f"Not enough samples per class for {kind}={need}. "
            f"Shortages (class: available): {shortages}. "
            f"(Exclude: {sorted(exclude)})"
        )

    
def extract_small(data, angle=60, N=100, random=False):
    y = _label_tensor(data)
    exclude = set() # {6, 9} if data in (mnist, usps) else set() 
    _assert_enough_per_class(y, N, exclude_labels=exclude, need_two_sets=False)
    idxs = []
    for c, cls_idx in _class_indices(y, exclude).items():
        idxs.extend(cls_idx[:N])
    rot = rotate_random_degree if random else (lambda t: rotate_degree(t, angle))
    out = []
    for i in idxs:
        x, lab = data[i]
        out.append((torch.squeeze(rot(x)), int(lab)))
    return out

def extract_disjoint_sets(data, angle=60, N=100, random=False):
    y = _label_tensor(data)
    exclude = set() #{6, 9} if data in (mnist, usps) else set()
    _assert_enough_per_class(y, N, exclude_labels=exclude, need_two_sets=True)
    idx_by_class = _class_indices(y, exclude)

    label_idx, test_idx = [], []
    for c, idxs in idx_by_class.items():
        label_idx.extend(idxs[:N])       
        test_idx.extend(idxs[N:2*N])     

    labelset, testset = [], []
    for i in label_idx:
        x, lab = data[i]
        x0 = rotate_degree(x, 0)
        labelset.append((torch.squeeze(x0), int(lab)))
    for i in test_idx:
        x, lab = data[i]
        xr = rotate_random_degree(x) if random else rotate_degree(x, angle)
        testset.append((torch.squeeze(xr), int(lab)))

    return {'labelset': labelset, 'testset': testset}

def prepare_bispec(data, angle=60, N=100, random=False):
    if data in (mnist, usps):
        data = extract_small(data, angle, N=N, random=random)

    bis = PolarBispec()
    xs = [d[0] for d in data]
    ys = [int(d[1]) for d in data]

    bs = []
    for x in xs:
        b = bis.bispec(x)
        bs.append(torch.stack([b.real, b.imag]))
    return {'ys': ys, 'xs': xs, 'bs': bs}

## OT FUNCTIONS
def get_image_np(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    return img

def computeOT(data1, data2, feature = 'xs',  method = 'sinkhorn_epsilon_scaling', reg = 0.1, num_itermax = 100000, is_verbose = False):
    N = len(data1['ys'])
    M = ot.dist(torch.stack(data1[feature]).reshape((N, -1)), torch.stack(data2[feature]).reshape((N, -1)))

    M_normalized = M / M.max()

    M_normalized = M_normalized.to(torch.float64)
    
    a = torch.ones(N, dtype=torch.float64, device=M_normalized.device) * (1 / N)
    b = torch.ones(N, dtype=torch.float64, device=M_normalized.device) * (1 / N)
    if reg == None:
        W = ot.emd(a, b, M_normalized)
    else:
        W = ot.sinkhorn(a, b, M_normalized, reg = reg, method = method, numItermax=num_itermax, verbose=is_verbose)

    # ## SANITY CHECKS 
    # print("max:", torch.max(W))
    # print("min:", torch.min(W))

    # # row and column marginals 
    # print("Row sums (should each be 1/N):", torch.sum(W, axis=1))
    # print("Column sums (should each be 1/N):", torch.sum(W, axis=0))

    # print("total sum:", torch.sum(W))

    return W

def evalOT(data1, data2, feature = 'xs', method = 'sinkhorn_epsilon_scaling', reg = 0.1, num_itermax = 100000, is_verbose = False, show_img = False):
    N = len(data1['ys'])
    W = computeOT(data1, data2, feature = feature, method = method, reg = reg, num_itermax = num_itermax, is_verbose =is_verbose)

    device = W.device
    y1 = torch.as_tensor(data1['ys'], device=device, dtype=torch.long)
    y2 = torch.as_tensor(data2['ys'], device=device, dtype=torch.long)

    topk_idx = W.topk(10, dim=1).indices
    top1_idx = topk_idx[:, 0] 
    pred_top1 = y2[top1_idx] 
    hits_top1 = (pred_top1 == y1)
    accuracy = hits_top1.double().mean().item()
    accurate = int(hits_top1.sum().item())

    K = int(max(y1.max().item(), y2.max().item())) + 1
    cm_idx = y1 * K + pred_top1
    conf_mat = torch.bincount(cm_idx, minlength=K*K).reshape(K, K).double()

    print("number of accurate matches: ", accurate)
    print('fraction accurate:', accuracy)

    if show_img:
        print("ACCURATE MATCHES")
        for i in hits_top1.nonzero(as_tuple=False).squeeze(1).tolist():
            if i % 10 == 0:
                fig, axs = plt.subplots(1, 6, figsize=(10, 5))
                axs[0].imshow(data1['xs'][i], cmap='gray')
                for j in range(5):
                    axs[j+1].imshow(data2['xs'][topk_idx[i, j].item()], cmap='gray')
                for ax in axs: ax.axis('off')
                plt.tight_layout()
                plt.show()
                
        print("INACCURATE MATCHES")
        for i in (~hits_top1).nonzero(as_tuple=False).squeeze(1).tolist():
            if i % 10 == 0:
                fig, axs = plt.subplots(1, 6, figsize=(10, 5))
                axs[0].imshow(data1['xs'][i], cmap='gray')
                for j in range(5):
                    axs[j+1].imshow(data2['xs'][topk_idx[i, j].item()], cmap='gray')
                for ax in axs: ax.axis('off')
                plt.tight_layout()
                plt.show()
    
    rep_images_data1 = {}  # For rows (true labels)
    rep_images_data2 = {}  # For columns (predicted labels)
    
    for i in range(N):
        label1 = data1['ys'][i]
        label2 = data2['ys'][i]
        if label1 not in rep_images_data1:
            rep_images_data1[label1] = data1['xs'][i]
        if label2 not in rep_images_data2:
            rep_images_data2[label2] = data2['xs'][i]
    
    # heatmap
    df_cm = pd.DataFrame(conf_mat.numpy(), index=[i for i in range(10)],
                         columns=[i for i in range(10)])
    fig, ax = plt.subplots(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, ax=ax, cbar=True)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.tick_top()  
    
    for i in range(10):
        if i in rep_images_data1:
            img = get_image_np(rep_images_data1[i])
            im = OffsetImage(img, cmap='gray', zoom=0.6)
            # Place image at data coordinate (x=0, y=i+0.5), then offset left by 30 points
            ab = AnnotationBbox(im, (0, i+0.5), xybox=(-30, 0),
                                xycoords='data', boxcoords="offset points",
                                frameon=False)
            ax.add_artist(ab)
    
    for j in range(10):
        if j in rep_images_data2:
            img = get_image_np(rep_images_data2[j])
            im = OffsetImage(img, cmap='gray', zoom=0.6)
            ab = AnnotationBbox(im, (j+0.5, 0), xybox=(0, 20),
                                xycoords='data', boxcoords="offset points",
                                frameon=False)
            ax.add_artist(ab)
    
    plt.show()

    return accuracy, conf_mat

def test_ot_once(data1, data2, feature = 'xs', method = 'sinkhorn_epsilon_scaling', reg = 0.1, num_itermax = 100_000, is_verbose = True, return_confmat =True):
    N = len(data1['ys'])
    M = ot.dist(torch.stack(data1[feature]).reshape((N, -1)), torch.stack(data2[feature]).reshape((N, -1)))
    M_normalized = M / M.max()
    M_normalized = M_normalized.to(torch.float64)
    a = torch.ones(N, dtype=torch.float64, device=M_normalized.device) * (1 / N)
    b = torch.ones(N, dtype=torch.float64, device=M_normalized.device) * (1 / N)

    t0 = time.perf_counter()
    if reg == None:
        W, log = ot.emd(a, b, M_normalized, log=True)
        niter = None
        last_errs = None
    else:
        W, log = ot.sinkhorn(a, b, M_normalized, reg = reg, method = method, numItermax=num_itermax, verbose=is_verbose, log=True)
        errs = log.get("err", [])
        last5 = errs[-5:]
        last_errs = [float(e) if not torch.is_tensor(e) else float(e.item()) for e in last5]
        niter = int(log.get("niter", 0)) if "niter" in log else None
    
    elapsed = time.perf_counter() - t0

    device = W.device
    y1 = torch.as_tensor(data1['ys'], device=device, dtype=torch.long)
    y2 = torch.as_tensor(data2['ys'], device=device, dtype=torch.long)

    top1_idx = W.argmax(dim=1)   
    pred_top1 = y2[top1_idx] 
    hits_top1 = (pred_top1 == y1)
    accuracy = hits_top1.double().mean().item()
    accurate = int(hits_top1.sum().item())

    K = int(max(y1.max().item(), y2.max().item())) + 1
    cm_idx = y1 * K + pred_top1
    conf_mat = torch.bincount(cm_idx, minlength=K*K).reshape(K, K).double()

    per_class_percent_correct = []
    per_class_support = []

    for c in range(K):
        mask = (y1 == c)
        support = mask.sum().item()
        per_class_support.append(support)
        if support > 0:
            correct_c = hits_top1[mask].sum().item()
            pct_c = 100.0 * correct_c / support
        else:
            pct_c = 0.0
        per_class_percent_correct.append(pct_c)

    metrics = {
        "n": N,
        "correct": int(accurate),
        "accuracy": float(accuracy),
        "niter": None if niter is None else int(niter),
        "last_errs": None if last_errs is None else last_errs,
        "elapsed_sec": float(elapsed),
        "method": method,
        "reg": None if reg is None else float(reg),
        "feature": feature,
        "per_class_percent_correct": per_class_percent_correct,
        "per_class_support": per_class_support,
    }
    
    if return_confmat:
        metrics["confmat"] = conf_mat.cpu().tolist()
        
    return metrics
    