import numpy as np
import pandas as pd
from fovi.sensing.coords import num_sampling_coords
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img-res', type=int, default=64)
parser.add_argument('--patch-size', type=int, default=8)
parser.add_argument('--fov', type=float, default=16)
args = parser.parse_args()

# which # radii to try, for each resolution in terms of the desired_res (img_res//patch_size)**2 (ie number of patches)
radii_by_res = {
    25: np.arange(2,10),
    36: np.arange(2,10),
    64: np.arange(2,11),
    100: np.arange(4,15),
    144: np.arange(6,17),
    256: np.arange(19,30),
}

def get_nearest_radii(desired_res, radii_by_res):
    """Map desired_res to the nearest key in radii_by_res."""
    keys = np.array(sorted(radii_by_res.keys()))
    nearest_key = keys[np.argmin(np.abs(keys - desired_res))]
    return radii_by_res[nearest_key]

desired_res = (args.img_res//args.patch_size)**2
radii = get_nearest_radii(desired_res, radii_by_res)

cmf_a_by_radii = {'radii':[], 'cmf_a':[], 'diff':[]}
for radii in radii:
    diffs = []
    cmf_as = np.logspace(-3, 3, 10000, base=10)
    for cmf_a in cmf_as:
        n = num_sampling_coords(args.fov, cmf_a, radii)
        diffs.append(np.abs(desired_res - n))
    
    cmf_a_by_radii['radii'].append(radii)
    cmf_a = cmf_as[np.argmin(diffs)]
    cmf_a_by_radii['cmf_a'].append(cmf_a)
    cmf_a_by_radii['diff'].append(np.min(diffs))
cmf_a_by_radii = pd.DataFrame(cmf_a_by_radii)
cmf_a_by_radii = cmf_a_by_radii[cmf_a_by_radii['diff'] == 0]

print(cmf_a_by_radii)