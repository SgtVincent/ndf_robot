from hashlib import new
import sys
import os
import numpy as np
import argparse
import torch
import scipy.ndimage.morphology
import matplotlib.pyplot as plt
DM_ROOT = os.environ.get('DENSE_MATCHING_ROOT')
DM_PRETRAINED_DIR = os.path.join(DM_ROOT, "pre_trained_models")
sys.path.append(DM_ROOT)

from validation.test_parser import define_model_parser, define_pdcnet_parser
from utils_flow.visualization_utils import overlay_semantic_mask, make_sparse_matching_plot
from validation.utils import matches_from_flow


def default_dense_corr_args(arg_input=""):

    parser = argparse.ArgumentParser(
        description='Test models on a pair of images')
    parser.add_argument('--model', type=str, default="PWarpCSFNet_WS",
                        help='Model to use')
    parser.add_argument('--pre_trained_model', type=str, help='Name of the pre-trained-model',
                        default="pfpascal")

    parser.add_argument('--flipping_condition', default=False, action="store_true",
                        help='Apply flipping condition for semantic data and GLU-Net-based networks? ')
    parser.add_argument('--optim_iter', type=int, default=3,
                        help='number of optim iter for global GOCor, when applicable')
    parser.add_argument('--local_optim_iter', dest='local_optim_iter', type=int, default=None,
                        help='number of optim iter for local GOCor, when applicable')
    parser.add_argument('--path_to_pre_trained_models', type=str, default=DM_PRETRAINED_DIR,
                        help='path to the folder containing the pre trained model weights, or '
                        'path to the model checkpoint.')

    # add subparser for model types
    subparsers = parser.add_subparsers(dest='network_type')
    define_pdcnet_parser(subparsers)
    args = parser.parse_args(arg_input)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu
    args.device = torch.device("cpu")

    return args


def match_contact_points(query_image, reference_image, estimated_flow, contact_pixels):
    mask = np.zeros(estimated_flow.shape[-2:], dtype=int)[np.newaxis, ...]
    mask_indices = contact_pixels.astype(int)[:, [1, 0]]  # (x,y) -> (row, col)
    mask[:, mask_indices[:, 0], mask_indices[:, 1]] = 1
    mask = torch.tensor(mask, device=estimated_flow.device) == 1
    # print(estimated_flow.shape)
    # print(mask.shape)
    mkpts_q, mkpts_r = matches_from_flow(estimated_flow, mask)
    # print(mkpts_q)
    # print(mkpts_r)

    confidence_values = np.ones(mkpts_q.shape[0])
    import matplotlib.cm as cm
    color = cm.jet(confidence_values)
    out = make_sparse_matching_plot(
        query_image, reference_image, mkpts_q, mkpts_r, color, margin=10)

    # plt.figure(figsize=(16, 8))
    # plt.imshow(out)
    # plt.show()
    return out, mkpts_q


def project_points_to_pixels(pts, ext_mat, int_mat):
    pts = np.array(pts)
    pts_homo = np.concatenate(
        [pts, np.ones((pts.shape[0], 1))], axis=1)  # (N,4)
    proj_mat = int_mat @ ext_mat[:3, :]
    pix_homo = proj_mat @ pts_homo.T  # (3, N)
    pixels = (pix_homo[:2, :] / pix_homo[2, :]).T  # (N, 2)

    return pixels


def filter_outliers(pts, mask, th=5):

    def bwdist_manhattan(mask, idx):
        pos_idx = np.argwhere(mask == 1)
        dists = np.abs((idx - pos_idx)).sum(1)
        closest_idx = np.argmin(dists)
        return pos_idx[closest_idx], dists[closest_idx]

    new_pts = []
    for pt in pts:  # (x, y) format
        idx, dist = bwdist_manhattan(mask, pt[::-1])  # (x, y) to (r, c)
        if dist <= th:
            new_pts.append(idx[::-1])  # (r, c) to (x, y)

    return np.stack(new_pts)
