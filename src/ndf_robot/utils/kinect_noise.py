from importlib.resources import path
import numpy as np 
import cv2 
import os 
import os.path as osp
import matplotlib.pyplot as plt
from ndf_robot.utils import path_util

def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_):
    size_filt_ = 9

    xx = np.linspace(0, size_filt_ - 1, size_filt_)
    yy = np.linspace(0, size_filt_ - 1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf ** 2 + yf ** 2)
    vals = sqr_radius * 1.2 ** 2

    vals[vals == 0] = 1
    weights_ = 1 / vals

    fill_weights = 1 / (1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0

    disp_rows, disp_cols = disp.shape
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):
        for c in range(0, lim_cols):

            if dot_pattern_[r + center, c + center] > 0:

                # c and r are the top left corner
                window = disp[r:r + size_filt_, c:c + size_filt_]
                dot_win = dot_pattern_[r:r + size_filt_, c:c + size_filt_]

                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0
                n_thresh = np.sum(dot_win) / 255.0

                if n_valids > n_thresh / 1.2:

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window < invalid_disp_, dot_win, 0),
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2:
                        accu = window[center, center]

                        assert (accu < invalid_disp_)

                        out_disp[r + center, c + center] = round((accu) * 8.0) / 8.0

                        interpolation_window = interpolation_map[r:r + size_filt_, c:c + size_filt_]
                        disp_data_window = out_disp[r:r + size_filt_, c:c + size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes == 1] = fill_weights[substitutes == 1]

                        disp_data_window[substitutes == 1] = out_disp[r + center, c + center]

    return out_disp

def add_noise(depth):
    
    scale_factor  = 100     # converting depth from m to cm
    focal_length  = 480.0   # focal length of the camera used
    baseline_m    = 0.075   # baseline in m
    invalid_disp_ = 99999999.9
    dot_pattern_ = cv2.imread(
        osp.join(path_util.get_ndf_data(), "noise_model", "kinect-pattern_3x3.png"), 0)
    depth_interp = add_gaussian_shifts(depth)

    disp_ = focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0) / 8.0

    out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

    depth = focal_length * baseline_m / out_disp
    # depth[out_disp == invalid_disp_] = 0
    (h, w) = np.shape(depth)
    # The depth here needs to converted to cms so scale factor is introduced
    # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects
    def safe_round(x):
        x_r = np.round(x)
        x_r[x_r == 0] = 1
        return x_r
    
    noisy_depth = (35130 / safe_round(
        (35130 / safe_round(depth * scale_factor)) + np.random.normal(size=(h, w)) * (1.0 / 6.0) + 0.5)) / scale_factor

    # Displaying side by side the orignal depth map and the noisy depth map with barron noise cvpr 2013 model
    return noisy_depth


if __name__ == "__main__":

    cam_idx = 0
    demo_file = "/home/junting/repo/ndf_robot/src/ndf_robot/data/demos/bottle/grasp_side_place_shelf_start_upright_all_methods_multi_instance/grasp_demo_2bbd2b37776088354e23e9314af9ae57.npz"
    demo_image_dir = "/home/junting/repo/ndf_robot/src/ndf_robot/data/demo_images_no_occlusion/bottle/grasp_side_place_shelf_start_upright_all_methods_multi_instance/grasp_demo_2bbd2b37776088354e23e9314af9ae57"
    demo_rgb_path = os.path.join(demo_image_dir, f"rgb_cam_{cam_idx}.png")
    demo_depth_path = os.path.join(demo_image_dir, f"depth_cam_{cam_idx}.npy")

    demo_depth_img = np.load(demo_depth_path)
    depth_noisy = add_noise(demo_depth_img)    

    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(121)
    plt.imshow(demo_depth_img)
    plt.axis('off')
    plt.title("original depth")
    fig.add_subplot(122)
    plt.imshow(depth_noisy)
    plt.axis('off')
    plt.title("kinect noise depth")
    plt.show()

    
