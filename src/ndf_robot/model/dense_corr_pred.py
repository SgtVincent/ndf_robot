import sys
import os
import cv2
import numpy as np
from sympy import denom
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
DM_ROOT = "/home/junting/repo/DenseMatching"
DM_PRETRAINED_DIR = os.path.join(DM_ROOT, "pre_trained_models")
sys.path.append(DM_ROOT)

from test_models import crop_image_according_to_mask, boolean_string, pad_to_same_shape
from utils_flow.visualization_utils import overlay_semantic_mask, make_sparse_matching_plot
from ndf_robot.utils import dense_corr_utils
from validation.utils import matches_from_flow
from utils_flow.util_optical_flow import flow_to_image
from model_selection import select_model
from utils_flow.pixel_wise_mapping import remap_using_flow_fields

class DenseCorrPredictor:

    def __init__(self, args, visualize=False) -> None:
        self.args = args
        self.network, _ = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, args.local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)
        self.visualize=visualize
        self.estimate_uncertainty = False # not using prob model 

    def predict(self, query_image, reference_image, query_mask, reference_mask, ref_pixels):
        # crop 
        ref_img_cropped, ref_cropbox = crop_image_according_to_mask(reference_image, reference_mask)
        ref_pixels_cropped = np.copy(ref_pixels)
        ref_pixels_cropped[:, 0] -= ref_cropbox[2]
        ref_pixels_cropped[:, 1] -= ref_cropbox[0]

        query_img_cropped, query_cropbox = crop_image_according_to_mask(query_image, query_mask)
        _, query_pts_cropped = self.generate_query_pixels(
            query_img_cropped, ref_img_cropped, ref_pixels_cropped)

        query_pixels = np.copy(query_pts_cropped)
        query_pixels[:, 0] += query_cropbox[2]
        query_pixels[:, 1] += query_cropbox[0]
        
        if self.visualize:

            confidence_values = np.ones(query_pixels.shape[0])
            color = cm.jet(confidence_values)
            matching = make_sparse_matching_plot(
            query_image, reference_image, query_pixels, ref_pixels, color, margin=10)
            cv2.imshow("PwarpC-dense matching", matching[:,:,::-1])
            cv2.waitKey(1000)
            # plt.figure(figsize=(16, 8))
            # plt.imshow(matching)
            # plt.axis('off')
            # # plt.savefig(os.path.join(save_dir_new,query_img_name + '_matching.png'))
            # plt.show()
        
        return query_pixels

    def generate_query_pixels(self, query_image, reference_image, ref_pixels):

        ref_image_shape = reference_image.shape[:2]

        # pad both images to the same size, to be processed by network
        query_image_, reference_image_ = pad_to_same_shape(
            query_image, reference_image)
        # convert numpy to torch tensor and put it in right format
        query_image_ = torch.from_numpy(
            query_image_).permute(2, 0, 1).unsqueeze(0)
        reference_image_ = torch.from_numpy(
            reference_image_).permute(2, 0, 1).unsqueeze(0)

        # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
        # specific pre-processing (/255 and rescaling) are done within the function.

        # pass both images to the network, it will pre-process the images and ouput the estimated flow
        # in dimension 1x2xHxW
        if self.estimate_uncertainty:
            if self.flipping_condition:
                raise NotImplementedError(
                    'No flipping condition with PDC-Net for now')

            estimated_flow, uncertainty_components = self.network.estimate_flow_and_confidence_map(
                query_image_, reference_image_, mode='channel_first')
            confidence_map = uncertainty_components['p_r'].squeeze(
            ).detach().cpu().numpy()
            confidence_map = confidence_map[: ref_image_shape[0],
                                            : ref_image_shape[1]]
        else:
            if self.args.flipping_condition and 'GLUNet' in self.args.model:
                estimated_flow = self.network.estimate_flow_with_flipping_condition(
                    query_image_, reference_image_, mode='channel_first')
            else:
                estimated_flow = self.network.estimate_flow(
                    query_image_, reference_image_, mode='channel_first')
        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        estimated_flow_numpy = estimated_flow_numpy[: ref_image_shape[0],
                                                    : ref_image_shape[1]]
        # removes the padding

        warped_query_image = remap_using_flow_fields(
            query_image, estimated_flow_numpy[:, :, 0],
            estimated_flow_numpy[:, :, 1]).astype(
            np.uint8)

        if self.estimate_uncertainty:
            color = [255, 102, 51]
            # fig, axis = plt.subplots(1, 5, figsize=(30, 30))

            # confident_mask = (confidence_map > 0.50).astype(np.uint8)
            # confident_warped = overlay_semantic_mask(
            #     warped_query_image, ann=255 - confident_mask*255, color=color)
            # axis[2].imshow(confident_warped)
            # axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
            #                   .format(self.args.model, self.args.pre_trained_model))
            # axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
            # axis[4].set_title('Confident regions')
        else:
            contact_pts_matching, query_pts = dense_corr_utils.match_contact_points(
                query_image, reference_image, estimated_flow, ref_pixels)
            # fig, axis = plt.subplots(1, 5, figsize=(30, 30))
            # axis[2].imshow(warped_query_image)
            # axis[2].set_title('Warped query image according to estimated flow by {}_{}'.format(
            #     self.args.model, self.args.pre_trained_model))
        # axis[0].imshow(query_image)
        # axis[0].set_title('Query image')
        # axis[1].imshow(reference_image)
        # axis[1].set_title('Reference image')

        # axis[3].imshow(flow_to_image(estimated_flow_numpy))
        # axis[3].set_title('Estimated flow {}_{}'.format(
        #     self.args.model, self.args.pre_trained_model))

        # axis[-1].imshow(contact_pts_matching)
        # axis[-1].set_title('Contact points matching')

        # if self.visualize:
        #     plt.show()

        # plt.close(fig)
        return estimated_flow, query_pts

if __name__ == "__main__":
    
    # paths definition 
    cam_idx = 0
    demo_file = "/home/junting/repo/ndf_robot/src/ndf_robot/data/demos/bottle/grasp_side_place_shelf_start_upright_all_methods_multi_instance/grasp_demo_2bbd2b37776088354e23e9314af9ae57.npz"
    demo_image_dir = "/home/junting/repo/ndf_robot/src/ndf_robot/data/demo_images_no_occlusion/bottle/grasp_side_place_shelf_start_upright_all_methods_multi_instance/grasp_demo_2bbd2b37776088354e23e9314af9ae57"
    demo_rgb_path = os.path.join(demo_image_dir, f"rgb_cam_{cam_idx}.png")
    demo_seg_path = os.path.join(demo_image_dir, f"seg_cam_{cam_idx}.png")
    
    query_image_dir = "/home/junting/repo/ndf_robot/src/ndf_robot/data/images/bottle/ed8aff7768d2cc3e45bcca2603f7a948"
    query_rgb_path = os.path.join(query_image_dir, f"rgb_cam_{cam_idx}.png")
    query_seg_path = os.path.join(query_image_dir, f"seg_cam_{cam_idx}.png")

    # reading data 
    demo_rgb_img = cv2.imread(demo_rgb_path)[..., ::- 1]
    demo_seg_img = cv2.imread(demo_seg_path, cv2.IMREAD_GRAYSCALE)
    
    # demo_mask = (demo_seg_img > 2) & (demo_seg_img < 255)
    demo_data = np.load(demo_file, allow_pickle=True)
    
    query_rgb_img = cv2.imread(query_rgb_path)[..., ::- 1]
    query_seg_img = cv2.imread(query_seg_path, cv2.IMREAD_GRAYSCALE)
    # query_mask = (query_seg_img > 2) & (query_seg_img < 255)

    int_mats = demo_data['intrinsic_matrices']
    ext_mats = demo_data['extrinsic_matrices']
    int_mat = int_mats[cam_idx]
    ext_mat = ext_mats[cam_idx]
    # in pybullet format https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.cb0co8y2vuvc
    contact_pts_pybullet = demo_data['contact_points']
    contact_pts = [pts_pybullet[5] for pts_pybullet in contact_pts_pybullet]
    print(f"contact points: {contact_pts}")
    contact_pixels = dense_corr_utils.project_points_to_pixels(contact_pts, ext_mat, int_mat)

    # initialize model 
    model_args = dense_corr_utils.default_dense_corr_args()
    dense_corr_pred = DenseCorrPredictor(model_args, visualize=True)

    # predict 
    dense_corr_pred.predict(query_rgb_img, demo_rgb_img, query_seg_img, demo_seg_img, contact_pixels)