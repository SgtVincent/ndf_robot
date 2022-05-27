import os
import os.path as osp
from pydoc import describe
import random
# from tkinter import Image
from PIL import Image
import numpy as np
import time
import signal
import torch
import argparse
import shutil
import copy
import open3d as o3d
import matplotlib.pyplot as plt
import pybullet as p

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat


import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import np2img

from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.opt.optimizer_multimode import MultiModalityOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import path_util
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world,
    constraint_grasp_open, safeCollisionFilterPair, object_is_still_grasped,
    get_ee_offset, post_process_grasp_point, process_demo_data_rack,
    process_demo_data_shelf, process_xq_data, process_xq_rs_data,
    safeRemoveConstraint,)


def main(args, global_dict):
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={
                  'self_collision': False, 'seed': args.seed})
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_ndf_config(),
                            'eval_cfgs', args.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' %
                 config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(
        path_util.get_ndf_config(), args.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict['shapenet_obj_dir']
    obj_class = global_dict['object_class']
    demo_imgs_save_dir = global_dict['demo_imgs_save_dir']

    test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(
    ), '%s_test_object_split.txt' % obj_class), dtype=str).tolist()
    if obj_class == 'mug':
        avoid_shapenet_ids = bad_shapenet_mug_ids_list + cfg.MUG.AVOID_SHAPENET_IDS
    elif obj_class == 'bowl':
        avoid_shapenet_ids = bad_shapenet_bowls_ids_list + cfg.BOWL.AVOID_SHAPENET_IDS
    elif obj_class == 'bottle':
        avoid_shapenet_ids = bad_shapenet_bottles_ids_list + cfg.BOTTLE.AVOID_SHAPENET_IDS
    else:
        test_shapenet_ids = []

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10
    p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
    p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)

    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    preplace_horizontal_tf_list = cfg.PREPLACE_HORIZONTAL_OFFSET_TF
    preplace_horizontal_tf = util.list2pose_stamped(
        cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
    preplace_offset_tf = util.list2pose_stamped(cfg.PREPLACE_OFFSET_TF)

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        load_shelf = True
    else:
        load_shelf = False

    # get filenames of all the demo files
    demo_filenames = os.listdir(global_dict['demo_load_dir'])
    assert len(
        demo_filenames), 'No demonstrations found in path: %s!' % global_dict['demo_load_dir']

    # strip the filenames to properly pair up each demo file
    grasp_demo_filenames_orig = [
        osp.join(global_dict['demo_load_dir'],
                 fn) for fn in demo_filenames if 'grasp_demo' in fn]  # use the grasp names as a reference

    place_demo_filenames = []
    grasp_demo_filenames = []
    for i, fname in enumerate(grasp_demo_filenames_orig):
        shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
        place_fname = osp.join(
            '/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
        if osp.exists(place_fname):
            grasp_demo_filenames.append(fname)
            place_demo_filenames.append(place_fname)
        else:
            log_warn(
                'Could not find corresponding placement demo: %s, skipping ' %
                place_fname)

    if args.n_demos > 0:
        gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
        gp_fns = random.sample(gp_fns, args.n_demos)
        grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
        grasp_demo_filenames, place_demo_filenames = list(
            grasp_demo_filenames), list(place_demo_filenames)
        log_warn('USING ONLY %d DEMONSTRATIONS' % len(grasp_demo_filenames))
        print(grasp_demo_filenames, place_demo_filenames)
    else:
        log_warn('USING ALL %d DEMONSTRATIONS' % len(grasp_demo_filenames))

    # grasp_demo_filenames = grasp_demo_filenames[:args.num_demo]
    # place_demo_filenames = place_demo_filenames[:args.num_demo]

    # reset
    robot.arm.reset(force_reset=True)
    robot.cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z],
        dist=0.9,
        yaw=45,
        pitch=-25,
        roll=0)

    cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
    cam_info = {}
    cam_info['pose_world'] = []
    for cam in cams.cams:
        cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

    def hide_link(obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    # load all the demo data into simulator and save rgb, depth, seg images
    for iteration, fname in enumerate(grasp_demo_filenames):

        ################ load demo metadata file ###############################
        print('Loading demo from fname: %s' % fname)
        # grasp demo files and place demo files are paired
        grasp_demo_fn = grasp_demo_filenames[iteration]
        place_demo_fn = place_demo_filenames[iteration]
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)

        # start_ee_pose = grasp_data['ee_pose_world'].tolist()
        # end_ee_pose = place_data['ee_pose_world'].tolist()
        # place_rel_mat = util.get_transform(
        #     pose_frame_target=util.list2pose_stamped(end_ee_pose),
        #     pose_frame_source=util.list2pose_stamped(start_ee_pose)
        # )
        # place_rel_mat = util.matrix_from_pose(place_rel_mat)

        # task_dict = {grasp_demo_fn.split(
        #     '/')[-1]: grasp_data, place_demo_fn.split('/')[-1]: place_data}
        task_dict = {grasp_demo_fn: grasp_data, place_demo_fn: place_data}
        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])
        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(
            path_util.get_ndf_descriptions(),
            'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(grasp_data['table_urdf'].item())
        table_id = robot.pb_client.load_urdf(tmp_urdf_fname,
                                             cfg.TABLE_POS,
                                             table_ori,
                                             scaling=cfg.TABLE_SCALING)

        if iteration == 0:
            if obj_class == 'mug':
                rack_link_id = 0
                shelf_link_id = 1
            elif obj_class in ['bowl', 'bottle']:
                rack_link_id = None
                shelf_link_id = 0

        # generate reference images for two tasks separately
        for task_file_path, task_data in task_dict.items():

            ###################### load a demo object ###########################
            task_fn = task_file_path.split('/')[-1]
            # obj_shapenet_id = random.sample(test_object_ids, 1)[0]
            try:  # some demo files have empty shapenet_id
                obj_shapenet_id = task_data['shapenet_id'][0]
            except:
                obj_shapenet_id = task_fn.split("_")[2].split(".")[0]
            id_str = f'Shapenet ID: {obj_shapenet_id}'
            log_info(id_str)

            viz_dict = {}  # will hold information that's useful for post-run visualizations

            if obj_class in ['bottle', 'jar', 'bowl', 'mug']:
                upright_orientation = common.euler2quat(
                    [np.pi / 2, 0, 0]).tolist()
            else:
                upright_orientation = common.euler2quat([0, 0, 0]).tolist()

            # for testing, use the "normalized" object
            obj_obj_file = osp.join(
                shapenet_obj_dir, obj_shapenet_id,
                'models/model_normalized.obj')
            obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

            scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
            scale_default = cfg.MESH_SCALE_DEFAULT
            if args.rand_mesh_scale:
                mesh_scale = [np.random.random() * (scale_high -
                                                    scale_low) + scale_low] * 3
            else:
                mesh_scale = [scale_default] * 3

            # region: object pose random initialization
            # if args.any_pose:
            #     if obj_class in ['bowl', 'bottle']:
            #         rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
            #         ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            #     else:
            #         rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
            #         ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            #     pos = [
            #         np.random.random() * (x_high - x_low) + x_low,
            #         np.random.random() * (y_high - y_low) + y_low,
            #         table_z]
            #     pose = pos + ori
            #     rand_yaw_T = util.rand_body_yaw_transform(
            #         pos, min_theta=-np.pi, max_theta=np.pi)
            #     pose_w_yaw = util.transform_pose(util.list2pose_stamped(
            #         pose), util.pose_from_matrix(rand_yaw_T))
            #     pos, ori = util.pose_stamped2list(
            #         pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
            # else:
            #     pos = [np.random.random() * (x_high - x_low) + x_low,
            #            np.random.random() * (y_high - y_low) + y_low, table_z]
            #     pose = util.list2pose_stamped(pos + upright_orientation)
            #     rand_yaw_T = util.rand_body_yaw_transform(
            #         pos, min_theta=-np.pi, max_theta=np.pi)
            #     pose_w_yaw = util.transform_pose(
            #         pose, util.pose_from_matrix(rand_yaw_T))
            #     pos, ori = util.pose_stamped2list(
            #         pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
            # endregion: object pose random initialization

            viz_dict['shapenet_id'] = obj_shapenet_id
            viz_dict['obj_obj_file'] = obj_obj_file
            if 'normalized' not in shapenet_obj_dir:
                viz_dict['obj_obj_norm_file'] = osp.join(
                    shapenet_obj_dir + '_normalized', obj_shapenet_id,
                    'models/model_normalized.obj')
            else:
                viz_dict['obj_obj_norm_file'] = osp.join(
                    shapenet_obj_dir, obj_shapenet_id,
                    'models/model_normalized.obj')
            viz_dict['obj_obj_file_dec'] = obj_obj_file_dec
            viz_dict['mesh_scale'] = mesh_scale

            # convert mesh with vhacd
            if not osp.exists(obj_obj_file_dec):
                # Volumetric Hierarchical Approximate Decomposition algorithm:
                # This can import a concave Wavefront .obj file and export a new Wavefront obj file
                # that contains the convex decomposed parts
                p.vhacd(
                    obj_obj_file,
                    obj_obj_file_dec,
                    'log.txt',
                    concavity=0.0025,
                    alpha=0.04,
                    beta=0.05,
                    gamma=0.00125,
                    minVolumePerCH=0.0001,
                    resolution=1000000,
                    depth=20,
                    planeDownsampling=4,
                    convexhullDownsampling=4,
                    pca=0,
                    mode=0,
                    convexhullApproximation=1
                )

            robot.arm.go_home(ignore_physics=True)
            # robot.arm.move_ee_xyz([0, 0, 0.2])

            # if args.any_pose:
            #     robot.pb_client.set_step_sim(True)
            # if obj_class in ['bowl']:
            # NOTE: if not set simulator to step mode, bottles will fall down
            robot.pb_client.set_step_sim(True)

            # read position and orientation from metadata
            pos, ori = task_data['obj_pose_world'][
                : 3], task_data['obj_pose_world'][
                3:]
            obj_id = robot.pb_client.load_geom(
                'mesh',
                mass=0.01,
                mesh_scale=mesh_scale,
                visualfile=obj_obj_file_dec,
                collifile=obj_obj_file_dec,
                base_pos=pos,
                base_ori=ori)
            p.changeDynamics(obj_id, -1, lateralFriction=0.5)

            if obj_class == 'bowl':
                safeCollisionFilterPair(
                    bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1,
                    linkIndexB=rack_link_id, enableCollision=False)
                safeCollisionFilterPair(
                    bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1,
                    linkIndexB=shelf_link_id, enableCollision=False)
                robot.pb_client.set_step_sim(False)

            if args.any_pose:
                robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(
                obj_id, table_id, -1, -1, enableCollision=True)
            p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
            time.sleep(1.5)

            # hide_link(table_id, rack_link_id)

            ############ teleport robot arm to ground truth pose ###########

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(
                    bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id,
                    linkIndexA=i, linkIndexB=-1, enableCollision=False,
                    physicsClientId=robot.pb_client.get_client_id())
                safeCollisionFilterPair(
                    bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id,
                    linkIndexA=i, linkIndexB=-1, enableCollision=False,
                    physicsClientId=robot.pb_client.get_client_id())

            task_jnt_pos = task_data['robot_joints']
            robot.arm.eetool.open()
            robot.pb_client.set_step_sim(True)
            robot.arm.set_jpos(task_jnt_pos, ignore_physics=True)
            robot.arm.eetool.close(ignore_physics=True)
            time.sleep(0.2)

            # get object point cloud
            # depth_imgs = []
            # seg_idxs = []
            # obj_pcd_pts = []
            # table_pcd_pts = []
            # rack_pcd_pts = []

            obj_pose_world = p.getBasePositionAndOrientation(obj_id)
            obj_pose_world = util.list2pose_stamped(
                list(obj_pose_world[0]) + list(obj_pose_world[1]))
            viz_dict['start_obj_pose'] = util.pose_stamped2list(obj_pose_world)

            ################# save images from simulator ########################
            if args.save_images or args.debug:

                task_imgs_save_dir = osp.join(
                    demo_imgs_save_dir, task_fn.split('.')[0])
                util.safe_makedirs(task_imgs_save_dir)
                
                # saved to demo files 
                int_mats = []
                ext_mats = []
                # rgb_images = []
                for i, cam in enumerate(cams.cams):
                    # get image and raw point cloud
                    rgb, depth, seg = cam.get_images(
                        get_rgb=True, get_depth=True, get_seg=True)
                    int_mats.append(cam.cam_int_mat)
                    ext_mats.append(np.linalg.inv(cam.cam_ext_mat))
                    # rgb_images.append(rgb)
                    

                    # Add noise
                    if args.depth_noise == "gaussian":
                        min_depth = depth.min()
                        # noise type: random noise of pixel-wise < 2% noise
                        # given Intel Realsense noise is < 2%
                        depth_noise = np.multiply((np.random.rand(
                            depth.shape[0],
                            depth.shape[1]) - 0.5),
                            depth) * 2 * args.gaussian_std
                        depth += depth_noise
                        depth[depth <= 0.0] = min_depth / 2.0  # min depth clip

                    if args.save_images:  # save images
                        Image.fromarray(rgb.astype(np.uint8)).save(
                            osp.join(task_imgs_save_dir, f"rgb_cam_{i}.png"))
                        # NOTE: depth info lost after saving to png
                        # Image.fromarray(depth.astype(np.uint8)).save(
                        #     osp.join(obj_image_save_dir, f"depth_cam_{i}.png"))
                        np.save(osp.join(task_imgs_save_dir, f"seg_cam_{i}.npy"), depth)
                        Image.fromarray(seg.astype(np.uint8)).save(
                            osp.join(task_imgs_save_dir, f"seg_cam_{i}.png"))
                    else:  # debug

                        pcd_pts, pcd_rgb = cam.get_pcd(
                            in_world=True, rgb_image=rgb, depth_image=depth,
                            depth_min=0.0, depth_max=np.inf)
                        pcd_o3d = o3d.t.geometry.PointCloud()
                        pcd_o3d.point["positions"] = o3d.core.Tensor(
                            pcd_pts, dtype=o3d.core.float32)
                        intrinsic_mat = o3d.core.Tensor(
                            cam.cam_int_mat, dtype=o3d.core.float32)
                        # NOTE: cam.cam_ext_mat: camera frame -> world frame
                        extrinsic_mat = o3d.core.Tensor(
                            np.linalg.inv(cam.cam_ext_mat),
                            dtype=o3d.core.float32)
                        # extrinsic_mat = o3d.core.Tensor(
                        #     cam.view_matrix, dtype=o3d.core.float32)

                        depth_project = pcd_o3d.project_to_depth_image(
                            width=depth.shape[1],
                            height=depth.shape[0],
                            intrinsics=intrinsic_mat,
                            extrinsics=extrinsic_mat,
                            depth_scale=cam.depth_scale,
                            depth_max=10.0).as_tensor().numpy().squeeze()

                        fig = plt.figure(figsize=(10, 10))
                        fig.add_subplot(1, 2, 1)
                        plt.imshow(depth)
                        fig.add_subplot(1, 2, 2)
                        plt.imshow(depth_project)
                        plt.show(block=True)

                    # region: save point cloud, not used now
                    # pts_raw, _ = cam.get_pcd(
                    #     in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0,
                    #     depth_max=np.inf)

                    # # flatten and find corresponding pixels in segmentation mask
                    # flat_seg = seg.flatten()
                    # flat_depth = depth.flatten()
                    # obj_inds = np.where(flat_seg == obj_id)
                    # table_inds = np.where(flat_seg == table_id)
                    # seg_depth = flat_depth[obj_inds[0]]

                    # obj_pts = pts_raw[obj_inds[0], :]
                    # obj_pcd_pts.append(util.crop_pcd(obj_pts))
                    # table_pts = pts_raw[table_inds[0],
                    #                     :][::int(table_inds[0].shape[0]/500)]
                    # table_pcd_pts.append(table_pts)

                    # if rack_link_id is not None:
                    #     rack_val = table_id + ((rack_link_id+1) << 24)
                    #     rack_inds = np.where(flat_seg == rack_val)
                    #     if rack_inds[0].shape[0] > 0:
                    #         rack_pts = pts_raw[rack_inds[0], :]
                    #         rack_pcd_pts.append(rack_pts)

                    # depth_imgs.append(seg_depth)
                    # seg_idxs.append(obj_inds)
                    # endregion

                # target_obj_pcd_obs = np.concatenate(
                #     obj_pcd_pts, axis=0)  # object shape point cloud

            ###### get closest points on object as contact query points #######
            
            if args.save_contact_pts:
                demo_update_dict = {}  # extra information to write back to demo files
                ee_pose_world = task_data['ee_pose_world'].tolist()

                gripper_closest_points = p.getClosestPoints(
                    bodyA=obj_id,
                    bodyB=robot.arm.robot_id,
                    distance=0.0025,
                    linkIndexA=-1,
                    linkIndexB=right_pad_id)

                # sort by distance to the object
                sorted(gripper_closest_points, key=lambda pt_info: pt_info[8])
                # gripper_contact_pose = copy.deepcopy(ee_pose_world)
                # if len(gripper_closest_points):
                #     for i, pt in enumerate(gripper_closest_points):
                #         print(pt[8])
                #     gripper_contact_pose[: 3] = np.asarray(
                #         gripper_closest_points[0][5])
                
                # write back updated to demo files 
                task_data_dict = dict(task_data)
                task_data_dict.update({
                    "contact_points": gripper_closest_points,
                    "intrinsic_matrices": int_mats,
                    "extrinsic_matrices": ext_mats,
                })
                np.savez(task_file_path, **task_data_dict)
            
            robot.arm.go_home(ignore_physics=True)
            # deleted unused code here
            robot.pb_client.remove_body(obj_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--demo_imgs_save_dir', type=str, default='demo_images')
    parser.add_argument('--demo_exp', type=str, default='test_bottle')
    parser.add_argument('--exp', type=str, default='debug_eval')
    parser.add_argument('--object_class', type=str, default='bottle')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_vis_per_model', action='store_true')
    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--rand_mesh_scale', action='store_true')
    parser.add_argument('--only_test_ids', action='store_true')
    parser.add_argument(
        '--all_cat_model', action='store_true',
        help='True if we want to use a model that was trained on multipl categories')
    parser.add_argument(
        '--n_demos', type=int, default=0,
        help='if some integer value greater than 0, we will only use that many demonstrations')
    parser.add_argument('--acts', type=str, default='all')
    parser.add_argument(
        '--old_model', action='store_true',
        help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    parser.add_argument(
        '--save_all_opt_results', action='store_true',
        help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)

    # custom arguments
    parser.add_argument('--depth_noise', type=str,
                        default="none", choices=["none", "gaussian"])
    parser.add_argument(
        '--gaussian_std', type=float, default=0.02,
        help="ratio of std dev of gaussian noise to mean on depth map)")
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--save_contact_pts', action='store_true')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, util.signal_handler)

    obj_class = args.object_class
    shapenet_obj_dir = osp.join(
        path_util.get_ndf_obj_descriptions(),
        obj_class + '_centered_obj_normalized')

    demo_load_dir = osp.join(path_util.get_ndf_data(),
                             'demos', obj_class, args.demo_exp)
    demo_imgs_save_dir = osp.join(
        path_util.get_ndf_data(),
        args.demo_imgs_save_dir, obj_class, args.demo_exp)

    expstr = 'exp--' + str(args.exp)
    if args.depth_noise == "gaussian":
        expstr = expstr + f"_gaussian_noise_{args.gaussian_std}"

    modelstr = 'model--' + str(args.model_path)
    seedstr = 'seed--' + str(args.seed)
    full_experiment_name = '_'.join([expstr, modelstr, seedstr])

    vnn_model_path = osp.join(
        path_util.get_ndf_model_weights(), args.model_path + '.pth')

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        demo_imgs_save_dir=demo_imgs_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path
    )
    if not args.save_images:
        print("[Warning]: --save_images flag is False, no images will be saved")
    if not args.save_contact_pts:
        print("[Warning]: --save_contact_pts flag is False, no images will be saved")

    main(args, global_dict)
