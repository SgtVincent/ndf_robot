import os
import os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import shutil
import open3d as o3d
import pybullet as p
from scipy.io import savemat, loadmat
import cv2

from scipy.interpolate import griddata
from scipy import ndimage
from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat
import matplotlib.pyplot as plt
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import np2img

from ndf_robot.opt.optimizer import OccNetOptimizer
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


def vispcd(pcd_list):
    o3d.visualization.draw_geometries(pcd_list)


def add_gaussian_shifts(depth, std=1 / 2.0):
    rows, cols = depth.shape
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates
    xx = np.linspace(0, cols - 1, cols)
    yy = np.linspace(0, rows - 1, rows)

    # get xpixels and ypixels
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp


def makepcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


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

                    cur_valid_dots = np.multiply(
                        np.where(window < invalid_disp_, dot_win, 0),
                        np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2:
                        accu = window[center, center]

                        assert (accu < invalid_disp_)

                        out_disp[r + center,
                                 c + center] = round((accu) * 8.0) / 8.0

                        interpolation_window = interpolation_map[r:r + size_filt_,
                                                                 c:c + size_filt_]
                        disp_data_window = out_disp[r:r + size_filt_,
                                                    c:c + size_filt_]

                        substitutes = np.where(
                            interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes ==
                                             1] = fill_weights[substitutes == 1]

                        disp_data_window[substitutes ==
                                         1] = out_disp[r + center, c + center]

    return out_disp


def noise_depth(depth):
    scale_factor = 100     # converting depth from m to cm
    focal_length = 480.0   # focal length of the camera used
    baseline_m = 0.075   # baseline in m
    invalid_disp_ = 99999999.9
    dot_pattern_ = cv2.imread(
        "/home/rui/projects/sp2_ws/simkinect/data/kinect-pattern_3x3.png", 0)
    depth_interp = add_gaussian_shifts(depth)
    # depth_interp = depth)

    disp_ = focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0) / 8.0

    out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

    depth = focal_length * baseline_m / out_disp
    # depth[out_disp == invalid_disp_] = 0
    (h, w) = np.shape(depth)
    # The depth here needs to converted to cms so scale factor is introduced
    # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects
    noisy_depth = (35130 / np.round((35130 / np.round(depth * scale_factor)
                                     ) + np.random.normal(size=(h, w)) * (1.0 / 6.0) + 0.5)) / scale_factor

    # Displaying side by side the orignal depth map and the noisy depth map with barron noise cvpr 2013 model
    return noisy_depth


def plot_colormap(data):

    fig, (ax0) = plt.subplots(1, 1)

    c = ax0.pcolor(data)
    ax0.set_title('default: no edges')

    fig.tight_layout()
    plt.show()


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
    config_fname = osp.join(
        path_util.get_ndf_config(),
        'eval_cfgs', args.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' % config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(
        path_util.get_ndf_config(),
        args.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict['shapenet_obj_dir']
    obj_class = global_dict['object_class']
    eval_save_dir = global_dict['eval_save_dir']

    eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
    eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)

    test_shapenet_ids = np.loadtxt(
        osp.join(
            path_util.get_ndf_share(),
            '%s_test_object_split.txt' % obj_class),
        dtype=str).tolist()
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

    if args.dgcnn:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='dgcnn',
            return_features=True,
            sigmoid=True,
            acts=args.acts).cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='pointnet',
            return_features=True,
            sigmoid=True).cuda()

    if not args.random:
        checkpoint_path = global_dict['vnn_checkpoint_path']
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        pass

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
            '/'.join(fname.split('/')[: -1]),
            'place_demo_' + shapenet_id_npz)
        if osp.exists(place_fname):
            grasp_demo_filenames.append(fname)
            place_demo_filenames.append(place_fname)
        else:
            log_warn(
                'Could not find corresponding placement demo: %s, skipping ' %
                place_fname)

    success_list = []
    place_success_list = []
    place_success_teleport_list = []
    grasp_success_list = []

    demo_shapenet_ids = []

    # get info from all demonstrations
    demo_target_info_list = []
    demo_rack_target_info_list = []

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

    grasp_demo_filenames = grasp_demo_filenames[:args.num_demo]
    place_demo_filenames = place_demo_filenames[:args.num_demo]

    max_bb_volume = 0
    place_xq_demo_idx = 0
    grasp_data_list = []
    place_data_list = []
    demo_rel_mat_list = []

    # load all the demo data and look at objects to help decide on query points
    for i, fname in enumerate(grasp_demo_filenames):
        print('Loading demo from fname: %s' % fname)
        grasp_demo_fn = grasp_demo_filenames[i]
        place_demo_fn = place_demo_filenames[i]
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)

        grasp_data_list.append(grasp_data)
        place_data_list.append(place_data)

        start_ee_pose = grasp_data['ee_pose_world'].tolist()
        end_ee_pose = place_data['ee_pose_world'].tolist()
        place_rel_mat = util.get_transform(
            pose_frame_target=util.list2pose_stamped(end_ee_pose),
            pose_frame_source=util.list2pose_stamped(start_ee_pose)
        )
        place_rel_mat = util.matrix_from_pose(place_rel_mat)
        demo_rel_mat_list.append(place_rel_mat)

        if i == 0:
            optimizer_gripper_pts, rack_optimizer_gripper_pts, shelf_optimizer_gripper_pts = process_xq_data(
                grasp_data, place_data, shelf=load_shelf)
            optimizer_gripper_pts_rs, rack_optimizer_gripper_pts_rs, shelf_optimizer_gripper_pts_rs = process_xq_rs_data(
                grasp_data, place_data, shelf=load_shelf)

            if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
                print('Using shelf points')
                place_optimizer_pts = shelf_optimizer_gripper_pts
                place_optimizer_pts_rs = shelf_optimizer_gripper_pts_rs
            else:
                print('Using rack points')
                place_optimizer_pts = rack_optimizer_gripper_pts
                place_optimizer_pts_rs = rack_optimizer_gripper_pts_rs

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            target_info, rack_target_info, shapenet_id = process_demo_data_shelf(
                grasp_data, place_data, cfg=None)
        else:
            target_info, rack_target_info, shapenet_id = process_demo_data_rack(
                grasp_data,
                place_data,
                cfg=None)

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            rack_target_info['demo_query_pts'] = place_optimizer_pts
        demo_target_info_list.append(target_info)
        demo_rack_target_info_list.append(rack_target_info)
        demo_shapenet_ids.append(shapenet_id)

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_optimizer_pts,
        query_pts_real_shape=place_optimizer_pts_rs,
        opt_iterations=args.opt_iterations)

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=optimizer_gripper_pts,
        query_pts_real_shape=optimizer_gripper_pts_rs,
        opt_iterations=args.opt_iterations)
    grasp_optimizer.set_demo_info(demo_target_info_list)
    place_optimizer.set_demo_info(demo_rack_target_info_list)

    # get objects that we can use for testing
    test_object_ids = []
    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(
        shapenet_obj_dir)] if obj_class == 'mug' else os.listdir(shapenet_obj_dir)
    for s_id in shapenet_id_list:
        valid = s_id not in demo_shapenet_ids and s_id not in avoid_shapenet_ids
        if args.only_test_ids:
            valid = valid and (s_id in test_shapenet_ids)

        if valid:
            test_object_ids.append(s_id)

    if args.single_instance:
        test_object_ids = [demo_shapenet_ids[0]]

    # reset
    robot.arm.reset(force_reset=True)
    robot.cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z],
        dist=0.9,
        yaw=45,
        pitch=-25,
        roll=0)
# cfg.N_CAMERAS
    cams = MultiCams(cfg.CAMERA, robot.pb_client, cfg.N_CAMERAS)
    cam_info = {}
    cam_info['pose_world'] = []
    for cam in cams.cams:
        cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

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

    if obj_class == 'mug':
        rack_link_id = 0
        shelf_link_id = 1
    elif obj_class in ['bowl', 'bottle']:
        rack_link_id = None
        shelf_link_id = 0

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        placement_link_id = shelf_link_id
    else:
        placement_link_id = rack_link_id

    def hide_link(obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    viz_data_list = []
    for iteration in range(0, 28):
        # load a test object
        obj_shapenet_id = test_object_ids[iteration]
        # obj_shapenet_id = random.sample(test_object_ids, 1)[0]
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        log_info(id_str)

        viz_dict = {}  # will hold information that's useful for post-run visualizations
        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        if obj_class in ['bottle', 'jar', 'bowl', 'mug']:
            upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(
            shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

        scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
        scale_default = cfg.MESH_SCALE_DEFAULT
        if False:
            mesh_scale = [
                np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3
        temp_dict2d = loadmat(
            '/media/rui/data/pwarpc_generate_data/evaluation/bottle/' +
            str(iteration).zfill(4) + '.mat')
        rand_yaw_T = temp_dict2d['rand_yaw_T']
        pos_rand = temp_dict2d['pos_rand']
        pos_rand = list(pos_rand[0])
        if args.any_pose:
            if obj_class in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                table_z]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(
                pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(
                util.list2pose_stamped(pose),
                util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[
                : 3], util.pose_stamped2list(pose_w_yaw)[
                3:]
        else:
            # pos_rand = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
            pose = util.list2pose_stamped(pos_rand + upright_orientation)
            # rand_yaw_T = util.rand_body_yaw_transform(pos_rand, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(
                pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[
                : 3], util.pose_stamped2list(pose_w_yaw)[
                3:]
        this_save_dict = {'rand_yaw_T': rand_yaw_T, 'pos_rand': pos_rand}
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
        robot.arm.move_ee_xyz([0, 0, 0.2])

        if args.any_pose:
            robot.pb_client.set_step_sim(True)
        if obj_class in ['bowl']:
            robot.pb_client.set_step_sim(True)

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

        o_cid = None
        if args.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        hide_link(table_id, rack_link_id)

        # get object point cloud
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []
        rack_pcd_pts = []

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(
            list(obj_pose_world[0]) + list(obj_pose_world[1]))
        viz_dict['start_obj_pose'] = util.pose_stamped2list(obj_pose_world)
        for i, cam in enumerate(cams.cams):
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(
                get_rgb=True, get_depth=True, get_seg=True)
            # cv2.imwrite(os.path.join('/media/rui/data/pwarpc_generate_data/evaluation/',args.object_class,str(iteration).zfill(4)+'.png'), rgb)
            # cv2.imwrite(os.path.join('/media/rui/data/pwarpc_generate_data/evaluation/', args.object_class,
            #                          str(iteration).zfill(4) + '_mask.png'), seg)
            #
            # savemat(os.path.join('/media/rui/data/pwarpc_generate_data/evaluation/',args.object_class,str(iteration).zfill(4)+'.mat'),this_save_dict)

            debug_depth = depth
            depth = noise_depth(depth)
            pts_raw, _ = cam.get_pcd(
                in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0,
                depth_max=np.inf)

            # get 3d info
            pt_temp = loadmat(
                '/home/rui/Downloads/bottle_pts2/bottle_pts2/ndf_from_WS_pfpascal_cropped/'
                + str(iteration).zfill(4) + '_contact_pts.mat')
            p3dP = cam.get_pix_3dpt(
                [int(pt_temp['contact_pts'][0, 1])],
                [int(pt_temp['contact_pts'][0, 0])],
                in_world=True, filter_depth=False, k=1, ktype='median',
                depth_min=None, depth_max=None)
            # flatten and find corresponding pixels in segmentation mask
            fl_d = depth.flatten()
            notdeleted_ind = fl_d != 0
            flat_seg = seg.flatten()
            flat_seg = flat_seg[notdeleted_ind]
            flat_depth = depth.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            table_inds = np.where(flat_seg == table_id)
            seg_depth = flat_depth[obj_inds[0]]

            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))
            table_pts = pts_raw[table_inds[0],
                                :][::int(table_inds[0].shape[0]/500)]
            table_pcd_pts.append(table_pts)

            if rack_link_id is not None:
                rack_val = table_id + ((rack_link_id+1) << 24)
                rack_inds = np.where(flat_seg == rack_val)
                if rack_inds[0].shape[0] > 0:
                    rack_pts = pts_raw[rack_inds[0], :]
                    rack_pcd_pts.append(rack_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        target_obj_pcd_obs = np.concatenate(
            obj_pcd_pts, axis=0)  # object shape point cloud
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(
            target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        if obj_class == 'mug':
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        if obj_class == 'bowl':
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(
                    bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id,
                    linkIndexA=i, linkIndexB=rack_link_id,
                    enableCollision=False)
                safeCollisionFilterPair(
                    bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id,
                    linkIndexA=i, linkIndexB=shelf_link_id,
                    enableCollision=False)
            safeCollisionFilterPair(
                bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1,
                linkIndexB=rack_link_id, enableCollision=False)
            safeCollisionFilterPair(
                bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1,
                linkIndexB=shelf_link_id, enableCollision=False)

        # optimize grasp pose
        pre_grasp_ee_pose_mats, best_idx = grasp_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True)
        pre_grasp_ee_pose = util.pose_stamped2list(
            util.pose_from_matrix(pre_grasp_ee_pose_mats[best_idx]))
        viz_dict['start_ee_pose'] = pre_grasp_ee_pose

        ########################### grasp post-process #############################
        new_grasp_pt = post_process_grasp_point(
            pre_grasp_ee_pose, target_obj_pcd_obs,
            thin_feature=(not args.non_thin_feature),
            grasp_viz=False, grasp_dist_thresh=args.grasp_dist_thresh,
            p2d_prior=p3dP)
        pre_grasp_ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # optimize placement pose
        rack_pose_mats, best_rack_idx = place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False)
        rack_relative_pose = util.pose_stamped2list(
            util.pose_from_matrix(rack_pose_mats[best_rack_idx]))

        ee_end_pose = util.transform_pose(
            pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
            pose_transform=util.list2pose_stamped(rack_relative_pose))
        pre_ee_end_pose2 = util.transform_pose(
            pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
        pre_ee_end_pose1 = util.transform_pose(
            pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf)

        ee_end_pose_list = util.pose_stamped2list(ee_end_pose)
        pre_ee_end_pose1_list = util.pose_stamped2list(pre_ee_end_pose1)
        pre_ee_end_pose2_list = util.pose_stamped2list(pre_ee_end_pose2)

        obj_start_pose = obj_pose_world
        obj_end_pose = util.transform_pose(
            pose_source=obj_start_pose, pose_transform=util.list2pose_stamped(
                rack_relative_pose))
        obj_end_pose_list = util.pose_stamped2list(obj_end_pose)
        viz_dict['final_obj_pose'] = obj_end_pose_list

        # save visualizations for debugging / looking at optimizaiton solutions
        if args.save_vis_per_model:
            analysis_dir = args.model_path + '_' + str(obj_shapenet_id)
            eval_iter_dir = osp.join(eval_save_dir, analysis_dir)
            if not osp.exists(eval_iter_dir):
                os.makedirs(eval_iter_dir)
            for f_id, fname in enumerate(grasp_optimizer.viz_files):
                new_viz_fname = fname.split('/')[-1]
                viz_index = int(new_viz_fname.split('.html')[0].split('_')[-1])
                new_fname = osp.join(eval_iter_dir, new_viz_fname)
                if args.save_all_opt_results:
                    shutil.copy(fname, new_fname)
                else:
                    if viz_index == best_idx:
                        shutil.copy(fname, new_fname)
            for f_id, fname in enumerate(place_optimizer.viz_files):
                new_viz_fname = fname.split('/')[-1]
                viz_index = int(new_viz_fname.split('.html')[0].split('_')[-1])
                new_fname = osp.join(eval_iter_dir, new_viz_fname)
                if args.save_all_opt_results:
                    shutil.copy(fname, new_fname)
                else:
                    if viz_index == best_rack_idx:
                        shutil.copy(fname, new_fname)

        viz_data_list.append(viz_dict)
        viz_sample_fname = osp.join(
            eval_iter_dir, 'overlay_visualization_data.npz')
        np.savez(viz_sample_fname, viz_dict=viz_dict,
                 viz_data_list=viz_data_list)

        # reset object to placement pose to detect placement success
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, table_id, -1,
                                placement_link_id, enableCollision=False)
        robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        robot.pb_client.reset_body(
            obj_id, obj_end_pose_list[: 3],
            obj_end_pose_list[3:])

        time.sleep(1.0)
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, '%d.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        safeCollisionFilterPair(obj_id, table_id, -1,
                                placement_link_id, enableCollision=True)
        robot.pb_client.set_step_sim(False)
        time.sleep(1.0)

        obj_surf_contacts = p.getContactPoints(
            obj_id, table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        place_success_teleport_list.append(place_success_teleport)

        time.sleep(1.0)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        robot.pb_client.reset_body(obj_id, pos, ori)

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):

            # reset everything
            robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(
                obj_id, table_id, -1, -1, enableCollision=True)
            if args.any_pose:
                robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            if args.any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                robot.pb_client.set_step_sim(False)
            robot.arm.go_home(ignore_physics=True)

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
            robot.arm.eetool.open()

            if jnt_pos is None or grasp_jnt_pos is None:
                jnt_pos = ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = ik_helper.get_feasible_ik(pre_grasp_ee_pose)

                if jnt_pos is None or grasp_jnt_pos is None:
                    jnt_pos = ik_helper.get_ik(pre_pre_grasp_ee_pose)
                    grasp_jnt_pos = ik_helper.get_ik(pre_grasp_ee_pose)

                    if jnt_pos is None or grasp_jnt_pos is None:
                        jnt_pos = robot.arm.compute_ik(
                            pre_pre_grasp_ee_pose[: 3],
                            pre_pre_grasp_ee_pose[3:])
                        # this is the pose that's at the grasp, where we just need to close the fingers
                        grasp_jnt_pos = robot.arm.compute_ik(
                            pre_grasp_ee_pose[: 3],
                            pre_grasp_ee_pose[3:])

            if grasp_jnt_pos is not None and jnt_pos is not None:
                if g_idx == 0:
                    robot.pb_client.set_step_sim(True)
                    robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    robot.arm.eetool.close(ignore_physics=True)
                    time.sleep(0.2)
                    grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(
                        eval_grasp_imgs_dir, '%d.png' % iteration)
                    np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    continue

                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    plan1 = ik_helper.plan_joint_motion(
                        robot.arm.get_jpos(), jnt_pos)
                    plan2 = ik_helper.plan_joint_motion(jnt_pos, grasp_jnt_pos)
                    if plan1 is not None and plan2 is not None:
                        grasp_plan = plan1 + plan2

                        robot.arm.eetool.open()
                        for jnt in plan1:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.025)
                        robot.arm.set_jpos(plan1[-1], wait=True)
                        for jnt in plan2:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.04)
                        robot.arm.set_jpos(grasp_plan[-1], wait=True)

                        # get pose that's straight up
                        offset_pose = util.transform_pose(
                            pose_source=util.list2pose_stamped(np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()),
                            pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
                        )
                        offset_pose_list = util.pose_stamped2list(offset_pose)
                        offset_jnts = ik_helper.get_feasible_ik(
                            offset_pose_list)

                        # turn ON collisions between robot and object, and close fingers
                        for i in range(p.getNumJoints(robot.arm.robot_id)):
                            safeCollisionFilterPair(
                                bodyUniqueIdA=robot.arm.robot_id,
                                bodyUniqueIdB=obj_id, linkIndexA=i,
                                linkIndexB=-1, enableCollision=True,
                                physicsClientId=robot.pb_client.get_client_id())
                            safeCollisionFilterPair(
                                bodyUniqueIdA=robot.arm.robot_id,
                                bodyUniqueIdB=table_id, linkIndexA=i,
                                linkIndexB=rack_link_id, enableCollision=False,
                                physicsClientId=robot.pb_client.get_client_id())

                        time.sleep(0.8)
                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[
                            0]
                        jnt_pos_before_grasp = robot.arm.get_jpos()
                        soft_grasp_close(robot, finger_joint_id, force=50)
                        safeRemoveConstraint(o_cid)
                        time.sleep(0.8)
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False)
                        time.sleep(0.8)

                        if g_idx == 1:
                            grasp_success = object_is_still_grasped(
                                robot, obj_id, right_pad_id, left_pad_id)

                            if grasp_success:
                                # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                                safeCollisionFilterPair(
                                    obj_id, table_id, -1, -1,
                                    enableCollision=True)
                                robot.arm.eetool.open()
                                p.resetBasePositionAndOrientation(
                                    obj_id, obj_pos_before_grasp, ori)
                                soft_grasp_close(
                                    robot, finger_joint_id, force=40)
                                robot.arm.set_jpos(
                                    jnt_pos_before_grasp, ignore_physics=True)
                                cid = constraint_grasp_close(robot, obj_id)

                        #########################################################################################################

                        if offset_jnts is not None:
                            offset_plan = ik_helper.plan_joint_motion(
                                robot.arm.get_jpos(), offset_jnts)

                            if offset_plan is not None:
                                for jnt in offset_plan:
                                    robot.arm.set_jpos(jnt, wait=False)
                                    time.sleep(0.04)
                                robot.arm.set_jpos(offset_plan[-1], wait=True)

                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False)
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, rack_link_id,
                            enableCollision=False)
                        time.sleep(1.0)

        if grasp_success:
            ####################################### get place pose ###########################################

            pre_place_jnt_pos1 = ik_helper.get_feasible_ik(
                pre_ee_end_pose1_list)
            pre_place_jnt_pos2 = ik_helper.get_feasible_ik(
                pre_ee_end_pose2_list)
            place_jnt_pos = ik_helper.get_feasible_ik(ee_end_pose_list)

            if place_jnt_pos is not None and pre_place_jnt_pos2 is not None and pre_place_jnt_pos1 is not None:
                plan1 = ik_helper.plan_joint_motion(
                    robot.arm.get_jpos(), pre_place_jnt_pos1)
                plan2 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos1, pre_place_jnt_pos2)
                plan3 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos2, place_jnt_pos)

                if plan1 is not None and plan2 is not None and plan3 is not None:
                    place_plan = plan1 + plan2

                    for jnt in place_plan:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.035)
                    robot.arm.set_jpos(place_plan[-1], wait=True)

                ################################################################################################################

                    # turn ON collisions between object and rack, and open fingers
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=True)
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, rack_link_id,
                        enableCollision=True)

                    for jnt in plan3:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.075)
                    robot.arm.set_jpos(plan3[-1], wait=True)

                    p.changeDynamics(
                        obj_id, -1, linearDamping=5, angularDamping=5)
                    constraint_grasp_open(cid)
                    robot.arm.eetool.open()

                    time.sleep(0.2)
                    for i in range(p.getNumJoints(robot.arm.robot_id)):
                        safeCollisionFilterPair(
                            bodyUniqueIdA=robot.arm.robot_id,
                            bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1,
                            enableCollision=False,
                            physicsClientId=robot.pb_client.get_client_id())
                    robot.arm.move_ee_xyz([0, 0.075, 0.075])
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=False)
                    time.sleep(4.0)

                    # observe and record outcome
                    obj_surf_contacts = p.getContactPoints(
                        obj_id, table_id, -1, placement_link_id)
                    touching_surf = len(obj_surf_contacts) > 0
                    obj_floor_contacts = p.getContactPoints(
                        obj_id, robot.arm.floor_id, -1, -1)
                    touching_floor = len(obj_floor_contacts) > 0
                    place_success = touching_surf and not touching_floor

        robot.arm.go_home(ignore_physics=True)

        place_success_list.append(place_success)
        grasp_success_list.append(grasp_success)
        log_str = 'Iteration: %d, ' % iteration
        kvs = {}
        kvs['Place Success'] = sum(
            place_success_list) / float(len(place_success_list))
        kvs['Place [teleport] Success'] = sum(
            place_success_teleport_list) / float(len(place_success_teleport_list))
        kvs['Grasp Success'] = sum(
            grasp_success_list) / float(len(grasp_success_list))
        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        id_str = ', shapenet_id: %s' % obj_shapenet_id
        log_info(log_str + id_str)

        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        if not osp.exists(eval_iter_dir):
            os.makedirs(eval_iter_dir)
        sample_fname = osp.join(eval_iter_dir, 'success_rate_eval_implicit.npz')
        np.savez(
            sample_fname,
            obj_shapenet_id=obj_shapenet_id,
            success=success_list,
            grasp_success=grasp_success,
            place_success=place_success,
            place_success_teleport=place_success_teleport,
            grasp_success_list=grasp_success_list,
            place_success_list=place_success_list,
            place_success_teleport_list=place_success_teleport_list,
            start_obj_pose=util.pose_stamped2list(obj_start_pose),
            best_place_obj_pose=obj_end_pose_list,
            ee_transforms=pre_grasp_ee_pose_mats,
            obj_transforms=rack_pose_mats,
            mesh_file=obj_obj_file,
            distractor_info=None,
            args=args.__dict__,
            global_dict=global_dict,
            cfg=util.cn2dict(cfg),
            obj_cfg=util.cn2dict(obj_cfg)
        )

        robot.pb_client.remove_body(obj_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    parser.add_argument('--demo_exp', type=str, default='debug_label')
    parser.add_argument('--exp', type=str, default='debug_eval')
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--opt_iterations', type=int, default=250)
    parser.add_argument('--num_demo', type=int, default=12,
                        help='number of demos use')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_vis_per_model', action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--noise_decay', type=float, default=0.75)
    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--dgcnn', action='store_true')
    parser.add_argument('--random', action='store_true',
                        help='utilize random weights')
    parser.add_argument('--early_weight', action='store_true',
                        help='utilize early weights')
    parser.add_argument('--late_weight', action='store_true',
                        help='utilize late weights')
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
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--start_iteration', type=int, default=0)

    args = parser.parse_args()

    signal.signal(signal.SIGINT, util.signal_handler)

    obj_class = args.object_class
    shapenet_obj_dir = osp.join(
        path_util.get_ndf_obj_descriptions(),
        obj_class + '_centered_obj_normalized')

    demo_load_dir = osp.join(
        path_util.get_ndf_data(),
        'demos', obj_class, args.demo_exp)

    expstr = 'exp--' + str(args.exp)
    modelstr = 'model--' + str(args.model_path)
    seedstr = 'seed--' + str(args.seed)
    full_experiment_name = '_'.join([expstr, modelstr, seedstr])
    eval_save_dir = osp.join(
        path_util.get_ndf_eval_data(),
        args.eval_data_dir, full_experiment_name)
    util.safe_makedirs(eval_save_dir)

    vnn_model_path = osp.join(
        path_util.get_ndf_model_weights(),
        args.model_path + '.pth')

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path
    )

    main(args, global_dict)
