
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, CamModel, CamModelsContainer
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,  CamModelParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from datetime import datetime
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2CAM, matrix_to_quaternion
from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings
import torch.nn.functional as F
import math

def training(dataset, opt, pipe, cm, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,cm)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    frames = scene.getTrainCameras()
    ##### TODO: get Camera model
    container = CamModelsContainer(len(frames), cm)
    d_cam2world_all = container.d_cam2world_all()
    models = container.get_models()
    ###
    
    ### temporary added
    torch.autograd.set_detect_anomaly(True)


    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        get_all = get_loss(models[viewpoint_cam.uid], viewpoint_cam, gaussians, gt_image, pipe, bg, opt)
        loss = get_all["losses"]
        Ll1 = get_all["Ll1"]
        viewspace_point_tensor = get_all["viewspace_points"]
        visibility_filter = get_all["visibility_filter"]
        radii = get_all["radii"]
        transformed_gaussians = get_all["gaussians"]

        loss.backward()

        iter_end.record()

        if iteration < opt.iterations:
            models[viewpoint_cam.uid].regularize(iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report_cam(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, get_render, models, gaussians, pipe, background)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                ##cam model save
                cam_model_path = os.path.join(dataset.model_path, "cam_model/iteration_{}".format(iteration))
                # print(srgs.model_path)
                container.save_cam(os.path.join(cam_model_path, "cam_model.pth"))

            # Densification
            ### TODO: change
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                #### cam_model update
                container.optimizer_step(viewpoint_cam.uid)
                # models[viewpoint_cam.uid].optimizer.step()
                # models[viewpoint_cam.uid].optimizer.zero_grad(set_to_none = True)

                if iteration % 100 == 0:
                    deltas = models[viewpoint_cam.uid].get_deltas()
                    # print("delta_r : ",deltas[0])
                    # print("delta_t : ",deltas[2])

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def update_viewpoint_cam(viewpoint_cam, model):

    ### consider as world 2 cam
    d_cam2world = model.d_cam2world

    # print(d_cam2world)
    w2c = torch.matmul(d_cam2world, viewpoint_cam.world_view_transform.transpose(0,1))
    # w2c = viewpoint_cam.w2c.cuda()

    # R = w2c[:3,:3].T
    # T = w2c[:3, 3]

    # world_view_transform = getWorld2View2CAM(R, T, viewpoint_cam.trans, viewpoint_cam.scale).cuda() # .transpose(0,1)

    return w2c

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M_")
        args.model_path = os.path.join("./output/", current_time_str+unique_str[0:3])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def training_report_cam(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, models, gaussians, pipe, background):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    image = torch.clamp(renderFunc(models[viewpoint.uid], viewpoint, gaussians, viewpoint.original_image.cuda(), pipe, background)["image"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def get_loss(model, viewpoint_camera, gaussians, gt_img, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None):

    transformed_gaussians = transform_to_frame(model, viewpoint_camera, gaussians)

    rendervar = transformed_params2rendervar(gaussians, transformed_gaussians, pipe)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    ###TODO: change world view transform
    world2c = torch.eye(4).cuda()
    ###
    
    projection_matrix = viewpoint_camera.projection_matrix
    # RGB Rendering
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        ## changed
        # viewmatrix=viewpoint_camera.world_view_transform,
        viewmatrix = world2c,
        # projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix = projection_matrix,
        sh_degree=gaussians.active_sh_degree,
        campos = world2c.inverse()[3, :3],
        # campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )


    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points.retain_grad()
    # im, radius = Renderer(raster_settings=raster_settings)(**rendervar)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    im, radii = rasterizer(
        means3D = rendervar['means3D'],
        means2D = screenspace_points,
        shs = rendervar['shs'],
        colors_precomp = rendervar['colors_precomp'],
        opacities = rendervar['opacities'],
        scales = rendervar['scales'],
        rotations = rendervar['rotations'],
        cov3D_precomp = rendervar['cov3D_precomp']
    )
    # variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # losses = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    Ll1 = l1_loss(im, gt_img)
    losses = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(im, gt_img))

    # weighted_losses = {k: v * 1 for k, v in losses.items()}
    # loss = sum(weighted_losses.values())


    ## viewspace_points: screenspace_points,
    ## visibility_filter : radii > 0,
    return { "losses": losses,
    "Ll1": Ll1,
    "viewspace_points": screenspace_points, # screenspace_points,
    "visibility_filter": radii > 0,
    "radii": radii,
    "gaussians": transformed_gaussians,
    }

def get_render(model, viewpoint_camera, gaussians, gt_img, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):

    transformed_gaussians = transform_to_frame(model, viewpoint_camera, gaussians)

    rendervar = transformed_params2rendervar(gaussians, transformed_gaussians, pipe)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    world2c = torch.eye(4).cuda()

    projection_matrix = viewpoint_camera.projection_matrix


    # RGB Rendering
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        ## changed
        # viewmatrix=viewpoint_camera.world_view_transform,
        viewmatrix = world2c,
        # projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix = projection_matrix,
        sh_degree=gaussians.active_sh_degree,
        campos = world2c.inverse()[3, :3],
        # campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # screenspace_points.retain_grad()
    # rendervar['means2D'].retain_grad()
    # im, radius = Renderer(raster_settings=raster_settings)(**rendervar)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    im, radii = rasterizer(
        means3D = rendervar['means3D'],
        means2D = rendervar['means2D'],
        shs = rendervar['shs'],
        colors_precomp = rendervar['colors_precomp'],
        opacities = rendervar['opacities'],
        scales = rendervar['scales'],
        rotations = rendervar['rotations'],
        cov3D_precomp = rendervar['cov3D_precomp']
    )
    # variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # losses = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    # weighted_losses = {k: v * 1 for k, v in losses.items()}
    # loss = sum(weighted_losses.values())


    ## viewspace_points: screenspace_points,
    ## visibility_filter : radii > 0,
    return { "image": im,
    "viewspace_points": screenspace_points, # screenspace_points,
    "visibility_filter": radii > 0,
    "radii": radii,
    }

def transform_to_frame(model, viewpoint_cam, gaussians):
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_gaussians: Transformed Gaussians (dict containing means3D & unnorm_rotations)
    """
    ### calculate world 2 cam
    # world2cam = viewpoint_cam.world_view_transform.transpose(0,1)

    world2cam = update_viewpoint_cam(viewpoint_cam, model)

    ### calculate w2c quaternion
    delta_r_ = matrix_to_quaternion(world2cam[:3,:3])

    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    transform_rots = True # Anisotropic Gaussians
    
    # Get Centers and Unnorm Rots of Gaussians in World Frame
    pts = gaussians.get_xyz
    rots = gaussians.get_rotation
    
    transformed_gaussians = {}
    # Transform Centers of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (world2cam @ pts4.T).T[:, :3]
    transformed_gaussians['means3D'] = transformed_pts
    # transformed_gaussians['means3D']= pts

    transformed_rots = quat_mult(delta_r_, rots)
    # transformed_rots = quat_mult(rots, delta_r_)
    transformed_gaussians['rotations'] = transformed_rots
    # transformed_gaussians['rotations'] = rots

    return transformed_gaussians

def transformed_params2rendervar(gaussians, transformed_gaussians, pipe, override_color=None):

    rendervar = {}
    cov3D_precomp = None
    shs=None
    colors_precomp=None

    # Initialize Render Variables
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
            dir_pp = (transformed_gaussians['means3D'] - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = gaussians.get_features
    else:
        colors_precomp = override_color

    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # screenspace_points.retain_grad()

    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        # 'means2D': torch.zeros_like(gaussians.get_xyz, requires_grad=True, device="cuda") + 0,
        'means2D': screenspace_points,
        'shs': shs,
        'colors_precomp': colors_precomp,
        # 'opacities': torch.sigmoid(params['logit_opacities']),
        'opacities': gaussians.get_opacity,
        # 'scales': torch.exp(log_scales),
        'scales': gaussians.get_scaling,
        'rotations': F.normalize(transformed_gaussians['rotations']),
        'cov3D_precomp': cov3D_precomp
    }
    return rendervar

def quat_mult(q1, q2):
    ## changed T -> mT
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    cm = CamModelParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000, 100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000, 100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), cm, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
