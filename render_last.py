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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, CamModelParams
from gaussian_renderer import GaussianModel
from scene import Scene, GaussianModel, CamModel, CamModelsContainer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2CAM, matrix_to_quaternion
from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings
import torch.nn.functional as F
import math

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_cam_models=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        gt = view.original_image[0:3, :, :]
        ## change gaussians
        if train_cam_models:
            # transformed_gaussians = transform_to_frame(train_cam_models[view.uid], view, gaussians, gaussians_grad=False, camera_grad=False)
            rendering = get_render(train_cam_models[view.uid], view, gaussians, gt, pipeline, background)["image"]
        else:
            rendering = render(view, gaussians, pipeline, background)["render"]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, cam_cfg : CamModelParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, cam_cfg, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train_noise", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        #### update view
        train_cams = scene.getTrainCameras()
        train_cam_models = load_cams(train_cams, cam_cfg, dataset.model_path, scene.loaded_iter)
        test_cams = scene.getTestCameras()

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, train_cams, gaussians, pipeline, background, train_cam_models)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, test_cams, gaussians, pipeline, background)


def load_cams(train_cams, cam_cfg, model_path, loaded_iter):
    model_path = os.path.join(model_path, "cam_model",
                                            "iteration_" + str(loaded_iter),
                                            "cam_model.pth")

    container = CamModelsContainer(len(train_cams), cam_cfg)
    container.load_cam_all(model_path)
    d_cam2world_all = container.d_cam2world_all()
    models = container.get_models()
    
    new_train_cams = []

    # for viewpoint_cam, model in zip(train_cams, models):

    #     new_viewpoint_cam = update_viewpoint_cam(viewpoint_cam, model)

    #     new_train_cams.append(new_viewpoint_cam)

    return models

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
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    cm = CamModelParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), cm, args.skip_train, args.skip_test)