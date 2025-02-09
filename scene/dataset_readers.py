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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getWorld2View2CAM
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import utils.camera_utils as cam_util
from scene.gaussian_model import BasicPointCloud
from scene.cam_model import CamModel, CamModelsContainer
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    w2c: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def getNerfppNormCAM(cam_info):
    def get_center_and_diag_torch(cam_centers):
        # cam_centers 리스트를 하나의 텐서로 결합
        cam_centers = torch.cat(cam_centers, dim=1)
        # 평균 카메라 중심 계산
        avg_cam_center = torch.mean(cam_centers, dim=1, keepdim=True)
        center = avg_cam_center
        # 각 카메라 중심까지의 거리 계산
        dist = torch.norm(cam_centers - center, dim=0, keepdim=True)
        # 최대 거리(대각선) 계산
        diagonal = torch.max(dist)
        return center.flatten(), diagonal.item()

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2CAM(cam.R, cam.T)
        C2W = torch.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag_torch(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def add_noise_to_rotation_torch(c2w, noise_scale_rotation=0.01, noise_scale_translation=0.01):
    # c2w에서 회전 부분만 추출
    rotation_matrix = c2w[:3, :3]
    
    # 회전 축과 각도에 노이즈 추가 (예시로 작은 값 사용)
    axis_noise = torch.randn(3) * noise_scale_rotation
    angle_noise = torch.randn(1) * noise_scale_rotation
    
    # 회전 축을 정규화
    axis = axis_noise / torch.norm(axis_noise)
    
    # 로드리게스 공식을 사용하여 노이즈가 추가된 회전 행렬 생성
    theta = torch.norm(angle_noise)
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    I = torch.eye(3)
    R_noise = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)
    
    # 새로운 회전 행렬을 원래 c2w 행렬에 적용
    new_rotation_matrix = torch.matmul(rotation_matrix, R_noise)
    new_c2w = c2w.clone()
    new_c2w[:3, :3] = new_rotation_matrix

    translation_noise = torch.randn(3) * noise_scale_translation
    new_c2w[:3, 3] += translation_noise

    return new_c2w

def add_noise_to_rotation(c2w, noise_scale_rotation=0.15, noise_scale_translation=0.15):
    rotation_matrix = c2w[:3, :3]
    
    # PyTorch를 사용하여 노이즈 생성
    axis_noise = (torch.randn(3) * noise_scale_rotation).numpy()
    angle_noise = (torch.randn(1) * noise_scale_rotation).numpy()
    
    # 회전 축 정규화
    axis = axis_noise / np.linalg.norm(axis_noise)
    
    # 로드리게스 공식을 사용하여 노이즈가 추가된 회전 행렬 생성
    theta = np.linalg.norm(angle_noise)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    I = np.eye(3)
    R_noise = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # 새로운 회전 행렬을 원래 c2w 행렬에 적용
    new_rotation_matrix = np.dot(rotation_matrix, R_noise)
    new_c2w = c2w.copy()
    new_c2w[:3, :3] = new_rotation_matrix

    # 변환 벡터에 노이즈 추가 (PyTorch에서 생성 후 NumPy로 변환)
    translation_noise = (torch.randn(3) * noise_scale_translation).numpy()
    new_c2w[:3, 3] += translation_noise

    return new_c2w

def add_se3_noise_to_rotation(c2w, noise_scale_rotation=0.05):

    se3_noise = torch.randn(1,6)*noise_scale_rotation

    pose_noise = cam_util.lie.se3_to_SE3(se3_noise)

    new_c2w = cam_util.pose.compose([pose_noise, torch.from_numpy(c2w).unsqueeze(dim=0).float()]).numpy()

    new_c2w = np.squeeze(new_c2w, axis=0)
    new_c2w = np.vstack([new_c2w, np.array([0, 0, 0, 1])])

    return new_c2w

def add_quat_noise_to_rotation(c2w, noise_scale_rotation=0.01, noise_scale_translation=0.01):

    ##TODO

    new_2w = c2w

    return new_c2w

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    torch.manual_seed(42)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            c2w = add_noise_to_rotation(c2w)
            # c2w = add_se3_noise_to_rotation(c2w[:3,:])

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], w2c=w2c))
            
    return cam_infos

def readCamerasFromTransformsCAM(path, transformsfile, white_background, cfg_cam, extension=".png", load_path=None, loaded_iter=None, is_test=None):
    cam_infos = []
    torch.manual_seed(42)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            c2w = torch.from_numpy(c2w).to(dtype=torch.float32)
            if is_test != True:
                c2w = add_noise_to_rotation_torch(c2w)
                # print("noise is aded")

            ### add cammodel
            ## TODO: c2w + d_c2w
            # c2w = c2w @ d_cam2world_all[idx]
            # new_c2w = torch.matmul(c2w, d_cam2world_all[idx])

            # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3,:3].T
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
 
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            # cammodel = models[idx]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], w2c=w2c))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval,  cfg_cam, extension=".png", load_path=None, loaded_iter=None, is_test=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfoCAM(path, white_background, eval, cfg_cam, extension=".png", load_path=None, loaded_iter=None, is_test=None):
    print("Reading Training Transforms")
    train_cam_infos= readCamerasFromTransformsCAM(path, "transforms_train.json", white_background, cfg_cam, extension, load_path, loaded_iter)
    print("Reading Test Transforms")
    test_cam_infos= readCamerasFromTransformsCAM(path, "transforms_test.json", white_background, cfg_cam, extension, load_path, loaded_iter, is_test=True)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        # train_container.extend(test_container)
        # test_container.clear()

    nerf_normalization = getNerfppNormCAM(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "CamOpt": readNerfSyntheticInfo
}

