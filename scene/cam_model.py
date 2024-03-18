import torch
from utils.graphics_utils import quaternion_to_matrix, matrix_to_quaternion
from utils.system_utils import mkdir_p
import utils.camera_utils as cam_util
import os
from plyfile import PlyData, PlyElement
import copy

class CamModelsContainer:
    def __init__(self, n_frames, args):
        self.models = [CamModel(n_frame=i, args=args) for i in range(n_frames)]
    
    # def obj_cam2world_all(self):
    #     return [model.obj_cam2world() for model in self.models]

    def d_cam2world_all(self):
        return [model.d_cam2world for model in self.models]

    def d_cam2world(self, idx):
        return self.models[idx].d_cam2world
    
    def regularize_all(self, iteration, idx):
        # for model in self.models:
        #     model.regularize(iteration)
        self.models[idx].regularize(iteration)
    
    def get_deltas_all(self):
        return [model.get_deltas() for model in self.models]
    
    def capture_all(self):
        return [model.capture() for model in self.models]
    
    # def load_cam_all(self, paths):
    #     # if len(paths) != len(self.models):
    #     #     raise ValueError("The number of paths must match the number of models.")
    #     for model, path in zip(self.models, paths):
    #         model.load_cam(path)

    def get_models(self):
        return self.models

    def optimizer_step(self, idx):
        for uid, model in enumerate(self.models):
            if uid == idx:
                model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        # self.models[idx].optimizer.step()
        # self.models[idx].optimizer.zero_grad()

    def extend(self, other_container):
        self.models.extend(other_container.models)

    def clear(self):
        self.models = []

    def save_cam(self, path):
        mkdir_p(os.path.dirname(path))

        models_state = self.capture_all()

        torch.save(models_state, path)

    def load_cam_all(self, path):
        models_state = torch.load(path)
        for model, state in zip(self.models, models_state):
            model.restore(state)

    def copy(self):
        return copy.deepcopy(self)

class CamModel():
    
    def __init__(self, n_frame, args):
        ## obj -> cam2world
        self.lr = args.cammodel_lr
        self.delta_r = torch.tensor([1., 0., 0., 0.], device='cuda', requires_grad=True)
        self.delta_s = torch.tensor([1., 1., 1.], device='cuda', requires_grad=True)
        self.delta_t = torch.tensor([0., 0., 0.], device='cuda', requires_grad=True)
        self.optimizer = torch.optim.Adam([self.delta_r, self.delta_s, self.delta_t], lr=self.lr)
        self.lambda_reg = args.cammodel_lambda_reg


    def scale_to_mat3(self):
        d_s = torch.eye(3)
        d_s[0, 0] = self.delta_s[0]
        d_s[1, 1] = self.delta_s[1]
        d_s[2, 2] = self.delta_s[2]
        return d_s

    #### TODO: cam2world -> world2cam
    @property
    def d_cam2world(self):
        d_cam2world = torch.eye(4).cuda()
        # d_s_mat = self.scale_to_mat3()
        d_r_mat = quaternion_to_matrix(self.delta_r)
        # d_sr_mat = torch.matmul(d_s_mat, d_r_mat)
        # d_box2world[:3, :3] = d_sr_mat
        d_cam2world[:3, :3] = d_r_mat
        d_cam2world[:3, 3] = self.delta_t
        return d_cam2world

    def regularize(self, iteration):
        loss = torch.norm(self.delta_r - torch.tensor([1., 0., 0., 0], device='cuda', requires_grad=False)) + torch.norm(self.delta_s - 1) + torch.norm(self.delta_t)
        loss = self.lambda_reg * loss
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def get_deltas(self):
        deltas = []
        with torch.no_grad():
            deltas.append(torch.norm(self.delta_r.detach().cpu() - torch.tensor([1., 0., 0., 0.])).item())
            deltas.append(torch.norm(self.delta_s.detach().cpu() - 1.).item())
            deltas.append(torch.norm(self.delta_t.detach().cpu()).item())
        return deltas

    def capture(self):
        return (
            self.delta_r,
            self.delta_s,
            self.delta_t,
        )

    def restore(self,state):
        # print(state)
        # self.delta_r = state['delta_r']
        # self.delta_s = state['delta_s']
        # self.delta_t = state['delta_t']
        self.delta_r = state[0]
        self.delta_s = state[1]
        self.delta_t = state[2]

class se3_CamModelsContainer:
    def __init__(self, n_frames, args):
        self.models = [se3_CamModel(n_frame=i, args=args) for i in range(n_frames)]
    

    def d_pose_all(self):
        return [model.d_pose for model in self.models]

    def d_pose(self, idx):
        return self.models[idx].d_pose
    
    def regularize_all(self, iteration, idx):
        # for model in self.models:
        #     model.regularize(iteration)
        self.models[idx].regularize(iteration)
    
    def capture_all(self):
        return [model.capture() for model in self.models]

    def get_models(self):
        return self.models

    def optimizer_step(self, idx):
        for uid, model in enumerate(self.models):
            if uid == idx:
                model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        # self.models[idx].optimizer.step()
        # self.models[idx].optimizer.zero_grad()

    def extend(self, other_container):
        self.models.extend(other_container.models)

    def clear(self):
        self.models = []

    def save_cam(self, path):
        mkdir_p(os.path.dirname(path))

        models_state = self.capture_all()

        torch.save(models_state, path)

    def load_cam_all(self, path):
        models_state = torch.load(path)
        for model, state in zip(self.models, models_state):
            model.restore(state)

    def copy(self):
        return copy.deepcopy(self)

class se3_CamModel():
    
    def __init__(self, n_frame, args):
        ## obj -> cam2world
        self.lr = args.cammodel_lr
        self.delta = torch.tensor([0., 0., 0., 0., 0., 0.], device='cuda', requires_grad=True)
        self.optimizer = torch.optim.Adam([self.delta], lr=self.lr)
        self.lambda_reg = args.cammodel_lambda_reg

    @property
    def d_pose(self):
        return cam_util.lie.se3_to_SE3(self.delta)

    def regularize(self, iteration):
        loss = torch.norm(self.delta - torch.tensor([1., 0., 0., 0., 0., 0.], device='cuda', requires_grad=False))
        loss = self.lambda_reg * loss
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad()

    def capture(self):
        return self.delta

    def restore(self, state):
        self.delta = state

## TODO: change to quat
class quat_CamModelsContainer:
    def __init__(self, n_frames, args):
        self.models = [se3_CamModel(n_frame=i, args=args) for i in range(n_frames)]
    

    def d_pose_all(self):
        return [model.d_pose for model in self.models]

    def d_pose(self, idx):
        return self.models[idx].d_pose
    
    def regularize_all(self, iteration, idx):
        # for model in self.models:
        #     model.regularize(iteration)
        self.models[idx].regularize(iteration)
    
    def capture_all(self):
        return [model.capture() for model in self.models]

    def get_models(self):
        return self.models

    def optimizer_step(self, idx):
        for uid, model in enumerate(self.models):
            if uid == idx:
                model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        # self.models[idx].optimizer.step()
        # self.models[idx].optimizer.zero_grad()

    def extend(self, other_container):
        self.models.extend(other_container.models)

    def clear(self):
        self.models = []

    def save_cam(self, path):
        mkdir_p(os.path.dirname(path))

        models_state = self.capture_all()

        torch.save(models_state, path)

    def load_cam_all(self, path):
        models_state = torch.load(path)
        for model, state in zip(self.models, models_state):
            model.restore(state)

    def copy(self):
        return copy.deepcopy(self)

class quat_CamModel():
    
    def __init__(self, n_frame, args):
        ## obj -> cam2world
        self.lr = args.cammodel_lr
        self.delta = torch.tensor([0., 0., 0., 0., 0., 0.], device='cuda', requires_grad=True)
        self.optimizer = torch.optim.Adam([self.delta], lr=self.lr)
        self.lambda_reg = args.cammodel_lambda_reg

    @property
    def d_pose(self):
        return cam_util.lie.se3_to_SE3(self.delta)

    def regularize(self, iteration):
        loss = torch.norm(self.delta - torch.tensor([1., 0., 0., 0., 0., 0.], device='cuda', requires_grad=False))
        loss = self.lambda_reg * loss
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad()

    def capture(self):
        return self.delta

    def restore(self, state):
        self.delta = state
