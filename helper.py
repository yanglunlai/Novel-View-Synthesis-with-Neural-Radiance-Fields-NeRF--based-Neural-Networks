"""
Some helper functions for training a NeRF model

- posenc: 
    positional encoding for input positoin and view direction vectors

- get_rays:
    generate rays_origin and rays_direction vectors (H, W, channels)

- render_rays:
    generate H*W*N_sample points and run foward pass of NeRF model
    output each pixel's density and rgb color.

- Inference:
    Given a 4x4 transformation matrix(pose), generate a novel view.
    (Output an image with H*W pixels)

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from IPython.display import HTML
from base64 import b64encode


import torchvision
import torchvision.transforms as transforms
import json
import imageio
import natsort
from skimage.transform import resize
import os

def load_train_val_data(GOOGLE_DRIVE_DATASET_PATH, scene_name='lego'):
    """

    This function loads training data and returns training data(poses|ground-truth images), 
    testing data (testpose|testimg), and focal length
    
    args:
        GOOGLE_DRIVE_DATASET_PATH:
            The path you upload /nerf_synthetic.
            We use /drive/Umich/EECS598/EECS 598 mini project/Dataset/nerf_example/nerf_synthetic
        scene_name:
            The scene you want to train. e.g. /lego, /ship, /drum, ... etc.
    Returns
        poses: a numpy array of shape (N, 4, 4) where N is the number of training images in the /train folder.
        images: a numpy array of shape (N, H, W, 3) stores the ground-truth for training predictions from training poses. 
        testpose: a numpy array of shape (4, 4) 
        testimg:  a numpy array of shape (H, W, 3)
        focal: a float scalar
    """
            
    train_data_dir = os.path.join(GOOGLE_DRIVE_DATASET_PATH, scene_name,'train')
    val_data_dir = os.path.join(GOOGLE_DRIVE_DATASET_PATH, scene_name,'val')
    train_pose_dir = os.path.join(GOOGLE_DRIVE_DATASET_PATH, scene_name,'transforms_train.json')
    val_pose_dir = os.path.join(GOOGLE_DRIVE_DATASET_PATH, scene_name,'transforms_val.json')

    def imread(f,H=100,W = 100):
            if f.endswith('png'):
                return resize(imageio.imread(f, ignoregamma=True),(H,W))
            else:
                return resize(imageio.imread(f),(H,W))

    train_imgfiles = [os.path.join(train_data_dir, f) for f in natsort.natsorted(os.listdir(train_data_dir),reverse=False) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    images =  [imread(f)[...,:3] for f in train_imgfiles]
    images = np.stack(images, -1).transpose(3,0,1,2).astype(np.float32) 

    val_imgfiles = [os.path.join(val_data_dir, f) for f in natsort.natsorted(os.listdir(val_data_dir),reverse=False) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    test_imgs =  [imread(f)[...,:3] for f in val_imgfiles]
    test_imgs = np.stack(test_imgs, -1).transpose(3,0,1,2).astype(np.float32) 

    print("training images.shape ", images.shape)
    print("showing test_imgs[35]: ")
    plt.imshow(test_imgs[35])
    plt.show()

    # Load the poses 
    pose = json.load(open(train_pose_dir))
    test_pose = json.load(open(val_pose_dir))
    poses = np.zeros((100,4,4), dtype=np.float32)
    test_poses = np.zeros((100,4,4),dtype=np.float32)
    for i in range(100):
        poses[i,...] = np.array(pose['frames'][i]['transform_matrix']).reshape(4,4)
        test_poses[i,...] = np.array(test_pose['frames'][i]['transform_matrix']).reshape(4,4)

    H, W = images.shape[1:3]
    testimg, testpose = test_imgs[35],test_poses[35]
    # data = np.load('tiny_nerf_data.npz')
    # focal = data['focal']
    
    focal = 138.88887889922103 

    return poses, images, testpose, testimg, focal

def posenc(x,L_embed):
    """
    This function performs positional encodings
    Uses alternative sine/cosine functions to encode input positions/directions

    args: 
      x: tensor of shape(3,1) [x, y, z] position
      L_embed: int that specifies the frequency of positional encoding
    """
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            # alternatively append sin and cosine
            rets.append(fn(2.**i * x))
    # return tf.concat(rets, -1)
    return torch.concat(rets, -1)


def get_rays(H, W, focal, c2w):
    """
    This function generates rays origins & directions
    given focal length and a camera to world tranformation matrix

    args:
      H: the height of image
      W: the weight of image
      focal: the focal length camera (unit:mm)
      c2w: the transformation matrix of camera frame to world frame

    returns:
      rays_o: the camera pinhole coordinate w.r.t world frame
      rays_d: the directional vector that of each sensor pixel pass through the pinhole o
    """
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1).to('cuda:0')  # (H,W,3) # pin-hole model to get 2D viewing direction (theta, phi, -1) ,-1 means the camera is put at z=-1
    # rays_direction :
    rays_d = torch.sum(dirs[..., np.newaxis, :] * torch.tensor(c2w[:3,:3]).to('cuda:0'), -1)#.to('cuda:0') # dirs:(100,100,3) -np.newaxis-> (100, 100, 1, 3) -> *mm(c2w) (broadcast) -> (100, 100, 3, 3) -reduce_sum-> (100, 100, 3)
    # rays_origin
    rays_o = torch.broadcast_to(torch.tensor(c2w[:3,-1]), rays_d.shape).to('cuda:0') # c2w:(3,) -> (100,100,3)
    return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    """
    This function render ray sampling points and passes them to the NeRF model
    using 'mini-batch'. The chunk size is specified but feel free to modify 
    if you want.

    If rand is True: for each sample point along each ray, we add a noise 
    with random uniform distribution on it. (So the distances between points are not the identical)

    args:
      network_fn: model
      rays_o: the camera pinhole coordinate w.r.t world frame
      rays_d: the directional vector that of each sensor pixel pass through the pinhole o
      near: the heuristic start point that contains rendered objects
      far: the heuristic end point that contains rendered objects
    return 
      rgb_map: returned rgb value for each pixel in the rendered images
      depth_map: returned depth value for each pixel in the rendered images
      acc_map:
    """
    def batchify(fn, chunk=1024*32):
        return lambda inputs : torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    # Compute 3D query points
    """
    Note:
    # z_vals: (100,100,64)
    broadcasting: z_vals(the starting point at each interval), torch.rand (the delta value in each interval)
    """
    z_vals = torch.linspace(start=near, end=far, steps=N_samples).repeat(rays_o.shape[0], rays_o.shape[1], 1).to('cuda:0') 
    if rand:
      z_vals += (torch.rand(rays_o.shape[0], rays_o.shape[1], N_samples) * (far-near)/N_samples).to('cuda:0')
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None].to('cuda:0') # sample point positions (100,100,64,3)
    dirs = rays_d.unsqueeze(2).repeat(1,1,N_samples,1) # sampled point directional vectors (100,100,64,3)
    
    # concatenate the pts and dirs to one input data
    pts_flat = pts.reshape(-1,3) # (chunk,3)
    pts_flat = posenc(pts_flat, L_embed=10) # (chunk, 3+2*pts_L_embed*3)
    dirs_flat  = dirs.reshape(-1,3) # (chunk,3) 
    dirs_flat = posenc(dirs_flat, L_embed=4)# (chunk, 3+2*dirs_L_embed*3)
    input = torch.cat([pts_flat,dirs_flat], -1) # (chunk, 3+ 2*dirs_L_embed*3+ 2*pts_L_embed*3)

    # Run network
    raw = batchify(network_fn)(input)
    raw = raw.reshape((pts.shape[0], pts.shape[1], pts.shape[2], 4))  

    # Compute opacities and colors
    sigma_a = raw[...,3].to('cuda:0')
    rgb = raw[...,:3].to('cuda:0')    

    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.broadcast_to(torch.tensor([1e10]), z_vals[...,:1].shape).to('cuda:0')], -1) 
    alpha = 1.-torch.exp(-sigma_a * dists)  .to('cuda:0')
    weights = alpha * torch.cumprod(1.-alpha + 1e-10, dim = -1).to('cuda:0')
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2) 
    depth_map = torch.sum(weights * z_vals, -1) 
    acc_map = torch.sum(weights, -1)
    return rgb_map, depth_map, acc_map

def Inference(model, focal, N_samples, H,W, testpose, testimg, filename):
    """
    This function predicts(infers) an image given a camera pose(testpose) 
    ans save this picture as filename.png
    args:
        model: The NeRF model
        focal: A float scalar specifying the focal length of the camera.
        N_samples: An integer specifying the number of sample points each ray.
        H: An integer specifying the height of the image
        W: An integer specifying the width of the image
        testpose: A numpy array of shape (4, 4) indicating the input test pose.
        testimg: The ground-truth of the predicted test-img
        filename: The image name without .png
    Returns
        No return values
    """    
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, torch.tensor(testpose[:3,:4]).to('cuda:0'))
        rgb, depth, acc = render_rays_video(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        img = np.clip(rgb.detach().cpu().numpy(),0,1)
        # ground-truth image for visualization
        # gt_image = test_images[pose_index]
        gt_image = testimg
        plt.figure(3, figsize=(20,6))
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(gt_image)
        plt.savefig(filename)
        plt.show()
    


def render_rays_video(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    """
    Same as "render_rays" above but reduce the "chunk" size in batchify.
    This is for "rendering video since CUDA has limited memory.

    args:
        network_fn: the NeRF model object (forward function)
        rays_o: a tensor of shape (H,W,3) where H,W is images.shape. pinhole origin
        rays_d: a tensor of shape (H,W,3) where H,W is images.shape. The directions of rays
        near: tnear in the paper. The startpoint of the integral
        far:  tfar in the paper. The endpoint of the integral
        N_samples: an integer: the number of sample points per ray.

    Returns

        rgb_map: a tensor of shape (H*W*N_samples, 3) stores the rgb values of each pixel
        depth_map:
        acc_map:
    """
    def batchify(fn, chunk=1024*32):
        return lambda inputs : torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    # Compute 3D query points
    z_vals = torch.linspace(start=near, end=far, steps=N_samples).repeat(rays_o.shape[0], rays_o.shape[1], 1).to('cuda:0') 
    if rand:
      z_vals += (torch.rand(rays_o.shape[0], rays_o.shape[1], N_samples) * (far-near)/N_samples).to('cuda:0')
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None].to('cuda:0') # sample point positions (100,100,64,3)
    dirs = rays_d.unsqueeze(2).repeat(1,1,N_samples,1) # sampled point directional vectors (100,100,64,3)
    
    # concatenate the pts and dirs to one input data
    pts_flat = pts.reshape(-1,3) # (chunk,3)
    pts_flat = posenc(pts_flat, L_embed=10) # (chunk, 3+2*pts_L_embed*3)
    dirs_flat  = dirs.reshape(-1,3) # (chunk,3) 
    dirs_flat = posenc(dirs_flat, L_embed=4)# (chunk, 3+2*dirs_L_embed*3)
    input = torch.cat([pts_flat,dirs_flat], -1) # (chunk, 3+ 2*dirs_L_embed*3+ 2*pts_L_embed*3)

    # Run network
    raw = batchify(network_fn)(input)
    raw = raw.reshape((pts.shape[0], pts.shape[1], pts.shape[2], 4))  

    # Compute opacities and colors
    sigma_a = raw[...,3].to('cuda:0')
    rgb = raw[...,:3].to('cuda:0')    

    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.broadcast_to(torch.tensor([1e10]), z_vals[...,:1].shape).to('cuda:0')], -1) 
    alpha = 1.-torch.exp(-sigma_a * dists)  .to('cuda:0')
    weights = alpha * torch.cumprod(1.-alpha + 1e-10, dim = -1).to('cuda:0')
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2) 
    depth_map = torch.sum(weights * z_vals, -1) 
    acc_map = torch.sum(weights, -1)
    return rgb_map, depth_map, acc_map

trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=torch.float32)

# rotation about x axis
rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=torch.float32)

# Rotation about "y" axis
rot_theta = lambda th : torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    """
    The function takes in "theta, phi, and radius" and generate a 
    4x4 c2w transformation matrix. 
    c2w means camera to world transformation.

    args:
        theta: a float scalar. The horizontal view angle.

        phi: a float scalar. The vertical view angle.

        radius: a float scalar. The distance from the pinhole to the scene origin 

    Returns
        c2w: a number array of shape (4, 4) representing the transformation matrix
        Later we pass c2w to get_rays() and get rays origins and directions.
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float32) @ c2w
    return c2w

def Render_Video(model, focal, H, W, N_samples, filename):
    """
    The function renders and save the video. (filename.mp4)

    args:
        model: The NeRF model object
        focal: A float scalar specifying the focal length of this camera
        H:     A integer scalar: the height of each output image
        W:     A integer scala: the width of each output image
        N_samples: An integer: the number of sample points along each ray
        filename: str: the video name "without .mp4" you want to save.

    Returns:
        No return value.
    """
    with torch.no_grad():
        frames = []
        for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
            torch.cuda.empty_cache()
            c2w = pose_spherical(th, -30., 4.).to('cuda:0')
            rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])#.to('cuda:0'))
            rgb, depth, acc = render_rays_video(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            frames.append((255*np.clip(rgb.detach().cpu().numpy(),0,1)).astype(np.uint8))
        import imageio
        # f = 'video.mp4'
        f = filename + '.mp4'
        imageio.mimwrite(f, frames, fps=30, quality=7)

def play_video(filename):
    """
    The function loads and plays the video you specify. (filename.mp4)

    args:
        filename: The video name "without .mp4" you want to play. 

    Returns:
        No return value.
    """

    mp4 = open(filename + '.mp4','rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls autoplay loop>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
