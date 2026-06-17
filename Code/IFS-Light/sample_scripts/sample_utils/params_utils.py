import numpy as np
import pandas as pd
import torch as th
import torch
import glob, os, sys
import cv2
import imageio
from collections import defaultdict

import sys
sys.path.insert(0, '/home/user/IFS-Light/sample_scripts/sample_utils/')
import shadow_utils
import vis_utils

def params_to_model(shape, exp, pose, cam, lights):

    from model_3d.FLAME import FLAME
    from model_3d.FLAME.config import cfg as flame_cfg
    from model_3d.FLAME.utils.renderer import SRenderY
    import model_3d.FLAME.utils.util as util

    flame = FLAME.FLAME(flame_cfg.model).cuda()
    verts, landmarks2d, landmarks3d = flame(shape_params=shape, 
            expression_params=exp, 
            pose_params=pose)
    renderer = SRenderY(image_size=256, obj_filename=flame_cfg.model.topology_path, uv_size=flame_cfg.model.uv_size).cuda()

    ## projection
    landmarks2d = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
    landmarks3d = util.batch_orth_proj(landmarks3d, cam); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
    trans_verts = util.batch_orth_proj(verts, cam); trans_verts[:,:,1:] = -trans_verts[:,:,1:]

    ## rendering
    shape_images = renderer.render_shape(verts, trans_verts, lights=lights)

    # opdict = {'verts' : verts,}
    # os.makedirs('./rendered_obj', exist_ok=True)
    # save_obj(renderer=renderer, filename=(f'./rendered_obj/{i}.obj'), opdict=opdict)
    
    return {"shape_images":shape_images, "landmarks2d":landmarks2d, "landmarks3d":landmarks3d}

def save_obj(renderer, filename, opdict):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    import model_3d.FLAME.utils.util as util
    i = 0
    vertices = opdict['verts'][i].cpu().numpy()
    faces = renderer.faces[0].cpu().numpy()
    colors = np.ones(shape=vertices.shape) * 127.5

    # save coarse mesh
    util.write_obj(filename, vertices, faces, colors=colors)

# def get_R_normals(n_step):
#     src = np.array([0, 0, 2.50])
#     dst = np.array([0, 0, 6.50])
#     rvec = np.linspace(src, dst, n_step)
#     R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
#     R = np.stack(R, axis=0)
#     return R

def get_R_normals(n_step):
    if n_step % 2 == 0:
        fh = sh = n_step//2
    else:
        fh = int(n_step//2)
        sh = fh + 1
        
    src = np.array([0, 6.50, 0])
    # dst = np.array([0, 2.50, 0])
    dst = np.array([0, 1.00, 0])
    rvec_f = np.linspace(src, dst, fh)
    
    src = rvec_f[-1]
    # dst = np.array([0, rvec_f[-1][1], -8.00])
    dst = np.array([0, 2.50, -8.00])
    rvec_s = np.linspace(rvec_f[-1], dst, sh)
    # print(rvec_f.shape, rvec_s.shape)
    rvec = np.concatenate((rvec_f, rvec_s), axis=0)
    # print(rvec_f)
    # print(rvec_s)
    # print(rvec)
    # print(rvec.shape)
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    R = np.stack(R, axis=0)
    return R

def grid_sh(n_grid, sh=None, sx=[-1, 1], sy=[1, 0], sh_scale=1.0, use_sh=False):
    sh_light = []
    sh_original = sh.cpu().numpy().copy().reshape(-1, 9, 3)
    print(f"[#] Buiding grid sh with : span_x={sx}, span_y={sy}, n_grid={n_grid}")
    print(f"[#] Given sh : \n{sh_original}")
    # sx is from left(negative) -> right(positive)
    # sy is from top(positive) -> bottom(negative)
    # print(sx, sy)
    # print(np.linspace(sx[0], sx[1], n_grid))
    # exit()
    for ix, lx in enumerate(np.linspace(sx[0], sx[1], n_grid)):
        for iy, ly in enumerate(np.linspace(sy[0], sy[1], n_grid)):
            l = np.array((lx, ly, 1))
            l = l / np.linalg.norm(l)
            
            if use_sh:
                tmp_light = sh_original.copy()
            else:
                tmp_light = np.zeros((1, 9, 3))
                tmp_light[0:1, 0:1, :] = sh_original[0:1, 0:1, :] * sh_scale
                
            # if iy in [1, 2, 3]:
                # print("IN", iy)
                # print(tmp_light)
                # tmp_light = tmp_light * sh_scale 
                # print(tmp_light)
            
            tmp_light[0:1, 1:2, :] = l[0]
            tmp_light[0:1, 2:3, :] = l[1]
            tmp_light[0:1, 3:4, :] = l[2]
            # if iy in [0, 1, 2, 3]:
            tmp_light = tmp_light * sh_scale 
            sh_light.append(tmp_light)
        # exit()
    sh_light = np.concatenate(sh_light, axis=0)
    sh_light = np.concatenate((sh_original.reshape(-1, 9, 3), sh_light))
    print(f"[#] Out grid sh : \n{sh_light.shape}")
    return sh_light

def load_flame_mask(part='face'):
    #NOTE: Modify the path to masking the FLAME model (ear, neck, etc.)
    f_mask = np.load("/home/user/IFS-Light/experimental/masking_flame/FLAME_masks_face-id.pkl", allow_pickle=True, encoding='latin1')
    v_mask = np.load('/home/user/IFS-Light/experimental/masking_flame/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
    mask={
        'v_mask':v_mask[part].tolist(),
        'f_mask':f_mask[part].tolist() 
    }
    return mask        

def init_deca(useTex=False, extractTex=True, device='cuda', 
              deca_mode='only_renderer', mask=None, deca_obj=None):
    
    # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
    sys.path.insert(1, '/home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/DECA/')

    from decalib import deca
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = extractTex
    deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    return deca_obj

def sh_to_ld(sh):
    #NOTE: Roughly Convert the SH to light direction
    sh = sh.reshape(-1, 9, 3)
    ld = th.mean(sh[0:1, 1:4, :], dim=2)
    return ld

def render_shadow_mask(sh_light, cam, verts, deca):
    sys.path.insert(0, '/home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
    from decalib.utils import util
    
    shadow_mask_all = []
    if verts.shape[0] >= 2:
        tmp = []
        for i in range(1, verts.shape[0]):
            tmp.append(th.allclose(verts[[0]], verts[[i]]))
        assert all(tmp);
        
    depth_image, alpha_image = deca.render.render_depth(verts.cuda())   # Depth : B x 1 x H x W
    _, _, h, w = depth_image.shape
    depth_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
    depth_grid = np.repeat(np.stack((depth_grid), axis=-1)[None, ...], repeats=sh_light.shape[0], axis=0)   # B x H x W x 2
    depth_grid = np.concatenate((depth_grid, depth_image.permute(0, 2, 3, 1)[..., 0:1].cpu().numpy()), axis=-1) # B x H x W x 3
    depth_grid[..., 2] *= 256
    depth_grid = th.tensor(depth_grid).cuda()
    shadow_mask = th.clone(depth_grid[:, :, :, 2])
    # print(shadow_mask.shape, sh_light.shape)
    for i in range(sh_light.shape[0]):
        each_depth_grid = depth_grid[i].clone()
        #NOTE: Render the shadow mask from light direction
        ld = sh_to_ld(sh=th.tensor(sh_light[[i]])).cuda()
        ld = util.batch_orth_proj(ld[None, ...], cam[None, ...].cuda()); ld[:,:,1:] = -ld[:,:,1:]    # This fn takes pts=Bx3, cam=Bx3
        ray = ld.view(3).cuda()
        ray[2] *= 0.5
        n = 256
        ray = ray / th.norm(ray)
        mxaxis = max(abs(ray[0]), abs(ray[1]))
        shift = ray / mxaxis * th.arange(n).view(n, 1).cuda()
        coords = each_depth_grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)

        output = th.nn.functional.grid_sample(
            th.tensor(np.tile(each_depth_grid[:, :, 2].view(1, 1, n, n).cpu().numpy(), [n, 1, 1, 1])).cuda(),
            coords[..., :2] / (n - 1) * 2 - 1,
            align_corners=True)
        diff = coords[..., 2] - output[:, 0] 
        shadow_mask[i] *= (th.min(diff, dim=0)[0] > -0.1) * 0.5 + 0.5
        
    #     print(shadow_mask.shape)
    #     print(th.max(shadow_mask))
    #     print(th.min(shadow_mask))
    #     import torchvision
    #     torchvision.utils.save_image(shadow_mask[[i]]/255.0, f'inf{i}.png')
    #     torchvision.utils.save_image(shadow_mask[[i]], f'inf2{i}.png')
    # torchvision.utils.save_image(shadow_mask[:, None, ...]/255.0, f'infall.png')
    return th.clip(shadow_mask, 0, 255.0)/255.0

def render_deca_gridSH(deca_params, idx, n, render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None, repeat=True,
                deca_obj=None):
    '''
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param idx: index of data in batch to render
    :param n: n of repeated tensor (For interpolation)
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/wsy/difareli_code/preprocess_scripts/Relighting_preprocessing_tools/DECA/')

        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    testdata = datasets.TestData([deca_params['raw_image_path'][0]], iscrop=True, face_detector='fan', sample_step=10)
    if repeat:
        codedict = {'shape':deca_params['shape'][[idx]].repeat(n, 1).to(device).float(),
                    'pose':deca_params['pose'][[idx]].repeat(n, 1).to(device).float(),
                    'exp':deca_params['exp'][[idx]].repeat(n, 1).to(device).float(),
                    'cam':deca_params['cam'][[idx]].repeat(n, 1).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    'tform':testdata[idx]['tform'][None].to(device).reshape(-1, 3, 3).repeat(n, 1, 1).to(device).float(),
                    'images':testdata[idx]['image'].to(device)[None,...].float().repeat(n, 1, 1, 1),
                    'tex':deca_params['albedo'][[idx]].repeat(n, 1).to(device).float(),
                    'detail':deca_params['detail'][[idx]].repeat(n, 1).to(device).float(),
        }
        # print(codedict['pose'])
        # print(codedict['light'])
        # exit()
        original_image = deca_params['raw_image'][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
    else:
        codedict = {'shape':th.tensor(deca_params['shape']).to(device).float(),
                    'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':th.tensor(deca_params['exp']).to(device).float(),
                    'cam':th.tensor(deca_params['cam']).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    'tform':testdata[idx]['tform'].to(device).reshape(-1, 3, 3).float(),
                    'images':th.stack([testdata[i]['image'] for i in range(len(deca_params['raw_image_path']))]).to(device).float(),
                    'tex':th.tensor(deca_params['albedo']).to(device).float(),
                    'detail':(deca_params['detail']).to(device).float(),
        }
        original_image = deca_params['raw_image'].to(device).float() / 255.0
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    elif render_mode == 'template_shape':
        use_template = True
        mean_cam = th.tensor(avg_dict['cam'])[None, ...].repeat(n, 1).to(device).float()
        tform = th.tensor(avg_dict['tform'])[None, ...].repeat(n, 1).to(device).reshape(-1, 3, 3).float()
        tform_inv = th.inverse(tform).transpose(1,2)
    else: raise NotImplementedError
    orig_opdict, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict['shape_images']
    return rendered_image, orig_visdict

def render_deca(deca_params, idx, src_id, dst_id, n,  render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None, repeat=True,
                deca_obj=None, step=0, mean_color = 0):
    '''
    TODO: Adding the rendering with template shape, might need to load mean of camera/tform
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param idx: index of data in batch to render
    :param n: n of repeated tensor (For interpolation)
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/wsy/difareli_code/preprocess_scripts/Relighting_preprocessing_tools/DECA/')

        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    testdata = datasets.TestData([deca_params['raw_image_path'][0]], iscrop=True, face_detector='fan', sample_step=10)
    if repeat:
        codedict = {'shape':deca_params['shape'][[idx]].repeat(n, 1).to(device).float(),
                    'pose':deca_params['pose'][[idx]].repeat(n, 1).to(device).float(),
                    # 'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':deca_params['exp'][[idx]].repeat(n, 1).to(device).float(),
                    'cam':deca_params['cam'][[idx]].repeat(n, 1).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':deca_params['tform'][[idx]].repeat(n, 1).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].repeat(n, 1, 1).to(device).float(),
                    'images':testdata[idx]['image'].to(device)[None,...].float().repeat(n, 1, 1, 1),
                    'tex':deca_params['albedo'][[idx]].repeat(n, 1).to(device).float(),
                    'detail':deca_params['detail'][[idx]].repeat(n, 1).to(device).float(),
        }
        # print(codedict['pose'])
        # print(codedict['light'])
        # exit()
        original_image = deca_params['raw_image'][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
    else:
        codedict = {'shape':th.tensor(deca_params['shape']).to(device).float(),
                    'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':th.tensor(deca_params['exp']).to(device).float(),
                    'cam':th.tensor(deca_params['cam']).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':th.tensor(deca_params['tform']).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].to(device).float(),
                    'images':th.stack([testdata[i]['image'] for i in range(len(deca_params['raw_image_path']))]).to(device).float(),
                    'tex':th.tensor(deca_params['albedo']).to(device).float(),
                    'detail':(deca_params['detail']).to(device).float(),
        }
        original_image = deca_params['raw_image'].to(device).float() / 255.0
        
    if rotate_normals:
        codedict.update({'R_normals': th.tensor(deca_params['R_normals']).to(device).float()})
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    elif render_mode == 'template_shape':
        use_template = True
        mean_cam = th.tensor(avg_dict['cam'])[None, ...].repeat(n, 1).to(device).float()
        tform = th.tensor(avg_dict['tform'])[None, ...].repeat(n, 1).to(device).reshape(-1, 3, 3).float()
        tform_inv = th.inverse(tform).transpose(1,2)
    else: raise NotImplementedError
    orig_opdict, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict['shape_images']

  

    #叠加方向光的阴影 问题是如何把阴影变为5步融合进去 

    # input_image = '/home/wsy/FairMPdata/aligned_images/valid/'+str(src_id)
    # if step == 0:
    #     reference_image = '/home/wsy/FairMPdata/aligned_images/valid/'+str(src_id)
    # else:
    #     reference_image = '/home/wsy/FairMPdata/aligned_images/valid/'+str(dst_id)
    # input_mask = '/home/wsy/multi_PIE_skinmask/'+str(src_id).split('_')[0]+'/res_'+str(src_id)
    # final_shading = render_direction_shadow(input_image, reference_image, input_mask)
    # # cv2.imwrite('final_shading.png', 255.0*final_shading)
    # # final_shading = cv2.imread('final_shading.png')/255.0
    # # final_shading = np.reshape(np.transpose(final_shading, (2,0,1)), (1,3,256,256))
    # rendered_image = th.tensor(final_shading).cuda() * rendered_image * 1.2

    input_image = '/home/user/IFS-Light/preprocess_scripts/ffhq/aligned_images/valid/'+ str(src_id)
    if step == 0:
        reference_image = '/home/user/IFS-Light/preprocess_scripts/ffhq/aligned_images/valid/' + str(src_id)
    else:
        reference_image = '/home/user/IFS-Light/preprocess_scripts/ffhq/aligned_images/valid/' + str(dst_id)
    #input_mask = '/home/wsy/GeomConsistentFR-main/FFHQ_skin_masks/res_' + str(src_id)
    input_mask = '/home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/face-parsing.PyTorch/res/ffhq/vis/res_' + str(src_id)
    final_shading = render_direction_shadow(input_image, reference_image, input_mask)
    cv2.imwrite('shading.png', 255.0*final_shading)
    final_shading = cv2.imread('shading.png')
    rendered_image = th.tensor(final_shading).cuda() * rendered_image

    return rendered_image, orig_visdict

def render_direction_shadow(input_image, reference_image, input_mask):
    model = shadow_utils.RelightNet()
    model.load_state_dict(torch.load('/home/user/IFS-Light/sample_scripts/model_logs/ours/model_epoch106.pth'))
    model = model.float()
    model = model.cuda()
    model.eval()
  
   
    epoch = 200
    #相机的内参参数，x、y轴的焦距设为700，主点位于图像中心
    intrinsic_matrix = np.zeros((1, 3, 3))
    intrinsic_matrix[:, 0, 0] = 700.0
    intrinsic_matrix[:, 1, 1] = 700.0
    intrinsic_matrix[:, 2, 2] = 1.0
    intrinsic_matrix[:, 0, 2] = model.img_width/2.0
    intrinsic_matrix[:, 1, 2] = model.img_height/2.0
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)

    with torch.no_grad():
        curr_input_image = torch.reshape(torch.from_numpy(imageio.imread(input_image)/255.0), (1, 256, 256, 3)) 
        curr_reference_image = torch.reshape(torch.from_numpy(imageio.imread(reference_image)/255.0), (1, 256, 256, 3))
        curr_training_lighting = torch.from_numpy(np.zeros((model.batch_size, 4)))
        curr_img_name = input_image
        curr_mask_fill_nose = torch.from_numpy(np.reshape(imageio.imread(input_mask) , (256, 256, 1)))/255.0
        #获取参考图像的光照
        albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(curr_reference_image.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask_fill_nose.cuda(), torch.reshape(curr_training_lighting[:, 1:4].float().cuda(), (model.batch_size, 3, 1, 1)), torch.reshape(curr_training_lighting[:, 0].float().cuda(), (model.batch_size, 1, 1)))
        
        #获取输入图像的光照
        albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals, estimated_unit_light_direction, estimated_ambient_light = model(curr_input_image.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask_fill_nose.cuda(), torch.reshape(estimated_unit_light_direction.float().cuda(), (model.batch_size, 3, 1, 1)), torch.reshape(estimated_ambient_light.float().cuda(), (model.batch_size, 1, 1)))
        
        final_shading = final_shading.cpu().numpy()
        #full_shading = full_shading.cpu().numpy()
        curr_mask_fill_nose_3_channels = np.zeros((model.img_height, model.img_width, 3))
        curr_mask_fill_nose_3_channels[:, :, 0] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
        curr_mask_fill_nose_3_channels[:, :, 1] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
        curr_mask_fill_nose_3_channels[:, :, 2] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))

        final_shading = final_shading[0, :, :]*np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
        #full_shading = full_shading[0, :, :]*np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
    return final_shading



def render_deca_rotateSH(deca_params, render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None,
                deca_obj=None):
    '''
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/DECA/')

        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    num_ren = deca_params['light'].shape[0]
    idx = 0
    # Expand the shape to match the light
    testdata = datasets.TestData([deca_params['raw_image_path'][idx]], iscrop=True, face_detector='fan', sample_step=10)
    
    codedict = {'shape':deca_params['shape'][[idx]].repeat(num_ren, 1).to(device).float(),
                'pose':deca_params['pose'][[idx]].repeat(num_ren, 1).to(device).float(),
                'exp':deca_params['exp'][[idx]].repeat(num_ren, 1).to(device).float(),
                'cam':deca_params['cam'][[idx]].repeat(num_ren, 1).to(device).float(),
                'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                'tform':testdata[idx]['tform'][None].repeat(num_ren, 1, 1).to(device).float(),
                'images':testdata[idx]['image'].to(device)[None,...].float().repeat(num_ren, 1, 1, 1),
                'tex':deca_params['albedo'][[idx]].repeat(num_ren, 1).to(device).float(),
                'detail':deca_params['detail'][[idx]].repeat(num_ren, 1).to(device).float(),
    }
    # for k in codedict.keys():
    #     print(k, codedict[k].shape)
    # original_image = deca_params['raw_image'].to(device).float() / 255.0
    original_image = deca_params['raw_image'][[idx]].to(device).float().repeat(num_ren, 1, 1, 1) / 255.0
    # print("ORIG: ", original_image.shape)
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    else: raise NotImplementedError
    orig_opdict, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict['shape_images']
    return rendered_image, orig_visdict


def render_deca_videos(deca_params, render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None,
                deca_obj=None):
    '''
    TODO: Adding the rendering with template shape, might need to load mean of camera/tform
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/DECA/')

        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    # testdata = datasets.TestData([deca_params['raw_image_path'][0]], iscrop=True, face_detector='fan', sample_step=10)
    testdata = datasets.TestData(deca_params['raw_image_path'], iscrop=True, face_detector='fan', sample_step=10)
    codedict = {'shape':th.tensor(deca_params['shape']).to(device).float(),
                'pose':th.tensor(deca_params['pose']).to(device).float(),
                'exp':th.tensor(deca_params['exp']).to(device).float(),
                'cam':th.tensor(deca_params['cam']).to(device).float(),
                'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                'tform':th.tensor(deca_params['tform']).to(device).reshape(-1, 3, 3).float(),
                'images':th.stack([testdata[i]['image'] for i in range(len(deca_params['raw_image_path']))]).to(device).float(),
                'tex':th.tensor(deca_params['albedo']).to(device).float(),
                'detail':th.tensor(deca_params['detail']).to(device).float(),
    }
    original_image = deca_params['raw_image'].to(device).float() / 255.0
        
    if rotate_normals:
        codedict.update({'R_normals': th.tensor(deca_params['R_normals']).to(device).float()})
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    else: raise NotImplementedError
    orig_opdict, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict['shape_images']
    return rendered_image, orig_visdict

def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64)

    return params_s

def normalize(arr, min_val=None, max_val=None, a=-1, b=1):
    '''
    Normalize any vars to [a, b]
    :param a: new minimum value
    :param b: new maximum value
    :param arr: np.array shape=(N, #params_dim) e.g. deca's params_dim = 159
    ref : https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    '''
    if max_val is None and min_val is None:
        max_val = np.max(arr, axis=0)    
        min_val = np.min(arr, axis=0)

    arr_norm = ((b-a) * (arr - min_val) / (max_val - min_val)) + a
    return arr_norm, min_val, max_val

def denormalize(arr_norm, min_val, max_val, a=-1, b=1):
    arr_denorm = (((arr_norm - a) * (max_val - min_val)) / (b - a)) + min_val
    return arr_denorm

def load_params(path, params_key):
    '''
    Load & Return the params
    Input : 
    :params path: path of the pre-computed parameters
    :params params_key: list of parameters name e.g. ['pose', 'light']
    Return :
    :params params_s: the dict-like of {'0.jpg':}
    '''

    params = {}
    for k in params_key:
        for p in glob.glob(f'{path}/*{k}-anno.txt'):
            # Params
            if k in p:
                print(f'Key=> {k} : Filename=>{p}')
                params[k] = read_params(path=p)

    params_s = swap_key(params)

    all_params = []
    for img_name in params_s:
        each_img = []
        for k in params_key:
            each_img.append(params_s[img_name][k])
        all_params.append(np.concatenate(each_img))
    all_params = np.stack(all_params, axis=0)
    return params_s, all_params
    
def get_params_set(set, params_key, path="/home/user/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/test_images/params/"):
    if set == 'itw':
        # In-the-wild
        sys.path.insert(0, '/home/user/IFS-Light/sample_scripts/cond_utils/arcface/')
        sys.path.insert(0, '/home/user/IFS-Light/sample_scripts/cond_utils/arcface/detector/')
        sys.path.insert(0, '../../cond_utils/deca/')
        from cond_utils.arcface import get_arcface_emb
        from cond_utils.deca import get_deca_emb

        itw_path = "../../itw_images/aligned/"
        device = 'cuda:0'
        # ArcFace
        faceemb_itw, emb = get_arcface_emb.get_arcface_emb(img_path=itw_path, device=device)

        # DECA
        deca_itw = get_deca_emb.get_deca_emb(img_path=itw_path, device=device)

        assert deca_itw.keys() == faceemb_itw.keys()
        params_itw = {}
        for img_name in deca_itw.keys():
            params_itw[img_name] = deca_itw[img_name]
            params_itw[img_name].update(faceemb_itw[img_name])
            
        params_set = params_itw
            
    elif set == 'valid' or set == 'train':
        # Load params
        if params_key is None:
            params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']

        if set == 'train':
            params_train, params_train_arr = load_params(path=f"{path}/{set}/", params_key=params_key)
            params_set = params_train
        elif set == 'valid':
            params_valid, params_valid_arr = load_params(path=f"{path}/{set}/", params_key=params_key)
            params_set = params_valid
        else:
            raise NotImplementedError

    else: raise NotImplementedError

    return params_set

def preprocess_cond(deca_params, k, cfg):
    if k != 'light':
        return deca_params
    else:
        num_SH = cfg.relighting.num_SH
        params = deca_params[k]
        params = params.reshape(params.shape[0], 9, 3)
        params = params[:, :num_SH, :]
        # params = params.flatten(start_dim=1)
        params = params.reshape(params.shape[0], -1)
        deca_params = params
        return deca_params
    
def write_params(path, params, keys):
    tmp = {}
    for k in keys:
        if th.is_tensor(params[k]):
            tmp[k] = params[k].cpu().numpy()
        else:
            tmp[k] = params[k]
            
    np.save(file=path, arr=tmp)
        