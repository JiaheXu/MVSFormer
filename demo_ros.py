import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import time

import torch.nn.parallel
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset, DataLoader, SequentialSampler

import datasets.data_loaders as module_data
import misc.fusion as fusion
from base.parse_config import ConfigParser
from datasets.data_io import read_pfm, save_pfm
from misc.gipuma import gipuma_filter
from utils import *
from torchvision import transforms
import PIL.Image as IM

# ros things
import message_filters
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct



class mvsformer():

    def __init__(self, args, config):
        
        self.args = args
        self.config = config
        self.mode = None

        # model
        # build models architecture, then print to console
        if config['arch']['args']['vit_args'].get('twin', False):
            from models.mvsformer_model import TwinMVSNet
            self.model = TwinMVSNet(config['arch']['args'])
        else:
            from models.mvsformer_model import DINOMVSNet
            self.model = DINOMVSNet(config['arch']['args'])

        print('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(str(config.resume))
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key.replace('module.', '')] = val
        self.model.load_state_dict(new_state_dict, strict=True)

        # prepare models for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        # temp setting
        if hasattr(self.model, 'vit_args') and 'height' in self.model.vit_args and 'width' in self.model.vit_args:
            self.model.vit_args['height'] = args.max_h // 2 # Todo check max_h
            self.model.vit_args['width'] = args.max_w // 2

        self.times = []

        self.tmp = None
        # get tmp
        if args.tmps is not None:
            self.tmp = [float(a) for a in args.tmps.split(',')]
        else:
            self.tmp = args.tmp
        
        self.max_h, self.max_w = args.max_h, args.max_w
        
        
        # camera params        
        self.cameraMatrix1 = []
        self.cameraMatrix2 = []
        
        self.distCoeffs1 = []
        self.distCoeffs2 = []
        # 2k setting, need to modify into yaml file in the future 1242 * 2208
        self.cameraMatrix1.append( np.array([
            [1056.7081298828125, 0., 1102.6148681640625],
            [0. ,1056.7081298828125,  612.7953491210938],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( np.array([
            [1056.7081298828125, 0., 1102.6148681640625],
            [0. ,1056.7081298828125,  612.7953491210938],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        # 1080p setting, need to modify into yaml file in the future 1080*1920
        self.cameraMatrix1.append( np.array([
            [1064.29833984375, 0., 958.1749267578125],
            [0. ,1064.29833984375,  532.0037231445312],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( np.array([
            [1064.29833984375, 0., 958.1749267578125],
            [0. ,1064.29833984375,  532.0037231445312],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        # 720p setting, need to modify into yaml file in the future 720*1280
        self.cameraMatrix1.append( np.array([
            [525.5062255859375, 0., 637.7863159179688],
            [0. ,525.5062255859375,  354.01727294921875],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( np.array([
            [525.5062255859375, 0., 637.7863159179688],
            [0. ,525.5062255859375,  354.01727294921875],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.imageSize = []
        self.imageSize.append( (1242, 2208) )
        self.imageSize.append( (1080, 1920) )
        self.imageSize.append( (720, 1280) )
        self.R = np.array([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],            
        ])
        self.T = np.array([ 
            [-0.12],
            [0.],
            [0.]            
        ])

        self.extrinsic = []
        self.extrinsic.append( np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0 ,1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]            
        ]) )
        self.extrinsic.append( np.array([
            [1.0, 0.0, 0.0, -0.12],
            [0.0 ,1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]            
        ]) )


        # self.Q = []
        # for i in range ( len( self.imageSize ) ):
        #     R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(self.cameraMatrix1[i], self.distCoeffs1[i], self.cameraMatrix2[i], self.distCoeffs2[i], self.imageSize[i], self.R, self.T)
        #     self.Q.append(Q)

        #Todo: feed a startup all zero image to the network
        self.cam1_sub = message_filters.Subscriber(args.left_topic, Image)
        self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        self.conf_map_sub = message_filters.Subscriber(args.conf_map_topic, Image)

        self.disparity_pub = rospy.Publisher("zed2/disparity", PointCloud2, queue_size=1)

        self.point_cloud_pub = rospy.Publisher("zed2/point_cloud2", PointCloud2, queue_size=1)

        # self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub], 10, 1, allow_headerless=True)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub, self.depth_sub, self.conf_map_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def load_image(self, img):
        img = img.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        new_h, new_w = max_h, max_w

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def callback(self, cam1_msg, cam2_msg, depth_msg, conf_map_msg):

        print("callback")
        image1 = bridge.imgmsg_to_cv2(cam1_msg) # bgra
        # print("cam1_msg: ", cam1_msg.encoding)
        image1_np = np.array(image1[:,:,0:3])
        
        image2 = bridge.imgmsg_to_cv2(cam2_msg)
        image2_np = np.array(image2[:,:,0:3])

        image_depth = bridge.imgmsg_to_cv2(depth_msg)
        image_depth_np = np.array(image_depth)

        # rgb = bgr[...,::-1].copy()
        # bgr = rgb[...,::-1].copy()
        # gbr = rgb[...,[2,0,1]].copy()

        image1_rgb = image1_np[...,::-1].copy()
        image2_rgb = image2_np[...,::-1].copy()

        images = [image1_rgb, image2_rgb]
        
        res_idx = 0
        if(image1_rgb.shape[0] == 1080):
            res_idx = 1
        if(image1_rgb.shape[0] == 720):
            res_idx = 2
        
        imgs = []
        depth_values = None
        depth_ms = None
        mask = None
        proj_matrices = []


        # todo change following numbers
        local_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        depth_min =  425.0
        depth_interval =  2.6500000000000004
        interval_scale =  1.06
        ndepths = 192
        for i, img in enumerate(images): # img in rgb order

            intrinsics = self.cameraMatrix1[res_idx]
            extrinsics = self.extrinsic[i]
            
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)
            # resize to standard height or width
            # c_h, c_w = img.shape[:2]
            # if (c_h != s_h) or (c_w != s_w):
            #     scale_h = 1.0 * s_h / c_h
            #     scale_w = 1.0 * s_w / c_w
            #     img = cv2.resize(img, (s_w, s_h))
            #     intrinsics[0, :] *= scale_w
            #     intrinsics[1, :] *= scale_h

            img = IM.fromarray(img)
            imgs.append(local_transforms(img))
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)

        # all
        imgs = torch.stack(imgs)  # [V,3,H,W]
        imgs = imgs.unsqueeze(0) # [1,V,3,H,W]

        proj_matrices = np.stack(proj_matrices)
        proj_matrices = torch.from_numpy(proj_matrices)
        proj_matrices = proj_matrices.unsqueeze(0)
        # print("proj_matrices: ", proj_matrices.shape)

        stage0_pjmats = proj_matrices.clone()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage1_pjmats = proj_matrices.clone()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage2_pjmats = proj_matrices.clone()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 1
        stage3_pjmats = proj_matrices.clone()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage4_pjmats = proj_matrices.clone()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        proj_matrices_ms = {
            "stage1": stage0_pjmats,
            "stage2": stage1_pjmats,
            "stage3": stage2_pjmats,
            "stage4": stage3_pjmats,
            "stage5": stage4_pjmats
        }

        depth_values = torch.from_numpy(depth_values)
        depth_values = depth_values.unsqueeze(0)

        print("imgs: ", imgs.shape)
        print("depth_values: ", depth_values.shape)
        sample = {
                    "imgs": imgs,
                    "proj_matrices": proj_matrices_ms,
                    "depth_values": depth_values
                }
        
        # print("image1.shape: ", image1_np.shape)
        # print("image2.shape: ", image2_np.shape)
        # cv2.imwrite('img1.png', image1_np)
        # cv2.imwrite('img2.png', image2_np)

        with torch.no_grad():
            # print("sample: ",sample["imgs"].shape)
            torch.cuda.synchronize()
            start_time = time.time()
            sample_cuda = tocuda(sample)
            num_stage = 3 if args.no_refinement else 4
            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
            # print("cam_params: ", cam_params)
            # print("imgs.shape: ", imgs.shape)
            # imgs = imgs.unsqueeze(0)
            B, V, _, H, W = imgs.shape

            depth_interval = sample_cuda['depth_values'][:, 1] - sample_cuda['depth_values'][:, 0]
            # filenames = sample["filename"]
            # with torch.cuda.amp.autocast():

            # print("sample_cuda['depth_values']: ", sample_cuda['depth_values'].shape)
            outputs = self.model.forward(imgs, cam_params, sample_cuda['depth_values'], tmp=self.tmp)
            
            # print("output: ", outputs)
            
            torch.cuda.synchronize()

            end_time = time.time()
            self.times.append(end_time - start_time)
            
            depth_est_cuda = outputs['refined_depth']

            outputs = tensor2numpy(outputs)
            del sample_cuda


    def run(self):
        rospy.spin()  



def demo(args, config):
    init_kwags = {
        "data_path": args.testpath,
        # "data_list": testlist,
        "mode": "test",
        "num_srcs": args.num_view,
        "num_depths": args.numdepth,
        "interval_scale": Interval_Scale,
        "shuffle": False,
        "batch_size": 1,
        "fix_res": args.fix_res,
        "max_h": args.max_h,
        "max_w": args.max_w,
        "dataset_eval": args.dataset,
        "iterative": False,  # iterative inference
        "refine": not args.no_refinement,
        "use_short_range": args.use_short_range,
        "num_workers": 4,
    }

    rospy.init_node("mvsformer_node")
    mvsformer_node = mvsformer(args, config)
    mvsformer_node.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
    parser.add_argument('--model', default='mvsnet', help='select model')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--config', default=None, type=str, help='config file path (default: None)')

    parser.add_argument('--dataset', default='dtu', help='select dataset')
    parser.add_argument('--testpath', help='testing data dir for some scenes')
    parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
    parser.add_argument('--testlist', help='testing scene list')
    parser.add_argument('--exp_name', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

    parser.add_argument('--resume', default=None, help='load a specific checkpoint')
    parser.add_argument('--outdir', default='/home/wmlce/mount_194/DTU_MVS_outputs', help='output dir')
    parser.add_argument('--display', action='store_true', help='display depth images and masks')

    parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

    parser.add_argument('--ndepths', type=str, default=None, help='ndepths')
    parser.add_argument('--depth_interals_ratio', type=str, default=None, help='depth_interals_ratio')
    parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
    parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
    parser.add_argument('--no_refinement', action="store_true", help='depth refinement in last stage')
    parser.add_argument('--full_res', action="store_true", help='full resolution prediction')

    parser.add_argument('--interval_scale', type=float, required=True, help='the depth interval scale')
    parser.add_argument('--num_view', type=int, default=5, help='num of view')
    parser.add_argument('--max_h', type=int, default=864, help='testing max h')
    parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
    parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
    parser.add_argument('--depth_scale', type=float, default=1.0, help='depth scale')
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature of softmax')

    parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
    parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

    parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "pcd", "dpcd"],
                        help="filter method")

    # filter
    parser.add_argument('--prob_threshold', type=str, default='0.5,0.5,0.5,0.5', help='prob confidence')
    parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
    parser.add_argument('--thres_disp', type=float, default=1.0, help='threshold of disparity')
    parser.add_argument('--downsample', type=float, default=None, help='downsampling point cloud')

    ## dpcd filter
    parser.add_argument('--dist_base', type=float, default=4.0, help='threshold of disparity')
    parser.add_argument('--rel_diff_base', type=float, default=1300.0, help='downsampling point cloud')

    # filter by gimupa
    parser.add_argument('--fusibile_exe_path', type=str, default='./fusibile/fusibile')
    parser.add_argument('--disp_threshold', type=float, default='0.2')
    parser.add_argument('--num_consistent', type=float, default='3')

    # tank templet
    parser.add_argument('--use_short_range', action='store_true')

    # confidence
    parser.add_argument('--combine_conf', action='store_true')
    parser.add_argument('--tmp', default=1.0, type=float)
    parser.add_argument('--tmps', default=None, type=str)
    parser.add_argument('--save_all_confs', action='store_true')

    parser.add_argument('--left_topic', type=str, default="/zed2/zed_node/left/image_rect_color", help="left cam topic")
    parser.add_argument('--right_topic', type=str, default="/zed2/zed_node/right/image_rect_color", help="right cam topic")
    parser.add_argument('--depth_topic', type=str, default="/zed2/zed_node/depth/depth_registered", help="depth cam topic")
    parser.add_argument('--conf_map_topic', type=str, default="/zed2/zed_node/confidence/confidence_map", help="depth confidence map topic")

    parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")
    args = parser.parse_args()
    if args.testpath_single_scene:
        args.testpath = os.path.dirname(args.testpath_single_scene)

    Interval_Scale = args.interval_scale
    print("***********Interval_Scale**********\n", Interval_Scale)

    # args.outdir = args.outdir + f'_{args.max_w}x{args.max_h}'
    os.makedirs(args.outdir, exist_ok=True)


    config = ConfigParser.from_args(parser, mkdir=False)
    if args.ndepths is not None:
        config['arch']['args']['ndepths'] = [int(d) for d in args.ndepths.split(',')]
    if args.depth_interals_ratio is not None:
        config['arch']['args']['depth_interals_ratio'] = [float(d) for d in args.depth_interals_ratio.split(',')]

    # step1. save all the depth maps and the masks in outputs directory
    demo(args, config)


    # Todo
    # # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    # if args.filter_method == "pcd" or args.filter_method == "dpcd":
    #     # support multi-processing, the default number of worker is 4
    #     pcd_filter(testlist)

    # elif args.filter_method == 'gipuma':
    #     prob_threshold = args.prob_threshold
    #     prob_threshold = [float(p) for p in prob_threshold.split(',')]
    #     gipuma_filter(testlist, args.outdir, prob_threshold, args.disp_threshold, args.num_consistent,
    #                   args.fusibile_exe_path)
    # else:
    #     raise NotImplementedError

    
