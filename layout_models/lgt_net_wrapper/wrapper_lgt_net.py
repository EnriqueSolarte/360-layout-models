import pdb
import glob
import json
import os
import sys
import yaml
import pathlib
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from copy import deepcopy
from pathlib import Path


class WrapperLGTNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_lgt_net_path()
        from layout_models.models.LGTNet.models.build import build_model
        from layout_models.models.LGTNet.utils.logger import get_logger

        self.logger = get_logger()

        # ! Setting cuda-device
        self.device = torch.device(
            f"cuda:{cfg.TRAIN.DEVICE}" if torch.cuda.is_available() else "cpu"
        )

        # Loaded trained model
        # assert os.path.isfile(cfg.CKPT.DIR), f"Not found {cfg.CKPT.DIR}"
        self.logger.info("Loading LGTNet...")
        model, _, _, _ = build_model(cfg, self.logger)
        self.net = model
        self.logger.info(f"ckpt: {cfg.CKPT.DIR}")
        self.logger.info("LGTNet Wrapper Successfully initialized")

    @staticmethod
    def set_lgt_net_path():
        ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent

        lgt_dir = os.path.join(ROOT_DIR, "models", "LGTNet")
        if lgt_dir not in sys.path:
            assert os.path.isdir(lgt_dir), f"Not found {lgt_dir}"
            sys.path.append(lgt_dir)

    def estimate_within_list_ly(self, list_ly):
        from mvl_challenge.models.LGTNET.postprocessing.post_process import post_process
        from mvl_challenge.models.LGTNET.utils.misc import tensor2np_d, tensor2np
        from mvl_challenge.models.LGTNET.utils.conversion import depth2xyz, uv2pixel
        from mvl_challenge.models.LGTNET.utils.boundary import corners2boundaries
        layout_dataloader = DataLoader(
            MVImageLayout([(ly.img_fn, ly.idx) for ly in list_ly]),
            batch_size=self.cfg.runners.mvl.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.mvl.num_workers,
            pin_memory=True if self.device != "cpu" else False,
            worker_init_fn=lambda x: np.random.seed(),
        )
        self.net.eval()
        evaluated_data = {}
        for x in tqdm(layout_dataloader, desc=f"Estimating layout..."):
            with torch.no_grad():
                y_bon_est_cpu = np.zeros(
                    (self.cfg.runners.mvl.batch_size, 2, 1024))
                dt = self.net(x["images"].to(self.device))
                # pdb.set_trace()
                if self.cfg.post_processing != 'original':
                    dt['processed_xyz'] = post_process(
                        tensor2np(dt['depth']), type_name=self.cfg.post_processing)
                dt_np = tensor2np_d(dt)
                for i in range(len(dt_np['depth'])):
                    dt_depth = dt_np['depth'][i]
                    dt_xyz = depth2xyz(np.abs(dt_depth))
                    dt_ratio = dt_np['ratio'][i][0]
                    if 'processed_xyz' in dt:
                        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][i], step=None, visible=False,
                                                           length=1024)
                        # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                        # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                        dt_ceiling = uv2pixel(dt_boundaries[1])
                        dt_floor = uv2pixel(dt_boundaries[0])
                        if len(dt_ceiling) != 1024 or len(dt_floor) != 1024:
                            # C_XY[:,0] non repeat index
                            dt_ceiling_index = np.unique(
                                dt_ceiling[:, 0], return_index=True)
                            # F_XY[:,0] non repeat index
                            dt_floor_index = np.unique(
                                dt_floor[:, 0], return_index=True)
                            dt_ceiling_pixel = np.transpose(
                                dt_ceiling[dt_ceiling_index[1], :])
                            dt_floor_pixel = np.transpose(
                                dt_floor[dt_floor_index[1], :])
                            dt_ceiling_pixel = complementary_element(
                                dt_ceiling_pixel, [512, 1024])
                            dt_floor_pixel = complementary_element(
                                dt_floor_pixel, [512, 1024])
                        else:
                            dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                            dt_floor_pixel = np.transpose(dt_floor)[1]
                        # pdb.set_trace()

                    else:
                        dt_boundaries = corners2boundaries(
                            dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=1024)
                        dt_ceiling_pixel = np.transpose(
                            uv2pixel(dt_boundaries[1]))[1]
                        dt_floor_pixel = np.transpose(
                            uv2pixel(dt_boundaries[0]))[1]

                    y_bon_est_cpu[i, 0, :] = dt_ceiling_pixel
                    y_bon_est_cpu[i, 1, :] = dt_floor_pixel
                y_bon_est = torch.tensor(
                    (y_bon_est_cpu / 512 - 0.5) * np.pi).to(self.device)
            for y_, idx in zip(y_bon_est.cpu(), x["idx"]):
                evaluated_data[idx] = y_
        [ly.set_phi_coords(phi_coords=evaluated_data[ly.idx])
         for ly in list_ly]

    def valid_iou_loop(self, only_val=False):
        from mvl_challenge.models.LGTNET.postprocessing.post_process import post_process
        from mvl_challenge.models.LGTNET.utils.misc import tensor2np_d, tensor2np
        from mvl_challenge.models.LGTNET.utils.conversion import depth2xyz, uv2pixel
        from mvl_challenge.models.LGTNET.utils.boundary import corners2boundaries
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        total_eval = {}
        invalid_cnt = 0

        for _ in trange(len(iterator_valid_iou), desc="IoU Validation epoch %d" % self.current_epoch):
            x, y_bon_ref, std, u_range, eval_range = next(iterator_valid_iou)
            u_range = u_range.int()
            eval_range = eval_range.int()
            with torch.no_grad():
                y_bon_est_cpu = np.zeros((y_bon_ref.size(dim=0), 2, 1024))
                dt = self.net(x.to(self.device))
                # pdb.set_trace()
                if self.cfg.post_processing != 'original':
                    dt['processed_xyz'] = post_process(
                        tensor2np(dt['depth']), type_name=self.cfg.post_processing)
                dt_np = tensor2np_d(dt)
                for i in range(len(dt_np['depth'])):
                    dt_depth = dt_np['depth'][i]
                    dt_xyz = depth2xyz(np.abs(dt_depth))
                    dt_ratio = dt_np['ratio'][i][0]
                    if 'processed_xyz' in dt:
                        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][i], step=None, visible=False,
                                                           length=1024)
                        # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                        # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                        dt_ceiling = uv2pixel(dt_boundaries[1])
                        dt_floor = uv2pixel(dt_boundaries[0])
                        if len(dt_ceiling) != 1024 or len(dt_floor) != 1024:
                            # C_XY[:,0] non repeat index
                            dt_ceiling_index = np.unique(
                                dt_ceiling[:, 0], return_index=True)
                            # F_XY[:,0] non repeat index
                            dt_floor_index = np.unique(
                                dt_floor[:, 0], return_index=True)
                            dt_ceiling_pixel = np.transpose(
                                dt_ceiling[dt_ceiling_index[1], :])
                            dt_floor_pixel = np.transpose(
                                dt_floor[dt_floor_index[1], :])
                            dt_ceiling_pixel = complementary_element(
                                dt_ceiling_pixel, [512, 1024])
                            dt_floor_pixel = complementary_element(
                                dt_floor_pixel, [512, 1024])
                        else:
                            dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                            dt_floor_pixel = np.transpose(dt_floor)[1]
                        # pdb.set_trace()

                    else:
                        dt_boundaries = corners2boundaries(
                            dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=1024)
                        dt_ceiling_pixel = np.transpose(
                            uv2pixel(dt_boundaries[1]))[1]
                        dt_floor_pixel = np.transpose(
                            uv2pixel(dt_boundaries[0]))[1]

                    y_bon_est_cpu[i, 0, :] = dt_ceiling_pixel
                    y_bon_est_cpu[i, 1, :] = dt_floor_pixel
                y_bon_est = torch.tensor(
                    (y_bon_est_cpu / 512 - 0.5) * np.pi).to(self.device)
                # pdb.set_trace()

                true_eval = {"2DIoU": [], "3DIoU": []}
                for gt, est, ind, e_ind in zip(y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy(), u_range.cpu().numpy(), eval_range.cpu().numpy()):
                    eval_2d3d_iuo_from_tensors(
                        est[None], gt[None], true_eval, e_ind)

                if ind[1] == 0:
                    loss = compute_weighted_L1(y_bon_est.to(
                        self.device), y_bon_ref.to(self.device), std.to(self.device))
                else:
                    loss = compute_weighted_L1(y_bon_est[:, ind[0]:(ind[1]+1)].to(
                        self.device), y_bon_ref[:, ind[0]:(ind[1]+1)].to(self.device), std[:, ind[0]:(ind[1]+1)].to(self.device))

                local_eval = dict(
                    loss=loss,
                )
                local_eval["2DIoU"] = torch.FloatTensor(
                    [true_eval["2DIoU"]]).mean()
                local_eval["3DIoU"] = torch.FloatTensor(
                    [true_eval["3DIoU"]]).mean()
            try:
                for k, v in local_eval.items():
                    if v.isnan():
                        continue
                    total_eval[k] = total_eval.get(k, 0) + v.item() * x.size(0)
            except:
                invalid_cnt += 1
                pass

        if only_val:
            scaler_value = self.cfg.runners.valid_iou.batch_size * \
                (len(iterator_valid_iou) - invalid_cnt)
            curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
            curr_score_2d_iou = total_eval["2DIoU"] / scaler_value
            self.logger.info(
                f"3D-IoU score(ceiling 2Diou in pp): {curr_score_3d_iou:.4f}")
            self.logger.info(
                f"2D-IoU score(floor 2Diou in pp): {curr_score_2d_iou:.4f}")
            return {"2D-IoU": curr_score_2d_iou, "3D-IoU": curr_score_3d_iou}

        scaler_value = self.cfg.runners.valid_iou.batch_size * \
            (len(iterator_valid_iou) - invalid_cnt)
        '''
        for k, v in total_eval.items():
            k = "valid_IoU/%s" % k
            self.tb_writer.add_scalar(
                k, v / scaler_value, self.current_epoch)
        '''
        # Save best validation loss model
        curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
        curr_score_2d_iou = total_eval["2DIoU"] / scaler_value

        # ! Saving current score
        self.curr_scores['iou_valid_scores'] = dict(
            best_3d_iou_score=curr_score_3d_iou,
            best_2d_iou_score=curr_score_2d_iou
        )

        if self.best_scores.get("best_iou_valid_score") is None:
            self.logger.info(f"Best 3D-IoU score: {curr_score_3d_iou:.4f}")
            self.logger.info(f"Best 2D-IoU score: {curr_score_2d_iou:.4f}")
            self.best_scores["best_iou_valid_score"] = dict(
                best_3d_iou_score=curr_score_3d_iou,
                best_2d_iou_score=curr_score_2d_iou
            )
        else:
            best_3d_iou_score = self.best_scores["best_iou_valid_score"]['best_3d_iou_score']
            best_2d_iou_score = self.best_scores["best_iou_valid_score"]['best_2d_iou_score']

            self.logger.info(
                f"3D-IoU: Best: {best_3d_iou_score:.4f} vs Curr:{curr_score_3d_iou:.4f}")
            self.logger.info(
                f"2D-IoU: Best: {best_2d_iou_score:.4f} vs Curr:{curr_score_2d_iou:.4f}")

            if best_3d_iou_score < curr_score_3d_iou:
                self.logger.info(
                    f"New 3D-IoU Best Score {curr_score_3d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_3d_iou_score'] = curr_score_3d_iou
                # self.save_model("best_3d_iou_valid.pth")

            if best_2d_iou_score < curr_score_2d_iou:
                self.logger.info(
                    f"New 2D-IoU Best Score {curr_score_2d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_2d_iou_score'] = curr_score_2d_iou
                # self.save_model("best_2d_iou_valid.pth")

    def save_model(self, filename):
        if not self.cfg.model.get("save_ckpt", True):
            return

        # ! Saving the current model
        state_dict = OrderedDict(
            {
                "args": self.cfg,
                "kwargs": {
                    "backbone": self.net.backbone,
                    "use_rnn": self.net.use_rnn,
                },
                "state_dict": self.net.state_dict(),
            }
        )
        torch.save(state_dict, os.path.join(
            self.dir_ckpt, filename))

    def prepare_for_training(self):
        # self.is_training = True
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()
        # self.set_optimizer()
        # self.set_scheduler()
        # self.set_train_dataloader()
        self.set_log_dir()
        save_cfg(os.path.join(self.dir_ckpt, 'cfg.yaml'), self.cfg)

    def set_log_dir(self):
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        create_directory(output_dir, delete_prev=False)
        self.logger.info(f"Output directory: {output_dir}")
        self.dir_log = os.path.join(output_dir, 'log')
        self.dir_ckpt = os.path.join(output_dir, 'ckpt')
        os.makedirs(self.dir_log, exist_ok=True)
        os.makedirs(self.dir_ckpt, exist_ok=True)

    def set_valid_dataloader(self):
        self.logger.info("Setting IoU Validation Dataloader")
        self.valid_iou_loader = DataLoader(
            MVLDataLoader(self.cfg.runners.valid_iou),
            # MVLDataLoader(self.cfg.runners.train),
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())

    def set_valid_dataloader_lsun(self):
        self.logger.info("Setting IoU Validation Dataloader")
        # lsun_dataset = LSUNDataset(self.cfg.mvl_dir, 'validation', [512, 512], False)
        lsun_dataset = LSUNPreprocDataset(
            self.cfg.mvl_dir, 'validation', [256, 256], False)
        print('lsun_dataset_val:', len(lsun_dataset))
        self.valid_iou_loader = DataLoader(
            lsun_dataset,
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())

    def set_valid_dataloader_mp3d_layout(self):
        self.logger.info("Setting IoU Validation Dataloader")
        mp3d_layout_dataset = Matterport3Dlayoutdataset(
            subset='validation', root_dir=self.cfg.mvl_dir)
        print('mp3d_dataset_val:', len(mp3d_layout_dataset))
        self.valid_iou_loader = DataLoader(
            mp3d_layout_dataset,
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())

    def set_multi_valid_dataloader_mp3d(self, mp3d_yaml_dir):
        self.logger.info("Setting IoU Validation Dataloader")
        # MVL_dataset = MVLDataLoader(self.cfg.runners.valid_iou)
        # MVL_dataset = MVLDataLoader(self.cfg.runners.train)
        # print('MVL_dataset_val:', len(MVL_dataset))
        with open(mp3d_yaml_dir, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # mp3d_dataset = Matterport3DDataset(**config['dataset_args']['test'])
        mp3d_dataset = MP3Dcroppanodataset(**config['dataset_args']['val'])
        print('mp3d_dataset_val:', len(mp3d_dataset))
        # concat_dataset = ConcatDataset([MVL_dataset, mp3d_dataset])
        # print('concat_dataset_val:', len(concat_dataset))

        self.valid_iou_loader = DataLoader(
            # concat_dataset,
            mp3d_dataset,
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())

    def set_multi_valid_dataloader(self):
        self.logger.info("Setting IoU Validation Dataloader")
        MVL_dataset = MVLDataLoader(self.cfg.runners.valid_iou)
        # MVL_dataset = MVLDataLoader(self.cfg.runners.train)
        MVL_dataset_2 = MVLDataLoader(self.cfg.runners.valid_iou_2)
        print('MVL_dataset_val:', len(MVL_dataset))
        print('MVL_dataset_val:', len(MVL_dataset_2))
        concat_dataset = ConcatDataset([MVL_dataset, MVL_dataset_2])
        print('concat_dataset_val:', len(concat_dataset))

        self.valid_iou_loader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())

    def plot_predict_and_gt(self):
        from mvl_challenge.models.LGTNET.postprocessing.post_process import post_process
        from mvl_challenge.models.LGTNET.utils.misc import tensor2np_d, tensor2np
        from mvl_challenge.models.LGTNET.utils.conversion import depth2xyz, uv2pixel
        from mvl_challenge.models.LGTNET.utils.boundary import corners2boundaries
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        raw_data = json.load(open(self.cfg.val_scene_list))
        list_rooms = list(raw_data.keys())
        list_frames = [raw_data[room] for room in list_rooms]
        list_frames = [item for sublist in list_frames for item in sublist]
        iter_list_frame = iter(list_frames)
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        self.logger.info(f"Output directory: {output_dir}")
        dst_dir = output_dir + '/inference_img'
        # dst_dir_gt = dst_dir + '/gt/'
        dst_dir_est = dst_dir + '/predict/'
        # pathlib.Path(dst_dir_gt).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dst_dir_est).mkdir(parents=True, exist_ok=True)
        image_fn = self.cfg.mvl_dir + '/img'

        for _ in trange(len(iterator_valid_iou), desc="plot image epoch %d" % self.current_epoch):
            x, y_bon_ref, std = next(iterator_valid_iou)

            with torch.no_grad():
                y_bon_est_cpu = np.zeros((y_bon_ref.size(dim=0), 2, 1024))
                dt = self.net(x.to(self.device))
                if self.cfg.post_processing != 'original':
                    dt['processed_xyz'] = post_process(
                        tensor2np(dt['depth']), type_name=self.cfg.post_processing)
                dt_np = tensor2np_d(dt)
                for i in range(len(dt_np['depth'])):
                    dt_depth = dt_np['depth'][i]
                    dt_xyz = depth2xyz(np.abs(dt_depth))
                    dt_ratio = dt_np['ratio'][i][0]
                    if 'processed_xyz' in dt:
                        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][i], step=None, visible=False,
                                                           length=1024)
                        # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                        # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                        dt_ceiling = uv2pixel(dt_boundaries[1])
                        dt_floor = uv2pixel(dt_boundaries[0])
                        if len(dt_ceiling) != 1024 or len(dt_floor) != 1024:
                            # C_XY[:,0] non repeat index
                            dt_ceiling_index = np.unique(
                                dt_ceiling[:, 0], return_index=True)
                            # F_XY[:,0] non repeat index
                            dt_floor_index = np.unique(
                                dt_floor[:, 0], return_index=True)
                            dt_ceiling_pixel = np.transpose(
                                dt_ceiling[dt_ceiling_index[1], :])
                            dt_floor_pixel = np.transpose(
                                dt_floor[dt_floor_index[1], :])
                            dt_ceiling_pixel = complementary_element(
                                dt_ceiling_pixel, [512, 1024])
                            dt_floor_pixel = complementary_element(
                                dt_floor_pixel, [512, 1024])
                        else:
                            dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                            dt_floor_pixel = np.transpose(dt_floor)[1]
                        # pdb.set_trace()

                    else:
                        dt_boundaries = corners2boundaries(
                            dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=1024)
                        dt_ceiling_pixel = np.transpose(
                            uv2pixel(dt_boundaries[1]))[1]
                        dt_floor_pixel = np.transpose(
                            uv2pixel(dt_boundaries[0]))[1]

                    y_bon_est_cpu[i, 0, :] = dt_ceiling_pixel
                    y_bon_est_cpu[i, 1, :] = dt_floor_pixel
                y_bon_est = torch.tensor(
                    (y_bon_est_cpu / 512 - 0.5) * np.pi).to(self.device)
                for gt, est in zip(y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy()):
                    img_name = next(iter_list_frame)
                    img_path = image_fn + '/' + img_name
                    if os.path.exists(img_path + '.jpg'):
                        img_path = img_path + '.jpg'
                        img = read_image(img_path, [512, 1024])*255
                        img_c = img.copy()
                        img_name = img_name + '.jpg'
                    elif os.path.exists(img_path + '.png'):
                        img_path = img_path + '.png'
                        img = read_image(img_path, [512, 1024])*255
                        img_c = img.copy()
                        img_name = img_name + '.png'
                    gt_pixel = ((gt/np.pi + 0.5)*img.shape[0]).round()
                    est_pixel = ((est/np.pi + 0.5)*img.shape[0]).round()
                    v_x = np.linspace(
                        0, img.shape[1] - 1, img.shape[1]).astype(int)

                    gt_pixel_ceiling = np.vstack(
                        (v_x, gt_pixel[0])).transpose()
                    # pdb.set_trace()
                    plotXY(img, gt_pixel_ceiling, color=(255, 0, 0))
                    gt_pixel_floor = np.vstack((v_x, gt_pixel[1])).transpose()
                    plotXY(img, gt_pixel_floor, color=(255, 0, 0))

                    est_pixel_ceiling = np.vstack(
                        (v_x, est_pixel[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(255, 160, 0))
                    # (255,255,0) yellow
                    # (0,255,255) blue
                    # (0,255,0) green
                    # (255,160,0) orange
                    # plotXY(img, est_pixel_ceiling, color=(0,0,255))
                    est_pixel_floor = np.vstack(
                        (v_x, est_pixel[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(255, 160, 0))
                    # plotXY(img, est_pixel_floor, color=(0,0,255))

                    imwrite(dst_dir_est+img_name, (img).astype(np.uint8))
                    # imwrite(dst_dir_est+img_name,(img_c).astype(np.uint8))

    def plot_predict_and_gt_mp3d(self):
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        self.logger.info(f"Output directory: {output_dir}")
        dst_dir_est = output_dir + '/inference_img/'
        # dst_dir_gt = dst_dir + '/gt/'
        # pathlib.Path(dst_dir_gt).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dst_dir_est).mkdir(parents=True, exist_ok=True)
        i = 0

        for _ in trange(len(iterator_valid_iou), desc="plot image epoch %d" % self.current_epoch):
            # x, y_bon_ref, std = next(iterator_valid_iou)
            x, y_bon_ref, std, u_range, eval_range = next(iterator_valid_iou)
            with torch.no_grad():
                y_bon_est_cpu = np.zeros((y_bon_ref.size(dim=0), 2, 1024))
                dt = self.net(x.to(self.device))

                y_bon_est = self.predict_to_phi_coords(dt)
                for image, gt, est in zip(x, y_bon_ref, y_bon_est):
                    img = image.detach().cpu().numpy().transpose([1, 2, 0])
                    img = (img.copy()*255).astype(np.uint8)
                    # pdb.set_trace()
                    gt_pixel = ((gt/np.pi + 0.5)*img.shape[0]).round()
                    est_pixel = ((est/np.pi + 0.5)*img.shape[0]).round().cpu()
                    v_x = np.linspace(
                        0, img.shape[1] - 1, img.shape[1]).astype(int)

                    gt_pixel_ceiling = np.vstack(
                        (v_x, gt_pixel[0])).transpose()
                    # pdb.set_trace()
                    plotXY(img, gt_pixel_ceiling, color=(255, 0, 0))
                    gt_pixel_floor = np.vstack((v_x, gt_pixel[1])).transpose()
                    plotXY(img, gt_pixel_floor, color=(255, 0, 0))

                    est_pixel_ceiling = np.vstack(
                        (v_x, est_pixel[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(0, 255, 255))
                    # (255,255,0) yellow
                    # (0,255,255) blue
                    # (0,255,0) green
                    # (255,160,0) orange
                    # plotXY(img, est_pixel_ceiling, color=(0,0,255))
                    est_pixel_floor = np.vstack(
                        (v_x, est_pixel[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(0, 255, 255))
                    # plotXY(img, est_pixel_floor, color=(0,0,255))

                    imwrite(dst_dir_est+"img_%d.jpg" %
                            i, (img).astype(np.uint8))
                    i += 1

    def gt_est_pixel_error(self):
        from mvl_challenge.models.LGTNET.postprocessing.post_process import post_process
        from mvl_challenge.models.LGTNET.utils.misc import tensor2np_d, tensor2np
        from mvl_challenge.models.LGTNET.utils.conversion import depth2xyz, uv2pixel
        from mvl_challenge.models.LGTNET.utils.boundary import corners2boundaries
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        ceiling_lonlat_error = np.array([0])
        floor_lonlat_error = np.array([0])
        ceiling_pixel_error = np.array([0])
        floor_pixel_error = np.array([0])
        shape = [512, 1024]
        for _ in trange(len(iterator_valid_iou), desc="compute phi_coords_error %d" % self.current_epoch):
            x, y_bon_ref, std = next(iterator_valid_iou)

            with torch.no_grad():
                y_bon_est_cpu = np.zeros((y_bon_ref.size(dim=0), 2, 1024))
                dt = self.net(x.to(self.device))
                if self.cfg.post_processing != 'original':
                    dt['processed_xyz'] = post_process(
                        tensor2np(dt['depth']), type_name=self.cfg.post_processing)
                dt_np = tensor2np_d(dt)
                for i in range(len(dt_np['depth'])):
                    dt_depth = dt_np['depth'][i]
                    dt_xyz = depth2xyz(np.abs(dt_depth))
                    dt_ratio = dt_np['ratio'][i][0]
                    if 'processed_xyz' in dt:
                        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][i], step=None, visible=False,
                                                           length=1024)
                        # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                        # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                        dt_ceiling = uv2pixel(dt_boundaries[1])
                        dt_floor = uv2pixel(dt_boundaries[0])
                        if len(dt_ceiling) != 1024 or len(dt_floor) != 1024:
                            # C_XY[:,0] non repeat index
                            dt_ceiling_index = np.unique(
                                dt_ceiling[:, 0], return_index=True)
                            # F_XY[:,0] non repeat index
                            dt_floor_index = np.unique(
                                dt_floor[:, 0], return_index=True)
                            dt_ceiling_pixel = np.transpose(
                                dt_ceiling[dt_ceiling_index[1], :])
                            dt_floor_pixel = np.transpose(
                                dt_floor[dt_floor_index[1], :])
                            dt_ceiling_pixel = complementary_element(
                                dt_ceiling_pixel, [512, 1024])
                            dt_floor_pixel = complementary_element(
                                dt_floor_pixel, [512, 1024])
                        else:
                            dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                            dt_floor_pixel = np.transpose(dt_floor)[1]
                        # pdb.set_trace()

                    else:
                        dt_boundaries = corners2boundaries(
                            dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=1024)
                        dt_ceiling_pixel = np.transpose(
                            uv2pixel(dt_boundaries[1]))[1]
                        dt_floor_pixel = np.transpose(
                            uv2pixel(dt_boundaries[0]))[1]

                    y_bon_est_cpu[i, 0, :] = dt_ceiling_pixel
                    y_bon_est_cpu[i, 1, :] = dt_floor_pixel
                y_bon_est = torch.tensor(
                    (y_bon_est_cpu / 512 - 0.5) * np.pi).to(self.device)
                for gt, est in zip(y_bon_ref, y_bon_est):
                    est_pixel = torch.round(
                        (est/np.pi + 0.5)*shape[0]).int().to(self.device)
                    gt_pixel = torch.round(
                        (gt/np.pi + 0.5)*shape[0]).int().to(self.device)

                    v_x = torch.linspace(
                        0, shape[1]-1, shape[1]).int().to(self.device)
                    # pdb.set_trace()
                    gt_pixel_ceiling = torch.transpose(
                        torch.vstack((v_x, gt_pixel[0])), 0, 1)
                    gt_pixel_floor = torch.transpose(
                        torch.vstack((v_x, gt_pixel[1])), 0, 1)
                    est_pixel_ceiling = torch.transpose(
                        torch.vstack((v_x, est_pixel[0])), 0, 1)
                    est_pixel_floor = torch.transpose(
                        torch.vstack((v_x, est_pixel[1])), 0, 1)

                    # gt_xyz_ceiling = XY2xyz(gt_pixel_ceiling, shape, mode='torch')
                    gt_lonlat_ceiling = XY2lonlat(
                        gt_pixel_ceiling, shape, mode='torch')
                    # gt_xyz_floor = XY2xyz(gt_pixel_floor, shape, mode='torch')
                    gt_lonlat_floor = XY2lonlat(
                        gt_pixel_floor, shape, mode='torch')

                    # est_xyz_ceiling = XY2xyz(est_pixel_ceiling, shape, mode='torch')
                    est_lonlat_ceiling = XY2lonlat(
                        est_pixel_ceiling, shape, mode='torch')
                    # est_xyz_floor = XY2xyz(est_pixel_floor, shape, mode='torch')
                    est_lonlat_floor = XY2lonlat(
                        est_pixel_floor, shape, mode='torch')

                    ceiling_each_error = torch.zeros((1024)).to(self.device)
                    floor_each_error = torch.zeros((1024)).to(self.device)

                    for i in range(v_x.size(dim=0)):
                        ceiling_point_error = torch.norm(
                            est_lonlat_ceiling[i] - gt_lonlat_ceiling[i])
                        floor_point_error = torch.norm(
                            est_lonlat_floor[i] - gt_lonlat_floor[i])
                        # pdb.set_trace()
                        ceiling_each_error[i] = ceiling_point_error
                        floor_each_error[i] = floor_point_error
                    avg_each_ceiling_pixel_error = np.mean(
                        torch.abs(est_pixel_ceiling - gt_pixel_ceiling).cpu().numpy())
                    avg_each_floor_pixel_error = np.mean(
                        torch.abs(est_pixel_floor - gt_pixel_floor).cpu().numpy())
                    avg_ceiling_lonlat_error = np.mean(
                        ceiling_each_error[:].cpu().numpy())
                    avg_floor_lonlat_error = np.mean(
                        floor_each_error[:].cpu().numpy())
                    ceiling_pixel_error = np.append(
                        ceiling_pixel_error,  avg_each_ceiling_pixel_error)
                    floor_pixel_error = np.append(
                        floor_pixel_error,  avg_each_floor_pixel_error)
                    ceiling_lonlat_error = np.append(
                        ceiling_lonlat_error,  avg_ceiling_lonlat_error)
                    floor_lonlat_error = np.append(
                        floor_lonlat_error,  avg_floor_lonlat_error)
                    pdb.set_trace()
        avg_ceiling_pixel_error = np.mean(ceiling_pixel_error[1:])
        avg_floor_pixel_error = np.mean(floor_pixel_error[1:])
        avg_ceiling_lonlat_error = np.mean(ceiling_lonlat_error[1:])/np.pi*180
        avg_floor_lonlat_error = np.mean(floor_lonlat_error[1:])/np.pi*180
        # pdb.set_trace()
        '''
        logging.info(
            f"ceiling pixel error: {avg_ceiling_pixel_error:.4f}")
        logging.info(
            f"floor pixel error: {avg_floor_pixel_error:.4f}")
        logging.info(
            f"ceiling lonlat error: {avg_ceiling_lonlat_error:.4f}")
        logging.info(
            f"floor lonlat error: {avg_floor_lonlat_error:.4f}")
        '''
        self.curr_scores['valid_pixel_error'] = dict(
            Ceiling_pixel_error=avg_ceiling_pixel_error,
            Floor_pixel_error=avg_floor_pixel_error
        )

        if self.best_scores.get("best_pixel_error") is None:
            self.logger.info(
                f"Best ceiling pixel error: {avg_ceiling_pixel_error:.4f}")
            self.logger.info(
                f"Best floor pixel error: {avg_floor_pixel_error:.4f}")
            self.best_scores["best_pixel_error"] = dict(
                best_ceiling_pixel_error=avg_ceiling_pixel_error,
                best_floor_pixel_error=avg_floor_pixel_error
            )
        else:
            best_ceiling_pixel_error = self.best_scores["best_pixel_error"]['best_ceiling_pixel_error']
            best_floor_pixel_error = self.best_scores["best_pixel_error"]['best_floor_pixel_error']

            self.logger.info(
                f"ceiling_pixel_error: Best: {best_ceiling_pixel_error:.4f} vs Curr:{avg_ceiling_pixel_error:.4f}")
            self.logger.info(
                f"floor_pixel_error: Best: {best_floor_pixel_error:.4f} vs Curr:{avg_floor_pixel_error:.4f}")

            if best_ceiling_pixel_error > avg_ceiling_pixel_error:
                self.logger.info(
                    f"New ceiling pixel error {avg_ceiling_pixel_error: 0.4f}")
                self.best_scores["best_pixel_error"]['best_ceiling_pixel_error'] = avg_ceiling_pixel_error
                self.save_model("best_ceiling_pixel_error.pth")

            if best_floor_pixel_error < avg_floor_pixel_error:
                self.logger.info(
                    f"New floor pixel error {avg_floor_pixel_error: 0.4f}")
                self.best_scores["best_pixel_error"]['best_floor_pixel_error'] = avg_floor_pixel_error
                self.save_model("best_floor_pixel_error.pth")

    @torch.no_grad()
    def predict_to_phi_coords(self, dt):
        from mvl_challenge.models.LGTNET.postprocessing.post_process import post_process
        from mvl_challenge.models.LGTNET.utils.misc import tensor2np_d, tensor2np
        from mvl_challenge.models.LGTNET.utils.conversion import depth2xyz, uv2pixel
        from mvl_challenge.models.LGTNET.utils.boundary import corners2boundaries
        if self.cfg.post_processing != 'original':
            dt['processed_xyz'] = post_process(
                tensor2np(dt['depth']), type_name=self.cfg.post_processing)
        dt_np = tensor2np_d(dt)
        y_bon_est_cpu = np.zeros((len(dt_np['depth']), 2, 1024))
        for i in range(len(dt_np['depth'])):
            dt_depth = dt_np['depth'][i]
            dt_xyz = depth2xyz(np.abs(dt_depth))
            dt_ratio = dt_np['ratio'][i][0]
            if 'processed_xyz' in dt:
                dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][i], step=None, visible=False,
                                                   length=1024)
                # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                dt_ceiling = uv2pixel(dt_boundaries[1])
                dt_floor = uv2pixel(dt_boundaries[0])
                if len(dt_ceiling) != 1024 or len(dt_floor) != 1024:
                    # C_XY[:,0] non repeat index
                    dt_ceiling_index = np.unique(
                        dt_ceiling[:, 0], return_index=True)
                    # F_XY[:,0] non repeat index
                    dt_floor_index = np.unique(
                        dt_floor[:, 0], return_index=True)
                    dt_ceiling_pixel = np.transpose(
                        dt_ceiling[dt_ceiling_index[1], :])
                    dt_floor_pixel = np.transpose(
                        dt_floor[dt_floor_index[1], :])
                    dt_ceiling_pixel = complementary_element(
                        dt_ceiling_pixel, [512, 1024])
                    dt_floor_pixel = complementary_element(
                        dt_floor_pixel, [512, 1024])
                else:
                    dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                    dt_floor_pixel = np.transpose(dt_floor)[1]
                # pdb.set_trace()

            else:
                dt_boundaries = corners2boundaries(
                    dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=1024)
                # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                dt_ceiling = uv2pixel(dt_boundaries[1])
                dt_floor = uv2pixel(dt_boundaries[0])
                if len(dt_ceiling) != 1024 or len(dt_floor) != 1024:
                    # C_XY[:,0] non repeat index
                    dt_ceiling_index = np.unique(
                        dt_ceiling[:, 0], return_index=True)
                    # F_XY[:,0] non repeat index
                    dt_floor_index = np.unique(
                        dt_floor[:, 0], return_index=True)
                    dt_ceiling_pixel = np.transpose(
                        dt_ceiling[dt_ceiling_index[1], :])
                    dt_floor_pixel = np.transpose(
                        dt_floor[dt_floor_index[1], :])
                    dt_ceiling_pixel = complementary_element(
                        dt_ceiling_pixel, [512, 1024])
                    dt_floor_pixel = complementary_element(
                        dt_floor_pixel, [512, 1024])
                else:
                    dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                    dt_floor_pixel = np.transpose(dt_floor)[1]

            y_bon_est_cpu[i, 0, :] = dt_ceiling_pixel
            y_bon_est_cpu[i, 1, :] = dt_floor_pixel
        y_bon_est = torch.tensor(
            (y_bon_est_cpu / 512 - 0.5) * np.pi).to(self.device)

        return y_bon_est

    def save_predict(self):
        output_img_dir = "/media/Pluto/jonathan/mvl_toolkit/output_data/img/"
        output_label_dir = "/media/Pluto/jonathan/mvl_toolkit/output_data/label/"
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        shape = [512, 1024]
        i = 0
        for _ in trange(len(iterator_valid_iou), desc="save_predict %d" % self.current_epoch):
            x, y_bon_ref, std = next(iterator_valid_iou)

            with torch.no_grad():
                dt = self.net(x.to(self.device))
                y_bon_est = self.predict_to_phi_coords(dt)
                for image, gt, est in zip(x, y_bon_ref, y_bon_est):
                    est_pixel = torch.round(
                        (est/np.pi + 0.5)*shape[0]).int().cpu().numpy().tolist()
                    # gt_pixel = torch.round((gt/np.pi + 0.5)*shape[0]).int().to(self.device)
                    label = {"phi_coords": est_pixel}
                    img = image.detach().cpu().numpy().transpose([1, 2, 0])
                    imwrite(output_img_dir+"img_%s.jpg" %
                            (str(i)), (img*255).astype(np.uint8))
                    save_json_dict(output_label_dir +
                                   "label_%s.json" % (str(i)), label)
                    i += 1

    def plot_predict_pers(self):
        from mvl_challenge.models.LGTNET.postprocessing.post_process import post_process
        from mvl_challenge.models.LGTNET.utils.misc import tensor2np_d, tensor2np
        from mvl_challenge.models.LGTNET.utils.conversion import depth2xyz, uv2pixel
        from mvl_challenge.models.LGTNET.utils.boundary import corners2boundaries
        print_cfg_information(self.cfg)
        self.net.eval()
        src = '/media/Pluto/jonathan/mvl_toolkit/output_data/img'
        lst = sorted(glob.glob(src+'/*.png') + glob.glob(src+'/*.jpg'))
        dst_dir_est = '/media/Pluto/jonathan/mvl_toolkit/output_data/predict_pers'
        os.makedirs(dst_dir_est, exist_ok=True)
        equi_shape = [512, 640]
        i = 0
        for one in tqdm(lst[:20]):
            img = read_image(one, [512, 1024])
            img = img[0:256, 0:320, :]
            img = cv2.resize(img, dsize=(640, 512),
                             interpolation=cv2.INTER_AREA)
            x = torch.FloatTensor(img).permute(
                2, 0, 1)[None, ...].to(self.device)

            with torch.no_grad():
                y_bon_est_cpu = np.zeros((2, equi_shape[1]))
                dt = self.net(x.to(self.device))
                if self.cfg.post_processing != 'original':
                    dt['processed_xyz'] = post_process(
                        tensor2np(dt['depth']), type_name=self.cfg.post_processing)
                dt_np = tensor2np_d(dt)
                dt_depth = dt_np['depth'][0]
                # depth1, depth2 = np.split(dt_np['depth'][0], 2)
                # dt_depth = (depth1 + np.flip(depth2)) / 21
                dt_xyz = depth2xyz(np.abs(dt_depth))
                dt_ratio = dt_np['ratio'][0][0]
                if 'processed_xyz' in dt:
                    dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][0], step=None, visible=False,
                                                       length=equi_shape[1]*2)
                    # dt_ceiling_pixel = np.transpose(uv2pixel(dt_boundaries[1]))[1]
                    # dt_floor_pixel = np.transpose(uv2pixel(dt_boundaries[0]))[1]
                    dt_ceiling = uv2pixel(dt_boundaries[1], w=equi_shape[1]*2)
                    dt_floor = uv2pixel(dt_boundaries[0], w=equi_shape[1]*2)
                    if len(dt_ceiling) != equi_shape[1]*2 or len(dt_floor) != equi_shape[1]*2:
                        # C_XY[:,0] non repeat index
                        dt_ceiling_index = np.unique(
                            dt_ceiling[:, 0], return_index=True)
                        # F_XY[:,0] non repeat index
                        dt_floor_index = np.unique(
                            dt_floor[:, 0], return_index=True)
                        dt_ceiling_pixel = np.transpose(
                            dt_ceiling[dt_ceiling_index[1], :])
                        dt_floor_pixel = np.transpose(
                            dt_floor[dt_floor_index[1], :])
                        dt_ceiling_pixel = complementary_element(
                            dt_ceiling_pixel, equi_shape)
                        dt_floor_pixel = complementary_element(
                            dt_floor_pixel, equi_shape)
                    else:
                        dt_ceiling_pixel = np.transpose(dt_ceiling)[1]
                        dt_floor_pixel = np.transpose(dt_floor)[1]
                    # pdb.set_trace()

                else:
                    dt_boundaries = corners2boundaries(
                        dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=1024)
                    dt_ceiling_pixel = np.transpose(
                        uv2pixel(dt_boundaries[1]))[1]
                    dt_floor_pixel = np.transpose(
                        uv2pixel(dt_boundaries[0]))[1]

                y_bon_est_cpu[0, :] = dt_ceiling_pixel[0:640]
                y_bon_est_cpu[1, :] = dt_floor_pixel[0:640]
                v_x = np.linspace(
                    0, equi_shape[1] - 1, equi_shape[1]).astype(int)
                est_pixel_ceiling = np.vstack(
                    (v_x, y_bon_est_cpu[0])).transpose()
                plotXY(img, est_pixel_ceiling, color=(0, 1, 1))
                # (255,255,0) yellow
                # (0,255,255) blue
                # (0,255,0) green
                # (255,160,0) orange
                # plotXY(img, est_pixel_ceiling, color=(0,0,255))
                est_pixel_floor = np.vstack(
                    (v_x, y_bon_est_cpu[1])).transpose()
                plotXY(img, est_pixel_floor, color=(0, 1, 1))
                # plotXY(img, est_pixel_floor, color=(0,0,255))

                imwrite(dst_dir_est+"/img_%d.jpg" %
                        i, (img*255).astype(np.uint8))
                i += 1

    def plot_predict_and_gt_mp3d_layout(self):
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        self.logger.info(f"Output directory: {output_dir}")
        dst_dir_est = output_dir + '/inference_img/'
        pathlib.Path(dst_dir_est).mkdir(parents=True, exist_ok=True)
        i = 0

        equi_shape = [512, 1024]
        for _ in trange(len(iterator_valid_iou), desc="save_predict %d" % self.current_epoch):
            x, y_bon_ref, c_intrinsic, whole_map_ps = next(iterator_valid_iou)

            w = x.size(dim=-1)
            # '''
            with torch.no_grad():
                dt = self.net(x.to(self.device))
                y_bon_est = self.predict_to_phi_coords(dt)
                for image, gt, est in zip(x, y_bon_ref, y_bon_est):
                    image = image.detach().cpu().numpy().transpose([1, 2, 0])
                    img = (image.copy()*255).astype(np.uint8)
                    # pdb.set_trace()
                    gt_pixel = ((gt + 0.5)*img.shape[0]).round()
                    # gt_pixel = gt
                    est_pixel = ((est/np.pi + 0.5) *
                                 img.shape[0]).round().cpu().numpy()
                    v_x = np.linspace(
                        0, img.shape[1] - 1, img.shape[1]).astype(int)

                    gt_pixel_ceiling = np.vstack(
                        (v_x, gt_pixel[0])).transpose()
                    # pdb.set_trace()
                    plotXY(img, gt_pixel_ceiling, color=(255, 0, 0))
                    gt_pixel_floor = np.vstack((v_x, gt_pixel[1])).transpose()
                    plotXY(img, gt_pixel_floor, color=(255, 0, 0))

                    est_pixel_ceiling = np.vstack(
                        (v_x, est_pixel[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(0, 255, 255))
                    # (255,255,0) yellow
                    # (0,255,255) blue
                    # (0,255,0) green
                    # (255,160,0) orange
                    # plotXY(img, est_pixel_ceiling, color=(0,0,255))
                    est_pixel_floor = np.vstack(
                        (v_x, est_pixel[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(0, 255, 255))
                    # plotXY(img, est_pixel_floor, color=(0,0,255))
                    imwrite(dst_dir_est +
                            f"img_{i}_lsun.jpg", img.astype(np.uint8))
                    # imwrite(dst_dir_est + f"img_{i}_postprocess.jpg",img.astype(np.uint8))
                    i += 1
            # '''
            '''
            with torch.no_grad():
                dt = self.net(x.to(self.device))
                c_intrinsic = c_intrinsic.to(self.device)
                w = x.size(dim=-1)
                h = x.size(dim=-2)
                y_bon_ref = (y_bon_ref + 0.5)*(h-1)
                y_bon_est = np.zeros((len(dt['depth']),2,w))
                for i in range(len(dt['depth'])):
                    dt_depth = torch.abs(dt['depth'][i])
                    dt_ratio = dt['ratio'][i][0]
                    f_dt_xyz = pt('torch').equi_depth2XYZ(dt_depth, [h,w])
                    c_dt_xyz = pt('torch').equi_depth2XYZ(dt_depth, [h,w], ch=-dt_ratio)
                    f_pers_xyz = pt('torch').XYZ2pers_XYZ(f_dt_xyz, y_bon_ref[i][1], c_intrinsic[i])
                    c_pers_xyz = pt('torch').XYZ2pers_XYZ(c_dt_xyz, y_bon_ref[i][0], c_intrinsic[i])
                    pdb.set_trace()
                    f_dt_xy = pt('torch').XYZ2boundary(f_pers_xyz, c_intrinsic[i]).cpu().numpy()
                    c_dt_xy = pt('torch').XYZ2boundary(c_pers_xyz, c_intrinsic[i]).cpu().numpy()
                    
                    dt_floor_pixel = complementary_element(f_dt_xy, equi_shape)
                    dt_ceiling_pixel = complementary_element(c_dt_xy, equi_shape)
                    
                    y_bon_est[i,0,:] = dt_ceiling_pixel
                    y_bon_est[i,1,:] = dt_floor_pixel

                for image, gt, est in zip(x, y_bon_ref.cpu().numpy(), y_bon_est):
                    pdb.set_trace()
                    img = image.detach().cpu().numpy().transpose([1, 2, 0])
                    img = (img.copy()*255).astype(np.uint8)
                    #pdb.set_trace()
                    v_x = np.linspace(0, img.shape[1] - 1, img.shape[1]).astype(int)
                    gt = np.round((gt + 0.5)*img.shape[0])
                    gt_pixel_ceiling = np.vstack((v_x, gt[0])).transpose()
                    gt_pixel_floor = np.vstack((v_x, gt[1])).transpose()

                    plotXY(img, gt_pixel_ceiling, color=(255,0,0))
                    plotXY(img, gt_pixel_floor, color=(255,0,0))

                    est_pixel_ceiling = np.vstack((v_x, est[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(0,255,255))
                    est_pixel_floor = np.vstack((v_x, est[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(0,255,255))

                    imwrite(dst_dir_est + f"img_{i}.jpg",img.astype(np.uint8))
                    i+=1
            '''
            '''
            #pers2pano perspective predict
            with torch.no_grad():
                dt = self.net(x.to(self.device))
                y_bon_est = self.predict_to_phi_coords(dt)
                for image, gt, est, c_int in zip(x, y_bon_ref, y_bon_est, c_intrinsic):
                    e2c = Equirec2Cube(512, 1024, 256, c_int,CUDA=True)
                    img = image[None,...].to(self.device)
                    h = img.size(dim=2)
                    w = img.size(dim=3)
                    #img = (img.copy()*255).astype(np.uint8)
                    gt_pixel = ((gt + 0.5)*h/2).round().cpu().numpy()
                    est_pixel = ((est/np.pi + 0.5)*h).int()
                    zeros_img1 = torch.zeros((3,h,w)).to(self.device)
                    zeros_img2 = torch.zeros((3,h,w)).to(self.device)
                    
                    for j in range(w):
                        zeros_img1[0,est_pixel[0,j],j] = 0.5
                        zeros_img2[0,est_pixel[1,j],j] = 0.5
                    zeros_img1 = zeros_img1[None,...]
                    zeros_img2 = zeros_img2[None,...]
                    cubemap_tensor_img = e2c(img).to(self.device)
                    cubemap_tensor_bc = e2c(zeros_img1).to(self.device)
                    cubemap_tensor_bf = e2c(zeros_img2).to(self.device)

                    cube_img = cubemap_tensor_img[2].permute(1, 2, 0).cpu().numpy()
                    cube_bc = cubemap_tensor_bc[2].permute(1, 2, 0).cpu().numpy()
                    cube_bf = cubemap_tensor_bf[2].permute(1, 2, 0).cpu().numpy()
                    #(255,255,0) yellow
                    #(0,255,255) blue
                    #(0,255,0) green
                    #(255,160,0) orange
                    #plotXY(img, est_pixel_ceiling, color=(0,0,255))
                    cube_img = (cube_img.copy()*255).astype(np.uint8)
                    v_x = np.linspace(0, cube_img.shape[1] - 1, cube_img.shape[1]).astype(int)
                    gt_pixel_ceiling = np.vstack((v_x, gt_pixel[0])).transpose()
                    #pdb.set_trace()
                    plotXY(cube_img, gt_pixel_ceiling, color=(255,0,0))
                    gt_pixel_floor = np.vstack((v_x, gt_pixel[1])).transpose()
                    plotXY(cube_img, gt_pixel_floor, color=(255,0,0))

                    boundary = np.zeros((2,cube_img.shape[1]))
                    for j in range(cube_img.shape[1]):
                        c_index = np.where(cube_bc[:,j,0] > 0)[0]
                        f_index = np.where(cube_bf[:,j,0] > 0)[0]
                        if len(c_index) > 0:
                            boundary[0,j] = np.argmax(cube_bc[:,j,0])
                        else:
                            boundary[0,j] = -5
                        if len(f_index) > 0:
                            boundary[1,j] = np.argmax(cube_bf[:,j,0])
                        else:
                            boundary[1,j] = cube_img.shape[0]*1.05
                    #pdb.set_trace()
                    est_pixel_ceiling = np.vstack((v_x, boundary[0])).transpose()
                    plotXY(cube_img, est_pixel_ceiling, color=(0,255,255))  
                    est_pixel_floor = np.vstack((v_x, boundary[1])).transpose()
                    plotXY(cube_img, est_pixel_floor, color=(0,255,255))
                    image = image.squeeze().detach().cpu().numpy().transpose([1, 2, 0])
                    
                    #plotXY(img, est_pixel_floor, color=(0,0,255))
                    #imwrite(dst_dir_est + f"img_{i}_equi.jpg",(cube_img).astype(np.uint8))
                    imwrite(dst_dir_est + f"img_{i}.jpg",(image*255).astype(np.uint8))
                    #imwrite(dst_dir_est + f"img_{i}_postprocess.jpg",img.astype(np.uint8))
                    i+=1
            '''
            '''
            with torch.no_grad():
                dt = self.net(x.to(self.device))
                y_bon_est = self.predict_to_phi_coords(dt)
                for image, gt, est, c_int, map in zip(x, y_bon_ref, y_bon_est, c_intrinsic, whole_map_ps):
                    img = image.detach().cpu().numpy().transpose([1, 2, 0])
                    img = (img.copy()*255).astype(np.uint8)
                    map = map.detach().cpu().numpy().transpose([1, 2, 0])
                    map_b = np.zeros((2,map.shape[1]))
                    for j in range(map.shape[1]):
                        c_index = np.where(map[:,j,0] > 0)[0]
                        f_index = np.where(map[:,j,1] > 0)[0]
                        if len(c_index) > 0:
                            map_b[0,j] = np.argmax(map[:,j,0])
                        else:
                            map_b[0,j] = -5
                        if len(f_index) > 0:
                            map_b[1,j] = np.argmax(map[:,j,1])
                        else:
                            map_b[1,j] = map.shape[0]*1.05
                    
                    gt_pixel = map_b
                    est_pixel = ((est/np.pi + 0.5)*img.shape[0]).round().cpu().numpy()
                    v_x = np.linspace(0, img.shape[1] - 1, img.shape[1]).astype(int)

                    gt_pixel_ceiling = np.vstack((v_x, gt_pixel[0])).transpose()
                    #pdb.set_trace()
                    plotXY(img, gt_pixel_ceiling, color=(255,0,0))
                    gt_pixel_floor = np.vstack((v_x, gt_pixel[1])).transpose()
                    plotXY(img, gt_pixel_floor, color=(255,0,0))

                    est_pixel_ceiling = np.vstack((v_x, est_pixel[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(0,255,255))
                    #(255,255,0) yellow
                    #(0,255,255) blue
                    #(0,255,0) green
                    #(255,160,0) orange
                    #plotXY(img, est_pixel_ceiling, color=(0,0,255))
                    est_pixel_floor = np.vstack((v_x, est_pixel[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(0,255,255))
                    #plotXY(img, est_pixel_floor, color=(0,0,255))
                    imwrite(dst_dir_est + f"img_{i}_ps.jpg",img.astype(np.uint8))
                    #imwrite(dst_dir_est + f"img_{i}_postprocess.jpg",img.astype(np.uint8))
                    i+=1
            '''
