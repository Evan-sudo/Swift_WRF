import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import os
from deform_model import *
from torch.utils.data import DataLoader
from dataloader import *
from data_painter import *
from skimage.metrics import structural_similarity as ssimt



class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.num_points = num_points
        self.image_name = image_path.stem
        self.batch_size = 1
        BLOCK_H, BLOCK_W = 16, 16
        self.datadir = image_path
        #self.datadir ="/home/old/gaussian4d/workspace/6dgs/data/data_test6000"  
        spectrum_dir = os.path.join(self.datadir, "spectrum")
        image0 = Image.open(os.path.join(spectrum_dir, os.listdir(spectrum_dir)[0]))  
        self.W, self.H = image0.size
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.smooth = True
               
        yaml_file_path = os.path.join(self.datadir, 'gateway_info.yml')
        # with open(yaml_file_path, 'r') as file:
        #     data = yaml.safe_load(file)
        # self.r_o = data['gateway1']['position']
        # self.gateway_orientation = data['gateway1']['orientation']

        dataset = dataset_dict["rfid"]
        train_index = os.path.join(self.datadir, "train_index_3000.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")
       
        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type="rfid")

        self.train_set = dataset(self.datadir, train_index)
        self.test_set = dataset(self.datadir, test_index)
        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.log_dir = Path(f"./checkpoints/")
        
        
        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)
            self.deform = DeformModel()
            self.deform.train_setting()
            
        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path+"gaussian_model.pth.tar", map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)  
            self.deform.load_weights(model_path)
    
    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)
          
    
    def _load_images(self, image_paths):
        gt_images = []
        for path in sorted(os.listdir(image_paths)):  # 按文件名排序
            path_full = os.path.join(image_paths, path)
            image_tensor = image_path_to_tensor(path_full).to(self.device)  # 转换为 Tensor 并移动到设备
            gt_images.append(image_tensor)
        return gt_images    
    

    def train(self, stage):     
        psnr_list, iter_list = [], []
        if stage == "coarse":
            iterations = 10000
        elif stage == "fine":
            iterations = self.iterations
            if self.smooth:
                smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=2e6)
        progress_bar = tqdm(range(1, iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, iterations+1):
            if stage == "coarse":
                # Pick a random Camera
                try:
                    spectrum, tx_pos = next(self.train_iter_dataset)
                except:
                    self.dataset_init()
                    spectrum, tx_pos = next(self.train_iter_dataset)          
                gt_image = spectrum.cuda()
                render_pkg = self.gaussian_model.forward()
            elif stage == "fine":
                # Pick a random Camera
                self.gaussian_model._xyz.requires_grad_(False)
                try:
                    spectrum, tx_pos = next(self.train_iter_dataset)
                except:
                    self.dataset_init()
                    spectrum, tx_pos = next(self.train_iter_dataset)    
                tx_pos = tx_pos.cuda()      
                gt_image = spectrum.cuda()
                if self.smooth:
                    tx_pos += torch.randn(1, 3, device='cuda') * smooth_term(iter) * (1/(6000 ** (1/3)))
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(self.gaussian_model.get_xyz, tx_pos)
                render_pkg = self.gaussian_model.forward_dynamic(deformed_xyz, deformed_opacity, deformed_features)
                
            image_full = render_pkg["render"].squeeze(0)
            _, height, _ = image_full.shape
            pred_spectrum_real = image_full[0,:height, :]
            pred_spectrum_imag = image_full[1,:height, :]
            pred_spectrum = pred_spectrum_real + 1j * pred_spectrum_imag
            pred_spectrum = torch.abs(pred_spectrum).unsqueeze(0)
            

            Ll1 = l1_loss(pred_spectrum, gt_image)
            lambda_value= 0.3
            loss = (1.0 - lambda_value)*Ll1 + lambda_value*(1-ssim(pred_spectrum, gt_image))
            loss.backward()        
            with torch.no_grad():
                mse_loss = F.mse_loss(pred_spectrum, gt_image)
                psnr = 10 * math.log10(1.0 / mse_loss.item())

            self.gaussian_model.optimizer.step()
            self.gaussian_model.optimizer.zero_grad(set_to_none = True)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.gaussian_model.scheduler.step()
            self.deform.update_learning_rate(iter)
            psnr_list.append(psnr)
            iter_list.append(iter)
            
            ## test
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
                if iter % 20000 == 0:
                    self.test()
                    
                    
        end_time = time.time() - start_time
        progress_bar.close() 
        print("Training finished! Saving models!")      
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        self.deform.save_weights(self.log_dir)
    
    
    def test(self):
        """Evaluate the model on all video frames."""
        self.gaussian_model.eval()
        last_two = os.path.normpath(self.datadir).split(os.sep)[-2:]
        train_dir = os.path.join("./test", *last_two, "train")
        test_dir  = os.path.join("./test", *last_two, "test")
        
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        print("Evaluate testset!")
        save_img_idx = 0
        all_ssim = []
        pix_error = []
        all_psnr = []
        pred_list = []
        
        
         # validation on testset
        with torch.no_grad():
            for spectrum, tx_pos in self.test_iter: 
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(self.gaussian_model.get_xyz, tx_pos.cuda())
                render_pkg = self.gaussian_model.forward_dynamic(deformed_xyz, deformed_opacity, deformed_features)
                image_full = render_pkg["render"].squeeze(0)
                _, height, _ = image_full.shape
                pred_spectrum_real = image_full[0,:height, :]
                pred_spectrum_imag = image_full[1,:height, :]
                pred_spectrum = pred_spectrum_real + 1j * pred_spectrum_imag
                pred_spectrum = torch.abs(pred_spectrum).unsqueeze(0)
                
                pred_list.append(pred_spectrum.squeeze(0).cpu().numpy())
                
                pred_spectrum = pred_spectrum
                gt_spectrum = spectrum.cuda()    
                pixel_error = torch.mean(abs(pred_spectrum - gt_spectrum)).detach().cpu().numpy()
                #ssim_i = ssimt(pred_spectrum, gt_spectrum, data_range=1, multichannel=False).detach().cpu().numpy()
                ssim_i = ssimt(pred_spectrum.squeeze(0).detach().cpu().numpy(),gt_spectrum.squeeze(0).detach().cpu().numpy(),data_range=1,channel_axis=None)
                mse_loss = F.mse_loss(pred_spectrum, gt_spectrum).detach().cpu().numpy()
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                #print("Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error, ssim_i))
                #paint_spectrum_compare(pred_spectrum.squeeze(0).detach().cpu().numpy(), gt_spectrum.squeeze(0).detach().cpu().numpy(), save_path=os.path.join(test_dir, f'{save_img_idx}.png'))
                all_ssim.append(ssim_i)
                all_psnr.append(psnr)
                pix_error.append(pixel_error)
                save_img_idx += 1
                np.savetxt(os.path.join(test_dir, 'all_ssim.txt'), all_ssim, fmt='%.4f')
                np.savetxt(os.path.join(test_dir, 'all_err.txt'), pix_error, fmt='%.4f')
                np.savetxt(os.path.join(test_dir, 'all_psnr.txt'), all_psnr, fmt='%.4f')
               
        print("Avg pixel error", sum(pix_error) / len(pix_error))   
        median_ssim = np.median(all_ssim)
        print("Median SSIM:", median_ssim)     
          
         


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default="/home/old/gaussian4d/workspace/6dgs/data/data_test6000", help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='bunny', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    image_path = Path(args.dataset)

    trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
        iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
    print("Start coarse training!")
    trainer.train(stage="coarse")
    print("Start fine training!")
    psnr,ssim = trainer.train(stage="fine")    
    print(f"Segment: {args.data_name}, PSNR: {psnr:.4f}, SSIM: {ssim:.6f}") 
    return psnr, ssim   
  

if __name__ == "__main__":
    main(sys.argv[1:])
