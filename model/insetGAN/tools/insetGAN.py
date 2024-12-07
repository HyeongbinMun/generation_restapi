import torch
from torch.autograd import Variable as V
from facenet_pytorch import MTCNN
from torchvision import transforms
import pickle
import numpy as np
import os
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import lpips
from optim_utils import get_bounding_box_face, get_target_bounding_box_face, BicubicDownSample, save_tensor
import contextlib

import config_2 as config

class INSETGAN:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.mtcnn = MTCNN(device=self.device, keep_all=False, select_largest=False)
        self.mtcnn.requires_grad = False
        self.percept = lpips.LPIPS(net='vgg').to(self.device)
        self.loss_L1 = torch.nn.L1Loss().to(self.device)
        self.loss_L2 = torch.nn.MSELoss().to(self.device)

        os.makedirs(config.out_folder, exist_ok=True)

        # Load and set up networks
        self.G_canvas, self.latent_avg_canvas = self.load_network(
            config.home_dir + '/networks/DeepFashion_1024x768.pkl')
        self.G_inset, self.latent_avg_inset = self.load_network(config.home_dir + '/networks/ffhq.pkl')

        self.face_res = self.G_inset.synthesis.img_resolution
        self.downsampler_256 = BicubicDownSample(factor=1024 // 256, device=self.device)
        self.downsampler_128 = BicubicDownSample(factor=1024 // 128, device=self.device)
        self.downsampler_64 = BicubicDownSample(factor=1024 // 64, device=self.device)
        self.downsampler = BicubicDownSample(factor=256 // 64, device=self.device)

        self.loss_fn_dict = {
            'L2':                 self.l2_loss,
            'L1':                 self.l1_loss,
            'L1_gradient':        self.l1_loss,
            'L1_in':              self.l1_loss,
            'perceptual':         self.percep_loss,
            'perceptual_in':      self.percep_loss,
            'perceptual_face':    self.percep_loss,
            'perceptual_edge':    self.percep_edge_loss,
            'edge':               self.edge_loss,
            'mean_latent':        self.mean_latent_loss,
            'selected_body':      self.percep_loss,
            'selected_body_L1':   self.l1_loss,
        }

    def l2_loss(self, target, optim):
        res = float(target.shape[-1])
        return self.loss_L2(target, optim) / (res ** 2)

    def l1_loss(self, target, optim):
        res = float(target.shape[-1])
        return self.loss_L1(target, optim) / (res ** 2)

    def percep_loss(self, target, optim):
        return self.percept(target, optim).sum()

    def percep_edge_loss(self, target, optim, edge=8, bottom_multiplier=1):
        target_cp = target.clone()
        optim_cp = optim.clone()
        target_cp[:, :, edge:-edge, edge:-edge] = 0
        optim_cp[:, :, edge:-edge, edge:-edge] = 0
        return self.percep_loss(target_cp, optim_cp)

    def edge_loss(self, target, optim, edge=8, bottom_multiplier=1):
        res = float(target.shape[-1])
        return (self.loss_L2(target[:, :, :edge, :], optim[:, :, :edge, :]) / (edge * res)
                + self.loss_L2(target[:, :, -edge:, :], optim[:, :, -edge:, :]) / (edge * res) * bottom_multiplier
                + self.loss_L2(target[:, :, edge:-edge, :edge], optim[:, :, edge:-edge, :edge]) / (
                            edge * (res - 2 * edge))
                + self.loss_L2(target[:, :, edge:-edge, -edge:], optim[:, :, edge:-edge, -edge:]) / (
                            edge * (res - 2 * edge)))

    def mean_latent_loss(self, w, w_avg):
        return self.l2_loss(w, w_avg)

    def load_network(self, ckpt_path):
        """Load network and calculate average latent."""
        with open(ckpt_path, 'rb') as f:
            networks = pickle.Unpickler(f).load()
        G = networks['G_ema'].to(self.device)
        w_samples = G.mapping(torch.randn(10000, G.z_dim, device=self.device), None)
        w_samples = w_samples[:, :1, :]
        latent_avg = torch.mean(w_samples, axis=0).squeeze().unsqueeze(0).repeat(G.num_ws, 1).unsqueeze(0)
        return G, latent_avg

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale."""
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def run(self):
        """Main process for running INSETGAN."""
        # Set default values if attributes are missing in config
        trunc_inset = getattr(self.config, 'trunc_inset', 0.7)
        trunc_canvas = getattr(self.config, 'trunc_canvas', 0.7)

        # Generate initial random images for canvas and inset
        random_humans_w, random_outputs = self.generate_initial_images(
            self.G_canvas, self.latent_avg_canvas, self.config.seed_canvas, trunc_canvas
        )
        if self.config.output_seed_images:
            save_tensor(random_outputs, 'human', out_folder=self.config.out_folder)

        random_face_w, random_outputs = self.generate_initial_images(
            self.G_inset, self.latent_avg_inset, self.config.seed_inset, trunc_inset
        )
        if self.config.output_seed_images:
            save_tensor(random_outputs, 'face', out_folder=self.config.out_folder)

        # Select body and face indices
        bodies = self.config.selected_bodies if hasattr(self.config, 'selected_bodies') else range(len(random_humans_w))
        faces = self.config.selected_faces if hasattr(self.config, 'selected_faces') else range(len(random_face_w))

        for body, face in zip(bodies, faces):
            latent_w_canvas, latent_w_inset = self.prepare_latent_vectors(random_humans_w[body], random_face_w[face])
            self.optimize_and_save(latent_w_canvas, latent_w_inset, body, face)

        bodies = self.config.selected_bodies if hasattr(self.config, 'selected_bodies') else range(len(random_humans_w))
        faces = self.config.selected_faces if hasattr(self.config, 'selected_faces') else range(len(random_face_w))

        for idx in range(len(bodies)):
            body = bodies[idx]
            face = faces[idx]

            # 각각의 face와 body에 대한 시작 latent 준비
            latent_w_canvas = random_humans_w[body][0].detach().clone()
            latent_in_canvas = latent_w_canvas.unsqueeze(0).unsqueeze(0).repeat(1, self.G_canvas.num_ws, 1).to(
                self.device)

            latent_w_inset = random_face_w[face].unsqueeze(0).detach().clone()

            # 최적화를 위한 손실 초기화
            losses_w_inset = defaultdict(list)
            losses_w_canvas = defaultdict(list)

            # 얼굴과 캔버스 이미지 생성
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                with torch.no_grad():
                    gen_canvas = self.G_canvas.synthesis(latent_in_canvas, noise_mode='const')
                    hires_inset = self.G_inset.synthesis(latent_w_inset, noise_mode='const')

            # 얼굴 해상도에 따라 다운샘플링
            if self.face_res == 1024:
                input_inset = self.downsampler_256(hires_inset)
            else:
                input_inset = hires_inset

            # 최적화 진행
            self.optimize_and_save(latent_in_canvas, latent_w_inset, body, face)

    def generate_initial_images(self, G, latent_avg, seed, truncation):
        """Generate initial latent vectors and images with truncation adjustment."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(32, G.z_dim, device=self.device, generator=generator)

        with torch.no_grad():
            w_samples = G.mapping(z, None)
            trunc_insets = getattr(self.config, 'trunc_insets', None)

            # Adjust truncation based on config parameters
            if trunc_insets is not None:
                for i in range(min(len(trunc_insets), w_samples.shape[1])):  # 안전하게 인덱스 범위 조정
                    w_samples[:, i, :] = w_samples[:, i, :] * (1 - trunc_insets[i]) + latent_avg[:, i, :] * trunc_insets[i]
            else:
                w_samples = w_samples * (1 - truncation) + latent_avg * truncation

            images = G.synthesis(w_samples, noise_mode='const')
        return w_samples, images

    def prepare_latent_vectors(self, latent_canvas, latent_inset):
        """Prepare latent vectors for optimization."""
        latent_w_canvas = latent_canvas[0].unsqueeze(0).unsqueeze(0).repeat(1, self.G_canvas.num_ws, 1).to(self.device)
        latent_w_inset = latent_inset.unsqueeze(0).detach().clone()
        return latent_w_canvas, latent_w_inset

    def find_target_regions_for_canvas_and_insets(self, gen_canvas, hires_inset):
        """Find target bounding boxes and cropping information."""
        # Find bounding box for inset
        inset_bounding_box = get_bounding_box_face(self.mtcnn, hires_inset)

        # 다운샘플링 설정
        input_downsampled = self.downsampler_64(hires_inset) if self.face_res == 1024 else self.downsampler(hires_inset)

        # Find bounding box for canvas and calculate inset's position in canvas
        canvas_bounding_box = get_bounding_box_face(self.mtcnn, gen_canvas)
        gen_inset, crop_box = get_target_bounding_box_face(
            gen_canvas, canvas_bounding_box.squeeze(), inset_bounding_box.squeeze(), vertical=False
        )

        # Update crop box dimensions
        xmin, ymin, xmax, ymax = crop_box
        w_inset = xmax - xmin
        h_inset = ymax - ymin

        return {
            'inset_bounding_box': inset_bounding_box,
            'canvas_bounding_box': canvas_bounding_box,
            'gen_inset': gen_inset,
            'crop_box': crop_box,
            'w_inset': w_inset,
            'h_inset': h_inset,
            'input_downsampled': input_downsampled
        }

    def optimize_and_save(self, latent_w_canvas, latent_w_inset, body, face):
        """Optimize latent codes for the canvas and inset images and save results."""

        # Generate initial images for canvas and inset
        with torch.no_grad():
            gen_canvas = self.G_canvas.synthesis(latent_w_canvas, noise_mode='const')
            hires_inset = self.G_inset.synthesis(latent_w_inset, noise_mode='const')

        # Get target region information
        regions = self.find_target_regions_for_canvas_and_insets(gen_canvas, hires_inset)
        crop_box = regions['crop_box']
        selected_body = self.downsampler_128(gen_canvas).squeeze() if self.config.fix_canvas_from_start else None
        selected_face = regions['input_downsampled'] if self.config.fix_inset_from_start else None

        # Set up optimization
        opt_canvas = torch.optim.Adam([latent_w_canvas], lr=self.config.learning_rate_optim_canvas)
        opt_inset = torch.optim.Adam([latent_w_inset], lr=self.config.learning_rate_optim_inset)
        optim_canvas_step, optim_inset_step = self.config.start_canvas_optim, not self.config.start_canvas_optim

        best_inset_state, best_canvas_loss, best_inset_loss = None, 1e6, 1e6

        pbar = tqdm(range(self.config.num_optim_iter), position=1, leave=True)
        for j in pbar:
            # Set requires_grad for current step
            latent_w_inset.requires_grad = optim_inset_step
            latent_w_canvas.requires_grad = optim_canvas_step

            optimizer = opt_inset if optim_inset_step else opt_canvas
            optimizer.zero_grad()

            if j == self.config.fix_canvas_at_iter:
                selected_body = self.downsampler_128(gen_canvas).squeeze()

            if j == self.config.fix_inset_at_iter:
                selected_face = self.downsampler_64(hires_inset) if self.face_res == 1024 else self.downsampler(hires_inset)

            # Update latent for the canvas generator
            latent_in = latent_w_canvas.unsqueeze(0).repeat(self.G_canvas.num_ws, 1).unsqueeze(0)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                gen_canvas = self.G_canvas.synthesis(latent_in, noise_mode='const')
                hires_inset = self.G_inset.synthesis(latent_w_inset, noise_mode='const')

            # Prepare target inset
            target_inset = self.downsampler_256(hires_inset) if self.face_res == 1024 else hires_inset
            target_downsampled = self.downsampler(target_inset)

            # Update bounding box if needed
            if j % self.config.update_bbox_interval == 0 and j <= self.config.update_bbox_until:
                new_canvas_bounding_box = get_bounding_box_face(self.mtcnn, gen_canvas)
                if new_canvas_bounding_box is not None:
                    delta_bounding_box = new_canvas_bounding_box - regions['canvas_bounding_box']
                    canvas_bounding_box = regions['canvas_bounding_box'] + ((250 - j) / 250) * delta_bounding_box

            # Recalculate target region for the inset
            gen_inset, crop_box = get_target_bounding_box_face(
                gen_canvas, regions['canvas_bounding_box'].squeeze(), regions['inset_bounding_box'].squeeze(),
                vertical=False
            )

            # Compute losses
            total_loss = self.calculate_loss(gen_canvas, gen_inset, crop_box, selected_body, selected_face, j,
                                             latent_w_canvas, latent_w_inset, optim_inset_step, target_inset,
                                             target_downsampled)

            total_loss.backward()
            optimizer.step()

            # Save the best state based on loss
            if optim_inset_step and j > 0 and (regions['input_downsampled'] is not None):
                best_inset_loss = min(best_inset_loss, total_loss.item())
                best_inset_state = [latent_w_canvas.detach().clone(), latent_w_inset.detach().clone(), crop_box, j]

            # Switch optimization steps
            if j % self.config.switch_optimizers_every == 0:
                optim_inset_step = not optim_inset_step
                optim_canvas_step = not optim_canvas_step

    def calculate_loss(self, gen_canvas, gen_inset, crop_box, selected_body, selected_face, j, losses_w_inset,
                       losses_w_canvas):
        """Calculate total loss for the given iteration."""
        total_loss = 0
        loss_dict = losses_w_inset if j % 2 == 1 else losses_w_canvas
        loss_source = self.config.lambdas_w_inset if j % 2 == 1 else self.config.lambdas_w_canvas
        xmin, ymin, xmax, ymax = crop_box

        for loss_name, loss_weight in loss_source.items():
            # 손실 함수를 loss_fn_dict에서 가져옵니다.
            loss_fn = self.loss_fn_dict.get(loss_name, None)
            if loss_fn is None:
                continue  # 정의되지 않은 손실 함수는 건너뜁니다.

            # 적절한 target과 generated 텐서를 설정합니다.
            if loss_name == 'perceptual_in' and selected_face is not None:
                target, generated = V(selected_face[:, :, 2:ymax - 2, xmin + 2:xmax - 2]), gen_inset
            elif loss_name == 'selected_body' and selected_body is not None:
                target, generated = selected_body, gen_canvas
            else:
                target, generated = gen_canvas, gen_inset

            # 입력 텐서의 크기가 다른 경우 다운샘플링을 적용해 해상도를 맞춥니다.
            while target.shape != generated.shape:
                if target.shape[-1] > generated.shape[-1]:
                    target = self.downsampler(target)
                elif generated.shape[-1] > target.shape[-1]:
                    generated = self.downsampler(generated)
                else:
                    break  # 해상도가 같아질 때까지 반복

            # 손실을 계산하고, 결과에 가중치를 적용한 후 총 손실에 추가합니다.
            loss = loss_weight * loss_fn(target, generated).sum()
            total_loss += loss
            loss_dict[loss_name].append(loss.item())

        return total_loss

    def save_final_output(self, best_inset_state, body, face):
        """Generate and save the optimized images."""
        latent_w_canvas, latent_w_inset, crop_box, _ = best_inset_state
        final_canvas = self.G_canvas.synthesis(latent_w_canvas, noise_mode='const')
        final_inset = self.G_inset.synthesis(latent_w_inset, noise_mode='const')

        xmin, ymin, xmax, ymax = crop_box
        gen_paste = (final_canvas.squeeze() + 1) / 2 * 255
        gen_paste = gen_paste.cpu().clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(gen_paste)

        inset_img = (final_inset.squeeze() + 1) / 2 * 255
        inset_img = inset_img.cpu().clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        paste_im = Image.fromarray(inset_img).resize((ymax - ymin, xmax - xmin), Image.LANCZOS)

        im.paste(paste_im, (xmin, ymin))
        im.save(f"{self.config.out_folder}/{face}_{body}_optimized.png")
        print(f"Saved final optimized image for body {body} and face {face}.")

inset_gan = INSETGAN()
inset_gan.run()