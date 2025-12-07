import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from moviepy.editor import VideoFileClip
from einops import rearrange
from datetime import datetime
from src.dataset.test_preprocess import preprocess
from src.dataset.utils import save_videos_grid, save_videos_from_pil, seed_everything, get_head_exp_motion_bucketid
from src.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from src.pipelines.hunyuan_svd_pipeline import HunyuanLongSVDPipeline
from src.models.condition.unet_3d_svd_condition_ip import UNet3DConditionSVDModel, init_ip_adapters
from src.models.condition.coarse_motion import HeadExpression, HeadPose
from src.models.condition.refine_motion import IntensityAwareMotionRefiner
from src.models.condition.pose_guider import PoseGuider
from src.models.dinov2.models.vision_transformer import vit_large, ImageProjector


def create_soft_mask(size, border_ratio=0.1):
    """
    create a soft mask with edge blurring for smooth blending.
    size: (width, height)
    """
    w, h = size
    mask = np.ones((h, w), dtype=np.float32)
    
    # calculate the number of pixels for edge blurring
    border_w = int(w * border_ratio)
    border_h = int(h * border_ratio)
    
    # horizontal direction gradient
    if border_w > 0:
        mask[:, :border_w] = np.linspace(0, 1, border_w)[None, :]
        mask[:, -border_w:] = np.linspace(1, 0, border_w)[None, :]
    
    # vertical direction gradient
    if border_h > 0:
        mask[:border_h, :] *= np.linspace(0, 1, border_h)[:, None]
        mask[-border_h:, :] *= np.linspace(1, 0, border_h)[:, None]
        
    return mask[..., None] # add channel dimension (H, W, 1)


def paste_back_frame(original_img, generated_crop, crop_bbox, mask=None):
    """
    paste the generated cropped frame back to the original image
    original_img: original image numpy array
    generated_crop: generated 512x512 face frame
    crop_bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = crop_bbox
    target_w = x2 - x1
    target_h = y2 - y1
    
    # 1. resize the generated face back to the size of the original crop box
    generated_resized = cv2.resize(generated_crop, (target_w, target_h))
    
    # 2. if no mask is provided, create a new one
    if mask is None:
        mask = create_soft_mask((target_w, target_h))
        
    # 3. get the corresponding region in the original image
    background_region = original_img[y1:y2, x1:x2].astype(np.float32) / 255.0
    foreground_region = generated_resized.astype(np.float32) / 255.0
    
    # 4. blend: Result = Gen * Mask + BG * (1 - Mask)
    blended_region = foreground_region * mask + background_region * (1 - mask)
    
    # 5. paste back to the original image
    result_img = original_img.copy()
    result_img[y1:y2, x1:x2] = (blended_region * 255).astype(np.uint8)
    
    return result_img


@torch.no_grad()
def main(cfg, args):
    output_dir = f"{cfg.output_dir}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_path = args.video_path
    image_path = args.image_path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_video_path = os.path.join(output_dir, f'{timestamp}_{image_name}_{video_name}', 'cropped.mp4')
    print(f"Generating and writing to: {output_dir}/{timestamp}_{image_name}_{video_name}")

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.pretrained_model_name_or_path, 
        subfolder="vae",
        variant="fp16")
    
    val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, 
        subfolder="scheduler")
    
    unet = UNet3DConditionSVDModel.from_config(
        cfg.pretrained_model_name_or_path,
        subfolder="unet",
        variant="fp16")
    init_ip_adapters(unet, cfg.num_adapter_embeds, cfg.ip_motion_scale)
    
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, 
        block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda")
    
    motion_expression_model = HeadExpression(cfg.input_expression_dim).to('cuda')
    motion_headpose_model = HeadPose().to('cuda')    
    motion_proj = IntensityAwareMotionRefiner(input_dim=cfg.input_expression_dim, 
                                  output_dim=cfg.motion_expression_dim, 
                                  num_queries=cfg.num_queries).to(device="cuda")  

    image_encoder = vit_large(
        patch_size=14,
        num_register_tokens=4,
        img_size=526,
        init_values=1.0,
        block_chunks=0,
        backbone=True,
        layers_output=True,
        add_adapter_layer=[3, 7, 11, 15, 19, 23],
        visual_adapter_dim=384,                
    )
    image_proj = ImageProjector(cfg.num_img_tokens, cfg.num_queries, dtype=unet.dtype).to(device="cuda")      

    pose_guider_checkpoint_path = cfg.pose_guider_checkpoint_path
    unet_checkpoint_path = cfg.unet_checkpoint_path
    motion_proj_checkpoint_path = cfg.motion_proj_checkpoint_path
    dino_checkpoint_path = cfg.dino_checkpoint_path
    image_proj_checkpoint_path = cfg.image_proj_checkpoint_path
    motion_pose_checkpoint_path = cfg.motion_pose_checkpoint_path
    motion_expression_checkpoint_path = cfg.motion_expression_checkpoint_path

    state_dict = torch.load(dino_checkpoint_path)
    image_encoder.load_state_dict(state_dict, strict=True)    

    image_proj.load_weights(image_proj_checkpoint_path, strict=True)
    pose_guider.load_state_dict(torch.load(pose_guider_checkpoint_path, map_location="cpu"), strict=True)
    
    unet.load_state_dict(torch.load(unet_checkpoint_path, map_location="cpu"), strict=True)

    state_dict = torch.load(motion_proj_checkpoint_path, map_location="cpu")
    motion_proj.load_state_dict(state_dict, strict=True)

    motion_expression_checkpoint = torch.load(motion_expression_checkpoint_path, map_location='cuda')
    motion_expression_model.load_state_dict(motion_expression_checkpoint, strict=True)
    motion_pose_checkpoint = torch.load(motion_pose_checkpoint_path, map_location='cuda')
    motion_headpose_model.load_state_dict(motion_pose_checkpoint, strict=True)
    
    image_encoder.eval()
    image_proj.eval()    
    pose_guider.eval()   
    unet.eval()
    motion_proj.eval()
    motion_expression_model.eval()
    motion_headpose_model.eval()
    motion_expression_model.requires_grad_(False)
    motion_headpose_model.requires_grad_(False)
    
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    vae.to(weight_dtype)
    unet.to(weight_dtype)
    pose_guider.to(weight_dtype)
    image_encoder.to(weight_dtype)
    image_proj.to(weight_dtype)
    pipe = HunyuanLongSVDPipeline(
        unet=unet,
        image_encoder=image_encoder,
        image_proj=image_proj,
        vae=vae,
        pose_guider=pose_guider,
        scheduler=val_noise_scheduler,
    )
    pipe = pipe.to("cuda", dtype=unet.dtype)

    if cfg.use_arcface:
        arcface_session = ort.InferenceSession(cfg.arcface_model_path, providers=['CUDAExecutionProvider'])
    
    sample = preprocess(image_path, video_path, limit=cfg.frame_num, 
            image_size=cfg.arcface_img_size, area=cfg.area, det_path=cfg.det_path)
    
    original_image = sample['original_image'] # Numpy array (H, W, C), RGB format
    crop_bbox = sample['crop_bbox'] # [x1, y1, x2, y2]

    ref_img = sample['ref_img'].unsqueeze(0).to('cuda')
    transformed_images = sample['transformed_images'].unsqueeze(0).to('cuda')
    arcface_img = sample['arcface_image']
    lmk_list = sample['lmk_list']
    if not cfg.use_arcface or arcface_img is None:
        arcface_embeddings = np.zeros((1, cfg.arcface_img_size))
    else:
        arcface_img = arcface_img.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, ...]
        arcface_embeddings = arcface_session.run(None, {"data": arcface_img})[0]
        arcface_embeddings = arcface_embeddings / np.linalg.norm(arcface_embeddings)
    dwpose_images = sample['img_pose']
    motion_pose_images = sample['motion_pose_image']
    motion_face_images = sample['motion_face_image']
    driven_images = sample['driven_image']
    pose_cond_tensor_all = []
    driven_feat_all = []
    uncond_driven_feat_all = []
    num_frames_all = 0
    driven_video_all = []
    batch = cfg.n_sample_frames
    for idx in range(0, motion_pose_images.shape[0], batch):
        driven_video = driven_images[idx:idx+batch].to('cuda')
        motion_pose_image = motion_pose_images[idx:idx+batch].to('cuda')
        motion_face_image = motion_face_images[idx:idx+batch].to('cuda')
        pose_cond_tensor = dwpose_images[idx:idx+batch].to('cuda')
        lmks = lmk_list[idx:idx+batch]
        num_frames = motion_pose_image.shape[0]
        
        motion_bucket_id_head, motion_bucket_id_exp = get_head_exp_motion_bucketid(lmks)                

        motion_feature = motion_expression_model(motion_face_image)
        motion_bucket_id_head = torch.IntTensor([motion_bucket_id_head]).to('cuda')
        motion_bucket_id_exp = torch.IntTensor([motion_bucket_id_exp]).to('cuda')
        motion_feature_embed = motion_proj(motion_feature, motion_bucket_id_head, motion_bucket_id_exp)

        driven_pose_feat = motion_headpose_model(motion_pose_image * 2 + 1)
        driven_pose_feat_embed = torch.cat([driven_pose_feat['rotation'], driven_pose_feat['translation'] * 0], dim=-1)
        
        driven_feat = torch.cat([motion_feature_embed, driven_pose_feat_embed.unsqueeze(1).repeat(1, motion_feature_embed.shape[1], 1)], dim=-1)
        driven_feat = driven_feat.unsqueeze(0)
        uncond_driven_feat = torch.zeros_like(driven_feat)

        pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
        pose_cond_tensor = rearrange(pose_cond_tensor, 'b f c h w -> b c f h w')

        pose_cond_tensor_all.append(pose_cond_tensor)
        driven_feat_all.append(driven_feat)
        uncond_driven_feat_all.append(uncond_driven_feat)
        driven_video_all.append(driven_video)
        num_frames_all += num_frames

    driven_video_all = torch.cat(driven_video_all, dim=0)
    pose_cond_tensor_all = torch.cat(pose_cond_tensor_all, dim=2)
    uncond_driven_feat_all = torch.cat(uncond_driven_feat_all, dim=1)
    driven_feat_all = torch.cat(driven_feat_all, dim=1)

    driven_video_all_2 = []
    pose_cond_tensor_all_2 = []
    driven_feat_all_2 = []
    uncond_driven_feat_all_2 = []

    for i in range(cfg.pad_frames):
        weight = i / cfg.pad_frames
        driven_video_all_2.append(driven_video_all[:1])
        pose_cond_tensor_all_2.append(pose_cond_tensor_all[:, :, :1])
        driven_feat_all_2.append(driven_feat_all[:, :1] * weight)
        uncond_driven_feat_all_2.append(uncond_driven_feat_all[:, :1])

    driven_video_all_2.append(driven_video_all)
    pose_cond_tensor_all_2.append(pose_cond_tensor_all)
    driven_feat_all_2.append(driven_feat_all)
    uncond_driven_feat_all_2.append(uncond_driven_feat_all)

    for i in range(cfg.pad_frames):
        weight = i / cfg.pad_frames
        driven_video_all_2.append(driven_video_all[:1])
        pose_cond_tensor_all_2.append(pose_cond_tensor_all[:, :, :1])
        driven_feat_all_2.append(driven_feat_all[:, -1:] * (1 - weight))
        uncond_driven_feat_all_2.append(uncond_driven_feat_all[:, :1])

    driven_video_all = torch.cat(driven_video_all_2, dim=0)
    pose_cond_tensor_all = torch.cat(pose_cond_tensor_all_2, dim=2)
    driven_feat_all = torch.cat(driven_feat_all_2, dim=1)
    uncond_driven_feat_all = torch.cat(uncond_driven_feat_all_2, dim=1)

    num_frames_all += cfg.pad_frames * 2

    video = pipe(
        ref_img.clone(),
        transformed_images.clone(),
        pose_cond_tensor_all,
        driven_feat_all,
        uncond_driven_feat_all,
        height=cfg.height,
        width=cfg.width,
        num_frames=num_frames_all,
        decode_chunk_size=cfg.decode_chunk_size,
        motion_bucket_id=cfg.motion_bucket_id,
        fps=cfg.fps,
        noise_aug_strength=cfg.noise_aug_strength,
        min_guidance_scale1=cfg.min_appearance_guidance_scale,
        max_guidance_scale1=cfg.max_appearance_guidance_scale,
        min_guidance_scale2=cfg.min_motion_guidance_scale,
        max_guidance_scale2=cfg.max_motion_guidance_scale,
        overlap=cfg.overlap,
        shift_offset=cfg.shift_offset,
        frames_per_batch=cfg.n_sample_frames,
        num_inference_steps=cfg.num_inference_steps,
        i2i_noise_strength=cfg.i2i_noise_strength,
        arcface_embeddings=arcface_embeddings,
    ).frames
    video = (video*0.5 + 0.5).clamp(0, 1).cpu()
    if cfg.pad_frames > 0:
        video = video[:, :, cfg.pad_frames:-cfg.pad_frames]

    # start paste back processing
    save_video_path_paste_back = os.path.join(output_dir, f'{timestamp}_{image_name}_{video_name}', 'full_resolution.mp4')
    
    # convert video tensor dimension: (batch, channels, frames, h, w) -> (frames, h, w, channels)
    generated_frames = video[0].permute(1, 2, 3, 0).numpy() # (T, H, W, C)
    generated_frames = (generated_frames * 255).astype(np.uint8)
    
    final_frames = []
    
    # pre-compute mask to avoid duplicate computation in loop
    x1, y1, x2, y2 = crop_bbox
    soft_mask = create_soft_mask((x2 - x1, y2 - y1))
    
    print("Processing paste back...")
    for i in range(len(generated_frames)):
        gen_frame = generated_frames[i]
        
        # paste back frame
        full_frame = paste_back_frame(original_image, gen_frame, crop_bbox, soft_mask)
        
        final_frames.append(Image.fromarray(full_frame))

    video_clip = VideoFileClip(video_path)
    save_videos_grid(video, save_video_path, n_rows=1, fps=video_clip.fps)
    print(f"Saved generated video to {save_video_path}")
    save_videos_from_pil(final_frames, save_video_path_paste_back, fps=video_clip.fps)
    print(f"Saved paste-back video to {save_video_path_paste_back}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/hunyuan-portrait.yaml")
    parser.add_argument("--video_path", type=str, default="./driving_video.mp4")
    parser.add_argument("--image_path", type=str, default='./source_image.png')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, args)
