import os
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image
import onnxruntime as ort
from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from moviepy.editor import VideoFileClip
from einops import rearrange
from datetime import datetime
import random

from src.dataset.test_preprocess import preprocess
from src.dataset.utils import save_videos_grid, save_videos_from_pil, seed_everything, get_head_exp_motion_bucketid
from src.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from src.pipelines.hunyuan_svd_pipeline import HunyuanLongSVDPipeline
from src.models.condition.unet_3d_svd_condition_ip import UNet3DConditionSVDModel, init_ip_adapters
from src.models.condition.coarse_motion import HeadExpression, HeadPose
from src.models.condition.refine_motion import IntensityAwareMotionRefiner
from src.models.condition.pose_guider import PoseGuider
from src.models.dinov2.models.vision_transformer import vit_large, ImageProjector

# --- 1. Global Initialization (Load Models Once) ---
print("Initializing environment and loading models...")

CONFIG_PATH = "./config/hunyuan-portrait.yaml"
cfg = OmegaConf.load(CONFIG_PATH)

# Create output dirs
output_dir = cfg.output_dir
os.makedirs(output_dir, exist_ok=True)
tmp_path = './tmp_gradio'
os.makedirs(tmp_path, exist_ok=True)

if cfg.seed is not None:
    seed_everything(cfg.seed)

# Determine weight dtype
weight_dtype = torch.float16 if cfg.weight_dtype == "fp16" else torch.float32

# Load Models
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    cfg.pretrained_model_name_or_path, 
    subfolder="vae",
    variant="fp16"
).to("cuda", dtype=weight_dtype)

val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    cfg.pretrained_model_name_or_path, 
    subfolder="scheduler"
)

unet = UNet3DConditionSVDModel.from_config(
    cfg.pretrained_model_name_or_path,
    subfolder="unet",
    variant="fp16"
)
init_ip_adapters(unet, cfg.num_adapter_embeds, cfg.ip_motion_scale)
unet.load_state_dict(torch.load(cfg.unet_checkpoint_path, map_location="cpu"), strict=True)
unet.to("cuda", dtype=weight_dtype)

pose_guider = PoseGuider(
    conditioning_embedding_channels=320, 
    block_out_channels=(16, 32, 96, 256)
).to("cuda")
pose_guider.load_state_dict(torch.load(cfg.pose_guider_checkpoint_path, map_location="cpu"), strict=True)
pose_guider.to(dtype=weight_dtype)

motion_expression_model = HeadExpression(cfg.input_expression_dim).to('cuda')
motion_expression_checkpoint = torch.load(cfg.motion_expression_checkpoint_path, map_location='cuda')
motion_expression_model.load_state_dict(motion_expression_checkpoint, strict=True)
motion_expression_model.eval()
motion_expression_model.requires_grad_(False)

motion_headpose_model = HeadPose().to('cuda')
motion_pose_checkpoint = torch.load(cfg.motion_pose_checkpoint_path, map_location='cuda')
motion_headpose_model.load_state_dict(motion_pose_checkpoint, strict=True)
motion_headpose_model.eval()
motion_headpose_model.requires_grad_(False)

motion_proj = IntensityAwareMotionRefiner(
    input_dim=cfg.input_expression_dim, 
    output_dim=cfg.motion_expression_dim, 
    num_queries=cfg.num_queries
).to("cuda")
motion_proj.load_state_dict(torch.load(cfg.motion_proj_checkpoint_path, map_location="cpu"), strict=True)
motion_proj.eval()

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
image_encoder.load_state_dict(torch.load(cfg.dino_checkpoint_path), strict=True)
image_encoder.to("cuda", dtype=weight_dtype)
image_encoder.eval()

image_proj = ImageProjector(cfg.num_img_tokens, cfg.num_queries, dtype=unet.dtype).to("cuda")
image_proj.load_weights(cfg.image_proj_checkpoint_path, strict=True)
image_proj.to(dtype=weight_dtype)
image_proj.eval()

# Initialize Pipeline
pipe = HunyuanLongSVDPipeline(
    unet=unet,
    image_encoder=image_encoder,
    image_proj=image_proj,
    vae=vae,
    pose_guider=pose_guider,
    scheduler=val_noise_scheduler,
).to("cuda", dtype=unet.dtype)

# Initialize ArcFace
arcface_session = None
if cfg.use_arcface:
    arcface_session = ort.InferenceSession(cfg.arcface_model_path, providers=['CUDAExecutionProvider'])

print("Models loaded successfully!")

# --- 2. Helper Functions ---

def create_soft_mask(size, border_ratio=0.1):
    """create a soft mask with edge blurring for smooth blending."""
    w, h = size
    mask = np.ones((h, w), dtype=np.float32)
    border_w = int(w * border_ratio)
    border_h = int(h * border_ratio)
    
    if border_w > 0:
        mask[:, :border_w] = np.linspace(0, 1, border_w)[None, :]
        mask[:, -border_w:] = np.linspace(1, 0, border_w)[None, :]
    if border_h > 0:
        mask[:border_h, :] *= np.linspace(0, 1, border_h)[:, None]
        mask[-border_h:, :] *= np.linspace(1, 0, border_h)[:, None]
        
    return mask[..., None]

def paste_back_frame(original_img, generated_crop, crop_bbox, mask=None):
    """paste the generated cropped frame back to the original image"""
    x1, y1, x2, y2 = crop_bbox
    target_w = x2 - x1
    target_h = y2 - y1
    
    generated_resized = cv2.resize(generated_crop, (target_w, target_h))
    
    if mask is None:
        mask = create_soft_mask((target_w, target_h))
        
    background_region = original_img[y1:y2, x1:x2].astype(np.float32) / 255.0
    foreground_region = generated_resized.astype(np.float32) / 255.0
    
    blended_region = foreground_region * mask + background_region * (1 - mask)
    
    result_img = original_img.copy()
    result_img[y1:y2, x1:x2] = (blended_region * 255).astype(np.uint8)
    
    return result_img

# --- 3. Inference Function ---

@torch.no_grad()
def run_inference(image_input, video_input, 
                  steps, seed, frame_limit, fps, 
                  motion_bucket_id, noise_aug,
                  app_guide_min, app_guide_max,
                  mot_guide_min, mot_guide_max):
    
    print("Starting Inference...")
    
    # Handle Seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)
    print(f"Using seed: {seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save inputs to temp directory
    image_path = os.path.join(tmp_path, f"{timestamp}_src.png")
    
    # Handle image input (Gradio sends PIL Image or path)
    if isinstance(image_input, np.ndarray):
        Image.fromarray(image_input).save(image_path)
    elif isinstance(image_input, Image.Image):
        image_input.save(image_path)
    elif isinstance(image_input, str):
        image_path = image_input
    
    video_path = video_input
    
    # Prepare Output Paths
    job_dir = os.path.join(output_dir, f'{timestamp}_gradio_job')
    os.makedirs(job_dir, exist_ok=True)
    save_video_path_crop = os.path.join(job_dir, 'cropped.mp4')
    save_video_path_full = os.path.join(job_dir, 'full_resolution.mp4')

    # Preprocess
    print(f"Preprocessing (Frame Limit: {frame_limit})...")
    sample = preprocess(image_path, video_path, limit=frame_limit, 
            image_size=cfg.arcface_img_size, area=cfg.area, det_path=cfg.det_path)
    
    original_image = sample['original_image'] # Numpy array (H, W, C), RGB
    crop_bbox = sample['crop_bbox'] # [x1, y1, x2, y2]

    # Prepare Tensors
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
    
    # Feature Extraction Loop
    pose_cond_tensor_all = []
    driven_feat_all = []
    uncond_driven_feat_all = []
    num_frames_all = 0
    driven_video_all = []
    batch = cfg.n_sample_frames
    
    print("Extracting Motion Features...")
    for idx in range(0, motion_pose_images.shape[0], batch):
        driven_video = driven_images[idx:idx+batch].to('cuda')
        motion_pose_image = motion_pose_images[idx:idx+batch].to('cuda')
        motion_face_image = motion_face_images[idx:idx+batch].to('cuda')
        pose_cond_tensor = dwpose_images[idx:idx+batch].to('cuda')
        lmks = lmk_list[idx:idx+batch]
        num_frames = motion_pose_image.shape[0]
        
        motion_bucket_id_head, motion_bucket_id_exp = get_head_exp_motion_bucketid(lmks)                

        motion_feature = motion_expression_model(motion_face_image)
        motion_bucket_id_head_t = torch.IntTensor([motion_bucket_id_head]).to('cuda')
        motion_bucket_id_exp_t = torch.IntTensor([motion_bucket_id_exp]).to('cuda')
        motion_feature_embed = motion_proj(motion_feature, motion_bucket_id_head_t, motion_bucket_id_exp_t)

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

    # Padding Logic
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

    # Generation
    print(f"Running Video Generation Pipeline (Steps: {steps})...")
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
        motion_bucket_id=motion_bucket_id, # User defined
        fps=fps, # User defined
        noise_aug_strength=noise_aug, # User defined
        min_guidance_scale1=app_guide_min, # User defined
        max_guidance_scale1=app_guide_max,
        min_guidance_scale2=mot_guide_min, # User defined
        max_guidance_scale2=mot_guide_max,
        overlap=cfg.overlap,
        shift_offset=cfg.shift_offset,
        frames_per_batch=cfg.n_sample_frames,
        num_inference_steps=steps, # User defined
        i2i_noise_strength=cfg.i2i_noise_strength,
        arcface_embeddings=arcface_embeddings,
    ).frames

    video = (video*0.5 + 0.5).clamp(0, 1).cpu()
    if cfg.pad_frames > 0:
        video = video[:, :, cfg.pad_frames:-cfg.pad_frames]

    # Post-processing & Paste Back
    print("Processing Paste Back...")
    generated_frames = video[0].permute(1, 2, 3, 0).numpy() # (T, H, W, C)
    generated_frames = (generated_frames * 255).astype(np.uint8)
    
    final_frames = []
    x1, y1, x2, y2 = crop_bbox
    soft_mask = create_soft_mask((x2 - x1, y2 - y1))
    
    for i in range(len(generated_frames)):
        gen_frame = generated_frames[i]
        full_frame = paste_back_frame(original_image, gen_frame, crop_bbox, soft_mask)
        final_frames.append(Image.fromarray(full_frame))

    print(f"Saving videos to {job_dir} with FPS {fps}")
    save_videos_grid(video, save_video_path_crop, n_rows=1, fps=fps)
    save_videos_from_pil(final_frames, save_video_path_full, fps=fps)
    
    return save_video_path_crop, save_video_path_full

# --- 4. Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="HunyuanPortrait") as demo:
    gr.Markdown(
        """
        # 🎥 HunyuanPortrait Animation
        **Upload a source image and a driving video to animate the portrait.**
        """
    )
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            with gr.Group():
                inp_img = gr.Image(label="Source Image (Human Portrait)", type="pil", height=300)
                inp_vid = gr.Video(label="Driving Video", height=300)
            
            # Advanced Settings Accordion
            with gr.Accordion("⚙️ Advanced Settings / 高级设置", open=False):
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Denoising Steps (去噪步数)")
                    seed = gr.Number(value=-1, label="Seed (-1 for Random)", precision=0)
                
                with gr.Row():
                    frame_limit = gr.Slider(minimum=10, maximum=500, step=10, value=100, label="Max Frame Number (最大帧数)")
                    fps = gr.Slider(minimum=8, maximum=60, step=1, value=12.5, label="Output FPS")
                
                with gr.Row():
                    motion_bucket_id = gr.Slider(minimum=1, maximum=255, step=1, value=0, label="Motion Bucket ID (运动幅度)")
                    noise_aug = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Noise Augmentation")
                
                gr.Markdown("### Guidance Scales (引导系数)")
                with gr.Row():
                    app_guide_min = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.5, label="Min Appearance Guidance")
                    app_guide_max = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.5, label="Max Appearance Guidance")
                with gr.Row():
                    mot_guide_min = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.5, label="Min Motion Guidance")
                    mot_guide_max = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.5, label="Max Motion Guidance")

            btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
        
        # Right Column: Outputs
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Full Resolution (Paste Back)"):
                    out_full = gr.Video(label="Final Result", height=500, autoplay=True)
                with gr.TabItem("Cropped Animation (Face Only)"):
                    out_crop = gr.Video(label="Cropped Face", height=500)

    # Bind Event
    btn.click(
        fn=run_inference, 
        inputs=[
            inp_img, inp_vid, 
            steps, seed, frame_limit, fps, 
            motion_bucket_id, noise_aug,
            app_guide_min, app_guide_max,
            mot_guide_min, mot_guide_max
        ], 
        outputs=[out_crop, out_full]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
