#!/bin/bash
if [ ! -d models ]; then
    mkdir models/official
    mkdir models/rush
fi
wget --header="Authorization: Bearer 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" -O models/official/clip_g.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/clip_g.safetensors
wget --header="Authorization: Bearer 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" -O models/official/clip_l.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/clip_l.safetensors
wget --header="Authorization: Bearer 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" -O models/official/t5xxl.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/t5xxl_fp16.safetensors
wget --header="Authorization: Bearer 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" -O models/official/sd3_medium.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/main/sd3.5_medium.safetensors
wget --header="Authorization: Bearer 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" -O models/official/sd3_large.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors
wget --header="Authorization: Bearer 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" -O models/official/sd3_vae.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/main/vae/diffusion_pytorch_model.safetensors