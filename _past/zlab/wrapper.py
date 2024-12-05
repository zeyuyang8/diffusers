"""written by @anonymous"""
import torch
from sd3.sd3_train import SD3Agent
from sd3.utils import DreamBoothDataset, collate_fn

@torch.no_grad()
def get_sd3_agent(args):
    sd3_agent = SD3Agent()
    sd3_official_path = args.sd3_official_paths
    sd3_agent.load(
        clipl_weights=sd3_official_path["clip_l"],
        clipg_weights=sd3_official_path["clip_g"],
        t5_weights=sd3_official_path["t5"],
        sd3_weights=sd3_official_path["sd3"],
        vae_weights=sd3_official_path["sd3"],
        verbose=args.verbose,
    )
    sd3_agent.configure(args)
    return sd3_agent

def get_dreambooth_dataloader(args):
    dreambooth_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        resolution=args.resolution,
    )
    dreambooth_dataloader = torch.utils.data.DataLoader(
        dreambooth_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    return dreambooth_dataloader
