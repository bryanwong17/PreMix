import os
import sys
import tqdm
import wandb
import torch
import hydra
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torchvision import transforms

from source.dataset import RegionFilepathsDataset
from source.dataset_utils import stain_norm
from source.models import GlobalFeatureExtractor, LocalFeatureExtractor
from source.utils import initialize_wandb, initialize_df, collate_region_filepaths

@hydra.main(
    version_base="1.2.0", config_path="config/feature_extraction", config_name="default"
)
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(output_dir, "features", cfg.experiment_name, cfg.level)
    slide_features_dir = Path(features_dir, "slide")
    region_features_dir = Path(features_dir, "region")

    if not cfg.resume:
        if features_dir.exists():
            print(f"{features_dir} already exists! deleting it...")
            shutil.rmtree(features_dir)
            print("done")
            features_dir.mkdir(parents=False)
            slide_features_dir.mkdir()
            if cfg.save_region_features:
                region_features_dir.mkdir()
        else:
            features_dir.mkdir(parents=True, exist_ok=True)
            slide_features_dir.mkdir(parents=True, exist_ok=True)
            if cfg.save_region_features:
                region_features_dir.mkdir(exist_ok=True)

    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")

    if cfg.level == "global":
        model = GlobalFeatureExtractor(
            pretrain_256=cfg.pretrain_256,
            pretrain_4096=cfg.pretrain_4096,
            last_output_layer=cfg.last_output_layer,
        )
    elif cfg.level == "local":
        model = LocalFeatureExtractor(
            pretrain_256=cfg.pretrain_256,
        )
    else:
        raise ValueError(f"cfg.level ({cfg.level} not supported")

    region_dir = Path(cfg.data_dir, cfg.dataset_name, "patches")
    if cfg.region_dir:
        region_dir = Path(cfg.region_dir)

    slide_ids = sorted([s.name for s in region_dir.iterdir()])
    print(f"{len(slide_ids)} slides with extracted patches found")

    if cfg.slide_list:
        with open(Path(cfg.slide_list), "r") as f:
            slide_ids = sorted([x.strip() for x in f.readlines()])
        print(f"restricting to {len(slide_ids)} slides from slide list .txt file")

    process_list_fp = None
    if (
        Path(features_dir.parent, f"process_list_{cfg.level}.csv").is_file()
        and cfg.resume
    ):
        process_list_fp = Path(output_dir, "features", f"process_list_{cfg.level}.csv")

    if process_list_fp is None:
        df = initialize_df(slide_ids)
    else:
        df = pd.read_csv(process_list_fp)

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)
    already_processed = len(df) - total

    region_dataset = RegionFilepathsDataset(df, region_dir, cfg.region_size, cfg.format, cfg.is_quarter_data)
    region_subset = torch.utils.data.Subset(
        region_dataset, indices=process_stack.index.tolist()
    )
    loader = torch.utils.data.DataLoader(
        region_subset, batch_size=1, shuffle=False, collate_fn=collate_region_filepaths
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    slide_ids, region_slide_ids = [], []
    slide_feature_paths, region_feature_paths = [], []
    x_coords, y_coords = [], []

    if cfg.wandb.enable:
        tqdm_output_fp = None
        tqdm_output_fp = Path(f"tqdm_{wandb.run.id}.log")
        tqdm_output_fp.unlink(missing_ok=True)
        tqdm_file = open(tqdm_output_fp, "a+") if tqdm_output_fp is not None else sys.stderr

    with tqdm.tqdm(
        loader,
        desc="Slide Encoding",
        unit=" slide",
        initial=already_processed,
        total=total + already_processed,
        ncols=80,
        position=0,
        leave=True,
        file=sys.stderr
    ) as t1:

        with torch.no_grad():

            for i, batch in enumerate(t1):

                idx, region_fps = batch
                slide_id = process_stack.loc[idx.item(), "slide_id"]
                features = []

                with tqdm.tqdm(
                    region_fps,
                    desc=(f"{slide_id}"),
                    unit=" region",
                    ncols=80 + len(slide_id),
                    position=1,
                    leave=False,
                ) as t2:

                    for fp in t2:

                        x_y = Path(fp).stem
                        x_str, y_str = x_y.split('_')
                        x, y = int(x_str), int(y_str)
                        x_coords.append(x)
                        y_coords.append(y)
                        img = np.array(Image.open(fp))

                        if cfg.stain_norm_macenko:
                            img = stain_norm(img)

                        img = transforms.functional.to_tensor(img)  # [3, 4096, 4096]
                        img = img.unsqueeze(0)  # [1, 3, 4096, 4096]
                        img = img.to(device, non_blocking=True)
                        
                        feature = model(img)
                        
                        if cfg.save_region_features:
                            save_path = Path(region_features_dir, f"{slide_id}_{x_y}.pt")
                            torch.save(feature, save_path)
                            region_feature_paths.append(save_path)
                            region_slide_ids.append(slide_id)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0).squeeze(1)
                save_path = Path(slide_features_dir, f"{slide_id}.pt")
                torch.save(stacked_features, save_path)

                df.loc[idx, "process"] = 0
                df.loc[idx, "status"] = "processed"
                df.to_csv(
                    Path(features_dir.parent, f"process_list_{cfg.level}.csv"),
                    index=False,
                )

                if cfg.wandb.enable:
                    wandb.log({"processed": already_processed + i + 1})

    if cfg.wandb.enable:
        tqdm_file.close()
        tqdm_output_fp.unlink()


if __name__ == "__main__":

    main()
