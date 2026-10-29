import argparse
import gc
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from oceanography_with_flowers.data.dataset import Pactcs30Dataset


def one_step_baseline(baseline_path):
    """
    Find baseline for any train/val/test split

    Args:
        baseline_path (str): Path to baseline config file
    """
    bl_cfg = OmegaConf.load(baseline_path)

    path = bl_cfg.path
    data_vars = bl_cfg.data_vars
    split = bl_cfg.split

    loss_fn = instantiate(bl_cfg.loss_fn)
    losses = []

    with torch.no_grad():
        for i in range(len(split) - 1):
            timespan = list(range(split[i], split[i + 1]))
            ds = Pactcs30Dataset(path, data_vars, timespan=timespan)
            print(f"Dataset {i + 1} loaded!\n", flush=True)

            bl_loss = 0.0
            mask = ds.get_mask()
            samples = len(ds)
            log_interval = max(1, samples // 5)

            for j in range(samples):
                feature, label = ds[j]
                C, _, _ = label.shape
                pred = feature[-C:]

                # loss is masked to stay consistent with training
                v_mask = mask.expand_as(label)
                loss = loss_fn(pred[v_mask], label[v_mask])
                bl_loss += loss.item()

                if j % log_interval == 0:
                    print(f"{100 * (j / samples):.1f}%", flush=True)

            losses.append(bl_loss / samples)

            del ds, mask
            gc.collect()
            print("", flush=True)

    print("Average Loss for Persistence Baseline:")
    print(f"Training Data:   {losses[0]:.6f}")
    print(f"Validation Data: {losses[1]:.6f}")
    print(f"Test Data:       {losses[2]:.6f}")

def rollout_baseline(baseline_path):
    bl_cfg = OmegaConf.load(baseline_path)

    path = bl_cfg.path
    data_vars = bl_cfg.data_vars
    split = bl_cfg.split

    loss_fn = instantiate(bl_cfg.loss_fn)
    ds = []
    total_loss = 0
    climatology_steps = split[1] - split[0]

    with torch.no_grad():
        for i in range(len(split) - 1):
            timespan = list(range(split[i], split[i + 1]))
            ds.append(Pactcs30Dataset(path, data_vars, timespan=timespan))
            print(f"Dataset {i + 1} loaded!\n", flush=True)

        mask = ds[0].get_mask()
        samples = len(ds[1])
        log_interval = max(1, samples // 5) 

        for j in range(samples):
            feature, label = ds[1][j]
            C, _, _ = label.shape
            label = feature[-C:]

            curr_day = j
            pred = torch.zeros(label.shape)

            for k in range(climatology_steps):
                past_state, _ = ds[0][curr_day]
                pred += past_state[-C:]
                if k % 4 == 0:
                    curr_day += 366
                else:
                    curr_day += 365

            pred /= climatology_steps

            # loss is masked to stay consistent with training
            v_mask = mask.expand_as(label)
            loss = loss_fn(pred[v_mask], label[v_mask])
            total_loss += loss.item()

            if j % log_interval == 0:
                print(f"{100 * (j / samples):.1f}%", flush=True)

        total_loss /= samples

    print(f"Average Loss for Climatology Baseline ({climatology_steps} years):")
    print(f"Test Data:       {total_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create baseline for train/val/test split")
    parser.add_argument(
        "--baseline-path",
        type=str,
        required=True,
        help="Path to baseline config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="one_step: persistance baseline\nrollout: climatology baseline"
    )
    args = parser.parse_args()

    match args.mode:
        case "one_step":
            one_step_baseline(args.baseline_path)
        case "rollout":
            rollout_baseline(args.baseline_path)
        case _:
            logging.error(f"mode {args.mode} did not match anything.")