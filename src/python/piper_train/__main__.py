import argparse
import json
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to pre-processed dataset directory")
    parser.add_argument("--checkpoint-epochs", type=int, default=1, help="Save checkpoint every N epochs (default: 1)")
    parser.add_argument("--quality", default="medium", choices=("x-low", "medium", "high"),
                        help="Quality/size of model (default: medium)")
    parser.add_argument("--resume_from_single_speaker_checkpoint",
                        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker "
                             "and resumes training")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--precision", type=int, choices=[16, 32], default=32, help="Precision for training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume training from")  # New Argument

    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    # Trainer configuration
    callbacks = []
    if args.checkpoint_epochs:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(args.dataset_dir / "checkpoints"),
            filename='epoch={epoch}-step={step}',
            every_n_epochs=args.checkpoint_epochs,
            save_top_k=-1,  # Save all checkpoints
            save_last=True  # Always save the last checkpoint
        )
        callbacks.append(checkpoint_callback)
        _LOGGER.debug("Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        precision=args.precision,
        callbacks=callbacks,
        default_root_dir=args.dataset_dir
    )

    # Model-specific arguments
    model_kwargs = {
        "num_symbols": num_symbols,
        "num_speakers": num_speakers,
        "sample_rate": sample_rate,
        "dataset": [dataset_path],
        "batch_size": args.batch_size,
        "validation_split": args.validation_split
    }

    if args.quality == "x-low":
        model_kwargs.update({
            "hidden_channels": 96,
            "inter_channels": 96,
            "filter_channels": 384
        })
    elif args.quality == "high":
        model_kwargs.update({
            "resblock": "1",
            "resblock_kernel_sizes": (3, 7, 11),
            "resblock_dilation_sizes": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            "upsample_rates": (8, 8, 2, 2),
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": (16, 16, 4, 4)
        })

    model = VitsModel(**model_kwargs)

    # Resuming from single-speaker checkpoint
    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        model.model_g.load_state_dict(model_single.model_g.state_dict(), strict=False)
        model.model_d.load_state_dict(model_single.model_d.state_dict(), strict=False)

    # Start training with optional checkpoint resumption
    if args.ckpt_path:
        _LOGGER.debug("Resuming training from checkpoint: %s", args.ckpt_path)
        trainer.fit(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main()
