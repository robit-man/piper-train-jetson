import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .dataset import PiperDataset, UtteranceCollate, Batch
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn

_LOGGER = logging.getLogger("vits.lightning")


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 4096,
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 16,
        lr_decay: float = 0.999875,
        grad_clip: Optional[float] = None,
        num_workers: int = 1,
        seed: int = 1234,
        validation_split: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if num_speakers > 1 and gin_channels <= 0:
            self.hparams.gin_channels = 512

        self.model_g = SynthesizerTrn(
            n_vocab=num_symbols,
            spec_channels=filter_length // 2 + 1,
            segment_size=segment_size // hop_length,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            n_speakers=num_speakers,
            gin_channels=gin_channels,
            use_sdp=use_sdp,
        )
        self.model_d = MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm)
        self.automatic_optimization = False
        self._train_dataset = None
        self._val_dataset = None
        self._load_datasets(validation_split)

    def _load_datasets(self, validation_split: float):
        if not self.hparams.dataset:
            return
        full_dataset = PiperDataset(self.hparams.dataset)
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        self._train_dataset, self._val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def forward(self, *args, **kwargs):
        return self.model_g(*args, **kwargs)

    def training_step(self, batch: Batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Generator step
        opt_g.zero_grad()
        loss_g, y_hat = self.generator_step(batch)
        self.manual_backward(loss_g)
        opt_g.step()

        # Discriminator step
        opt_d.zero_grad()
        loss_d = self.discriminator_step(batch, y_hat)
        self.manual_backward(loss_d)
        opt_d.step()

        self.log("train_loss_g", loss_g, prog_bar=True, on_step=True)
        self.log("train_loss_d", loss_d, prog_bar=True, on_step=True)

    def generator_step(self, batch: Batch, dump_folder: str = "./spectrogram_dumps"):
        # Create dump folder if it doesn't exist
        os.makedirs(dump_folder, exist_ok=True)

        # Extract inputs
        x = batch.phoneme_ids
        x_lengths = batch.phoneme_lengths
        y = batch.audios
        y_lengths = batch.audio_lengths

        # Validate lengths
        if x_lengths is None or y_lengths is None:
            raise ValueError("Input lengths (phoneme_lengths or audio_lengths) cannot be None.")
        if torch.any(x_lengths <= 0) or torch.any(y_lengths <= 0):
            raise ValueError("Invalid lengths detected: all lengths must be positive.")

        # Debug input lengths
        _LOGGER.debug(f"Phoneme lengths: {x_lengths}")
        _LOGGER.debug(f"Audio lengths: {y_lengths}")

        # Ensure raw audio is 2D: remove channel dimension if present
        if y.dim() == 3 and y.shape[1] == 1:
            y = y.squeeze(1)  # Shape: [B, 1, T] -> [B, T]
        elif y.dim() != 2:
            raise ValueError(f"Audio input has unexpected dimensions: {y.dim()}D")

        # Debug input audio shape
        _LOGGER.debug(f"Input audio shape: {y.shape}")

        # Convert raw audio to linear spectrogram
        window = torch.hann_window(self.hparams.win_length).to(y.device)
        spec = torch.stft(
            y,
            n_fft=self.hparams.filter_length,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            window=window,
            return_complex=True,
        )  # Output shape: [B, freq_bins, time_frames]

        # Compute magnitude of the complex spectrogram
        spec = torch.abs(spec)  # Shape: [B, freq_bins, time_frames]

        # Debug spectrogram shape
        _LOGGER.debug(f"Generated spectrogram shape: {spec.shape}")

        # Dump spectrograms
        for i in range(spec.size(0)):
            spec_numpy = spec[i].cpu().numpy()
            dump_path = os.path.join(dump_folder, f"spectrogram_{i}.npy")
            np.save(dump_path, spec_numpy)
            _LOGGER.debug(f"Saved spectrogram to: {dump_path}")

            # Optional: Save as an image
            plt.figure(figsize=(10, 4))
            plt.imshow(spec_numpy, aspect="auto", origin="lower", interpolation="none")
            plt.colorbar()
            plt.title(f"Spectrogram {i}")
            plt.xlabel("Time Frames")
            plt.ylabel("Frequency Bins")
            img_path = os.path.join(dump_folder, f"spectrogram_{i}.png")
            plt.savefig(img_path)
            plt.close()
            _LOGGER.debug(f"Saved spectrogram image to: {img_path}")

        # Pass the spectrogram into the model to generate y_hat
        y_hat, *_ = self.model_g(x, x_lengths, spec, y_lengths, None)

        # Convert raw audio to mel spectrogram for loss calculation
        y_mel = mel_spectrogram_torch(
            y,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )

        # Debug mel spectrogram shape
        _LOGGER.debug(f"Target mel spectrogram shape: {y_mel.shape}")

        # Convert predicted spectrogram to mel spectrogram
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),  # Remove channel dimension for mel calculation
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )

        # Debug predicted mel spectrogram shape
        _LOGGER.debug(f"Predicted mel spectrogram shape: {y_hat_mel.shape}")

        # Align dimensions
        min_length = min(y_mel.size(2), y_hat_mel.size(2))
        y_mel = y_mel[:, :, :min_length]
        y_hat_mel = y_hat_mel[:, :, :min_length]

        # Compute loss
        loss_mel = F.l1_loss(y_hat_mel, y_mel)

        return loss_mel, y_hat

    def discriminator_step(self, batch: Batch, y_hat):
        x, x_lengths, y, y_lengths = batch.phoneme_ids, batch.phoneme_lengths, batch.audios, batch.audio_lengths

        # Validate y_lengths
        if y_lengths is None:
            raise ValueError("Audio lengths (audio_lengths) cannot be None in discriminator_step.")
        if torch.any(y_lengths <= 0):
            raise ValueError("Invalid audio lengths detected: all lengths must be positive.")

        # Debug input lengths
        _LOGGER.debug(f"Discriminator Step - Phoneme lengths: {x_lengths}")
        _LOGGER.debug(f"Discriminator Step - Audio lengths: {y_lengths}")

        # Pass real and generated audio to discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        # Compute discriminator loss
        loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        return loss_d

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.model_g.parameters(), lr=self.hparams.learning_rate)
        opt_d = torch.optim.AdamW(self.model_d.parameters(), lr=self.hparams.learning_rate)
        return [opt_g, opt_d]

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.gin_channels > 0,
                segment_size=self.hparams.segment_size,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.gin_channels > 0,
                segment_size=self.hparams.segment_size,
            ),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--validation-split", type=float, default=0.1)
        parser.add_argument("--num-workers", type=int, default=4)
        return parent_parser
