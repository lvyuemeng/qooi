"""Learned state-discovery preparation and ML lifecycle methods."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.ai import vq_rssm
from qooi.ai.contracts import CodeSequence, WindowDataset
from qooi.research import states


@dataclass(frozen=True)
class PreparedStateDiscovery:
    config: states.LearnedStateConfig
    feature_frame: pl.DataFrame
    split_frame: pl.DataFrame
    windows: states.PreparedWindows
    provenance: states.WindowProvenance
    dataset: WindowDataset

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        config: states.LearnedStateConfig,
    ) -> PreparedStateDiscovery:
        return cls.from_frames((frame,), config)

    @classmethod
    def from_frames(
        cls,
        frames: tuple[pl.DataFrame, ...],
        config: states.LearnedStateConfig,
    ) -> PreparedStateDiscovery:
        if not frames:
            raise ValueError("at least one frame is required")
        feature_frames = []
        split_frames = []
        prepared_windows = []
        provenances = []
        required_columns = config.required_columns()
        for frame in frames:
            missing = [column for column in required_columns if column not in frame.columns]
            if missing:
                raise ValueError("missing required columns: " + ", ".join(missing))
            work = (
                frame.sort(config.columns.timestamp)
                if config.columns.timestamp in frame.columns
                else frame
            )
            feature_frame = config.window.features(
                work,
                config.columns,
                config.volatility_scaling,
            )
            split = config.window.split(feature_frame.height)
            split_frame = config.window.assign_split(feature_frame, split, config.columns)
            windows, provenance = config.window.windows(
                split_frame,
                config.columns,
                config.feature_columns,
                volatility_scale_column=config.volatility_scaling.output_column,
            )
            feature_frames.append(feature_frame)
            split_frames.append(split_frame)
            prepared_windows.append(windows)
            provenances.append(provenance)
        windows = states.PreparedWindows.concat(prepared_windows)
        provenance = states.WindowProvenance.concat(provenances)
        feature_frame = pl.concat(feature_frames, how="diagonal_relaxed")
        split_frame = pl.concat(split_frames, how="diagonal_relaxed")
        if len(windows.features) != len(provenance.row_index):
            raise ValueError("prepared windows and provenance counts must match")
        return cls(
            config=config,
            feature_frame=feature_frame,
            split_frame=split_frame,
            windows=windows,
            provenance=provenance,
            dataset=windows.to_dataset(),
        )

    def model_spec(self) -> vq_rssm.VqRssmSpec:
        config = self.config.vq_rssm
        return vq_rssm.VqRssmSpec(
            input_dim=config.input_dim or len(self.config.feature_columns),
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes,
            commitment_cost=config.commitment_cost,
            kl_anneal_steps=config.kl_anneal_steps,
            eps=config.eps,
        )

    def train_spec(self) -> vq_rssm.TrainSpec:
        config = self.config.train
        return vq_rssm.TrainSpec(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
        )

    def train_model(self) -> vq_rssm.TrainingResult:
        return vq_rssm.train(
            self.dataset,
            self.model_spec(),
            self.train_spec(),
            output_dir=self.config.checkpoint_dir,
        )

    def load_checkpoint(self) -> vq_rssm.VqRssmCheckpoint:
        return vq_rssm.load_checkpoint(self.config.checkpoint_path())

    def predict_codes(self, checkpoint: vq_rssm.VqRssmCheckpoint) -> CodeSequence:
        return vq_rssm.predict_codes(self.dataset, checkpoint)

    def predict_states(
        self,
        checkpoint: vq_rssm.VqRssmCheckpoint,
    ) -> states.StateSequence:
        return self.provenance.states_from_codes(
            self.predict_codes(checkpoint),
            state_column=self.config.state_column,
        )
