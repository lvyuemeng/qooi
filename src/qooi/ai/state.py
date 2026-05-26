"""Learned state-discovery preparation and ML lifecycle methods."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import polars as pl

from qooi.ai import vq_rssm
from qooi.ai.contracts import CodeSequence, SequenceDataset, WindowDataset
from qooi.research import states

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedStateDiscovery:
    config: states.LearnedStateConfig
    feature_frame: pl.DataFrame
    split_frame: pl.DataFrame
    windows: states.PreparedWindows
    provenance: states.WindowProvenance
    dataset: WindowDataset
    sequences: states.PreparedSequences
    sequence_provenance: states.WindowProvenance
    sequence_dataset: SequenceDataset

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
        prepared_sequences = []
        sequence_provenances = []
        required_columns = config.required_columns()
        total_start = time.perf_counter()
        total_input_rows = 0
        for index, frame in enumerate(frames, start=1):
            frame_start = time.perf_counter()
            total_input_rows += frame.height
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
            sequences, sequence_provenance = config.window.sequences(
                split_frame,
                config.columns,
                config.sequence,
                config.feature_columns,
                volatility_scale_column=config.volatility_scaling.output_column,
            )
            feature_frames.append(feature_frame)
            split_frames.append(split_frame)
            prepared_windows.append(windows)
            provenances.append(provenance)
            prepared_sequences.append(sequences)
            sequence_provenances.append(sequence_provenance)
            split_counts = split_frame.group_by("split").agg(pl.len().alias("rows"))
            split_rows = {
                row["split"]: row["rows"] for row in split_counts.iter_rows(named=True)
            }
            logger.info(
                "prepare_frame index=%s/%s input_rows=%s feature_rows=%s train_rows=%s "
                "valid_rows=%s test_rows=%s windows=%s elapsed_s=%.2f",
                index,
                len(frames),
                frame.height,
                feature_frame.height,
                split_rows.get("train", 0),
                split_rows.get("valid", 0),
                split_rows.get("test", 0),
                len(windows.features),
                time.perf_counter() - frame_start,
            )
        windows = states.PreparedWindows.concat(prepared_windows)
        provenance = states.WindowProvenance.concat(provenances)
        sequences = states.PreparedSequences.concat(prepared_sequences)
        sequence_provenance = states.WindowProvenance.concat(sequence_provenances)
        feature_frame = pl.concat(feature_frames, how="diagonal_relaxed")
        split_frame = pl.concat(split_frames, how="diagonal_relaxed")
        if len(windows.features) != len(provenance.row_index):
            raise ValueError("prepared windows and provenance counts must match")
        split_window_counts = {
            split: sum(1 for item in windows.splits if item == split)
            for split in ("train", "valid", "test")
        }
        logger.info(
            "prepare_done input_rows=%s feature_rows=%s windows=%s train_windows=%s "
            "valid_windows=%s test_windows=%s sequences=%s sequence_rows=%s elapsed_s=%.2f",
            total_input_rows,
            feature_frame.height,
            len(windows.features),
            split_window_counts["train"],
            split_window_counts["valid"],
            split_window_counts["test"],
            len(sequences.sequences),
            len(sequence_provenance.row_index),
            time.perf_counter() - total_start,
        )
        return cls(
            config=config,
            feature_frame=feature_frame,
            split_frame=split_frame,
            windows=windows,
            provenance=provenance,
            dataset=windows.to_dataset(),
            sequences=sequences,
            sequence_provenance=sequence_provenance,
            sequence_dataset=sequences.to_dataset(),
        )

    def model_spec(self) -> vq_rssm.VqRssmSpec:
        config = self.config.vq_rssm
        objective = self.config.objective
        return vq_rssm.VqRssmSpec(
            input_dim=config.input_dim or len(self.config.feature_columns),
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_codes=config.num_codes,
            commitment_cost=config.commitment_cost,
            kl_anneal_steps=config.kl_anneal_steps,
            eps=config.eps,
            objective_terms=objective.terms,
            reconstruct_weight=objective.reconstruct,
            vq_weight=objective.vq,
            kl_weight=objective.kl,
            future_weight=objective.future,
        )

    def sequence_runtime_config(self) -> vq_rssm.SequenceRuntimeConfig:
        return vq_rssm.SequenceRuntimeConfig(
            chunk=self.config.sequence.chunk,
            warmup=self.config.sequence.warmup,
            stride=self.config.sequence.stride,
            carry=self.config.sequence.carry,
        )

    def train_model(self) -> vq_rssm.TrainingResult:
        if self.config.input == "sequence":
            return vq_rssm.train_sequences(
                self.sequence_dataset,
                self.model_spec(),
                self.config.train,
                self.sequence_runtime_config(),
                output_dir=self.config.checkpoint_dir,
            )
        return vq_rssm.train(
            self.dataset,
            self.model_spec(),
            self.config.train,
            output_dir=self.config.checkpoint_dir,
        )

    def load_checkpoint(self, path=None) -> vq_rssm.VqRssmCheckpoint:
        return vq_rssm.load_checkpoint(path or self.config.checkpoint_path())

    def predict_codes(self, checkpoint: vq_rssm.VqRssmCheckpoint) -> CodeSequence:
        if self.config.input == "sequence":
            return vq_rssm.predict_sequence_codes(
                self.sequence_dataset,
                checkpoint,
                self.sequence_runtime_config(),
                train_cfg=self.config.train,
            )
        return vq_rssm.predict_codes(self.dataset, checkpoint, train_cfg=self.config.train)

    def predict_diagnostics(
        self, checkpoint: vq_rssm.VqRssmCheckpoint
    ) -> vq_rssm.InferenceDiagnostics:
        if self.config.input == "sequence":
            return vq_rssm.predict_sequence_diagnostics(
                self.sequence_dataset,
                checkpoint,
                self.sequence_runtime_config(),
                train_cfg=self.config.train,
            )
        return vq_rssm.predict_diagnostics(self.dataset, checkpoint, train_cfg=self.config.train)

    def predict_states(
        self,
        checkpoint: vq_rssm.VqRssmCheckpoint,
    ) -> states.StateSequence:
        provenance = (
            self.sequence_provenance
            if self.config.input == "sequence"
            else self.provenance
        )
        return provenance.states_from_codes(
            self.predict_codes(checkpoint),
            state_column=self.config.state_column,
        )
