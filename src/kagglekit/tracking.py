"""Experiment and submission tracking.

Teams that log rigorously make monotonic progress; teams that do not
rediscover their own failures. CSV-backed so the log lives in git next to
the code and survives any environment.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

EXPERIMENT_COLUMNS = ["id", "timestamp", "parent_id", "hypothesis", "change",
                      "oof_score", "fold_std", "delta_vs_parent",
                      "delta_over_noise_floor", "lb_score", "runtime_s",
                      "verdict", "notes"]

SUBMISSION_COLUMNS = ["sub_id", "timestamp", "experiment_id", "file",
                      "hypothesis", "oof_score", "public_lb", "private_lb"]


class ExperimentLog:
    """Append-only experiment record with pre-registered acceptance rules.

    Usage::

        log = ExperimentLog("experiments.csv", noise_floor=0.003)
        exp_id = log.record(hypothesis="TE crosses add signal",
                            change="added 9 target-encoded crosses",
                            oof_score=0.9503, fold_scores=[...],
                            parent_id=3)
    """

    def __init__(self, path: str, noise_floor: float | None = None,
                 accept_multiple: float = 1.0):
        self.path = path
        self.noise_floor = noise_floor
        self.accept_multiple = accept_multiple
        if os.path.exists(path):
            self.df = pd.read_csv(path)
            for c in EXPERIMENT_COLUMNS:
                if c not in self.df.columns:
                    self.df[c] = np.nan
        else:
            self.df = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

    def record(self, hypothesis: str, change: str, oof_score: float,
               fold_scores=None, parent_id=None, lb_score=None,
               runtime_s=None, notes: str = "") -> int:
        exp_id = int(self.df["id"].max()) + 1 if len(self.df) else 1
        fold_std = float(np.std(fold_scores, ddof=1)) if fold_scores is not None else np.nan
        delta = np.nan
        if parent_id is not None and (self.df["id"] == parent_id).any():
            parent = float(self.df.loc[self.df["id"] == parent_id, "oof_score"].iloc[0])
            delta = oof_score - parent
        d_over_floor = (delta / self.noise_floor
                        if self.noise_floor and not np.isnan(delta) else np.nan)
        verdict = ""
        if not np.isnan(delta) and self.noise_floor:
            verdict = ("keep" if delta > self.accept_multiple * self.noise_floor
                       else "revert" if delta < -self.accept_multiple * self.noise_floor
                       else "noise")
        row = dict(id=exp_id,
                   timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   parent_id=parent_id, hypothesis=hypothesis, change=change,
                   oof_score=oof_score, fold_std=fold_std, delta_vs_parent=delta,
                   delta_over_noise_floor=d_over_floor, lb_score=lb_score,
                   runtime_s=runtime_s, verdict=verdict, notes=notes)
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return exp_id

    def best(self) -> pd.Series:
        return self.df.loc[self.df["oof_score"].idxmax()]

    def summary(self, n: int = 10) -> pd.DataFrame:
        cols = ["id", "hypothesis", "oof_score", "delta_vs_parent",
                "delta_over_noise_floor", "verdict"]
        return self.df.sort_values("oof_score", ascending=False)[cols].head(n)


class SubmissionLog:
    """Track submissions and the OOF vs LB relationship, the health monitor of
    the validation scheme."""

    def __init__(self, path: str):
        self.path = path
        self.df = (pd.read_csv(path) if os.path.exists(path)
                   else pd.DataFrame(columns=SUBMISSION_COLUMNS))

    def record(self, experiment_id: int, file: str, hypothesis: str,
               oof_score: float, public_lb: float | None = None,
               private_lb: float | None = None) -> int:
        sub_id = int(self.df["sub_id"].max()) + 1 if len(self.df) else 1
        row = dict(sub_id=sub_id,
                   timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   experiment_id=experiment_id, file=file, hypothesis=hypothesis,
                   oof_score=oof_score, public_lb=public_lb, private_lb=private_lb)
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return sub_id

    def cv_lb_correlation(self) -> float:
        """Pearson correlation between OOF and public LB across submissions.

        Correlated: trust CV, spend submissions only on offline-undecidable
        questions. Uncorrelated: stop submitting and fix the CV.
        """
        d = self.df.dropna(subset=["oof_score", "public_lb"])
        if len(d) < 3:
            return float("nan")
        return float(np.corrcoef(d["oof_score"], d["public_lb"])[0, 1])
