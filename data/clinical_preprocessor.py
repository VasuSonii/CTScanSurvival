"""
data/clinical_preprocessor.py
===============================
Fits and applies the sklearn preprocessing pipeline for clinical tabular data.

Responsibilities
----------------
  1. Load the raw JSON / CSV metadata.
  2. Drop columns with > missing_threshold missing values.
  3. Separate targets (vital_status, vital_days_after_surgery) from features.
  4. Fit numeric (median impute + StandardScaler) and categorical
     (constant impute + OneHotEncoder) pipelines on the TRAIN cases only.
  5. Transform train and val cases separately (no leakage).
  6. Return a {case_id: torch.Tensor} lookup dict for use in the train loop.
  7. Persist the fitted pipeline to disk so it can be reloaded at inference.

Usage
-----
    from data.clinical_preprocessor import ClinicalPreprocessor

    cp = ClinicalPreprocessor(cfg)
    train_feats, val_feats, clinical_dim = cp.fit_transform(train_ids, val_ids)
    # train_feats / val_feats : dict[str, torch.Tensor]  shape (clinical_dim,)
    cp.save(path)           # persist fitted pipeline

    cp2 = ClinicalPreprocessor.load(path)
    feats = cp2.transform({"case_00042": raw_row})  # inference
"""

import json
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class _CastToStr:
    """Picklable replacement for lambda x: x.astype(str)."""
    def fit(self, X, y=None):        return self
    def transform(self, X):          return X.astype(str)
    def fit_transform(self, X, y=None): return self.transform(X)
    def get_params(self, deep=True): return {}

# Columns that are targets, identifiers, directly leak survival time, or encode
# tumour characteristics that the imaging pipeline already captures.
#
# ── Survival leakage ──────────────────────────────────────────────────────────
#   last_postop_egfr.days_after_nephrectomy — equals vital_days_after_surgery
#     for censored patients (last follow-up = end of observation).
#   last_postop_egfr.value — measured at the final follow-up timepoint, so its
#     existence implies the patient was alive at that time.
#
# ── Tumour characteristics ────────────────────────────────────────────────────
#   These fields are excluded because:
#   (a) OmniRad already encodes this information directly from the CT scan.
#       Including them alongside imaging would make it impossible to attribute
#       model performance to imaging vs. pathology labels.
#   (b) Several are post-surgical pathology findings (pathologic_size,
#       histologic_subtype, margins, grade) that are not available at the
#       time a scan-based model would run in practice.
#   (c) aua_risk_score is a clinical composite derived from pathology staging —
#       essentially a label that summarises prognosis, which the model should
#       learn to predict rather than receive as input.
_EXCLUDE_COLS = {
    # Targets and identifiers
    "case_id",
    "vital_status",
    "vital_days_after_surgery",
    # Survival-time leakage
    "last_postop_egfr.days_after_nephrectomy",
    "last_postop_egfr.value",
    # first_postop_egfr.value (kidney function reading) is safe to keep.
    # first_postop_egfr.days_after_nephrectomy is not — its value encodes a
    # lower bound on survival time: if it equals 184, the patient was alive at
    # day 184. Patients who die early simply have no measurement, and median
    # imputation does not remove this structural signal.
    "first_postop_egfr.days_after_nephrectomy",
    # Tumour characteristics (captured by imaging; post-surgical pathology)
    "aua_risk_score",
    "radiographic_size",
    "pathologic_size",
    "malignant",
    "tumor_histologic_subtype",
    "pathology_t_stage",
    "pathology_n_stage",
    "pathology_m_stage",
    "tumor_necrosis",
    "tumor_isup_grade",
    "positive_resection_margins",
    "sarcomatoid_features",
    "rhabdoid_features",
}


class ClinicalPreprocessor:
    """
    Wraps the full tabular preprocessing pipeline.

    Parameters
    ----------
    metadata_path     : path to kits.json (KiTS23 format) or a CSV file
    missing_threshold : columns with > this fraction missing are dropped
    """

    def __init__(
        self,
        metadata_path:     str,
        missing_threshold: float = 0.40,
    ):
        self.metadata_path     = metadata_path
        self.missing_threshold = missing_threshold
        self._pipeline:  Optional[ColumnTransformer] = None
        self._feat_names: list[str] = []
        self._drop_cols:  list[str] = []
        self._bool_cols:  list[str] = []
        self._df: Optional[pd.DataFrame] = None

    # ── Data loading ──────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        """Load metadata and flatten nested dicts (comorbidities, eGFR, etc.)."""
        if self.metadata_path.endswith(".csv"):
            df = pd.read_csv(self.metadata_path)
        else:
            with open(self.metadata_path) as f:
                data = json.load(f)
            # json_normalize flattens nested dicts:
            #   comorbidities.copd, last_preop_egfr.value, etc.
            df = pd.json_normalize(data)

        # Set case_id as index for fast lookup, keep as column too
        if "case_id" in df.columns:
            df = df.set_index("case_id", drop=False)
        return df

    # ── Fit + transform ───────────────────────────────────────────────────

    def fit_transform(
        self,
        train_ids: list[str],
        val_ids:   list[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        """
        Fit the pipeline on train_ids, transform both splits.

        Returns
        -------
        train_feats  : {case_id: Tensor(clinical_dim,)}
        val_feats    : {case_id: Tensor(clinical_dim,)}
        clinical_dim : int — feature vector length after preprocessing
        """
        if self._df is None:
            self._df = self._load()

        df = self._df

        # ── Drop high-missingness columns ─────────────────────────────────
        # Compute on train split only to avoid leakage
        train_df = df.loc[df["case_id"].isin(train_ids)] if "case_id" in df.columns \
                   else df.loc[train_ids]

        missing_frac  = train_df.isnull().mean()
        self._drop_cols = missing_frac[missing_frac > self.missing_threshold].index.tolist()
        logger.info(
            f"Dropping {len(self._drop_cols)} columns with >{self.missing_threshold*100:.0f}% "
            f"missing: {self._drop_cols}"
        )

        # ── Build feature matrix — exclude targets and identifiers ─────────
        drop = list(_EXCLUDE_COLS) + self._drop_cols
        X = df.drop(columns=[c for c in drop if c in df.columns])

        # Boolean columns (e.g. all comorbidity flags) are cast to int so
        # they go into the numeric pipeline as 0/1 rather than being
        # one-hot encoded as "True"/"False" which doubles feature count.
        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            X = X.copy()
            X[bool_cols] = X[bool_cols].astype("int64")

        numeric_cols     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        logger.info(
            f"Clinical features — numeric: {len(numeric_cols)}  "
            f"(incl. {len(bool_cols)} bool→int)  "
            f"categorical: {len(categorical_cols)}"
        )

        # ── Build and fit pipeline on train split only ─────────────────────
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("to_str",  _CastToStr()),
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self._pipeline = ColumnTransformer([
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ])
        # Store bool cols so _to_tensors can apply the same cast
        self._bool_cols = bool_cols

        # Fit ONLY on train rows
        X_train = X.loc[X.index.isin(train_ids)]
        self._pipeline.fit(X_train)

        # Feature names after encoding
        cat_names = (
            self._pipeline.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
            .tolist()
        )
        self._feat_names = numeric_cols + cat_names
        clinical_dim = len(self._feat_names)
        logger.info(f"Clinical feature dim after encoding: {clinical_dim}")

        train_feats = self._to_tensors(X, train_ids)
        val_feats   = self._to_tensors(X, val_ids)

        return train_feats, val_feats, clinical_dim

    def _to_tensors(
        self,
        X:    pd.DataFrame,
        ids:  list[str],
    ) -> dict[str, torch.Tensor]:
        """Transform a subset of rows and return {case_id: tensor}."""
        X_sub = X.loc[X.index.isin(ids)].copy()
        if self._bool_cols:
            present = [c for c in self._bool_cols if c in X_sub.columns]
            X_sub[present] = X_sub[present].astype("int64")
        arr      = self._pipeline.transform(X_sub).astype(np.float32)
        tensors  = {}
        for i, cid in enumerate(X_sub.index):
            tensors[cid] = torch.from_numpy(arr[i])
        # Warn for any ids that were missing from the metadata
        missing = set(ids) - set(tensors.keys())
        if missing:
            logger.warning(f"No clinical data for {len(missing)} cases: {missing}")
        return tensors

    # ── Persist ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the fitted pipeline + metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "pipeline":      self._pipeline,
            "feat_names":    self._feat_names,
            "drop_cols":     self._drop_cols,
            "bool_cols":     self._bool_cols,
            "metadata_path": self.metadata_path,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Clinical preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ClinicalPreprocessor":
        """Restore a fitted preprocessor from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(metadata_path=payload["metadata_path"])
        obj._pipeline   = payload["pipeline"]
        obj._feat_names = payload["feat_names"]
        obj._drop_cols  = payload["drop_cols"]
        obj._bool_cols  = payload.get("bool_cols", [])
        logger.info(f"Clinical preprocessor loaded ← {path}  "
                    f"(dim={len(obj._feat_names)})")
        return obj

    @property
    def feature_dim(self) -> int:
        return len(self._feat_names)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feat_names)


# ═════════════════════════════════════════════════════════════════════════════
# HECKTOR-specific preprocessor
# ═════════════════════════════════════════════════════════════════════════════

class ClinicalPreprocessorHector:
    """
    Tabular preprocessing pipeline for HECKTOR Task 2 clinical data.

    HECKTOR has very limited clinical features (~6 after dropping targets):
      Age, Gender, Tobacco Consumption, Alcohol Consumption, Treatment, M-stage

    Performance Status is dropped — typically >40% missing.
    Gender, Tobacco, Alcohol, Treatment are already numeric (0/1/2) but
    treated as categorical for proper one-hot encoding since they are codes
    not ordinal quantities.
    M-stage is string categorical (M0, M1, Mx).

    After encoding expect ~8-12 features total.  Use clinical_dim=32 output
    in ClinicalMLP — 128 would be overparameterised for this input size.
    """

    def __init__(self, metadata_path: str, missing_threshold: float = 0.40):
        self.metadata_path     = metadata_path
        self.missing_threshold = missing_threshold
        self._pipeline:   Optional[ColumnTransformer] = None
        self._feat_names: list[str] = []
        self._drop_cols:  list[str] = []
        self._df:         Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_path)
        if "PatientID" in df.columns:
            df = df.set_index("PatientID", drop=False)
        return df

    def fit_transform(
        self,
        train_ids: list[str],
        val_ids:   list[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        if self._df is None:
            self._df = self._load()
        df = self._df

        # Always-excluded: identifiers, targets, and high-missing cols
        always_drop = {"PatientID", "CenterID", "RFS", "Relapse", "Performance Status"}

        # Drop additional high-missingness columns (computed on train only)
        train_df = df.loc[df.index.isin(train_ids)]
        missing_frac = train_df.isnull().mean()
        high_missing = set(missing_frac[missing_frac > 0.9].index)
        self._drop_cols = list(always_drop | high_missing)

        X = df.drop(columns=[c for c in self._drop_cols if c in df.columns])

        # Age is the only truly continuous feature
        numeric_cols     = ["Age"] if "Age" in X.columns else []
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        logger.info(
            f"HECKTOR clinical features — numeric: {len(numeric_cols)}  "
            f"categorical: {len(categorical_cols)}  "
            f"(dropped {len(self._drop_cols)} cols)"
        )

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("to_str",  _CastToStr()),
            # _CastToStr converts NaN → "nan"; use string "nan" as missing marker
            ("imputer", SimpleImputer(
                missing_values="nan", strategy="constant", fill_value="Unknown"
            )),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self._pipeline = ColumnTransformer([
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ])

        X_train = X.loc[X.index.isin(train_ids)]
        self._pipeline.fit(X_train)

        cat_names = (
            self._pipeline.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
            .tolist()
        ) if categorical_cols else []
        self._feat_names = numeric_cols + cat_names
        clinical_dim     = len(self._feat_names)
        logger.info(f"HECKTOR clinical feature dim after encoding: {clinical_dim}")

        train_feats = self._to_tensors(X, train_ids)
        val_feats   = self._to_tensors(X, val_ids)
        return train_feats, val_feats, clinical_dim

    def _to_tensors(self, X: pd.DataFrame, ids: list[str]) -> dict[str, torch.Tensor]:
        X_sub = X.loc[X.index.isin(ids)].copy()
        arr   = self._pipeline.transform(X_sub).astype(np.float32)
        tensors = {}
        for i, cid in enumerate(X_sub.index):
            tensors[cid] = torch.from_numpy(arr[i])
        missing = set(ids) - set(tensors.keys())
        if missing:
            logger.warning(f"No HECKTOR clinical data for {len(missing)} cases: {missing}")
        return tensors

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "pipeline":      self._pipeline,
            "feat_names":    self._feat_names,
            "drop_cols":     self._drop_cols,
            "metadata_path": self.metadata_path,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"HECKTOR clinical preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ClinicalPreprocessorHector":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(metadata_path=payload["metadata_path"])
        obj._pipeline   = payload["pipeline"]
        obj._feat_names = payload["feat_names"]
        obj._drop_cols  = payload.get("drop_cols", [])
        logger.info(
            f"HECKTOR clinical preprocessor loaded ← {path}  "
            f"(dim={len(obj._feat_names)})"
        )
        return obj

    @property
    def feature_dim(self) -> int:
        return len(self._feat_names)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feat_names)