import math
import copy
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import construct_targets
from src.models.model import ClassificationModel


class Trainer:
    """ Classification Model trainer class. """

    def train(
            self,
            model: ClassificationModel,
            train_df: pd.DataFrame,
            eval_df: Optional[pd.DataFrame] = None,
            check_nan: bool = True
    ) -> Tuple[ClassificationModel, pd.DataFrame]:
        """ Fits the model in the provided dataset. """

        # Clean Dataframe.
        if check_nan and (train_df.isna().any().any() or (eval_df is not None and eval_df.isna().any().any())):
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        metrics_df = model.fit(train_df=train_df, eval_df=eval_df)
        return model, metrics_df

    def cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Evaluates the model using expanding-window temporal cross validation. """

        # Clean Dataframe.
        if df.isna().any().any():
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        temporal_df = self._sort_temporally(df=df)
        fold_indices = self._temporal_fold_indices(df=temporal_df, targets=construct_targets(df=temporal_df, target_type=model.target_type), k_folds=k_folds)

        metrics_df_per_fold = []
        for i, (train_ids, eval_ids) in tqdm(iterable=enumerate(fold_indices), desc='Evaluating Temporal CV', total=len(fold_indices)):
            train_df = temporal_df.iloc[train_ids]
            eval_df = temporal_df.iloc[eval_ids]
            fold_model = copy.deepcopy(model)
            _, metrics_df = self.train(model=fold_model, train_df=train_df, eval_df=eval_df, check_nan=False)
            metrics_df['Fold'] = i + 1
            metrics_df['Train End Date'] = train_df.iloc[-1]['Date'] if 'Date' in train_df else ''
            metrics_df['Eval Start Date'] = eval_df.iloc[0]['Date'] if 'Date' in eval_df else ''
            metrics_df['Eval End Date'] = eval_df.iloc[-1]['Date'] if 'Date' in eval_df else ''
            metrics_df['Samples'] = [train_df.shape[0], eval_df.shape[0]]
            metrics_df_per_fold.append(metrics_df)
        cv_df = pd.concat(metrics_df_per_fold, ignore_index=True, axis=0)
        return cv_df

    def sliding_cross_validation(
            self,
            model: ClassificationModel,
            df: pd.DataFrame,
            test_ratio: float,
            k_folds: int = 5
    ) -> pd.DataFrame:
        """ Evaluates the model using the Sliding k-fold cross validation method. """

        # Rectify the k-folds if case samples are too few.
        if df.isna().any().any():
            raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

        samples_per_fold = int(math.floor(df.shape[0]/k_folds))

        # Initialize sliding k-fold procedure.
        metrics_df_per_fold = []
        for i in tqdm(iterable=range(k_folds), desc='Evaluating Sliding K-Fold', total=k_folds):
            fold_df = df.iloc[-(i+1)*samples_per_fold:]
            train_df, eval_df = train_test_split(df=fold_df, test_size=test_ratio)
            fold_model = copy.deepcopy(model)
            _, metrics_df = self.train(model=fold_model, train_df=train_df, eval_df=eval_df, check_nan=False)
            metrics_df['Fold'] = i + 1
            metrics_df['Start Date'] = [train_df.iloc[-1]['Date'], eval_df.iloc[-1]['Date']]
            metrics_df['End Date'] = [train_df.iloc[0]['Date'], eval_df.iloc[0]['Date']]
            metrics_df['Samples'] = [train_df.shape[0], eval_df.shape[0]]
            metrics_df_per_fold.append(metrics_df)
        cv_df = pd.concat(metrics_df_per_fold, ignore_index=True, axis=0)
        return cv_df

    @staticmethod
    def _sort_temporally(df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' not in df:
            return df.reset_index(drop=True)
        return df.sort_values(by=['Date'], ascending=True, kind='stable').reset_index(drop=True)

    @staticmethod
    def _temporal_fold_indices(df: pd.DataFrame, targets: np.ndarray, k_folds: int) -> list[Tuple[np.ndarray, np.ndarray]]:
        if k_folds < 2:
            raise ValueError('k_folds must be greater than or equal to 2.')
        if df.shape[0] <= k_folds:
            raise ValueError(f'Cannot build {k_folds} temporal folds from {df.shape[0]} samples.')

        classes = set(np.unique(targets).tolist())
        initial_train_size = max(1, int(math.floor(df.shape[0] / (k_folds + 1))))
        while initial_train_size < df.shape[0] and set(np.unique(targets[:initial_train_size]).tolist()) != classes:
            initial_train_size += 1

        if df.shape[0] - initial_train_size < k_folds:
            raise ValueError(
                'Cannot build temporal folds with all classes represented in the initial training window.'
            )

        eval_blocks = np.array_split(np.arange(initial_train_size, df.shape[0]), k_folds)
        fold_indices = []
        for eval_ids in eval_blocks:
            if eval_ids.shape[0] == 0:
                continue
            train_ids = np.arange(0, int(eval_ids[0]))
            fold_indices.append((train_ids, eval_ids))

        if len(fold_indices) != k_folds:
            raise ValueError(f'Cannot build {k_folds} non-empty temporal folds.')
        return fold_indices
