import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    algorithm: str
    scaling_method: str = "standard"  # "standard", "minmax", "none"
    categorical_encoding: str = "onehot"  # "onehot", "label", "target", "vdm"
    handle_missing: str = "simple"  # "simple", "knn", "drop"
    outlier_detection: str = "none"  # "none", "iqr", "zscore", "isolation"
    cross_validation_folds: int = 10
    random_runs: int = 3
    test_size: float = 0.3
    k_neighbors: int = 5


def get_preprocessing_config(algorithm: str) -> PreprocessingConfig:
    configs = {
        "SMOTE": PreprocessingConfig(
            algorithm="SMOTE",
            scaling_method="none",  # В оригинальной статье не указано
            categorical_encoding="vdm",  # Value Distance Metric
            handle_missing="simple",
            outlier_detection="none",
            cross_validation_folds=10,
            random_runs=3,
            k_neighbors=5
        )
    }

    return configs.get(algorithm, configs["SMOTE"])


class SMOTEPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler = None
        self.categorical_encoder = None
        self.column_transformer = None
        self.feature_names_ = None
        self.categorical_features_ = None
        self.continuous_features_ = None
        self.target_encoder = None
        self.imputer = None

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:

        if self.config.handle_missing == "drop":
            X_processed = X.dropna()
            return X_processed

        elif self.config.handle_missing == "simple":
            imputer_continuous = SimpleImputer(strategy='median')
            imputer_categorical = SimpleImputer(strategy='most_frequent')

            X_processed = X.copy()

            if self.continuous_features_:
                X_processed[self.continuous_features_] = imputer_continuous.fit_transform(
                    X_processed[self.continuous_features_]
                )

            if self.categorical_features_:
                X_processed[self.categorical_features_] = imputer_categorical.fit_transform(
                    X_processed[self.categorical_features_]
                )

            return X_processed

        elif self.config.handle_missing == "knn":
            self.imputer = KNNImputer(n_neighbors=min(5, len(X) // 10))
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            return X_processed

        return X

    def _detect_outliers(self, X: pd.DataFrame) -> np.ndarray:

        if self.config.outlier_detection == "none":
            return np.ones(len(X), dtype=bool)

        outlier_mask = np.ones(len(X), dtype=bool)

        if self.config.outlier_detection == "iqr":
            # IQR method для непрерывных признаков
            for col in self.continuous_features_:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask &= (X[col] >= lower_bound) & (X[col] <= upper_bound)

        elif self.config.outlier_detection == "zscore":
            # Z-score method
            for col in self.continuous_features_:
                z_scores = np.abs(stats.zscore(X[col]))
                outlier_mask &= z_scores < 3

        return outlier_mask

    def _compute_vdm_distance(self, x1: pd.Series, x2: pd.Series, y: pd.Series) -> float:
        """
        Value Distance Metric (VDM)
        """
        distance = 0.0

        for feature in self.categorical_features_:
            if x1[feature] != x2[feature]:
                # Вычисляем VDM для данного признака
                classes = y.unique()
                vdm_dist = 0.0

                for class_label in classes:
                    # Количество экземпляров с x1[feature] в классе class_label
                    n_x1_class = len(y[(x1[feature] == x1[feature]) & (y == class_label)])
                    n_x1_total = len(y[x1[feature] == x1[feature]])

                    # Количество экземпляров с x2[feature] в классе class_label  
                    n_x2_class = len(y[(x2[feature] == x2[feature]) & (y == class_label)])
                    n_x2_total = len(y[x2[feature] == x2[feature]])

                    if n_x1_total > 0 and n_x2_total > 0:
                        prob_x1 = n_x1_class / n_x1_total
                        prob_x2 = n_x2_class / n_x2_total
                        vdm_dist += (prob_x1 - prob_x2) ** 2

                distance += vdm_dist

        return np.sqrt(distance)

    def _apply_scaling(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:

        if self.config.scaling_method == "none":
            return X

        X_scaled = X.copy()

        if self.continuous_features_:
            if fit:
                if self.config.scaling_method == "standard":
                    self.scaler = StandardScaler()
                elif self.config.scaling_method == "minmax":
                    self.scaler = MinMaxScaler()

                X_scaled[self.continuous_features_] = self.scaler.fit_transform(
                    X_scaled[self.continuous_features_]
                )
            else:
                if self.scaler is not None:
                    X_scaled[self.continuous_features_] = self.scaler.transform(
                        X_scaled[self.continuous_features_]
                    )

        return X_scaled

    def _encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:

        if not self.categorical_features_:
            return X

        X_encoded = X.copy()

        if self.config.categorical_encoding == "label":
            if fit:
                self.categorical_encoder = {}
                for col in self.categorical_features_:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    self.categorical_encoder[col] = le
            else:
                for col in self.categorical_features_:
                    if col in self.categorical_encoder:
                        le = self.categorical_encoder[col]
                        X_col = X_encoded[col].astype(str)

                        known_labels = set(le.classes_)
                        X_col = X_col.apply(lambda x: x if x in known_labels else le.classes_[0])
                        X_encoded[col] = le.transform(X_col)

        elif self.config.categorical_encoding == "onehot":
            if fit:
                self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

                encoded_data = self.categorical_encoder.fit_transform(
                    X_encoded[self.categorical_features_]
                )

                feature_names = self.categorical_encoder.get_feature_names_out(self.categorical_features_)

                X_encoded = X_encoded.drop(columns=self.categorical_features_)

                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X_encoded.index)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)

            else:
                if self.categorical_encoder is not None:
                    encoded_data = self.categorical_encoder.transform(
                        X_encoded[self.categorical_features_]
                    )
                    feature_names = self.categorical_encoder.get_feature_names_out(self.categorical_features_)

                    X_encoded = X_encoded.drop(columns=self.categorical_features_)
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X_encoded.index)
                    X_encoded = pd.concat([X_encoded, encoded_df], axis=1)

        return X_encoded

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SMOTEPreprocessor':

        X_processed = self._handle_missing_values(X)

        outlier_mask = self._detect_outliers(X_processed)
        if not np.all(outlier_mask):
            X_processed = X_processed[outlier_mask]
            y = y[outlier_mask]

        X_processed = self._encode_categorical(X_processed, fit=True)

        if self.config.categorical_encoding == "onehot":
            self.continuous_features_ = [col for col in X_processed.columns
                                         if col in self.continuous_features_]

        X_processed = self._apply_scaling(X_processed, fit=True)

        self.feature_names_ = X_processed.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Обработка пропущенных значений
        X_processed = self._handle_missing_values(X)

        # Кодирование категориальных признаков
        X_processed = self._encode_categorical(X_processed, fit=False)

        # Масштабирование
        X_processed = self._apply_scaling(X_processed, fit=False)

        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_preprocessing_report(self) -> Dict[str, Any]:
        report = {
            'algorithm': self.config.algorithm,
            'total_features': len(self.feature_names_) if self.feature_names_ else 0,
            'continuous_features': len(self.continuous_features_) if self.continuous_features_ else 0,
            'categorical_features': len(self.categorical_features_) if self.categorical_features_ else 0,
            'scaling_method': self.config.scaling_method,
            'categorical_encoding': self.config.categorical_encoding,
            'missing_value_handling': self.config.handle_missing,
            'outlier_detection': self.config.outlier_detection
        }

        return report
