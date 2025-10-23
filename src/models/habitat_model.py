"""
Main habitat modeling class for truffle habitat analysis.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class HabitatModel:
    """Main habitat modeling class for truffle habitat analysis."""
    
    def __init__(self, config: Dict[str, Any], models_dir: Path):
        self.config = config
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
        
        self.species_classifier = None
        self.suitability_predictor = None
        self.feature_importance_ = None
        self.training_data_ = None
        
    def train_models(self, data: pd.DataFrame, 
                    target_column: str = 'species',
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train habitat models on the provided data.
        
        Args:
            data: Habitat data DataFrame
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        if data.empty:
            raise ValueError("No data provided for training")
            
        logger.info("Starting model training")
        
        # Prepare features
        X, y, feature_names = self._prepare_features(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config['random_state'], 
            stratify=y if target_column == 'species' else None
        )
        
        # Train species classifier
        if target_column == 'species':
            self.species_classifier = self._train_species_classifier(X_train, y_train)
            species_results = self._evaluate_species_classifier(X_test, y_test)
        else:
            species_results = None
            
        # Train habitat suitability predictor
        self.suitability_predictor = self._train_suitability_predictor(X_train, y_train)
        suitability_results = self._evaluate_suitability_predictor(X_test, y_test)
        
        # Calculate feature importance
        self.feature_importance_ = self._calculate_feature_importance(feature_names)
        
        # Store training data
        self.training_data_ = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }
        
        # Save models
        self._save_models()
        
        results = {
            'species_classification': species_results,
            'habitat_suitability': suitability_results,
            'feature_importance': self.feature_importance_,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info("Model training complete")
        return results
        
    def _prepare_features(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for model training."""
        # Select numeric features
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column and other non-feature columns
        exclude_columns = [target_column, 'latitude', 'longitude', 'gbif_id', 'inat_id', 'source']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        X = data[feature_columns].fillna(data[feature_columns].median())
        y = data[target_column]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values, feature_columns
        
    def _train_species_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train species classification model."""
        logger.info("Training species classifier")
        
        # Use grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=self.config['random_state'])
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
        
    def _train_suitability_predictor(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train habitat suitability prediction model."""
        logger.info("Training habitat suitability predictor")
        
        # For suitability prediction, we'll use a composite score
        # This would typically be based on known habitat preferences
        suitability_scores = self._calculate_suitability_scores(y_train)
        
        # Use grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=self.config['random_state'])
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X_train, suitability_scores)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
        
    def _calculate_suitability_scores(self, species: np.ndarray) -> np.ndarray:
        """Calculate habitat suitability scores based on species."""
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        # based on known habitat preferences for each species
        
        species_scores = {
            'Tuber melanosporum': 0.9,  # Black truffle - high suitability
            'Tuber magnatum': 0.8,      # White truffle - high suitability
            'Tuber aestivum': 0.7,      # Summer truffle - medium-high suitability
            'Tuber borchii': 0.6,       # Bianchetto truffle - medium suitability
            'Tuber brumale': 0.5,       # Winter truffle - medium suitability
        }
        
        scores = np.array([species_scores.get(species, 0.3) for species in species])
        return scores
        
    def _evaluate_species_classifier(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate species classification model."""
        if self.species_classifier is None:
            return None
            
        y_pred = self.species_classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = self.species_classifier.score(X_test, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.species_classifier, X_test, y_test, cv=5, scoring='accuracy'
        )
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report
        }
        
        logger.info(f"Species classification accuracy: {accuracy:.3f}")
        return results
        
    def _evaluate_suitability_predictor(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate habitat suitability prediction model."""
        if self.suitability_predictor is None:
            return None
            
        # Calculate suitability scores for test data
        suitability_scores = self._calculate_suitability_scores(y_test)
        y_pred = self.suitability_predictor.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(suitability_scores, y_pred)
        mse = mean_squared_error(suitability_scores, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.suitability_predictor, X_test, suitability_scores, cv=5, scoring='r2'
        )
        
        results = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"Habitat suitability R²: {r2:.3f}")
        return results
        
    def _calculate_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Calculate and return feature importance."""
        importance_data = []
        
        if self.species_classifier is not None:
            species_importance = self.species_classifier.feature_importances_
            for name, importance in zip(feature_names, species_importance):
                importance_data.append({
                    'feature': name,
                    'species_importance': importance,
                    'suitability_importance': 0
                })
                
        if self.suitability_predictor is not None:
            suitability_importance = self.suitability_predictor.feature_importances_
            for i, (name, importance) in enumerate(zip(feature_names, suitability_importance)):
                if i < len(importance_data):
                    importance_data[i]['suitability_importance'] = importance
                else:
                    importance_data.append({
                        'feature': name,
                        'species_importance': 0,
                        'suitability_importance': importance
                    })
                    
        df = pd.DataFrame(importance_data)
        df['total_importance'] = df['species_importance'] + df['suitability_importance']
        df = df.sort_values('total_importance', ascending=False)
        
        return df
        
    def predict_species(self, X: np.ndarray) -> np.ndarray:
        """Predict species for given features."""
        if self.species_classifier is None:
            raise ValueError("Species classifier not trained")
            
        return self.species_classifier.predict(X)
        
    def predict_suitability(self, X: np.ndarray) -> np.ndarray:
        """Predict habitat suitability for given features."""
        if self.suitability_predictor is None:
            raise ValueError("Suitability predictor not trained")
            
        return self.suitability_predictor.predict(X)
        
    def predict_proba_species(self, X: np.ndarray) -> np.ndarray:
        """Predict species probabilities for given features."""
        if self.species_classifier is None:
            raise ValueError("Species classifier not trained")
            
        return self.species_classifier.predict_proba(X)
        
    def _save_models(self):
        """Save trained models to disk."""
        if self.species_classifier is not None:
            joblib.dump(self.species_classifier, self.models_dir / 'species_classifier.joblib')
            
        if self.suitability_predictor is not None:
            joblib.dump(self.suitability_predictor, self.models_dir / 'suitability_predictor.joblib')
            
        if hasattr(self, 'scaler'):
            joblib.dump(self.scaler, self.models_dir / 'feature_scaler.joblib')
            
        logger.info(f"Models saved to {self.models_dir}")
        
    def load_models(self):
        """Load trained models from disk."""
        try:
            self.species_classifier = joblib.load(self.models_dir / 'species_classifier.joblib')
            self.suitability_predictor = joblib.load(self.models_dir / 'suitability_predictor.joblib')
            self.scaler = joblib.load(self.models_dir / 'feature_scaler.joblib')
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Error loading models: {e}")
            
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[Path] = None):
        """Plot feature importance."""
        if self.feature_importance_ is None:
            logger.warning("No feature importance data available")
            return
            
        # Get top features
        top_features = self.feature_importance_.head(top_n)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Species importance
        ax1.barh(range(len(top_features)), top_features['species_importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Species Classification Importance')
        ax1.set_title('Top Features for Species Classification')
        
        # Suitability importance
        ax2.barh(range(len(top_features)), top_features['suitability_importance'])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_xlabel('Habitat Suitability Importance')
        ax2.set_title('Top Features for Habitat Suitability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
            
    def generate_habitat_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive habitat analysis report."""
        if data.empty:
            return {}
            
        report = {
            'dataset_summary': {
                'total_records': len(data),
                'species_count': data['species'].nunique(),
                'species_list': data['species'].unique().tolist()
            },
            'geographic_distribution': {
                'latitude_range': [float(data['latitude'].min()), float(data['latitude'].max())],
                'longitude_range': [float(data['longitude'].min()), float(data['longitude'].max())]
            }
        }
        
        # Environmental variable analysis
        env_cols = [col for col in data.columns if any(prefix in col for prefix in ['soil_', 'climate_'])]
        report['environmental_variables'] = {}
        
        for col in env_cols:
            if col in data.columns:
                values = data[col].dropna()
                if not values.empty:
                    report['environmental_variables'][col] = {
                        'count': len(values),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'percentile_25': float(values.quantile(0.25)),
                        'percentile_75': float(values.quantile(0.75))
                    }
                    
        # Species-specific analysis
        report['species_analysis'] = {}
        for species in data['species'].unique():
            species_data = data[data['species'] == species]
            report['species_analysis'][species] = {
                'record_count': len(species_data),
                'environmental_ranges': {}
            }
            
            for col in env_cols:
                if col in species_data.columns:
                    values = species_data[col].dropna()
                    if not values.empty:
                        report['species_analysis'][species]['environmental_ranges'][col] = {
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'mean': float(values.mean())
                        }
                        
        return report