"""
Feature engineering for truffle habitat analysis.
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for truffle habitat data."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from the raw habitat data.
        
        Args:
            data: Raw habitat data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if data.empty:
            return data
            
        logger.info("Engineering features for habitat data")
        
        df = data.copy()
        
        # Add topographic features
        df = self._add_topographic_features(df)
        
        # Add climate-derived features
        df = self._add_climate_features(df)
        
        # Add soil-derived features
        df = self._add_soil_features(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add interaction features
        df = self._add_interaction_features(df)
        
        # Add habitat suitability indicators
        df = self._add_habitat_suitability_features(df)
        
        logger.info(f"Feature engineering complete: {len(df.columns)} total features")
        return df
        
    def _add_topographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add topographic features."""
        # Note: In a real implementation, you would use DEM data to calculate these
        # For now, we'll add placeholder columns that would be calculated from elevation data
        
        # These would be calculated from SRTM or other DEM data
        df['elevation_m'] = np.nan  # Would be calculated from DEM
        df['slope_deg'] = np.nan    # Would be calculated from DEM
        df['aspect_deg'] = np.nan   # Would be calculated from DEM
        df['topographic_wetness_index'] = np.nan  # Would be calculated from DEM
        
        return df
        
    def _add_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add climate-derived features."""
        # Temperature seasonality
        if 'climate_bio4' in df.columns:
            df['temp_seasonality'] = df['climate_bio4']
            
        # Precipitation seasonality
        if 'climate_bio15' in df.columns:
            df['precip_seasonality'] = df['climate_bio15']
            
        # Aridity index (precipitation / potential evapotranspiration)
        if 'annual_precip_mm' in df.columns and 'climate_bio1' in df.columns:
            # Simplified PET calculation using temperature
            pet = 0.0023 * (df['climate_bio1'] + 17.8) * np.sqrt(df['annual_precip_mm'])
            df['aridity_index'] = df['annual_precip_mm'] / (pet + 1e-6)
            
        # Growing degree days (simplified)
        if 'climate_bio1' in df.columns:
            # Assume growing season when mean temp > 5°C
            df['growing_degree_days'] = np.maximum(0, df['climate_bio1'] - 5) * 365
            
        # Frost days (simplified)
        if 'climate_bio6' in df.columns:
            df['frost_days'] = np.maximum(0, -df['climate_bio6']) * 30  # Rough estimate
            
        return df
        
    def _add_soil_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add soil-derived features."""
        # Soil texture classification
        if all(col in df.columns for col in ['soil_sand_pct', 'soil_silt_pct', 'soil_clay_pct']):
            df['soil_texture_class'] = self._classify_soil_texture(
                df['soil_sand_pct'], df['soil_silt_pct'], df['soil_clay_pct']
            )
            
        # Cation exchange capacity per unit clay
        if 'soil_CEC' in df.columns and 'soil_clay_pct' in df.columns:
            df['cec_per_clay'] = df['soil_CEC'] / (df['soil_clay_pct'] + 1e-6)
            
        # Base saturation (simplified)
        if 'soil_CEC' in df.columns and 'soil_CaCO3_pct' in df.columns:
            # Rough estimate of base saturation
            df['base_saturation'] = np.minimum(100, df['soil_CaCO3_pct'] * 2)
            
        # Soil organic matter to nitrogen ratio
        if 'soil_OC_pct' in df.columns and 'soil_N_pct' in df.columns:
            df['cn_ratio'] = df['soil_OC_pct'] / (df['soil_N_pct'] + 1e-6)
            
        # Calcium carbonate availability
        if 'soil_CaCO3_pct' in df.columns:
            df['calcareous_soil'] = (df['soil_CaCO3_pct'] > 5).astype(int)
            
        return df
        
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        # Fruiting season indicators
        if 'month' in df.columns:
            df['fruiting_season'] = df['month'].apply(self._get_fruiting_season)
            df['is_fruiting_month'] = df['month'].apply(self._is_fruiting_month)
            
        # Seasonal temperature and precipitation
        if 'month' in df.columns and 'climate_bio1' in df.columns:
            df['seasonal_temp'] = df.apply(
                lambda row: self._get_seasonal_temp(row['month'], row['climate_bio1']), 
                axis=1
            )
            
        return df
        
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between environmental variables."""
        # pH-temperature interaction
        if 'soil_pH' in df.columns and 'climate_bio1' in df.columns:
            df['ph_temp_interaction'] = df['soil_pH'] * df['climate_bio1']
            
        # Precipitation-calcium carbonate interaction
        if 'annual_precip_mm' in df.columns and 'soil_CaCO3_pct' in df.columns:
            df['precip_caco3_interaction'] = df['annual_precip_mm'] * df['soil_CaCO3_pct']
            
        # Temperature-precipitation ratio
        if 'climate_bio1' in df.columns and 'annual_precip_mm' in df.columns:
            df['temp_precip_ratio'] = df['climate_bio1'] / (df['annual_precip_mm'] + 1e-6)
            
        return df
        
    def _add_habitat_suitability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add habitat suitability indicators based on known truffle preferences."""
        # pH suitability (most truffles prefer 7-8)
        if 'soil_pH' in df.columns:
            df['ph_suitability'] = np.where(
                (df['soil_pH'] >= 7.0) & (df['soil_pH'] <= 8.5),
                1, 0
            )
            
        # Temperature suitability (most truffles prefer 10-15°C annual mean)
        if 'climate_bio1' in df.columns:
            df['temp_suitability'] = np.where(
                (df['climate_bio1'] >= 8) & (df['climate_bio1'] <= 18),
                1, 0
            )
            
        # Precipitation suitability (most truffles prefer 500-1000mm annual)
        if 'annual_precip_mm' in df.columns:
            df['precip_suitability'] = np.where(
                (df['annual_precip_mm'] >= 400) & (df['annual_precip_mm'] <= 1200),
                1, 0
            )
            
        # Calcium carbonate suitability
        if 'soil_CaCO3_pct' in df.columns:
            df['caco3_suitability'] = np.where(
                (df['soil_CaCO3_pct'] >= 5) & (df['soil_CaCO3_pct'] <= 50),
                1, 0
            )
            
        # Overall habitat suitability score
        suitability_cols = ['ph_suitability', 'temp_suitability', 'precip_suitability', 'caco3_suitability']
        available_suitability_cols = [col for col in suitability_cols if col in df.columns]
        
        if available_suitability_cols:
            df['habitat_suitability_score'] = df[available_suitability_cols].sum(axis=1)
            df['habitat_suitability_pct'] = (df['habitat_suitability_score'] / len(available_suitability_cols)) * 100
            
        return df
        
    def _classify_soil_texture(self, sand: pd.Series, silt: pd.Series, clay: pd.Series) -> pd.Series:
        """Classify soil texture based on sand, silt, clay percentages."""
        def classify_texture(s, si, c):
            if pd.isna(s) or pd.isna(si) or pd.isna(c):
                return 'Unknown'
                
            # Normalize to 100%
            total = s + si + c
            if total == 0:
                return 'Unknown'
                
            s_norm = s / total * 100
            si_norm = si / total * 100
            c_norm = c / total * 100
            
            # USDA texture triangle classification
            if c_norm >= 40:
                return 'Clay'
            elif s_norm >= 85:
                return 'Sand'
            elif s_norm >= 70 and c_norm <= 15:
                return 'Loamy Sand'
            elif s_norm >= 43 and s_norm <= 85 and c_norm <= 20:
                return 'Sandy Loam'
            elif s_norm >= 23 and s_norm <= 52 and si_norm >= 28 and si_norm <= 50 and c_norm <= 27:
                return 'Loam'
            elif si_norm >= 50 and c_norm <= 27:
                return 'Silt Loam'
            elif si_norm >= 80:
                return 'Silt'
            elif c_norm >= 27 and c_norm <= 40:
                return 'Clay Loam'
            elif c_norm >= 20 and c_norm <= 35 and si_norm >= 15 and si_norm <= 52:
                return 'Sandy Clay Loam'
            elif c_norm >= 27 and c_norm <= 40 and si_norm >= 40:
                return 'Silty Clay Loam'
            elif c_norm >= 35 and s_norm >= 45:
                return 'Sandy Clay'
            elif c_norm >= 40 and si_norm >= 40:
                return 'Silty Clay'
            else:
                return 'Unknown'
                
        return pd.Series([classify_texture(s, si, c) for s, si, c in zip(sand, silt, clay)])
        
    def _get_fruiting_season(self, month: int) -> str:
        """Get fruiting season based on month."""
        if pd.isna(month):
            return 'Unknown'
            
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Unknown'
            
    def _is_fruiting_month(self, month: int) -> int:
        """Check if month is typically a fruiting month for truffles."""
        if pd.isna(month):
            return 0
            
        # Most truffles fruit in late autumn to early spring
        fruiting_months = [10, 11, 12, 1, 2, 3]
        return 1 if month in fruiting_months else 0
        
    def _get_seasonal_temp(self, month: int, annual_temp: float) -> float:
        """Get seasonal temperature based on month and annual temperature."""
        if pd.isna(month) or pd.isna(annual_temp):
            return np.nan
            
        # Simple seasonal adjustment
        seasonal_adjustments = {
            1: -2, 2: -1, 3: 1, 4: 3, 5: 6, 6: 9,
            7: 12, 8: 11, 9: 8, 10: 4, 11: 1, 12: -1
        }
        
        return annual_temp + seasonal_adjustments.get(month, 0)
        
    def prepare_ml_features(self, data: pd.DataFrame, 
                          target_column: str = 'species') -> tuple:
        """
        Prepare features for machine learning.
        
        Args:
            data: Habitat data DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Select numeric features
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column and other non-feature columns
        exclude_columns = [target_column, 'latitude', 'longitude', 'gbif_id', 'inat_id']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        X = data[feature_columns].fillna(data[feature_columns].median())
        y = data[target_column] if target_column in data.columns else None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['feature_scaler'] = scaler
        
        return X_scaled, y, feature_columns