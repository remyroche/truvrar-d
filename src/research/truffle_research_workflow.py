"""
Truffle Research Workflow - Complete 6-Phase Implementation

This module implements a comprehensive research workflow for truffle ecology analysis,
from data collection through model-agnostic inference, with full reproducibility
and automation capabilities.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import yaml
from dataclasses import dataclass, asdict
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..data_collectors import UnifiedDataCollector, load_collector_config
from ..data_collectors.harmonization import DataHarmonizer
from ..data_collectors.caching import DataCache

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for truffle research workflow."""
    
    # Research parameters
    truffle_species: List[str] = None
    tree_genera: List[str] = None
    study_regions: List[Tuple[float, float, float, float]] = None  # (min_lat, max_lat, min_lon, max_lon)
    
    # Data collection parameters
    occurrence_limit: int = 10000
    soil_variables: List[str] = None
    climate_variables: List[str] = None
    metagenomics_search_terms: List[str] = None
    
    # Analysis parameters
    test_size: float = 0.2
    random_state: int = 42
    min_samples_per_species: int = 10
    
    # Output parameters
    output_dir: Path = None
    cache_ttl_hours: int = 24
    enable_caching: bool = True
    enable_harmonization: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.truffle_species is None:
            self.truffle_species = [
                'Tuber melanosporum', 'Tuber magnatum', 'Tuber aestivum',
                'Tuber borchii', 'Tuber brumale', 'Tuber indicum'
            ]
        
        if self.tree_genera is None:
            self.tree_genera = [
                'Quercus', 'Corylus', 'Tilia', 'Populus', 'Salix',
                'Pinus', 'Carpinus', 'Fagus', 'Castanea'
            ]
        
        if self.study_regions is None:
            # Default: Europe and North America
            self.study_regions = [
                (35.0, 70.0, -15.0, 40.0),  # Europe
                (25.0, 50.0, -125.0, -65.0)  # North America
            ]
        
        if self.soil_variables is None:
            self.soil_variables = [
                'phh2o', 'cac03', 'soc', 'nitrogen', 'phosporus',
                'sand', 'silt', 'clay', 'bdod', 'cec', 'cfvo'
            ]
        
        if self.climate_variables is None:
            self.climate_variables = [
                'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6',
                'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12',
                'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19'
            ]
        
        if self.metagenomics_search_terms is None:
            self.metagenomics_search_terms = ['Tuber', 'truffle', 'ectomycorrhizal']
        
        if self.output_dir is None:
            self.output_dir = Path("truffle_research_output")


class TruffleResearchWorkflow:
    """Complete truffle research workflow implementation."""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collection system
        collector_config = load_collector_config()
        self.collector = UnifiedDataCollector(
            config=collector_config.get_all_configs(),
            data_dir=self.output_dir / "raw_data",
            enable_caching=config.enable_caching,
            enable_harmonization=config.enable_harmonization
        )
        
        # Initialize harmonizer
        self.harmonizer = DataHarmonizer(collector_config.get_all_configs())
        
        # Research state tracking
        self.research_state = {
            'phase': 0,
            'last_run': None,
            'data_sources_used': [],
            'total_records': 0,
            'quality_scores': {}
        }
        
        # Results storage
        self.results = {
            'occurrence_data': {},
            'environmental_data': {},
            'harmonized_data': None,
            'analysis_results': {},
            'model_performance': {}
        }
    
    def run_complete_workflow(self) -> Dict[str, Any]:
        """Run the complete 6-phase research workflow."""
        logger.info("Starting complete truffle research workflow")
        
        try:
            # Phase 1: Data Collection
            self._phase_1_data_collection()
            
            # Phase 2: Data Harmonization
            self._phase_2_data_harmonization()
            
            # Phase 3: Exploratory Data Analysis
            self._phase_3_exploratory_analysis()
            
            # Phase 4: Feature Engineering
            self._phase_4_feature_engineering()
            
            # Phase 5: Model Training & Evaluation
            self._phase_5_model_training()
            
            # Phase 6: Inference & Prediction
            self._phase_6_inference_prediction()
            
            # Save research state and results
            self._save_research_state()
            
            logger.info("Complete research workflow finished successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            self._save_research_state()
            raise
    
    def _phase_1_data_collection(self):
        """Phase 1: Collect data from all sources."""
        logger.info("Phase 1: Data Collection")
        
        # 1.1 Collect truffle occurrence data
        logger.info("Collecting truffle occurrence data...")
        truffle_data = self._collect_occurrence_data(
            species=self.config.truffle_species,
            data_type='truffle'
        )
        self.results['occurrence_data']['truffles'] = truffle_data
        
        # 1.2 Collect tree occurrence data
        logger.info("Collecting tree occurrence data...")
        tree_data = self._collect_occurrence_data(
            species=self.config.tree_genera,
            data_type='tree'
        )
        self.results['occurrence_data']['trees'] = tree_data
        
        # 1.3 Collect environmental data for all coordinates
        logger.info("Collecting environmental data...")
        environmental_data = self._collect_environmental_data()
        self.results['environmental_data'] = environmental_data
        
        # 1.4 Collect metagenomics data
        logger.info("Collecting metagenomics data...")
        metagenomics_data = self._collect_metagenomics_data()
        self.results['occurrence_data']['metagenomics'] = metagenomics_data
        
        self.research_state['phase'] = 1
        self.research_state['last_run'] = datetime.now().isoformat()
    
    def _collect_occurrence_data(self, species: List[str], data_type: str) -> Dict[str, Any]:
        """Collect occurrence data from GBIF and iNaturalist."""
        all_data = {}
        
        for source in ['gbif', 'inaturalist']:
            try:
                logger.info(f"Collecting {data_type} data from {source}")
                
                result = self.collector.collect(
                    source=source,
                    species=species,
                    limit=self.config.occurrence_limit,
                    has_coordinate=True
                )
                
                if isinstance(result, dict) and 'records_df' in result:
                    all_data[source] = result
                    self.research_state['data_sources_used'].append(f"{source}_{data_type}")
                    self.research_state['total_records'] += len(result['records_df'])
                else:
                    all_data[source] = {'records_df': result, 'metadata_df': pd.DataFrame()}
                    
            except Exception as e:
                logger.error(f"Error collecting {data_type} data from {source}: {e}")
                all_data[source] = {'records_df': pd.DataFrame(), 'metadata_df': pd.DataFrame()}
        
        return all_data
    
    def _collect_environmental_data(self) -> Dict[str, Any]:
        """Collect environmental data for all occurrence coordinates."""
        # Get all unique coordinates from occurrence data
        all_coords = self._extract_all_coordinates()
        
        if not all_coords:
            logger.warning("No coordinates found for environmental data collection")
            return {}
        
        environmental_data = {}
        
        # Collect soil data
        try:
            logger.info("Collecting soil data from SoilGrids...")
            soil_result = self.collector.collect(
                source='soilgrids',
                coordinates=all_coords,
                variables=self.config.soil_variables
            )
            environmental_data['soil'] = soil_result
        except Exception as e:
            logger.error(f"Error collecting soil data: {e}")
            environmental_data['soil'] = {'records_df': pd.DataFrame()}
        
        # Collect climate data
        try:
            logger.info("Collecting climate data from WorldClim...")
            climate_result = self.collector.collect(
                source='worldclim',
                coordinates=all_coords,
                variables=self.config.climate_variables
            )
            environmental_data['climate'] = climate_result
        except Exception as e:
            logger.error(f"Error collecting climate data: {e}")
            environmental_data['climate'] = {'records_df': pd.DataFrame()}
        
        # Collect geological data
        try:
            logger.info("Collecting geological data from GLiM...")
            geology_result = self.collector.collect(
                source='glim',
                coordinates=all_coords
            )
            environmental_data['geology'] = geology_result
        except Exception as e:
            logger.error(f"Error collecting geological data: {e}")
            environmental_data['geology'] = {'records_df': pd.DataFrame()}
        
        return environmental_data
    
    def _collect_metagenomics_data(self) -> Dict[str, Any]:
        """Collect metagenomics data from EBI."""
        metagenomics_data = {}
        
        for search_term in self.config.metagenomics_search_terms:
            try:
                logger.info(f"Collecting metagenomics data for '{search_term}'...")
                result = self.collector.collect(
                    source='ebi_metagenomics',
                    search_term=search_term,
                    limit=1000,
                    include_samples=True,
                    include_abundance=True
                )
                metagenomics_data[search_term] = result
            except Exception as e:
                logger.error(f"Error collecting metagenomics data for '{search_term}': {e}")
                metagenomics_data[search_term] = {'records_df': pd.DataFrame()}
        
        return metagenomics_data
    
    def _extract_all_coordinates(self) -> List[Tuple[float, float]]:
        """Extract all unique coordinates from occurrence data."""
        all_coords = set()
        
        # Extract from truffle data
        for source_data in self.results['occurrence_data'].get('truffles', {}).values():
            if isinstance(source_data, dict) and 'records_df' in source_data:
                df = source_data['records_df']
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    coords = df[['latitude', 'longitude']].dropna()
                    all_coords.update([(row['latitude'], row['longitude']) 
                                     for _, row in coords.iterrows()])
        
        # Extract from tree data
        for source_data in self.results['occurrence_data'].get('trees', {}).values():
            if isinstance(source_data, dict) and 'records_df' in source_data:
                df = source_data['records_df']
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    coords = df[['latitude', 'longitude']].dropna()
                    all_coords.update([(row['latitude'], row['longitude']) 
                                     for _, row in coords.iterrows()])
        
        return list(all_coords)
    
    def _phase_2_data_harmonization(self):
        """Phase 2: Harmonize and integrate all collected data."""
        logger.info("Phase 2: Data Harmonization")
        
        # Prepare data for harmonization
        data_dict = {}
        
        # Add occurrence data
        for data_type in ['truffles', 'trees']:
            for source, data in self.results['occurrence_data'].get(data_type, {}).items():
                if isinstance(data, dict) and 'records_df' in data and not data['records_df'].empty:
                    data_dict[f"{source}_{data_type}"] = data['records_df']
        
        # Add environmental data
        for env_type, data in self.results['environmental_data'].items():
            if isinstance(data, dict) and 'records_df' in data and not data['records_df'].empty:
                data_dict[f"env_{env_type}"] = data['records_df']
        
        # Harmonize all data
        if data_dict:
            harmonized_result = self.harmonizer.harmonize_data(data_dict)
            self.results['harmonized_data'] = harmonized_result
            
            # Update research state
            self.research_state['quality_scores'] = harmonized_result.get('quality_scores', {})
            
            logger.info(f"Harmonized {len(data_dict)} data sources")
            logger.info(f"Total harmonized records: {len(harmonized_result.get('records_df', pd.DataFrame()))}")
        else:
            logger.warning("No data available for harmonization")
            self.results['harmonized_data'] = {
                'records_df': pd.DataFrame(),
                'metadata_df': pd.DataFrame(),
                'summary_stats': {}
            }
        
        self.research_state['phase'] = 2
    
    def _phase_3_exploratory_analysis(self):
        """Phase 3: Exploratory data analysis."""
        logger.info("Phase 3: Exploratory Data Analysis")
        
        if self.results['harmonized_data'] is None:
            logger.warning("No harmonized data available for analysis")
            return
        
        df = self.results['harmonized_data']['records_df']
        
        if df.empty:
            logger.warning("Empty dataset for exploratory analysis")
            return
        
        # 3.1 Data quality assessment
        quality_analysis = self._analyze_data_quality(df)
        self.results['analysis_results']['quality_analysis'] = quality_analysis
        
        # 3.2 Geographic distribution analysis
        geographic_analysis = self._analyze_geographic_distribution(df)
        self.results['analysis_results']['geographic_analysis'] = geographic_analysis
        
        # 3.3 Species composition analysis
        species_analysis = self._analyze_species_composition(df)
        self.results['analysis_results']['species_analysis'] = species_analysis
        
        # 3.4 Environmental correlation analysis
        environmental_analysis = self._analyze_environmental_correlations(df)
        self.results['analysis_results']['environmental_analysis'] = environmental_analysis
        
        # 3.5 Generate exploratory plots
        self._generate_exploratory_plots(df)
        
        self.research_state['phase'] = 3
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        quality_metrics = {
            'total_records': len(df),
            'coordinate_coverage': (df['latitude'].notna() & df['longitude'].notna()).sum() / len(df) * 100,
            'temporal_coverage': df.select_dtypes(include=['datetime64']).notna().any(axis=1).sum() / len(df) * 100,
            'missing_data_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        }
        
        # Coordinate quality analysis
        if 'coord_quality' in df.columns:
            quality_metrics['coordinate_quality_distribution'] = df['coord_quality'].value_counts().to_dict()
        
        # Source distribution
        if 'data_source' in df.columns:
            quality_metrics['source_distribution'] = df['data_source'].value_counts().to_dict()
        
        return quality_metrics
    
    def _analyze_geographic_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic distribution of data."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return {}
        
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        
        if valid_coords.empty:
            return {}
        
        geographic_analysis = {
            'lat_range': [valid_coords['latitude'].min(), valid_coords['latitude'].max()],
            'lon_range': [valid_coords['longitude'].min(), valid_coords['longitude'].max()],
            'coordinate_count': len(valid_coords),
            'unique_locations': len(valid_coords[['latitude', 'longitude']].drop_duplicates())
        }
        
        # Regional distribution
        regional_counts = {}
        for min_lat, max_lat, min_lon, max_lon in self.config.study_regions:
            region_mask = (
                (valid_coords['latitude'] >= min_lat) & (valid_coords['latitude'] <= max_lat) &
                (valid_coords['longitude'] >= min_lon) & (valid_coords['longitude'] <= max_lon)
            )
            regional_counts[f"region_{min_lat}_{max_lat}_{min_lon}_{max_lon}"] = region_mask.sum()
        
        geographic_analysis['regional_distribution'] = regional_counts
        
        return geographic_analysis
    
    def _analyze_species_composition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze species composition and diversity."""
        species_analysis = {}
        
        if 'species_standardized' in df.columns:
            species_counts = df['species_standardized'].value_counts()
            species_analysis['species_counts'] = species_counts.to_dict()
            species_analysis['unique_species'] = len(species_counts)
            species_analysis['most_common_species'] = species_counts.index[0] if not species_counts.empty else None
        
        # Diversity metrics
        if 'shannon_diversity' in df.columns:
            species_analysis['shannon_diversity_stats'] = {
                'mean': df['shannon_diversity'].mean(),
                'std': df['shannon_diversity'].std(),
                'min': df['shannon_diversity'].min(),
                'max': df['shannon_diversity'].max()
            }
        
        return species_analysis
    
    def _analyze_environmental_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental variable correlations."""
        environmental_analysis = {}
        
        # Select numeric environmental columns
        env_cols = [col for col in df.columns if any(prefix in col.lower() for prefix in 
                   ['bio', 'soil_', 'temp', 'precip', 'ph', 'soc', 'sand', 'clay'])]
        
        if env_cols:
            env_data = df[env_cols].select_dtypes(include=[np.number])
            
            if not env_data.empty:
                # Calculate correlations
                correlation_matrix = env_data.corr()
                environmental_analysis['correlation_matrix'] = correlation_matrix.to_dict()
                
                # Environmental richness analysis
                if 'env_richness_index' in df.columns:
                    richness_stats = df['env_richness_index'].describe()
                    environmental_analysis['environmental_richness'] = richness_stats.to_dict()
        
        return environmental_analysis
    
    def _generate_exploratory_plots(self, df: pd.DataFrame):
        """Generate exploratory data analysis plots."""
        plots_dir = self.output_dir / "plots" / "exploratory"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Geographic distribution plot
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = df.dropna(subset=['latitude', 'longitude'])
            if not valid_coords.empty:
                plt.figure(figsize=(12, 8))
                plt.scatter(valid_coords['longitude'], valid_coords['latitude'], 
                           alpha=0.6, s=20)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title('Geographic Distribution of Data Points')
                plt.grid(True, alpha=0.3)
                plt.savefig(plots_dir / "geographic_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Species composition plot
        if 'species_standardized' in df.columns:
            species_counts = df['species_standardized'].value_counts().head(10)
            if not species_counts.empty:
                plt.figure(figsize=(12, 6))
                species_counts.plot(kind='bar')
                plt.title('Top 10 Species by Occurrence Count')
                plt.xlabel('Species')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(plots_dir / "species_composition.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Data quality plot
        if 'coord_quality' in df.columns:
            quality_counts = df['coord_quality'].value_counts()
            plt.figure(figsize=(10, 6))
            quality_counts.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Coordinate Quality Distribution')
            plt.ylabel('')
            plt.savefig(plots_dir / "coordinate_quality.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Exploratory plots saved to {plots_dir}")
    
    def _phase_4_feature_engineering(self):
        """Phase 4: Feature engineering and data preparation."""
        logger.info("Phase 4: Feature Engineering")
        
        if self.results['harmonized_data'] is None:
            logger.warning("No harmonized data available for feature engineering")
            return
        
        df = self.results['harmonized_data']['records_df']
        
        if df.empty:
            logger.warning("Empty dataset for feature engineering")
            return
        
        # 4.1 Create target variables
        target_variables = self._create_target_variables(df)
        self.results['analysis_results']['target_variables'] = target_variables
        
        # 4.2 Engineer environmental features
        environmental_features = self._engineer_environmental_features(df)
        self.results['analysis_results']['environmental_features'] = environmental_features
        
        # 4.3 Create interaction features
        interaction_features = self._create_interaction_features(df)
        self.results['analysis_results']['interaction_features'] = interaction_features
        
        # 4.4 Prepare final feature matrix
        feature_matrix = self._prepare_feature_matrix(df)
        self.results['analysis_results']['feature_matrix'] = feature_matrix
        
        self.research_state['phase'] = 4
    
    def _create_target_variables(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create target variables for modeling."""
        targets = {}
        
        # Binary truffle presence (1 if any truffle species, 0 otherwise)
        if 'species_standardized' in df.columns:
            truffle_species = [sp.lower() for sp in self.config.truffle_species]
            truffle_mask = df['species_standardized'].str.lower().str.contains('|'.join(truffle_species), na=False)
            targets['truffle_presence'] = truffle_mask.astype(int)
        
        # Specific species presence
        for species in self.config.truffle_species:
            species_mask = df['species_standardized'].str.lower().str.contains(species.lower(), na=False)
            targets[f'presence_{species.replace(" ", "_").lower()}'] = species_mask.astype(int)
        
        # Environmental richness (if available)
        if 'env_richness_index' in df.columns:
            targets['environmental_richness'] = df['env_richness_index']
        
        return targets
    
    def _engineer_environmental_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Engineer environmental features."""
        features = {}
        
        # Climate features
        climate_cols = [col for col in df.columns if col.startswith('bio')]
        if climate_cols:
            climate_data = df[climate_cols].select_dtypes(include=[np.number])
            if not climate_data.empty:
                features['mean_annual_temp'] = climate_data.get('bio1', pd.Series())
                features['annual_precipitation'] = climate_data.get('bio12', pd.Series())
                features['temp_seasonality'] = climate_data.get('bio4', pd.Series())
                features['precip_seasonality'] = climate_data.get('bio15', pd.Series())
        
        # Soil features
        soil_cols = [col for col in df.columns if col.startswith('soil_')]
        if soil_cols:
            soil_data = df[soil_cols].select_dtypes(include=[np.number])
            if not soil_data.empty:
                features['soil_ph'] = soil_data.get('soil_phh2o', pd.Series())
                features['soil_organic_carbon'] = soil_data.get('soil_soc', pd.Series())
                features['soil_texture_ratio'] = (
                    soil_data.get('soil_sand', pd.Series()) / 
                    (soil_data.get('soil_clay', pd.Series()) + 1e-6)
                )
        
        return features
    
    def _create_interaction_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create interaction features between environmental variables."""
        interactions = {}
        
        # Temperature-precipitation interactions
        if 'bio1' in df.columns and 'bio12' in df.columns:
            interactions['temp_precip_interaction'] = df['bio1'] * df['bio12']
        
        # Soil-climate interactions
        if 'soil_phh2o' in df.columns and 'bio1' in df.columns:
            interactions['ph_temp_interaction'] = df['soil_phh2o'] * df['bio1']
        
        # Elevation effects (if available)
        if 'elevation' in df.columns:
            if 'bio1' in df.columns:
                interactions['elevation_temp_interaction'] = df['elevation'] * df['bio1']
        
        return interactions
    
    def _prepare_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final feature matrix for modeling."""
        # Combine all features
        feature_cols = []
        
        # Add environmental features
        env_cols = [col for col in df.columns if any(prefix in col.lower() for prefix in 
                   ['bio', 'soil_', 'temp', 'precip', 'ph', 'soc', 'sand', 'clay'])]
        feature_cols.extend(env_cols)
        
        # Add derived features
        if 'env_richness_index' in df.columns:
            feature_cols.append('env_richness_index')
        
        if 'shannon_diversity' in df.columns:
            feature_cols.append('shannon_diversity')
        
        # Add coordinate features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            feature_cols.extend(['latitude', 'longitude'])
        
        # Create feature matrix
        feature_matrix = df[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        return feature_matrix
    
    def _phase_5_model_training(self):
        """Phase 5: Model training and evaluation."""
        logger.info("Phase 5: Model Training & Evaluation")
        
        if 'feature_matrix' not in self.results['analysis_results']:
            logger.warning("No feature matrix available for model training")
            return
        
        feature_matrix = self.results['analysis_results']['feature_matrix']
        target_variables = self.results['analysis_results']['target_variables']
        
        if feature_matrix.empty or not target_variables:
            logger.warning("Insufficient data for model training")
            return
        
        # Train models for each target variable
        model_results = {}
        
        for target_name, target_values in target_variables.items():
            if target_values.empty:
                continue
            
            logger.info(f"Training model for {target_name}")
            
            # Prepare data
            X = feature_matrix
            y = target_values
            
            # Remove rows with missing targets
            valid_mask = y.notna()
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < self.config.min_samples_per_species:
                logger.warning(f"Insufficient samples for {target_name}: {len(X_clean)}")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state,
                stratify=y_clean if y_clean.nunique() > 1 else None
            )
            
            # Train model
            if target_name == 'environmental_richness':
                # Regression model
                model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_results[target_name] = {
                    'model_type': 'regression',
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'feature_importance': dict(zip(X.columns, model.feature_importances_))
                }
            else:
                # Classification model
                model = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Evaluate
                report = classification_report(y_test, y_pred, output_dict=True)
                
                model_results[target_name] = {
                    'model_type': 'classification',
                    'model': model,
                    'classification_report': report,
                    'feature_importance': dict(zip(X.columns, model.feature_importances_))
                }
            
            logger.info(f"Model for {target_name} trained successfully")
        
        self.results['model_performance'] = model_results
        self.research_state['phase'] = 5
    
    def _phase_6_inference_prediction(self):
        """Phase 6: Inference and prediction."""
        logger.info("Phase 6: Inference & Prediction")
        
        if not self.results['model_performance']:
            logger.warning("No trained models available for inference")
            return
        
        # Generate predictions for the entire dataset
        feature_matrix = self.results['analysis_results']['feature_matrix']
        predictions = {}
        
        for target_name, model_info in self.results['model_performance'].items():
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(feature_matrix)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(feature_matrix)
            
            predictions[target_name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'model_type': model_info['model_type']
            }
        
        self.results['analysis_results']['predictions'] = predictions
        
        # Generate habitat suitability maps
        self._generate_habitat_suitability_maps()
        
        # Save models for future use
        self._save_trained_models()
        
        self.research_state['phase'] = 6
    
    def _generate_habitat_suitability_maps(self):
        """Generate habitat suitability maps."""
        logger.info("Generating habitat suitability maps...")
        
        # This would typically involve:
        # 1. Creating a grid of coordinates across the study area
        # 2. Extracting environmental features for each grid point
        # 3. Making predictions using trained models
        # 4. Creating visualization maps
        
        # For now, we'll create a simple summary
        suitability_summary = {
            'high_suitability_points': 0,
            'medium_suitability_points': 0,
            'low_suitability_points': 0
        }
        
        if 'truffle_presence' in self.results['analysis_results']['predictions']:
            preds = self.results['analysis_results']['predictions']['truffle_presence']['predictions']
            suitability_summary['high_suitability_points'] = (preds == 1).sum()
            suitability_summary['low_suitability_points'] = (preds == 0).sum()
        
        self.results['analysis_results']['habitat_suitability'] = suitability_summary
    
    def _save_trained_models(self):
        """Save trained models for future use."""
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for target_name, model_info in self.results['model_performance'].items():
            model_path = models_dir / f"model_{target_name.replace(' ', '_')}.joblib"
            joblib.dump(model_info['model'], model_path)
            logger.info(f"Saved model for {target_name} to {model_path}")
    
    def _save_research_state(self):
        """Save research state and results."""
        # Save research state
        state_path = self.output_dir / "research_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.research_state, f, indent=2, default=str)
        
        # Save results
        results_path = self.output_dir / "research_results.json"
        # Convert DataFrames to dict for JSON serialization
        serializable_results = self._make_results_serializable(self.results)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save configuration
        config_path = self.output_dir / "research_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        
        logger.info(f"Research state and results saved to {self.output_dir}")
    
    def _make_results_serializable(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        serializable = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                serializable[key] = self._make_results_serializable(value)
            elif isinstance(value, pd.DataFrame):
                serializable[key] = {
                    'data': value.to_dict('records'),
                    'columns': list(value.columns),
                    'shape': value.shape
                }
            elif isinstance(value, pd.Series):
                serializable[key] = {
                    'data': value.to_dict(),
                    'dtype': str(value.dtype)
                }
            else:
                serializable[key] = value
        
        return serializable


def create_research_config(
    truffle_species: List[str] = None,
    tree_genera: List[str] = None,
    study_regions: List[Tuple[float, float, float, float]] = None,
    output_dir: str = "truffle_research_output"
) -> ResearchConfig:
    """Create a research configuration with custom parameters."""
    return ResearchConfig(
        truffle_species=truffle_species,
        tree_genera=tree_genera,
        study_regions=study_regions,
        output_dir=Path(output_dir)
    )


def run_truffle_research(
    config: ResearchConfig = None,
    truffle_species: List[str] = None,
    output_dir: str = "truffle_research_output"
) -> Dict[str, Any]:
    """Run the complete truffle research workflow."""
    if config is None:
        config = create_research_config(
            truffle_species=truffle_species,
            output_dir=output_dir
        )
    
    workflow = TruffleResearchWorkflow(config)
    return workflow.run_complete_workflow()


if __name__ == "__main__":
    # Example usage
    config = create_research_config(
        truffle_species=['Tuber melanosporum', 'Tuber magnatum'],
        output_dir="example_truffle_research"
    )
    
    results = run_truffle_research(config)
    print("Research workflow completed successfully!")
    print(f"Results saved to: {config.output_dir}")