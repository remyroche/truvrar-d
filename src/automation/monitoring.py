"""
Data quality monitoring and alerting system.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for data quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityAlert:
    """Data quality alert."""
    timestamp: datetime
    level: AlertLevel
    message: str
    metric: str
    value: Any
    threshold: Any
    details: Optional[Dict[str, Any]] = None


class DataQualityMonitor:
    """Monitor data quality and generate alerts."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Quality thresholds
        self.thresholds = {
            'min_records_per_species': 5,
            'max_coordinate_uncertainty': 10000,  # meters
            'min_data_completeness': 0.5,  # 50%
            'max_duplicate_rate': 0.1,  # 10%
            'min_geographic_spread': 0.1,  # 10% of expected range
            'max_outlier_rate': 0.05,  # 5%
        }
        
        # Alert history
        self.alerts = []
        self.alert_history_path = self.output_dir / "alert_history.json"
        self._load_alert_history()
        
    def monitor_data_quality(self, data: pd.DataFrame, 
                           data_source: str = "unknown") -> List[DataQualityAlert]:
        """Monitor data quality and generate alerts."""
        logger.info(f"Monitoring data quality for {len(data)} records from {data_source}")
        
        current_alerts = []
        
        # Check basic data integrity
        current_alerts.extend(self._check_data_integrity(data, data_source))
        
        # Check species distribution
        current_alerts.extend(self._check_species_distribution(data, data_source))
        
        # Check coordinate quality
        current_alerts.extend(self._check_coordinate_quality(data, data_source))
        
        # Check data completeness
        current_alerts.extend(self._check_data_completeness(data, data_source))
        
        # Check for duplicates
        current_alerts.extend(self._check_duplicates(data, data_source))
        
        # Check geographic distribution
        current_alerts.extend(self._check_geographic_distribution(data, data_source))
        
        # Check for outliers
        current_alerts.extend(self._check_outliers(data, data_source))
        
        # Check temporal distribution
        current_alerts.extend(self._check_temporal_distribution(data, data_source))
        
        # Store alerts
        self.alerts.extend(current_alerts)
        self._save_alert_history()
        
        # Log summary
        alert_counts = {}
        for alert in current_alerts:
            alert_counts[alert.level.value] = alert_counts.get(alert.level.value, 0) + 1
            
        logger.info(f"Generated {len(current_alerts)} alerts: {alert_counts}")
        
        return current_alerts
        
    def _check_data_integrity(self, data: pd.DataFrame, 
                            data_source: str) -> List[DataQualityAlert]:
        """Check basic data integrity."""
        alerts = []
        
        # Check for empty dataset
        if data.empty:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                message="Empty dataset received",
                metric="record_count",
                value=0,
                threshold=1,
                details={"source": data_source}
            ))
            return alerts
            
        # Check for required columns
        required_columns = ['species', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.ERROR,
                message=f"Missing required columns: {missing_columns}",
                metric="missing_columns",
                value=missing_columns,
                threshold=[],
                details={"source": data_source}
            ))
            
        return alerts
        
    def _check_species_distribution(self, data: pd.DataFrame, 
                                  data_source: str) -> List[DataQualityAlert]:
        """Check species distribution quality."""
        alerts = []
        
        species_counts = data['species'].value_counts()
        
        # Check minimum records per species
        min_records = self.thresholds['min_records_per_species']
        low_count_species = species_counts[species_counts < min_records]
        
        if len(low_count_species) > 0:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"Species with low record counts: {low_count_species.to_dict()}",
                metric="min_records_per_species",
                value=low_count_species.to_dict(),
                threshold=min_records,
                details={"source": data_source}
            ))
            
        # Check for single species dominance
        total_records = len(data)
        max_species_count = species_counts.max()
        dominance_ratio = max_species_count / total_records
        
        if dominance_ratio > 0.8:  # 80% threshold
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"Single species dominance: {dominance_ratio:.2%}",
                metric="species_dominance",
                value=dominance_ratio,
                threshold=0.8,
                details={"source": data_source, "dominant_species": species_counts.idxmax()}
            ))
            
        return alerts
        
    def _check_coordinate_quality(self, data: pd.DataFrame, 
                                data_source: str) -> List[DataQualityAlert]:
        """Check coordinate quality."""
        alerts = []
        
        # Check coordinate ranges
        lat_range = data['latitude'].max() - data['latitude'].min()
        lon_range = data['longitude'].max() - data['longitude'].min()
        
        if lat_range < 0.1:  # Less than 0.1 degrees
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"Limited latitude range: {lat_range:.4f} degrees",
                metric="latitude_range",
                value=lat_range,
                threshold=0.1,
                details={"source": data_source}
            ))
            
        if lon_range < 0.1:  # Less than 0.1 degrees
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"Limited longitude range: {lon_range:.4f} degrees",
                metric="longitude_range",
                value=lon_range,
                threshold=0.1,
                details={"source": data_source}
            ))
            
        # Check coordinate uncertainty if available
        if 'coordinate_uncertainty' in data.columns:
            high_uncertainty = data[data['coordinate_uncertainty'] > self.thresholds['max_coordinate_uncertainty']]
            
            if len(high_uncertainty) > 0:
                uncertainty_rate = len(high_uncertainty) / len(data)
                alerts.append(DataQualityAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    message=f"High coordinate uncertainty: {uncertainty_rate:.2%} of records",
                    metric="coordinate_uncertainty",
                    value=uncertainty_rate,
                    threshold=0.1,  # 10% threshold
                    details={"source": data_source, "high_uncertainty_count": len(high_uncertainty)}
                ))
                
        return alerts
        
    def _check_data_completeness(self, data: pd.DataFrame, 
                               data_source: str) -> List[DataQualityAlert]:
        """Check data completeness."""
        alerts = []
        
        # Calculate completeness for each column
        completeness = data.notna().mean()
        low_completeness = completeness[completeness < self.thresholds['min_data_completeness']]
        
        if len(low_completeness) > 0:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"Low data completeness: {low_completeness.to_dict()}",
                metric="data_completeness",
                value=low_completeness.to_dict(),
                threshold=self.thresholds['min_data_completeness'],
                details={"source": data_source}
            ))
            
        return alerts
        
    def _check_duplicates(self, data: pd.DataFrame, 
                         data_source: str) -> List[DataQualityAlert]:
        """Check for duplicate records."""
        alerts = []
        
        # Check for exact duplicates
        exact_duplicates = data.duplicated().sum()
        duplicate_rate = exact_duplicates / len(data)
        
        if duplicate_rate > self.thresholds['max_duplicate_rate']:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"High duplicate rate: {duplicate_rate:.2%}",
                metric="duplicate_rate",
                value=duplicate_rate,
                threshold=self.thresholds['max_duplicate_rate'],
                details={"source": data_source, "duplicate_count": exact_duplicates}
            ))
            
        # Check for coordinate duplicates
        coord_duplicates = data.duplicated(subset=['latitude', 'longitude']).sum()
        coord_duplicate_rate = coord_duplicates / len(data)
        
        if coord_duplicate_rate > 0.2:  # 20% threshold
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"High coordinate duplicate rate: {coord_duplicate_rate:.2%}",
                metric="coordinate_duplicate_rate",
                value=coord_duplicate_rate,
                threshold=0.2,
                details={"source": data_source, "coordinate_duplicate_count": coord_duplicates}
            ))
            
        return alerts
        
    def _check_geographic_distribution(self, data: pd.DataFrame, 
                                     data_source: str) -> List[DataQualityAlert]:
        """Check geographic distribution."""
        alerts = []
        
        # Check for clustering (simplified)
        lat_std = data['latitude'].std()
        lon_std = data['longitude'].std()
        
        # Expected standard deviation for global distribution is around 30-40 degrees
        expected_std = 30
        lat_std_ratio = lat_std / expected_std
        lon_std_ratio = lon_std / expected_std
        
        if lat_std_ratio < self.thresholds['min_geographic_spread']:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.INFO,
                message=f"Limited latitude spread: {lat_std_ratio:.2%} of expected",
                metric="latitude_spread",
                value=lat_std_ratio,
                threshold=self.thresholds['min_geographic_spread'],
                details={"source": data_source, "latitude_std": lat_std}
            ))
            
        if lon_std_ratio < self.thresholds['min_geographic_spread']:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.INFO,
                message=f"Limited longitude spread: {lon_std_ratio:.2%} of expected",
                metric="longitude_spread",
                value=lon_std_ratio,
                threshold=self.thresholds['min_geographic_spread'],
                details={"source": data_source, "longitude_std": lon_std}
            ))
            
        return alerts
        
    def _check_outliers(self, data: pd.DataFrame, 
                       data_source: str) -> List[DataQualityAlert]:
        """Check for outliers in numeric columns."""
        alerts = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_columns = []
        
        for col in numeric_columns:
            if col in ['latitude', 'longitude']:  # Skip coordinates
                continue
                
            values = data[col].dropna()
            if len(values) < 10:  # Need at least 10 values
                continue
                
            # Use IQR method for outlier detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            outlier_rate = len(outliers) / len(values)
            
            if outlier_rate > self.thresholds['max_outlier_rate']:
                outlier_columns.append({
                    'column': col,
                    'outlier_rate': outlier_rate,
                    'outlier_count': len(outliers)
                })
                
        if outlier_columns:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"High outlier rates detected in columns: {[col['column'] for col in outlier_columns]}",
                metric="outlier_rate",
                value=outlier_columns,
                threshold=self.thresholds['max_outlier_rate'],
                details={"source": data_source}
            ))
            
        return alerts
        
    def _check_temporal_distribution(self, data: pd.DataFrame, 
                                   data_source: str) -> List[DataQualityAlert]:
        """Check temporal distribution if date information is available."""
        alerts = []
        
        if 'year' not in data.columns:
            return alerts
            
        # Check year distribution
        year_counts = data['year'].value_counts().sort_index()
        current_year = datetime.now().year
        
        # Check for very old data
        old_data = data[data['year'] < current_year - 20]
        if len(old_data) > len(data) * 0.5:  # More than 50% older than 20 years
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.INFO,
                message=f"High proportion of old data: {len(old_data)/len(data):.2%} older than 20 years",
                metric="data_age",
                value=len(old_data)/len(data),
                threshold=0.5,
                details={"source": data_source, "old_data_count": len(old_data)}
            ))
            
        # Check for recent data
        recent_data = data[data['year'] >= current_year - 2]
        if len(recent_data) < len(data) * 0.1:  # Less than 10% from last 2 years
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message=f"Low proportion of recent data: {len(recent_data)/len(data):.2%} from last 2 years",
                metric="recent_data",
                value=len(recent_data)/len(data),
                threshold=0.1,
                details={"source": data_source, "recent_data_count": len(recent_data)}
            ))
            
        return alerts
        
    def _load_alert_history(self):
        """Load alert history from file."""
        if self.alert_history_path.exists():
            try:
                with open(self.alert_history_path, 'r') as f:
                    alert_data = json.load(f)
                    self.alerts = [
                        DataQualityAlert(
                            timestamp=datetime.fromisoformat(alert['timestamp']),
                            level=AlertLevel(alert['level']),
                            message=alert['message'],
                            metric=alert['metric'],
                            value=alert['value'],
                            threshold=alert['threshold'],
                            details=alert.get('details')
                        ) for alert in alert_data
                    ]
            except Exception as e:
                logger.error(f"Error loading alert history: {e}")
                self.alerts = []
        else:
            self.alerts = []
            
    def _save_alert_history(self):
        """Save alert history to file."""
        try:
            alert_data = [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'level': alert.level.value,
                    'message': alert.message,
                    'metric': alert.metric,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'details': alert.details
                } for alert in self.alerts
            ]
            
            with open(self.alert_history_path, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")
            
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_level': {},
            'by_metric': {},
            'recent_alerts': []
        }
        
        for alert in recent_alerts:
            # Count by level
            level = alert.level.value
            summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
            
            # Count by metric
            metric = alert.metric
            summary['by_metric'][metric] = summary['by_metric'].get(metric, 0) + 1
            
            # Add to recent alerts (last 10)
            if len(summary['recent_alerts']) < 10:
                summary['recent_alerts'].append({
                    'timestamp': alert.timestamp.isoformat(),
                    'level': alert.level.value,
                    'message': alert.message,
                    'metric': alert.metric
                })
                
        return summary
        
    def get_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-1)."""
        if data.empty:
            return 0.0
            
        scores = []
        
        # Record count score
        record_score = min(1.0, len(data) / 100)  # Normalize to 100 records
        scores.append(record_score)
        
        # Completeness score
        completeness = data.notna().mean().mean()
        scores.append(completeness)
        
        # Species diversity score
        species_count = data['species'].nunique()
        species_score = min(1.0, species_count / 10)  # Normalize to 10 species
        scores.append(species_score)
        
        # Geographic spread score
        lat_range = data['latitude'].max() - data['latitude'].min()
        lon_range = data['longitude'].max() - data['longitude'].min()
        geo_score = min(1.0, (lat_range + lon_range) / 20)  # Normalize to 20 degrees
        scores.append(geo_score)
        
        # Duplicate rate score (inverted)
        duplicate_rate = data.duplicated().sum() / len(data)
        duplicate_score = 1.0 - duplicate_rate
        scores.append(duplicate_score)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.2, 0.2, 0.1]  # Completeness is most important
        quality_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return quality_score