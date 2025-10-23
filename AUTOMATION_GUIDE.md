# 🤖 Data Collection Automation Guide

This guide explains how to automate the downloading of data from various internet-based sources in the Global Truffle Habitat Atlas (GTHA).

## 📍 Where Data Downloading is Automated

### 1. **Main Data Collection Entry Points**

#### `src/data_processing/habitat_processor.py`
- **`collect_all_data()`** - Main orchestrator that coordinates all data sources
- **`_collect_occurrence_data()`** - Collects from GBIF and iNaturalist
- **`_collect_soil_data()`** - Downloads from SoilGrids API
- **`_collect_climate_data()`** - Retrieves from WorldClim

#### `src/data_collectors/` (Individual Source Collectors)
- **`gbif_collector.py`** - GBIF occurrence data
- **`inaturalist_collector.py`** - iNaturalist observations
- **`soilgrids_collector.py`** - Soil properties
- **`worldclim_collector.py`** - Climate variables

### 2. **Automation Framework**

#### `src/automation/` (New Automation System)
- **`scheduler.py`** - Scheduled data collection
- **`pipeline.py`** - Automated end-to-end processing
- **`monitoring.py`** - Data quality monitoring

## 🚀 Quick Start with Automation

### 1. **One-Time Data Collection**
```bash
# Collect data for specific species
python automate_data_collection.py \
  --mode one-time \
  --species "Tuber melanosporum" "Tuber magnatum" \
  --countries FR IT ES

# Quick demo with limited data
python automate_data_collection.py \
  --mode one-time \
  --species "Tuber melanosporum" \
  --countries FR
```

### 2. **Scheduled Data Collection**
```bash
# Daily collection at 2 AM
python automate_data_collection.py \
  --mode scheduled \
  --schedule-type daily \
  --hour 2 \
  --minute 0

# Weekly collection on Mondays at 3 AM
python automate_data_collection.py \
  --mode scheduled \
  --schedule-type weekly \
  --day monday \
  --hour 3

# Custom interval every 6 hours
python automate_data_collection.py \
  --mode scheduled \
  --schedule-type custom \
  --interval-hours 6
```

### 3. **Incremental Updates**
```bash
# Update with only new data
python automate_data_collection.py \
  --mode incremental \
  --species "Tuber melanosporum" "Tuber magnatum"
```

## ⚙️ Configuration Options

### 1. **Using Configuration File**
```bash
# Use custom configuration
python automate_data_collection.py \
  --mode one-time \
  --config-file automation_config.json
```

### 2. **Configuration Parameters**
```json
{
  "automation": {
    "default_species": ["Tuber melanosporum", "Tuber magnatum"],
    "default_countries": ["FR", "IT", "ES"],
    "scheduling": {
      "daily": {"hour": 2, "minute": 0, "enabled": true},
      "weekly": {"day": "monday", "hour": 3, "enabled": false}
    },
    "data_quality": {
      "monitoring_enabled": true,
      "alert_thresholds": {
        "min_records_per_species": 5,
        "max_coordinate_uncertainty": 10000
      }
    }
  }
}
```

## 🔧 Customizing Data Sources

### 1. **Adding New Data Sources**

Create a new collector in `src/data_collectors/`:

```python
from .base_collector import BaseCollector

class NewSourceCollector(BaseCollector):
    def __init__(self, config, data_dir):
        super().__init__(config, data_dir)
        self.base_url = config["new_source"]["base_url"]
    
    def collect(self, **kwargs):
        # Implement data collection logic
        pass
    
    def validate_data(self, data):
        # Implement data validation
        pass
```

### 2. **Modifying Existing Collectors**

Edit the collectors in `src/data_collectors/` to:
- Change API endpoints
- Modify data processing logic
- Add new data fields
- Implement custom filtering

### 3. **Parallel Data Collection**

Enable parallel processing for faster collection:

```python
# In habitat_processor.py
def _collect_data_parallel(self, species, countries):
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit collection tasks in parallel
        futures = [executor.submit(self._collect_single_source, source) 
                  for source in sources]
        # Process results
```

## 📊 Monitoring and Quality Control

### 1. **Data Quality Monitoring**
```python
from src.automation.monitoring import DataQualityMonitor

monitor = DataQualityMonitor(config, output_dir)
alerts = monitor.monitor_data_quality(data, "source_name")
quality_score = monitor.get_quality_score(data)
```

### 2. **Alert Configuration**
```json
{
  "data_quality": {
    "alert_thresholds": {
      "min_records_per_species": 5,
      "max_coordinate_uncertainty": 10000,
      "min_data_completeness": 0.5,
      "max_duplicate_rate": 0.1
    }
  }
}
```

### 3. **Quality Metrics**
- **Record Count**: Minimum records per species
- **Coordinate Quality**: Maximum uncertainty threshold
- **Data Completeness**: Minimum percentage of non-null values
- **Duplicate Rate**: Maximum acceptable duplicate percentage
- **Geographic Spread**: Minimum geographic distribution
- **Outlier Detection**: Statistical outlier identification

## 🔄 Advanced Automation Patterns

### 1. **Cron Job Integration**
```bash
# Add to crontab for daily collection at 2 AM
0 2 * * * /path/to/python /path/to/automate_data_collection.py --mode one-time --species "Tuber melanosporum" "Tuber magnatum"

# Weekly collection on Mondays at 3 AM
0 3 * * 1 /path/to/python /path/to/automate_data_collection.py --mode one-time --species "Tuber melanosporum" "Tuber magnatum" "Tuber aestivum"
```

### 2. **Docker Container**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run daily collection
CMD ["python", "automate_data_collection.py", "--mode", "scheduled", "--schedule-type", "daily"]
```

### 3. **Kubernetes CronJob**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: truffle-data-collection
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: data-collector
            image: truffle-atlas:latest
            command: ["python", "automate_data_collection.py", "--mode", "one-time"]
```

### 4. **Cloud Functions (AWS Lambda, Google Cloud Functions)**
```python
import json
from automate_data_collection import DataCollectionAutomation

def lambda_handler(event, context):
    automation = DataCollectionAutomation()
    results = automation.run_one_time_collection(
        species=["Tuber melanosporum", "Tuber magnatum"],
        countries=["FR", "IT", "ES"]
    )
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

## 📈 Performance Optimization

### 1. **Parallel Processing**
```python
# Enable parallel collection
python automate_data_collection.py \
  --mode one-time \
  --parallel \
  --species "Tuber melanosporum" "Tuber magnatum"
```

### 2. **Caching**
```python
# Implement caching for API responses
import requests_cache
requests_cache.install_cache('api_cache', expire_after=3600)  # 1 hour
```

### 3. **Rate Limiting**
```python
# Respect API rate limits
import time
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)  # 100 calls per minute
def make_api_request():
    # API call implementation
    pass
```

## 🚨 Error Handling and Recovery

### 1. **Retry Logic**
```python
def _make_request_with_retry(self, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 2. **Graceful Degradation**
```python
def collect_all_data(self, species, countries):
    all_data = []
    
    # Try each source, continue if one fails
    for source in self.sources:
        try:
            data = source.collect(species, countries)
            all_data.append(data)
        except Exception as e:
            logger.warning(f"Source {source} failed: {e}")
            continue
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
```

### 3. **Health Checks**
```python
def health_check(self):
    """Check if all data sources are accessible."""
    health_status = {}
    
    for source_name, collector in self.collectors.items():
        try:
            # Test API connectivity
            test_response = collector._make_request(collector.base_url + "/test")
            health_status[source_name] = "healthy"
        except Exception as e:
            health_status[source_name] = f"unhealthy: {e}"
    
    return health_status
```

## 📋 Best Practices

### 1. **Data Collection**
- Use appropriate rate limits for each API
- Implement proper error handling and retries
- Cache responses when possible
- Monitor data quality continuously

### 2. **Scheduling**
- Schedule during off-peak hours
- Use incremental updates when possible
- Monitor system resources
- Set up proper logging and alerting

### 3. **Storage**
- Use versioned data storage
- Implement data retention policies
- Backup critical data regularly
- Use appropriate file formats (CSV, Parquet, GeoJSON)

### 4. **Monitoring**
- Track collection success rates
- Monitor data quality metrics
- Set up alerts for failures
- Log all activities for debugging

## 🔍 Troubleshooting

### Common Issues:

1. **API Rate Limits**
   - Solution: Implement proper rate limiting and backoff
   - Check: `automation_config.json` for rate limit settings

2. **Network Timeouts**
   - Solution: Increase timeout values in configuration
   - Check: API timeout settings in config

3. **Data Quality Issues**
   - Solution: Adjust quality thresholds
   - Check: Data quality monitoring logs

4. **Memory Issues**
   - Solution: Process data in chunks
   - Check: System memory usage during collection

5. **Disk Space**
   - Solution: Implement data retention policies
   - Check: Output directory size

## 📞 Support

For issues with automation:
1. Check the logs in `logs/automation.log`
2. Review the configuration in `automation_config.json`
3. Test individual collectors separately
4. Check API connectivity and rate limits

---

**Happy Automating! 🤖🍄**