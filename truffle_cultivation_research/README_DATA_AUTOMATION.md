# Data Automation System
=======================

This document explains how to automate data downloading from various internet sources for truffle cultivation research.

## 🚀 Quick Start

```bash
# Start the data automation system
./scripts/start_data_automation.sh
```

## 📊 Data Sources

The system automatically downloads data from:

### Scientific Papers
- **OpenAlex**: Open access academic papers
- **Crossref**: Academic publications with DOI
- **PubMed**: Biomedical literature

### Patents
- **EPO OPS**: European Patent Office
- **WIPO PATENTSCOPE**: World Intellectual Property Organization

## ⚙️ Configuration

Edit `configs/data_fetching.yaml` to customize:

```yaml
# Search queries to monitor
queries:
  - '"Tuber melanosporum" AND mycorrhiz* AND hydroponic*'
  - '"Tuber aestivum" AND cultivation AND controlled environment'
  - 'truffle AND mycorrhiza AND nutrient*'
  # ... add your own queries

# How often to fetch data
fetch_interval_hours: 24

# Quality filters
min_confidence: 0.7
min_relevance_score: 0.5
```

## 🔄 Operation Modes

### 1. Real-time Monitoring
Continuously monitors for new data and processes it automatically.

```bash
python -m etl.automated_data_fetcher --mode monitor
```

### 2. Scheduled Fetching
Runs on a schedule (e.g., every 6 hours).

```bash
python -m etl.automated_data_fetcher --mode schedule
```

### 3. Batch Processing
Processes data once for specific queries.

```bash
python -m etl.automated_data_fetcher --mode batch --query "truffle cultivation"
```

## 📁 Data Storage

### Raw Data
- **Papers**: `data/raw/papers/` (JSON files)
- **Patents**: `data/raw/patents/` (JSON files)

### Database
- **SQLite**: `data/fetch_history.db` (tracks fetch history and processing status)

### Processed Data
- **Knowledge Graph**: Neo4j and GraphDB
- **Exports**: `data/exports/` (various formats)

## 🔧 Setup Commands

```bash
# Initial setup
python scripts/setup_data_automation.py

# Create default configuration
python scripts/setup_data_automation.py --create-config

# Test API connections
python scripts/setup_data_automation.py --test-connections

# Create systemd service (Linux)
python scripts/setup_data_automation.py --create-services
```

## 📊 Monitoring

### Check Processing Status
```bash
python scripts/process_fetched_data.py --stats
```

### View Logs
```bash
tail -f logs/*.log
```

### Database Queries
```bash
sqlite3 data/fetch_history.db
```

## 🐳 Docker Deployment

```bash
# Start with data fetcher
docker-compose up -d data-fetcher data-processor

# View logs
docker-compose logs -f data-fetcher
```

## 🔍 Query Examples

### Find Recent Papers
```sql
SELECT doi, title, source, fetch_time 
FROM fetched_papers 
WHERE fetch_time > datetime('now', '-7 days')
ORDER BY fetch_time DESC;
```

### Check Processing Status
```sql
SELECT 
    source,
    COUNT(*) as total,
    SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed,
    AVG(quality_score) as avg_quality
FROM fetched_papers 
GROUP BY source;
```

### Find High-Quality Papers
```sql
SELECT doi, title, quality_score 
FROM fetched_papers 
WHERE quality_score > 0.8 
ORDER BY quality_score DESC;
```

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Reduce `requests_per_minute` in config
   - Increase `delay_between_requests`

2. **Database Locked**
   - Check if another process is using the database
   - Restart the system

3. **No Data Fetched**
   - Check API connections
   - Verify queries are not too specific
   - Check rate limits

4. **Processing Errors**
   - Check logs in `logs/` directory
   - Verify dependencies are installed
   - Check database permissions

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m etl.automated_data_fetcher --mode monitor
```

## 📈 Performance Tuning

### For Large Datasets
- Increase `batch_size` in config
- Use multiple workers for processing
- Consider using a more powerful database

### For Real-time Processing
- Decrease `fetch_interval_hours`
- Use continuous processing mode
- Monitor system resources

## 🔐 Security

### API Keys
Store API keys in environment variables:
```bash
export OPENALEX_API_KEY="your_key"
export CROSSREF_API_KEY="your_key"
```

### Database Security
- Use strong passwords
- Limit database access
- Regular backups

## 📚 Advanced Usage

### Custom Entity Extraction
Modify `etl/automated_data_fetcher.py` to add custom entity extractors.

### Custom Quality Scoring
Implement custom quality scoring in `scripts/process_fetched_data.py`.

### Integration with Knowledge Graph
The system automatically ingests processed data into the knowledge graph.

## 🆘 Support

For issues or questions:
1. Check the logs
2. Review this documentation
3. Check the GitHub issues
4. Contact the development team