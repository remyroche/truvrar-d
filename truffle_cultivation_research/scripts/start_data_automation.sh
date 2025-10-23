#!/bin/bash
# Start Data Automation System
# ============================

set -e

echo "🍄 Starting Truffle Cultivation Data Automation System"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw/papers
mkdir -p data/raw/patents
mkdir -p data/processed
mkdir -p data/exports
mkdir -p logs

# Check if configuration exists
if [ ! -f "configs/data_fetching.yaml" ]; then
    echo "⚙️  Creating default configuration..."
    python scripts/setup_data_automation.py --create-config
fi

# Check if database exists
if [ ! -f "data/fetch_history.db" ]; then
    echo "🗄️  Initializing database..."
    python scripts/setup_data_automation.py
fi

# Test API connections
echo "🔗 Testing API connections..."
python scripts/setup_data_automation.py --test-connections

# Show current statistics
echo "📊 Current processing statistics:"
python scripts/process_fetched_data.py --stats

# Ask user what to do
echo ""
echo "What would you like to do?"
echo "1. Start real-time monitoring (recommended)"
echo "2. Run batch processing once"
echo "3. Start continuous processing"
echo "4. Show statistics only"
echo "5. Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Starting real-time monitoring..."
        echo "Press Ctrl+C to stop"
        python -m etl.automated_data_fetcher --mode monitor --config configs/data_fetching.yaml
        ;;
    2)
        echo "🔄 Running batch processing..."
        python -m etl.automated_data_fetcher --mode batch --config configs/data_fetching.yaml
        echo "✅ Batch processing completed"
        ;;
    3)
        echo "🔄 Starting continuous processing..."
        echo "Press Ctrl+C to stop"
        python scripts/process_fetched_data.py --mode continuous --interval 30
        ;;
    4)
        echo "📊 Processing statistics:"
        python scripts/process_fetched_data.py --stats
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Data automation system is running!"
echo ""
echo "Useful commands:"
echo "  - View logs: tail -f logs/*.log"
echo "  - Check stats: python scripts/process_fetched_data.py --stats"
echo "  - Stop monitoring: Press Ctrl+C"
echo "  - View database: sqlite3 data/fetch_history.db"