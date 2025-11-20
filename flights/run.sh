#!/bin/bash

# Airport Management System Startup Script

echo "🛫 Starting Airport Management System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Set default environment variables if not already set
export DATABRICKS_HOST="${DATABRICKS_HOST:-https://e2-demo-field-eng.cloud.databricks.com/}"
export DATABRICKS_TOKEN="${DATABRICKS_TOKEN:-dapif1528aa8af093d2c1671b6689c92bb68}"
export DATABRICKS_HTTP_PATH="${DATABRICKS_HTTP_PATH:-/sql/1.0/warehouses/862f1d757f0424f7}"
export FLIGHTS_TABLE="${FLIGHTS_TABLE:-ali_azzouz.airport.flights}"
export PASSENGERS_TABLE="${PASSENGERS_TABLE:-ali_azzouz.airport.passengers}"
export AIRPORTS_TABLE="${AIRPORTS_TABLE:-ali_azzouz.airport.airports}"
export AIRLINES_TABLE="${AIRLINES_TABLE:-ali_azzouz.airport.airlines}"

echo ""
echo "✅ Environment configured!"
echo "📊 Databricks Host: $DATABRICKS_HOST"
echo "📁 Flights Table: $FLIGHTS_TABLE"
echo "📁 Passengers Table: $PASSENGERS_TABLE"
echo ""
echo "🚀 Starting Flask application..."
echo "🌐 Open your browser to: http://localhost:5000"
echo ""

# Run the Flask app
python app.py

