import os

# Databricks Configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "https://e2-demo-field-eng.cloud.databricks.com/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DEFAULT_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/862f1d757f0424f7")

# Table Configuration
DEFAULT_FLIGHTS_TABLE = os.getenv("FLIGHTS_TABLE", "ali_azzouz.airport.flights")
DEFAULT_PASSENGERS_TABLE = os.getenv("PASSENGERS_TABLE", "ali_azzouz.airport.passengers")
DEFAULT_AIRPORTS_TABLE = os.getenv("AIRPORTS_TABLE", "ali_azzouz.airport.airports")
DEFAULT_AIRLINES_TABLE = os.getenv("AIRLINES_TABLE", "ali_azzouz.airport.airlines")
DEFAULT_AIRCRAFT_TABLE = os.getenv("AIRCRAFT_TABLE", "ali_azzouz.airport.aircraft")

# Flask Configuration
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
FLASK_RUN_HOST = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
FLASK_RUN_PORT = int(os.getenv("FLASK_RUN_PORT", "8000"))

