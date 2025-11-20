# ✈️ Airport Management System

A comprehensive airport management system built with Flask and Databricks, featuring real-time flight tracking, passenger management, and analytics dashboards.

## 🏗️ Project Structure

```
flights/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── run.sh                     # Shell script to run the application
├── app/                       # Application modules
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── database.py           # Database connections and initialization
│   └── routes.py             # API routes and endpoints
├── templates/                 # HTML templates
│   └── index.html            # Main application template
├── static/                    # Static assets
│   ├── css/
│   │   └── style.css         # Application styles
│   └── js/
│       └── app.js            # Frontend JavaScript
└── venv/                      # Virtual environment (gitignored)
```

## 🚀 Features

### Core Functionality
- **Flight Management**: Track, create, update, and delete flights
- **Passenger Records**: Manage passenger information and bookings
- **Airport Database**: Maintain airport information
- **Airline Management**: Manage airline data
- **Aircraft Inventory**: Track aircraft types and specifications

### Analytics Dashboard
- Total flights, passengers, and revenue metrics
- Top airports by flight volume
- Top airlines by passenger count
- Flight status distribution
- Passenger class distribution
- Revenue analysis by class and airline
- Top routes by revenue
- Average delay statistics by airport and airline

## 📋 Prerequisites

- Python 3.9+
- Databricks account with SQL Warehouse access
- Databricks personal access token

## 🔧 Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/flights
   ```

2. **Create a virtual environment** (if not already created)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables** (optional)
   
   Create a `.env` file or set environment variables:
   ```bash
   export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com/"
   export DATABRICKS_TOKEN="your-token-here"
   export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your-warehouse-id"
   ```

## 🏃 Running the Application

### Using Python directly
```bash
python app.py
```

### Using the shell script
```bash
chmod +x run.sh
./run.sh
```

The application will be available at `http://localhost:5000`

## 🗄️ Database

The application uses Databricks SQL Warehouse as the backend database. On first run, it will automatically:
- Create necessary tables (airports, airlines, flights, passengers, aircraft)
- Populate with sample data
- Skip initialization if tables already contain data

### Tables
- `airports`: Airport information (code, name, city, country)
- `airlines`: Airline information (code, name, country)
- `flights`: Flight records with departure/arrival times, status, delays
- `passengers`: Passenger bookings with ticket prices and seat assignments
- `aircraft`: Aircraft specifications

## 🔌 API Endpoints

### Airports
- `GET /api/airports` - List all airports
- `GET /api/airports/<code>` - Get specific airport
- `POST /api/airports` - Create new airport
- `PUT /api/airports/<code>` - Update airport
- `DELETE /api/airports/<code>` - Delete airport

### Airlines
- `GET /api/airlines` - List all airlines
- `GET /api/airlines/<code>` - Get specific airline
- `POST /api/airlines` - Create new airline
- `PUT /api/airlines/<code>` - Update airline
- `DELETE /api/airlines/<code>` - Delete airline

### Flights
- `GET /api/flights` - List all flights
- `GET /api/flights/<id>` - Get specific flight
- `POST /api/flights` - Create new flight
- `PUT /api/flights/<id>` - Update flight
- `DELETE /api/flights/<id>` - Delete flight

### Passengers
- `GET /api/passengers` - List all passengers
- `GET /api/passengers/<passport_id>` - Get specific passenger
- `POST /api/passengers` - Create new passenger
- `PUT /api/passengers/<passport_id>` - Update passenger
- `DELETE /api/passengers/<passport_id>` - Delete passenger

### Aircraft
- `GET /api/aircraft` - List all aircraft
- `GET /api/aircraft/<id>` - Get specific aircraft
- `POST /api/aircraft` - Create new aircraft
- `PUT /api/aircraft/<id>` - Update aircraft
- `DELETE /api/aircraft/<id>` - Delete aircraft

### Analytics
- `GET /api/kpis` - Get dashboard KPIs and analytics

## 🎨 Frontend

The frontend is built with:
- **Bootstrap 5**: Modern, responsive UI components
- **DataTables**: Interactive, sortable tables
- **Chart.js**: Beautiful, responsive charts
- **jQuery**: DOM manipulation and AJAX calls

## 📝 Configuration

Configuration is managed through `app/config.py` and can be overridden with environment variables:

- `DATABRICKS_HOST`: Databricks workspace URL
- `DATABRICKS_TOKEN`: Personal access token
- `DATABRICKS_HTTP_PATH`: SQL Warehouse HTTP path
- `FLIGHTS_TABLE`: Flights table name
- `PASSENGERS_TABLE`: Passengers table name
- `AIRPORTS_TABLE`: Airports table name
- `AIRLINES_TABLE`: Airlines table name
- `AIRCRAFT_TABLE`: Aircraft table name
- `FLASK_DEBUG`: Enable/disable debug mode
- `FLASK_HOST`: Host to bind the server
- `FLASK_PORT`: Port to run the server

## 🛠️ Development

### Project Organization

The project follows a modular structure:

1. **`app/config.py`**: Centralized configuration management
2. **`app/database.py`**: Database connection and initialization
3. **`app/routes.py`**: API endpoint definitions
4. **`templates/index.html`**: Main HTML template
5. **`static/css/style.css`**: Application styling
6. **`static/js/app.js`**: Frontend logic

### Adding New Features

1. Add configuration variables to `app/config.py`
2. Add database functions to `app/database.py`
3. Add API routes to `app/routes.py`
4. Update frontend in `templates/index.html` and `static/js/app.js`

## 🐛 Troubleshooting

### Database Connection Issues
- Verify Databricks credentials in `app/config.py`
- Ensure SQL Warehouse is running
- Check network connectivity to Databricks

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Port Already in Use
- Change the port in `app/config.py` or set `FLASK_PORT` environment variable

## 📄 License

This project is provided as-is for demonstration purposes.

## 👥 Contributing

This is a demonstration project. Feel free to fork and modify as needed.

## 📧 Support

For issues or questions, please refer to the Databricks documentation or Flask documentation.
