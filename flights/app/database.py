from functools import lru_cache
from datetime import datetime, timedelta
import random
from databricks import sql
from databricks.sdk.core import Config
from app.config import (
    DEFAULT_HTTP_PATH,
    DEFAULT_FLIGHTS_TABLE,
    DEFAULT_PASSENGERS_TABLE,
    DEFAULT_AIRPORTS_TABLE,
    DEFAULT_AIRLINES_TABLE,
    DEFAULT_AIRCRAFT_TABLE
)

cfg = Config()

@lru_cache(maxsize=1)
def get_connection(http_path):
    """Get a cached database connection"""
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )

def get_conn():
    """Get the default database connection"""
    return get_connection(DEFAULT_HTTP_PATH)

def table_exists(table_name):
    """Check if a table exists"""
    try:
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES LIKE '{table_name.split('.')[-1]}'")
            result = cursor.fetchall()
            return len(result) > 0
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False

def escape_sql_string(s):
    """Escape single quotes in SQL strings"""
    return s.replace("'", "''") if isinstance(s, str) else s

def initialize_database():
    """Initialize database with sample data if tables don't exist"""
    conn = get_conn()
    
    with conn.cursor() as cursor:
        # Check if tables exist and have data
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {DEFAULT_AIRPORTS_TABLE}")
            airports_count = cursor.fetchall()[0][0]
            if airports_count > 0:
                print(f"Tables already contain data ({airports_count} airports found). Skipping initialization.")
                return
        except Exception as e:
            print(f"Tables don't exist yet, will create them. ({e})")
        
        # Drop existing tables if they exist (to ensure clean schema)
        print("Dropping existing tables if any...")
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {DEFAULT_PASSENGERS_TABLE}")
            cursor.execute(f"DROP TABLE IF EXISTS {DEFAULT_FLIGHTS_TABLE}")
            cursor.execute(f"DROP TABLE IF EXISTS {DEFAULT_AIRCRAFT_TABLE}")
            cursor.execute(f"DROP TABLE IF EXISTS {DEFAULT_AIRLINES_TABLE}")
            cursor.execute(f"DROP TABLE IF EXISTS {DEFAULT_AIRPORTS_TABLE}")
        except Exception as e:
            print(f"Note: {e}")
        
        # Create airports table
        print("Creating airports table...")
        cursor.execute(f"""
            CREATE TABLE {DEFAULT_AIRPORTS_TABLE} (
                airport_code STRING,
                airport_name STRING,
                city STRING,
                country STRING
            )
        """)
        
        # Create airlines table
        print("Creating airlines table...")
        cursor.execute(f"""
            CREATE TABLE {DEFAULT_AIRLINES_TABLE} (
                airline_code STRING,
                airline_name STRING,
                country STRING
            )
        """)
        
        # Create flights table
        print("Creating flights table...")
        cursor.execute(f"""
            CREATE TABLE {DEFAULT_FLIGHTS_TABLE} (
                flight_id INT,
                flight_number STRING,
                airline STRING,
                origin STRING,
                destination STRING,
                departure_time TIMESTAMP,
                arrival_time TIMESTAMP,
                status STRING,
                aircraft_type STRING,
                delay_minutes INT
            )
        """)
        
        # Create passengers table
        print("Creating passengers table...")
        cursor.execute(f"""
            CREATE TABLE {DEFAULT_PASSENGERS_TABLE} (
                passport_id STRING,
                first_name STRING,
                last_name STRING,
                email STRING,
                phone STRING,
                sex STRING,
                flight_id INT,
                seat_number STRING,
                class STRING,
                booking_date TIMESTAMP,
                ticket_price DECIMAL(10,2)
            )
        """)
        
        # Create aircraft table
        print("Creating aircraft table...")
        cursor.execute(f"""
            CREATE TABLE {DEFAULT_AIRCRAFT_TABLE} (
                aircraft_id INT,
                aircraft_type STRING,
                manufacturer STRING,
                capacity INT,
                range_km INT
            )
        """)
        
        print("Initializing database with sample data...")
        
        # Insert airports
        airports_data = [
                ('JFK', 'John F. Kennedy International Airport', 'New York', 'USA'),
                ('LAX', 'Los Angeles International Airport', 'Los Angeles', 'USA'),
                ('ORD', 'O\'Hare International Airport', 'Chicago', 'USA'),
                ('ATL', 'Hartsfield-Jackson Atlanta International Airport', 'Atlanta', 'USA'),
                ('DFW', 'Dallas/Fort Worth International Airport', 'Dallas', 'USA'),
                ('DEN', 'Denver International Airport', 'Denver', 'USA'),
                ('LHR', 'London Heathrow Airport', 'London', 'UK'),
                ('CDG', 'Charles de Gaulle Airport', 'Paris', 'France'),
                ('DXB', 'Dubai International Airport', 'Dubai', 'UAE'),
                ('HND', 'Tokyo Haneda Airport', 'Tokyo', 'Japan'),
                ('SFO', 'San Francisco International Airport', 'San Francisco', 'USA'),
                ('MIA', 'Miami International Airport', 'Miami', 'USA'),
            ]
        
        airports_values = ", ".join([f"('{code}', '{escape_sql_string(name)}', '{city}', '{country}')" 
                                    for code, name, city, country in airports_data])
        cursor.execute(f"INSERT INTO {DEFAULT_AIRPORTS_TABLE} VALUES {airports_values}")
        
        # Insert airlines
        airlines_data = [
            ('AA', 'American Airlines', 'USA'),
            ('DL', 'Delta Air Lines', 'USA'),
            ('UA', 'United Airlines', 'USA'),
            ('BA', 'British Airways', 'UK'),
            ('EK', 'Emirates', 'UAE'),
            ('LH', 'Lufthansa', 'Germany'),
            ('AF', 'Air France', 'France'),
            ('JL', 'Japan Airlines', 'Japan'),
        ]
        
        airlines_values = ", ".join([f"('{code}', '{escape_sql_string(name)}', '{country}')" 
                                    for code, name, country in airlines_data])
        cursor.execute(f"INSERT INTO {DEFAULT_AIRLINES_TABLE} VALUES {airlines_values}")
        
        # Insert flights
        flight_statuses = ['Scheduled', 'Boarding', 'Departed', 'Arrived', 'Delayed', 'Cancelled']
        aircraft_types = ['Boeing 737', 'Boeing 777', 'Airbus A320', 'Airbus A380', 'Boeing 787']
        
        flights_values = []
        for i in range(1, 101):
            airline = random.choice([a[0] for a in airlines_data])
            origin = random.choice([a[0] for a in airports_data])
            destination = random.choice([a[0] for a in airports_data])
            while destination == origin:
                destination = random.choice([a[0] for a in airports_data])
            
            departure_time = datetime.now() + timedelta(days=random.randint(-30, 30), 
                                                       hours=random.randint(0, 23))
            arrival_time = departure_time + timedelta(hours=random.randint(2, 12))
            status = random.choice(flight_statuses)
            aircraft = random.choice(aircraft_types)
            
            # Generate realistic delays
            if status in ['Delayed', 'Cancelled']:
                delay_minutes = random.randint(15, 240)  # 15 min to 4 hours
            elif status in ['Departed', 'Arrived']:
                # 70% on time, 30% with minor delay
                delay_minutes = 0 if random.random() < 0.7 else random.randint(5, 60)
            else:  # Scheduled, Boarding
                delay_minutes = 0
            
            flights_values.append(
                f"({i}, '{airline}{1000+i}', '{airline}', '{origin}', '{destination}', "
                f"'{departure_time.strftime('%Y-%m-%d %H:%M:%S')}', "
                f"'{arrival_time.strftime('%Y-%m-%d %H:%M:%S')}', "
                f"'{status}', '{escape_sql_string(aircraft)}', {delay_minutes})"
            )
        
        # Insert in batches of 20
        batch_size = 20
        for i in range(0, len(flights_values), batch_size):
            batch = flights_values[i:i+batch_size]
            cursor.execute(f"INSERT INTO {DEFAULT_FLIGHTS_TABLE} VALUES {', '.join(batch)}")
        
        # Insert passengers
        first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'Robert', 'Lisa', 
                      'James', 'Mary', 'William', 'Patricia', 'Richard', 'Jennifer', 'Thomas']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
                     'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Wilson']
        sexes = ['Male', 'Female']
        classes = ['Economy', 'Business', 'First Class']
        
        passengers_values = []
        for i in range(1, 201):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            passport_id = f"P{random.randint(100000, 999999)}"
            email = f"{first_name.lower()}.{last_name.lower()}{i}@email.com"
            phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            sex = random.choice(sexes)
            flight_id = random.randint(1, 100)
            seat_number = f"{random.randint(1, 30)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}"
            passenger_class = random.choice(classes)
            booking_date = datetime.now() - timedelta(days=random.randint(1, 60))
            
            # Generate realistic ticket prices based on class
            if passenger_class == 'Economy':
                ticket_price = random.randint(150, 800)
            elif passenger_class == 'Business':
                ticket_price = random.randint(1200, 3500)
            else:  # First Class
                ticket_price = random.randint(3500, 8000)
            
            passengers_values.append(
                f"('{passport_id}', '{escape_sql_string(first_name)}', '{escape_sql_string(last_name)}', '{escape_sql_string(email)}', '{escape_sql_string(phone)}', '{sex}', "
                f"{flight_id}, '{seat_number}', '{passenger_class}', "
                f"'{booking_date.strftime('%Y-%m-%d %H:%M:%S')}', {ticket_price})"
            )
        
        # Insert in batches of 20
        for i in range(0, len(passengers_values), batch_size):
            batch = passengers_values[i:i+batch_size]
            cursor.execute(f"INSERT INTO {DEFAULT_PASSENGERS_TABLE} VALUES {', '.join(batch)}")
        
        # Insert aircraft
        aircraft_data = [
            (1, 'Boeing 737', 'Boeing', 180, 5600),
            (2, 'Boeing 777', 'Boeing', 396, 15200),
            (3, 'Boeing 787', 'Boeing', 330, 14140),
            (4, 'Airbus A320', 'Airbus', 180, 6100),
            (5, 'Airbus A330', 'Airbus', 277, 13400),
            (6, 'Airbus A380', 'Airbus', 555, 15200),
            (7, 'Boeing 747', 'Boeing', 416, 13450),
            (8, 'Airbus A350', 'Airbus', 440, 15000),
        ]
        
        aircraft_values = ", ".join([f"({id}, '{escape_sql_string(type)}', '{manufacturer}', {capacity}, {range_km})" 
                                    for id, type, manufacturer, capacity, range_km in aircraft_data])
        cursor.execute(f"INSERT INTO {DEFAULT_AIRCRAFT_TABLE} VALUES {aircraft_values}")
        
        print("Database initialized successfully!")

