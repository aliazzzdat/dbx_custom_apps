from flask import render_template, request, jsonify
from app.database import get_conn, escape_sql_string
from app.config import (
    DEFAULT_FLIGHTS_TABLE,
    DEFAULT_PASSENGERS_TABLE,
    DEFAULT_AIRPORTS_TABLE,
    DEFAULT_AIRLINES_TABLE,
    DEFAULT_AIRCRAFT_TABLE
)

def init_routes(app):
    """Initialize all routes for the Flask app"""
    
    @app.route('/')
    def index():
        """Render the main application page"""
        return render_template('index.html')

    # ========== Airport Routes ==========
    @app.route('/api/airports')
    def get_airports():
        """Get all airports"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_AIRPORTS_TABLE} ORDER BY airport_code")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return jsonify([dict(zip(columns, row)) for row in rows])

    @app.route('/api/airports/<string:airport_code>')
    def get_airport(airport_code):
        """Get a specific airport by code"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_AIRPORTS_TABLE} WHERE airport_code = '{airport_code}'")
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return jsonify(dict(zip(columns, row)))
            return jsonify({'error': 'Airport not found'}), 404

    @app.route('/api/airports', methods=['POST'])
    def create_airport():
        """Create a new airport"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                INSERT INTO {DEFAULT_AIRPORTS_TABLE} VALUES (
                    '{escape_sql_string(data['airport_code'])}', '{escape_sql_string(data['airport_name'])}',
                    '{escape_sql_string(data['city'])}', '{escape_sql_string(data['country'])}'
                )
            """)
            return jsonify({'success': True, 'airport_code': data['airport_code']})

    @app.route('/api/airports/<string:airport_code>', methods=['PUT'])
    def update_airport(airport_code):
        """Update an existing airport"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            # Delete old and insert new
            cursor.execute(f"DELETE FROM {DEFAULT_AIRPORTS_TABLE} WHERE airport_code = '{airport_code}'")
            cursor.execute(f"""
                INSERT INTO {DEFAULT_AIRPORTS_TABLE} VALUES (
                    '{escape_sql_string(data['airport_code'])}', '{escape_sql_string(data['airport_name'])}',
                    '{escape_sql_string(data['city'])}', '{escape_sql_string(data['country'])}'
                )
            """)
            return jsonify({'success': True})

    @app.route('/api/airports/<string:airport_code>', methods=['DELETE'])
    def delete_airport(airport_code):
        """Delete an airport"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {DEFAULT_AIRPORTS_TABLE} WHERE airport_code = '{airport_code}'")
            return jsonify({'success': True})

    # ========== Airline Routes ==========
    @app.route('/api/airlines')
    def get_airlines():
        """Get all airlines"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_AIRLINES_TABLE} ORDER BY airline_code")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return jsonify([dict(zip(columns, row)) for row in rows])

    @app.route('/api/airlines/<string:airline_code>')
    def get_airline(airline_code):
        """Get a specific airline by code"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_AIRLINES_TABLE} WHERE airline_code = '{airline_code}'")
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return jsonify(dict(zip(columns, row)))
            return jsonify({'error': 'Airline not found'}), 404

    @app.route('/api/airlines', methods=['POST'])
    def create_airline():
        """Create a new airline"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                INSERT INTO {DEFAULT_AIRLINES_TABLE} VALUES (
                    '{escape_sql_string(data['airline_code'])}', '{escape_sql_string(data['airline_name'])}',
                    '{escape_sql_string(data['country'])}'
                )
            """)
            return jsonify({'success': True, 'airline_code': data['airline_code']})

    @app.route('/api/airlines/<string:airline_code>', methods=['PUT'])
    def update_airline(airline_code):
        """Update an existing airline"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {DEFAULT_AIRLINES_TABLE} WHERE airline_code = '{airline_code}'")
            cursor.execute(f"""
                INSERT INTO {DEFAULT_AIRLINES_TABLE} VALUES (
                    '{escape_sql_string(data['airline_code'])}', '{escape_sql_string(data['airline_name'])}',
                    '{escape_sql_string(data['country'])}'
                )
            """)
            return jsonify({'success': True})

    @app.route('/api/airlines/<string:airline_code>', methods=['DELETE'])
    def delete_airline(airline_code):
        """Delete an airline"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {DEFAULT_AIRLINES_TABLE} WHERE airline_code = '{airline_code}'")
            return jsonify({'success': True})

    # ========== Flight Routes ==========
    @app.route('/api/flights')
    def get_flights():
        """Get all flights"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_FLIGHTS_TABLE} ORDER BY flight_id")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return jsonify([dict(zip(columns, row)) for row in rows])

    @app.route('/api/flights/<int:flight_id>')
    def get_flight(flight_id):
        """Get a specific flight by ID"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_FLIGHTS_TABLE} WHERE flight_id = {flight_id}")
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return jsonify(dict(zip(columns, row)))
            return jsonify({'error': 'Flight not found'}), 404

    @app.route('/api/flights', methods=['POST'])
    def create_flight():
        """Create a new flight"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            # Get max flight_id
            cursor.execute(f"SELECT MAX(flight_id) FROM {DEFAULT_FLIGHTS_TABLE}")
            max_id = cursor.fetchone()[0] or 0
            new_id = max_id + 1
            
            cursor.execute(f"""
                INSERT INTO {DEFAULT_FLIGHTS_TABLE} VALUES (
                    {new_id}, '{escape_sql_string(data['flight_number'])}', '{escape_sql_string(data['airline'])}',
                    '{escape_sql_string(data['origin'])}', '{escape_sql_string(data['destination'])}',
                    '{escape_sql_string(data['departure_time'])}', '{escape_sql_string(data['arrival_time'])}',
                    '{escape_sql_string(data['status'])}', '{escape_sql_string(data['aircraft_type'])}',
                    {data.get('delay_minutes', 0)}
                )
            """)
            return jsonify({'success': True, 'flight_id': new_id})

    @app.route('/api/flights/<int:flight_id>', methods=['PUT'])
    def update_flight(flight_id):
        """Update an existing flight"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                UPDATE {DEFAULT_FLIGHTS_TABLE}
                SET flight_number = '{escape_sql_string(data['flight_number'])}',
                    airline = '{escape_sql_string(data['airline'])}',
                    origin = '{escape_sql_string(data['origin'])}',
                    destination = '{escape_sql_string(data['destination'])}',
                    departure_time = '{escape_sql_string(data['departure_time'])}',
                    arrival_time = '{escape_sql_string(data['arrival_time'])}',
                    status = '{escape_sql_string(data['status'])}',
                    aircraft_type = '{escape_sql_string(data['aircraft_type'])}',
                    delay_minutes = {data.get('delay_minutes', 0)}
                WHERE flight_id = {flight_id}
            """)
            return jsonify({'success': True})

    @app.route('/api/flights/<int:flight_id>', methods=['DELETE'])
    def delete_flight(flight_id):
        """Delete a flight"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {DEFAULT_FLIGHTS_TABLE} WHERE flight_id = {flight_id}")
            return jsonify({'success': True})

    # ========== Passenger Routes ==========
    @app.route('/api/passengers')
    def get_passengers():
        """Get all passengers"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_PASSENGERS_TABLE} ORDER BY passport_id")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return jsonify([dict(zip(columns, row)) for row in rows])

    @app.route('/api/passengers/<string:passport_id>')
    def get_passenger(passport_id):
        """Get a specific passenger by passport ID"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_PASSENGERS_TABLE} WHERE passport_id = '{passport_id}'")
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return jsonify(dict(zip(columns, row)))
            return jsonify({'error': 'Passenger not found'}), 404

    @app.route('/api/passengers', methods=['POST'])
    def create_passenger():
        """Create a new passenger"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                INSERT INTO {DEFAULT_PASSENGERS_TABLE} VALUES (
                    '{escape_sql_string(data['passport_id'])}',
                    '{escape_sql_string(data['first_name'])}', '{escape_sql_string(data['last_name'])}',
                    '{escape_sql_string(data['email'])}', '{escape_sql_string(data['phone'])}', '{escape_sql_string(data['sex'])}',
                    {data['flight_id']}, '{escape_sql_string(data['seat_number'])}', '{escape_sql_string(data['class'])}',
                    '{escape_sql_string(data['booking_date'])}', {data['ticket_price']}
                )
            """)
            return jsonify({'success': True, 'passport_id': data['passport_id']})

    @app.route('/api/passengers/<string:passport_id>', methods=['PUT'])
    def update_passenger(passport_id):
        """Update an existing passenger"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            # Delete old and insert new (since we might be changing the primary key)
            cursor.execute(f"DELETE FROM {DEFAULT_PASSENGERS_TABLE} WHERE passport_id = '{passport_id}'")
            cursor.execute(f"""
                INSERT INTO {DEFAULT_PASSENGERS_TABLE} VALUES (
                    '{escape_sql_string(data['passport_id'])}',
                    '{escape_sql_string(data['first_name'])}', '{escape_sql_string(data['last_name'])}',
                    '{escape_sql_string(data['email'])}', '{escape_sql_string(data['phone'])}', '{escape_sql_string(data['sex'])}',
                    {data['flight_id']}, '{escape_sql_string(data['seat_number'])}', '{escape_sql_string(data['class'])}',
                    '{escape_sql_string(data['booking_date'])}', {data['ticket_price']}
                )
            """)
            return jsonify({'success': True})

    @app.route('/api/passengers/<string:passport_id>', methods=['DELETE'])
    def delete_passenger(passport_id):
        """Delete a passenger"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {DEFAULT_PASSENGERS_TABLE} WHERE passport_id = '{passport_id}'")
            return jsonify({'success': True})

    # ========== Aircraft Routes ==========
    @app.route('/api/aircraft')
    def get_aircraft():
        """Get all aircraft"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_AIRCRAFT_TABLE} ORDER BY aircraft_id")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return jsonify([dict(zip(columns, row)) for row in rows])

    @app.route('/api/aircraft/<int:aircraft_id>')
    def get_single_aircraft(aircraft_id):
        """Get a specific aircraft by ID"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DEFAULT_AIRCRAFT_TABLE} WHERE aircraft_id = {aircraft_id}")
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return jsonify(dict(zip(columns, row)))
            return jsonify({'error': 'Aircraft not found'}), 404

    @app.route('/api/aircraft', methods=['POST'])
    def create_aircraft():
        """Create a new aircraft"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            # Get max aircraft_id
            cursor.execute(f"SELECT MAX(aircraft_id) FROM {DEFAULT_AIRCRAFT_TABLE}")
            max_id = cursor.fetchone()[0] or 0
            new_id = max_id + 1
            
            cursor.execute(f"""
                INSERT INTO {DEFAULT_AIRCRAFT_TABLE} VALUES (
                    {new_id}, '{escape_sql_string(data['aircraft_type'])}', '{escape_sql_string(data['manufacturer'])}',
                    {data['capacity']}, {data['range_km']}
                )
            """)
            return jsonify({'success': True, 'aircraft_id': new_id})

    @app.route('/api/aircraft/<int:aircraft_id>', methods=['PUT'])
    def update_aircraft(aircraft_id):
        """Update an existing aircraft"""
        data = request.json
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                UPDATE {DEFAULT_AIRCRAFT_TABLE}
                SET aircraft_type = '{escape_sql_string(data['aircraft_type'])}',
                    manufacturer = '{escape_sql_string(data['manufacturer'])}',
                    capacity = {data['capacity']},
                    range_km = {data['range_km']}
                WHERE aircraft_id = {aircraft_id}
            """)
            return jsonify({'success': True})

    @app.route('/api/aircraft/<int:aircraft_id>', methods=['DELETE'])
    def delete_aircraft(aircraft_id):
        """Delete an aircraft"""
        conn = get_conn()
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {DEFAULT_AIRCRAFT_TABLE} WHERE aircraft_id = {aircraft_id}")
            return jsonify({'success': True})

    # ========== KPIs and Dashboard Routes ==========
    @app.route('/api/kpis')
    def get_kpis():
        """Get dashboard KPIs and analytics data"""
        conn = get_conn()
        with conn.cursor() as cursor:
            # Total flights
            cursor.execute(f"SELECT COUNT(*) FROM {DEFAULT_FLIGHTS_TABLE}")
            total_flights = cursor.fetchone()[0]
            
            # Total passengers
            cursor.execute(f"SELECT COUNT(*) FROM {DEFAULT_PASSENGERS_TABLE}")
            total_passengers = cursor.fetchone()[0]
            
            # Active flights
            cursor.execute(f"""
                SELECT COUNT(*) FROM {DEFAULT_FLIGHTS_TABLE} 
                WHERE status IN ('Scheduled', 'Boarding', 'Departed')
            """)
            active_flights = cursor.fetchone()[0]
            
            # Total airports
            cursor.execute(f"SELECT COUNT(*) FROM {DEFAULT_AIRPORTS_TABLE}")
            total_airports = cursor.fetchone()[0]
            
            # Total revenue
            cursor.execute(f"SELECT SUM(ticket_price) FROM {DEFAULT_PASSENGERS_TABLE}")
            total_revenue = float(cursor.fetchone()[0] or 0)
            
            # Average ticket price
            cursor.execute(f"SELECT AVG(ticket_price) FROM {DEFAULT_PASSENGERS_TABLE}")
            avg_ticket_price = float(cursor.fetchone()[0] or 0)
            
            # Top airports by flights
            cursor.execute(f"""
                SELECT origin as airport, COUNT(*) as count
                FROM {DEFAULT_FLIGHTS_TABLE}
                GROUP BY origin
                ORDER BY count DESC
                LIMIT 10
            """)
            top_airports = [{'airport': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Top airlines by passengers
            cursor.execute(f"""
                SELECT f.airline as airline, COUNT(p.passport_id) as count
                FROM {DEFAULT_FLIGHTS_TABLE} f
                JOIN {DEFAULT_PASSENGERS_TABLE} p ON f.flight_id = p.flight_id
                GROUP BY f.airline
                ORDER BY count DESC
                LIMIT 10
            """)
            top_airlines = [{'airline': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Flight status distribution
            cursor.execute(f"""
                SELECT status, COUNT(*) as count
                FROM {DEFAULT_FLIGHTS_TABLE}
                GROUP BY status
            """)
            flight_status = [{'status': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Passenger class distribution
            cursor.execute(f"""
                SELECT class, COUNT(*) as count
                FROM {DEFAULT_PASSENGERS_TABLE}
                GROUP BY class
            """)
            passenger_class = [{'class': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Revenue by class
            cursor.execute(f"""
                SELECT class, SUM(ticket_price) as revenue
                FROM {DEFAULT_PASSENGERS_TABLE}
                GROUP BY class
                ORDER BY revenue DESC
            """)
            revenue_by_class = [{'class': row[0], 'revenue': float(row[1])} for row in cursor.fetchall()]
            
            # Revenue by airline
            cursor.execute(f"""
                SELECT f.airline, SUM(p.ticket_price) as revenue
                FROM {DEFAULT_FLIGHTS_TABLE} f
                JOIN {DEFAULT_PASSENGERS_TABLE} p ON f.flight_id = p.flight_id
                GROUP BY f.airline
                ORDER BY revenue DESC
            """)
            revenue_by_airline = [{'airline': row[0], 'revenue': float(row[1])} for row in cursor.fetchall()]
            
            # Top routes by revenue
            try:
                cursor.execute(f"""
                    SELECT 
                        CONCAT(f.origin, ' → ', f.destination) as route,
                        SUM(p.ticket_price) as revenue,
                        COUNT(p.passport_id) as passengers
                    FROM {DEFAULT_FLIGHTS_TABLE} f
                    JOIN {DEFAULT_PASSENGERS_TABLE} p ON f.flight_id = p.flight_id
                    GROUP BY f.origin, f.destination
                    ORDER BY revenue DESC
                    LIMIT 10
                """)
                top_routes = [{'route': row[0], 'revenue': float(row[1]), 'passengers': row[2]} for row in cursor.fetchall()]
            except Exception:
                # Fallback using || operator
                cursor.execute(f"""
                    SELECT 
                        f.origin || ' → ' || f.destination as route,
                        SUM(p.ticket_price) as revenue,
                        COUNT(p.passport_id) as passengers
                    FROM {DEFAULT_FLIGHTS_TABLE} f
                    JOIN {DEFAULT_PASSENGERS_TABLE} p ON f.flight_id = p.flight_id
                    GROUP BY f.origin, f.destination
                    ORDER BY revenue DESC
                    LIMIT 10
                """)
                top_routes = [{'route': row[0], 'revenue': float(row[1]), 'passengers': row[2]} for row in cursor.fetchall()]
            
            # Average delay by airport
            cursor.execute(f"""
                SELECT origin as airport, AVG(delay_minutes) as avg_delay, COUNT(*) as flights
                FROM {DEFAULT_FLIGHTS_TABLE}
                WHERE delay_minutes > 0
                GROUP BY origin
                ORDER BY avg_delay DESC
                LIMIT 10
            """)
            delay_by_airport = [{'airport': row[0], 'avg_delay': float(row[1]), 'flights': row[2]} for row in cursor.fetchall()]
            
            # Average delay by airline
            cursor.execute(f"""
                SELECT airline, AVG(delay_minutes) as avg_delay, COUNT(*) as flights
                FROM {DEFAULT_FLIGHTS_TABLE}
                WHERE delay_minutes > 0
                GROUP BY airline
                ORDER BY avg_delay DESC
            """)
            delay_by_airline = [{'airline': row[0], 'avg_delay': float(row[1]), 'flights': row[2]} for row in cursor.fetchall()]
            
            return jsonify({
                'total_flights': total_flights,
                'total_passengers': total_passengers,
                'active_flights': active_flights,
                'total_airports': total_airports,
                'total_revenue': total_revenue,
                'avg_ticket_price': avg_ticket_price,
                'top_airports': top_airports,
                'top_airlines': top_airlines,
                'flight_status': flight_status,
                'passenger_class': passenger_class,
                'revenue_by_class': revenue_by_class,
                'revenue_by_airline': revenue_by_airline,
                'top_routes': top_routes,
                'delay_by_airport': delay_by_airport,
                'delay_by_airline': delay_by_airline
            })

