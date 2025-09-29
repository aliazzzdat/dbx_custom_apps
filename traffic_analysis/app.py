import os
import queue
import tempfile
import threading
import time
from collections import defaultdict, deque
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import supervision as sv
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sql import connect
from gradio import ChatMessage
from ultralytics import YOLO

CONFIG = {
    'SOURCE_POINTS': np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]]),
    'TARGET_WIDTH': 25,
    'TARGET_HEIGHT': 250,
    'DATABRICKS_HOST': os.getenv("DATABRICKS_HOST") or "e2-demo-field-eng.cloud.databricks.com",
    'HTTP_PATH': os.getenv("DATABRICKS_HTTP_PATH") or "/sql/1.0/warehouses/862f1d757f0424f7",
    'CATALOG': os.getenv("DATABRICKS_CATALOG") or "ali_azzouz",
    'SCHEMA': os.getenv("DATABRICKS_SCHEMA") or "traffic_analysis",
    'GENIE_SPACE_ID': "01f09ae643501c4cadb4e3ea77682487",
    'CACHE_DURATION': 30,
    'FRAME_UPDATE_INTERVAL': 5,
    'SPEED_VIOLATION_THRESHOLD': 130,
    'MIN_FRAMES_FOR_SPEED': 5,
    'COORDINATE_BUFFER_SIZE': 30,
    'DIRECTION_BOUNDARIES': {
        'left_threshold': 0.4,
        'right_threshold': 0.6
    }
}

CONFIG['TARGET_POINTS'] = np.array([
    [0, 0],
    [CONFIG['TARGET_WIDTH'] - 1, 0],
    [CONFIG['TARGET_WIDTH'] - 1, CONFIG['TARGET_HEIGHT'] - 1],
    [0, CONFIG['TARGET_HEIGHT'] - 1],
])

CONFIG['EVENTS_TABLE'] = f"{CONFIG['CATALOG']}.{CONFIG['SCHEMA']}.traffic_events"
CONFIG['STATS_TABLE'] = f"{CONFIG['CATALOG']}.{CONFIG['SCHEMA']}.traffic_aggregated"

cfg = None
table_update_queue = queue.Queue()
table_reader_running = False
_cached_events_data = None
_cache_timestamp = 0
conversation_id = None

try:
    token = os.getenv("DATABRICKS_TOKEN")
    username = os.getenv("DATABRICKS_USERNAME")
    
    if token or username:
        config_params = {"host": CONFIG['DATABRICKS_HOST']}
        if token:
            config_params["token"] = token
        elif username:
            config_params["username"] = username
        cfg = Config(**config_params)
    else:
        cfg = None
except Exception as e:
    cfg = None

def get_cache_key():
    return time.time()

def is_cache_valid():
    return (_cached_events_data is not None and 
            get_cache_key() - _cache_timestamp < CONFIG['CACHE_DURATION'])

SCHEMA_DEFINITIONS = {
    "traffic_events": {
        "columns": ["video_id", "frame_id", "timestamp", "car_id", "position_x", "position_y", "estimated_speed_kmh", "vehicle_type", "direction", "processing_datetime"],
        "dtypes": {"video_id": "object", "frame_id": "int64", "timestamp": "float64", "car_id": "int64", "position_x": "int64", "position_y": "int64", "estimated_speed_kmh": "int64", "vehicle_type": "object", "direction": "object", "processing_datetime": "datetime64[ns]"},
        "sql_types": {"video_id": "STRING", "frame_id": "BIGINT", "timestamp": "DOUBLE", "car_id": "BIGINT", "position_x": "BIGINT", "position_y": "BIGINT", "estimated_speed_kmh": "BIGINT", "vehicle_type": "STRING", "direction": "STRING", "processing_datetime": "TIMESTAMP"}
    },
    "traffic_aggregated": {
        "columns": ["metric", "value"],
        "dtypes": {"metric": "object", "value": "object"},
        "sql_types": {"metric": "STRING", "value": "STRING"}
    }
}

STATS_METRICS = {
    "total_vehicles": "Total Vehicles Detected", "total_cars": "Cars", "total_trucks": "Trucks",
    "total_events": "Total Detection Events", "events_per_vehicle": "Avg Events per Vehicle",
    "car_percentage": "Cars Percentage (%)", "truck_percentage": "Trucks Percentage (%)",
    "speed_violations": "Speed Violations (>130 km/h)", "speed_violation_rate": "Speed Violation Rate (%)",
    "avg_speed": "Average Speed (km/h)", "max_speed": "Maximum Speed (km/h)", "min_speed": "Minimum Speed (km/h)",
    "speed_std": "Speed Standard Deviation", "vehicles_per_minute": "Vehicles per Minute",
    "avg_vehicle_duration": "Avg Vehicle Duration (sec)", "fastest_vehicle_type": "Fastest Vehicle Type",
    "slowest_vehicle_type": "Slowest Vehicle Type", "vehicles_in": "Vehicles Going IN",
    "vehicles_out": "Vehicles Going OUT", "vehicles_stationary": "Stationary Vehicles",
    "in_out_ratio": "IN/OUT Ratio", "direction_balance": "Direction Balance"
}

def get_table_schema(table_name: str) -> dict:
    table_type = table_name.split('.')[-1].replace('traffic_', '')
    return SCHEMA_DEFINITIONS.get(f"traffic_{table_type}", {})

def create_dataframe(table_name: str, data: list = None) -> pd.DataFrame:
    schema = get_table_schema(table_name)
    columns = schema.get("columns", [])
    dtypes = schema.get("dtypes", {})
    
    if data is None or len(data) == 0:
        return pd.DataFrame({col: pd.Series(dtype=dtypes.get(col, "object")) for col in columns})
    df = pd.DataFrame(data, columns=columns)
    for col, dtype in dtypes.items():
        if col in df.columns:
            if dtype == "object":
                df[col] = df[col].fillna('unknown').astype(str)
            elif dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
            elif dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def get_databricks_connection():
    if cfg is None:
        return None
    try:
        connection = connect(
            server_hostname=CONFIG['DATABRICKS_HOST'],
            http_path=CONFIG['HTTP_PATH'],
            credentials_provider=lambda: cfg.authenticate,
        )
        return connection
    except Exception as e:
        return None

def test_databricks_connection():
    if cfg is None:
        return False, "No credentials configured"
    try:
        conn = get_databricks_connection()
        if conn is None:
            return False, "Connection failed"
        with conn.cursor() as cursor:
            cursor.execute("SELECT 'test' as connection_test")
            result = cursor.fetchone()
        conn.close()
        return True, "Connected"
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        return False, f"Error: {error_msg}"

def update_databricks_status():
    try:
        token = os.getenv("DATABRICKS_TOKEN")
        username = os.getenv("DATABRICKS_USERNAME")
        
        if not token and not username:
            return gr.update(value="🔴 No credentials configured")
        is_connected, message = test_databricks_connection()
        if is_connected:
            button_text = f"🟢 {message}"
        else:
            button_text = f"🔴 {message}"
        return gr.update(value=button_text)
    except Exception as e:
        return gr.update(value=f"🔴 Error: {str(e)}")

def table_exists(table_name: str, conn) -> bool:
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            return True
    except Exception:
        return False

def create_tables_if_not_exist():
    conn = get_databricks_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cursor:
            _create_table_if_not_exists(cursor, CONFIG['EVENTS_TABLE'], "events")
            _create_table_if_not_exists(cursor, CONFIG['STATS_TABLE'], "aggregated")
    except Exception:
        pass
    finally:
        if conn:
            conn.close()

def _create_table_if_not_exists(cursor, table_name: str, table_type: str):
    if not table_exists(table_name, cursor.connection):
        sql_types = get_table_schema(table_type).get("sql_types", {})
        columns_sql = ", ".join([f"{col} {dtype}" for col, dtype in sql_types.items()])
        cursor.execute(f"CREATE TABLE {table_name} ({columns_sql}) USING DELTA")

def read_table(table_name: str, conn) -> pd.DataFrame:
    if not table_exists(table_name, conn):
        create_tables_if_not_exist()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name}")
            
            try:
                arrow_table = cursor.fetchall_arrow()
                return arrow_table.to_pandas()
            except Exception:
                rows = cursor.fetchall()
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    return pd.DataFrame(rows, columns=columns)
                return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def batch_insert_to_table(table_name: str, df: pd.DataFrame, conn, operation_type: str = "data"):
    if df.empty or conn is None:
        return False, "No data to insert or no connection available"
    if not table_exists(table_name, conn):
        create_tables_if_not_exist()
    try:
        with conn.cursor() as cursor:
            columns = list(df.columns)
            columns_str = ", ".join(columns)
            values_list = []
            for _, row in df.iterrows():
                formatted_values = []
                for col in columns:
                    value = row[col]
                    if isinstance(value, str):
                        escaped_value = str(value).replace("'", "''")
                        formatted_values.append(f"'{escaped_value}'")
                    elif isinstance(value, (int, float)) and not pd.isna(value):
                        formatted_values.append(str(value))
                    elif pd.isna(value):
                        formatted_values.append("NULL")
                    else:
                        formatted_values.append(f"'{str(value)}'")
                values_list.append(f"({', '.join(formatted_values)})")
            batch_query = f"INSERT INTO {table_name} ({columns_str}) VALUES {', '.join(values_list)}"
            cursor.execute(batch_query)
        return True, f"Successfully inserted {len(df)} {operation_type} records"
    except Exception as e:
        return False, f"Error inserting {operation_type} to {table_name}: {str(e)}"

def read_tables():
    conn = get_databricks_connection()
    if conn is None:
        events_df = create_dataframe("events")
        stats_df = create_dataframe("aggregated")
        stats_df.loc[0] = ["No Databricks connection", "Check your credentials"]
        return events_df, stats_df
    try:
        with conn.cursor() as cursor:
            events_df = read_table(CONFIG['EVENTS_TABLE'], conn)
            stats_df = read_table(CONFIG['STATS_TABLE'], conn)
            return events_df, stats_df
    except Exception:
        events_df = create_dataframe("events")
        stats_df = create_dataframe("aggregated")
        stats_df.loc[0] = ["Error reading tables", "Connection failed"]
        return events_df, stats_df
    finally:
        if conn:
            conn.close()

def table_reader_thread():
    global table_reader_running, table_update_queue
    while table_reader_running:
        try:
            events_df, stats_df = read_tables()
            table_update_queue.put((events_df, stats_df))
            time.sleep(0.5)
        except Exception:
            time.sleep(1)

def start_table_reader():
    global table_reader_running
    table_reader_running = True
    thread = threading.Thread(target=table_reader_thread, daemon=True)
    thread.start()

def stop_table_reader():
    global table_reader_running
    table_reader_running = False

def get_latest_table_data():
    global table_update_queue
    try:
        return table_update_queue.get_nowait()
    except queue.Empty:
        return read_tables()

class SpeedEstimator:
    def __init__(self, confidence_threshold: float = 0.3, iou_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO("yolo11x.pt")
        self.byte_track = None
        self.view_transformer = None
        self.coordinates = defaultdict(lambda: deque(maxlen=CONFIG['COORDINATE_BUFFER_SIZE']))
        self.processing = False
        self.stop_processing = False
        self.realtime_events = create_dataframe("events")
        self.realtime_stats = create_dataframe("aggregated")
        self.video_id = None
        
    def initialize_for_video(self, video_path: str):
        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        self.byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=self.confidence_threshold)
        self.view_transformer = ViewTransformer(source=CONFIG['SOURCE_POINTS'], target=CONFIG['TARGET_POINTS'])
        self.coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        self.processing = True
        self.stop_processing = False
        self.realtime_events = create_dataframe("events")
        self.realtime_stats = create_dataframe("aggregated")
        self.video_id = os.path.basename(video_path).split('.')[0]
        return video_info
    
    def _determine_direction_from_original_position(self, x_position: int, frame_width: int) -> str:
        return "in" if x_position < frame_width // 2 else "out"
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float, video_id: str) -> Tuple[np.ndarray, dict]:
        if self.stop_processing:
            return frame, {}
        detections = self._detect_vehicles(frame)
        points = self._transform_detection_points(detections)
        self._update_vehicle_coordinates(detections, points)
        labels, frame_cars_data = self._process_vehicle_data(detections, points, frame_id, timestamp, video_id, frame.shape[1])
        annotated_frame = self._annotate_frame(frame, detections, labels)
        return annotated_frame, {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'cars_count': len(detections),
            'cars_data': frame_cars_data
        }

    def _detect_vehicles(self, frame: np.ndarray):
        result = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > self.confidence_threshold]
        polygon_zone = sv.PolygonZone(polygon=CONFIG['SOURCE_POINTS'])
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=self.iou_threshold)
        return self.byte_track.update_with_detections(detections=detections)

    def _transform_detection_points(self, detections):
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        return self.view_transformer.transform_points(points=points).astype(int)

    def _update_vehicle_coordinates(self, detections, points):
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            self.coordinates[tracker_id].append(y)

    def _process_vehicle_data(self, detections, points, frame_id, timestamp, video_id, frame_width):
        labels = []
        frame_cars_data = []
        class_names = self.model.names
        original_anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        processing_datetime = pd.Timestamp.now()
        for i, tracker_id in enumerate(detections.tracker_id):
            x_position_original = int(original_anchors[i][0]) if i < len(original_anchors) else 0
            direction = self._determine_direction_from_original_position(x_position_original, frame_width)
            speed = self._calculate_vehicle_speed(tracker_id)
            vehicle_type = self._get_vehicle_type(detections, i, class_names)
            label = f"#{tracker_id} {int(speed)} km/h {direction}" if speed > 0 else f"#{tracker_id} {direction}"
            labels.append(label)
            car_data = {
                'video_id': video_id,
                'frame_id': frame_id,
                'timestamp': timestamp,
                'car_id': tracker_id,
                'position_x': x_position_original,
                'position_y': int(original_anchors[i][1]) if i < len(original_anchors) else 0,
                'estimated_speed_kmh': int(speed),
                'vehicle_type': vehicle_type,
                'direction': direction,
                'processing_datetime': processing_datetime
            }
            frame_cars_data.append(car_data)
            new_row_df = create_dataframe("events", [car_data])
            self.realtime_events = pd.concat([self.realtime_events, new_row_df], ignore_index=True)
        
        return labels, frame_cars_data

    def _calculate_vehicle_speed(self, tracker_id):
        if len(self.coordinates[tracker_id]) < CONFIG['MIN_FRAMES_FOR_SPEED']:
            return 0
        coordinate_start = self.coordinates[tracker_id][-1]
        coordinate_end = self.coordinates[tracker_id][0]
        distance = abs(coordinate_start - coordinate_end)
        time_elapsed = len(self.coordinates[tracker_id]) / 30
        return distance / time_elapsed * 3.6 if time_elapsed > 0 else 0

    def _get_vehicle_type(self, detections, index, class_names):
        class_id = detections.class_id[index] if index < len(detections.class_id) else 2
        return class_names.get(class_id, "car")

    def _annotate_frame(self, frame, detections, labels):
        thickness = 2
        text_scale = 0.5
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness, trace_length=30, position=sv.Position.BOTTOM_CENTER
        )

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame
    
    def _calculate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics from a DataFrame and return as DataFrame format."""
        if df.empty:
            stats_data = self._get_empty_stats()
        else:
            vehicle_speeds = df.groupby('car_id')['estimated_speed_kmh'].max()
            vehicle_types = df.groupby('car_id')['vehicle_type'].first()
            vehicle_durations = df.groupby('car_id').size()
            
            total_vehicles = len(vehicle_speeds)
            speed_violations = len(vehicle_speeds[vehicle_speeds > CONFIG['SPEED_VIOLATION_THRESHOLD']])
            
            count_stats = self._calculate_count_statistics(df, total_vehicles, vehicle_types)
            speed_stats = self._calculate_speed_statistics(vehicle_speeds, total_vehicles)
            flow_stats = self._calculate_flow_statistics(df, total_vehicles, vehicle_durations, speed_violations)
            type_stats = self._calculate_type_statistics(vehicle_speeds, vehicle_types)
            direction_stats = self._calculate_direction_statistics(df)
            
            stats_data = {**count_stats, **speed_stats, **flow_stats, **type_stats, **direction_stats}
        
        expected_metrics = set(STATS_METRICS.keys())
        actual_metrics = set(stats_data.keys())
        
        missing_metrics = expected_metrics - actual_metrics
        if missing_metrics:
            raise ValueError(f"Missing required metrics in stats data: {missing_metrics}")
        
        extra_metrics = actual_metrics - expected_metrics
        if extra_metrics:
            raise ValueError(f"Unexpected metrics in stats data: {extra_metrics}")
        
        string_metrics = ['fastest_vehicle_type', 'slowest_vehicle_type', 'direction_balance']
        for metric in string_metrics:
            if metric in stats_data and not isinstance(stats_data[metric], str):
                raise ValueError(f"Metric '{metric}' should be string, got {type(stats_data[metric]).__name__}")
        stats_list = [(STATS_METRICS.get(key, key), str(value)) for key, value in stats_data.items()]
        return create_dataframe("aggregated", stats_list)

    def _get_empty_stats(self):
        empty_stats = {}
        for metric_key in STATS_METRICS.keys():
            if metric_key in ['fastest_vehicle_type', 'slowest_vehicle_type', 'direction_balance']:
                empty_stats[metric_key] = 'N/A'
            else:
                empty_stats[metric_key] = 0
        return empty_stats

    def _calculate_count_statistics(self, df: pd.DataFrame, total_vehicles: int, vehicle_types: pd.Series) -> dict:
        if total_vehicles == 0:
            return {
                'total_vehicles': 0, 'total_cars': 0, 'total_trucks': 0,
                'total_events': 0, 'events_per_vehicle': 0,
                'car_percentage': 0, 'truck_percentage': 0
            }
        
        total_cars = len(vehicle_types[vehicle_types == 'car'])
        total_trucks = len(vehicle_types[vehicle_types == 'truck'])
        total_events = len(df)
        events_per_vehicle = total_events / total_vehicles if total_vehicles > 0 else 0
        car_percentage = (total_cars / total_vehicles * 100) if total_vehicles > 0 else 0
        truck_percentage = (total_trucks / total_vehicles * 100) if total_vehicles > 0 else 0
        
        return {
            'total_vehicles': int(total_vehicles),
            'total_cars': int(total_cars),
            'total_trucks': int(total_trucks),
            'total_events': int(total_events),
            'events_per_vehicle': round(events_per_vehicle, 1),
            'car_percentage': round(car_percentage, 1),
            'truck_percentage': round(truck_percentage, 1)
        }

    def _calculate_speed_statistics(self, vehicle_speeds, total_vehicles):
        if total_vehicles == 0:
            return {'avg_speed': 0, 'max_speed': 0, 'min_speed': 0, 'speed_std': 0}
        
        return {
            'avg_speed': round(vehicle_speeds.mean(), 1),
            'max_speed': int(vehicle_speeds.max()),
            'min_speed': int(vehicle_speeds.min()),
            'speed_std': round(vehicle_speeds.std(), 1)
        }

    def _calculate_flow_statistics(self, df, total_vehicles, vehicle_durations, speed_violations):
        total_frames = df['frame_id'].max() - df['frame_id'].min() + 1 if len(df) > 0 else 1
        video_duration_minutes = total_frames / 30 / 60
        vehicles_per_minute = total_vehicles / video_duration_minutes if video_duration_minutes > 0 else 0
        avg_vehicle_duration = (vehicle_durations.mean() / 30) if total_vehicles > 0 else 0
        speed_violation_rate = (speed_violations / total_vehicles * 100) if total_vehicles > 0 else 0
        
        return {
            'vehicles_per_minute': round(vehicles_per_minute, 1),
            'avg_vehicle_duration': round(avg_vehicle_duration, 1),
            'speed_violations': int(speed_violations),
            'speed_violation_rate': round(speed_violation_rate, 1)
        }

    def _calculate_type_statistics(self, vehicle_speeds, vehicle_types):
        car_speeds = vehicle_speeds[vehicle_types == 'car']
        truck_speeds = vehicle_speeds[vehicle_types == 'truck']
        
        if len(car_speeds) > 0 and len(truck_speeds) > 0:
            fastest = 'car' if car_speeds.max() > truck_speeds.max() else 'truck'
            slowest = 'car' if car_speeds.mean() < truck_speeds.mean() else 'truck'
        elif len(car_speeds) > 0:
            fastest = slowest = 'car'
        elif len(truck_speeds) > 0:
            fastest = slowest = 'truck'
        else:
            fastest = slowest = 'N/A'
        
        return {
            'fastest_vehicle_type': fastest,
            'slowest_vehicle_type': slowest
        }

    def _calculate_direction_statistics(self, df):
        if 'direction' not in df.columns:
            return {'vehicles_in': 0, 'vehicles_out': 0, 'vehicles_stationary': 0, 'in_out_ratio': 0, 'direction_balance': 'N/A'}
        
        vehicle_directions = df.groupby('car_id')['direction'].last()
        vehicles_in = len(vehicle_directions[vehicle_directions == 'in'])
        vehicles_out = len(vehicle_directions[vehicle_directions == 'out'])
        vehicles_stationary = len(vehicle_directions[vehicle_directions == 'stationary'])
        
        in_out_ratio = round(vehicles_in / vehicles_out, 2) if vehicles_out > 0 else (float('inf') if vehicles_in > 0 else 0)
        
        if vehicles_in > vehicles_out:
            direction_balance = 'More vehicles going IN'
        elif vehicles_out > vehicles_in:
            direction_balance = 'More vehicles going OUT'
        else:
            direction_balance = 'Balanced traffic flow'
        
        return {
            'vehicles_in': int(vehicles_in),
            'vehicles_out': int(vehicles_out),
            'vehicles_stationary': int(vehicles_stationary),
            'in_out_ratio': in_out_ratio,
            'direction_balance': direction_balance
        }
    
    def _update_realtime_stats(self):
        try:
            if not self.realtime_events.empty:
                self.realtime_stats = self._calculate_stats(self.realtime_events)
            else:
                self.realtime_stats = create_dataframe("aggregated")
        except Exception:
            self.realtime_stats = create_dataframe("aggregated")
    
    def get_realtime_stats_dataframe(self) -> pd.DataFrame:
        if self.realtime_stats.empty:
            empty_df = create_dataframe("aggregated")
            no_data_row = {}
            for col in get_table_schema("aggregated").get("columns", []):
                if col == 'metric':
                    no_data_row[col] = "No data"
                elif col == 'value':
                    no_data_row[col] = "Process a video to see statistics"
                else:
                    no_data_row[col] = None
            empty_df.loc[0] = [no_data_row[col] for col in get_table_schema("aggregated").get("columns", [])]
            return empty_df
        
        return self.realtime_stats
    
    def get_realtime_events_dataframe(self) -> pd.DataFrame:
        if self.realtime_events.empty:
            empty_df = create_dataframe("events")
            no_data_row = {}
            for col in get_table_schema("events").get("columns", []):
                if col == 'video_id':
                    no_data_row[col] = "No data"
                elif col in ['frame_id', 'car_id', 'position_x', 'position_y', 'estimated_speed_kmh']:
                    no_data_row[col] = 0
                elif col == 'timestamp':
                    no_data_row[col] = 0.0
                elif col == 'vehicle_type':
                    no_data_row[col] = "Process a video to see events"
                elif col == 'direction':
                    no_data_row[col] = "unknown"
                elif col == 'processing_datetime':
                    no_data_row[col] = pd.NaT
                else:
                    no_data_row[col] = None
            empty_df.loc[0] = [no_data_row[col] for col in get_table_schema("events").get("columns", [])]
            return empty_df
        
        return self.realtime_events
    
    def append_to_databricks(self):
        conn = get_databricks_connection()
        if not conn:
            return False, "No Databricks connection available"
        
        if self.realtime_events.empty:
            return False, "No events data to append"
        
        try:
            success, message = batch_insert_to_table(CONFIG['EVENTS_TABLE'], self.realtime_events, conn, "events")
            if not success:
                return False, f"Error appending to Databricks: {message}"
            return True, f"Successfully appended {len(self.realtime_events)} events to Databricks"
        except Exception as e:
            return False, f"Error appending to Databricks: {str(e)}"
        finally:
            if conn:
                conn.close()
    
speed_estimator = SpeedEstimator()

current_processed_video = None
out = None

def process_video(video_file, confidence_threshold, iou_threshold):
    if video_file is None:
        yield None, "Please upload a video file first.", gr.update(), None, gr.update(interactive=False), gr.update(interactive=False)
        return
    
    try:
        _initialize_speed_estimator(confidence_threshold, iou_threshold)
        video_info = speed_estimator.initialize_for_video(video_file)
        video_id = os.path.basename(video_file).split('.')[0]
        
        _setup_output_video(video_info)
        
        frame_generator = sv.get_video_frames_generator(source_path=video_file)
        frame_count = 0
        total_frames = int(video_info.total_frames) if hasattr(video_info, 'total_frames') else 0
        
        try:
            for frame in frame_generator:
                if speed_estimator.stop_processing:
                    yield _get_final_yield_data("Processing stopped by user.")
                    break
                    
                timestamp = frame_count / video_info.fps
                annotated_frame, frame_data = speed_estimator.process_frame(frame, frame_count, timestamp, video_id)
                
                if speed_estimator.stop_processing:
                    yield _get_final_yield_data("Processing stopped by user.")
                    break
                
                _write_frame_to_video(annotated_frame)
                frame_count += 1
                
                if frame_count % CONFIG['FRAME_UPDATE_INTERVAL'] == 0:
                    speed_estimator._update_realtime_stats()
                    progress = _create_progress_message(frame_count, total_frames, frame_data.get('cars_count', 0))
                    yield _get_realtime_yield_data(progress, annotated_frame)
        
        finally:
            _cleanup_video_writer()
        
        speed_estimator._update_realtime_stats()
        yield _get_final_yield_data(f"Processing complete! {len(speed_estimator.get_realtime_events_dataframe())} events recorded.")
    
    except Exception as e:
        yield None, f"Error processing video: {str(e)}", None, None, gr.update(interactive=False), gr.update(interactive=False)

def _initialize_speed_estimator(confidence_threshold, iou_threshold):
    speed_estimator.stop_processing = False
    speed_estimator.processing = True
    speed_estimator.confidence_threshold = confidence_threshold
    speed_estimator.iou_threshold = iou_threshold

def _setup_output_video(video_info):
    global current_processed_video, out
    if current_processed_video and os.path.exists(current_processed_video):
        os.unlink(current_processed_video)
    
    current_processed_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(current_processed_video, fourcc, video_info.fps, (video_info.width, video_info.height))

def _write_frame_to_video(annotated_frame):
    global out
    if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
        out.write(annotated_frame)
    else:
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

def _cleanup_video_writer():
    global out
    out.release()

def _create_progress_message(frame_count, total_frames, cars_count):
    progress = f"Processing frame {frame_count}"
    if total_frames > 0:
        progress += f" of {total_frames} ({100*frame_count/total_frames:.1f}%)"
    progress += f" | {cars_count} vehicles detected"
    return progress


def _get_final_yield_data(message):
    final_stats_df = speed_estimator.get_realtime_stats_dataframe()
    final_events_df = speed_estimator.get_realtime_events_dataframe()
    append_btn_state = gr.update(interactive=has_realtime_data())
    show_video_btn_state = gr.update(interactive=True)
    return final_stats_df, message, None, final_events_df, show_video_btn_state, append_btn_state

def _get_realtime_yield_data(progress, annotated_frame):
    current_stats_df = speed_estimator.get_realtime_stats_dataframe()
    current_events_df = speed_estimator.get_realtime_events_dataframe()
    
    from PIL import Image
    if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
        pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(annotated_frame)
    
    append_btn_state = gr.update(interactive=has_realtime_data())
    show_video_btn_state = gr.update(interactive=False)
    return current_stats_df, progress, pil_image, current_events_df, show_video_btn_state, append_btn_state

def stop_processing():
    speed_estimator.stop_processing = True
    speed_estimator.processing = False
    return "Processing stopped. The current frame processing will stop at the next frame."


def reset_realtime_statistics():
    speed_estimator.realtime_events = create_dataframe("events")
    speed_estimator.realtime_stats = create_dataframe("aggregated")
    speed_estimator.coordinates = defaultdict(lambda: deque(maxlen=30))
    speed_estimator.processing = False
    speed_estimator.stop_processing = False
    speed_estimator.video_id = None
    
    return speed_estimator.get_realtime_stats_dataframe(), speed_estimator.get_realtime_events_dataframe()

def show_processed_video():
    global current_processed_video
    if current_processed_video and os.path.exists(current_processed_video):
        status_message = f"✅ Processed video loaded successfully: {os.path.basename(current_processed_video)}"
        return current_processed_video, status_message
    else:
        status_message = "⚠️ No processed video available. Please process a video first."
        return None, status_message


def _download_video_from_databricks(filename: str) -> tuple[str, str]:
    if cfg is None:
        raise Exception("Databricks credentials not configured. Please set DATABRICKS_TOKEN or DATABRICKS_USERNAME environment variable.")
    
    w = WorkspaceClient(config=cfg)
    volume_path = f"/Volumes/{CONFIG['CATALOG']}/{CONFIG['SCHEMA']}/videos/{filename}"
    
    response = w.files.download(volume_path)
    file_data = response.contents.read()
    
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.write(file_data)
    temp_video.close()
    
    return temp_video.name, f"Video downloaded successfully from {volume_path}!"


def load_sample_video():
    global current_processed_video
    reset_stats, reset_events = reset_realtime_statistics()
    
    try:
        temp_path, success_message = _download_video_from_databricks("vehicles.mp4")
        current_processed_video = temp_path
        
        return temp_path, success_message, reset_stats, reset_events, gr.update(interactive=True), gr.update(interactive=False)
        
    except Exception as e:
        return None, f"Sample video not found. Volume download failed: {str(e)}. Please check if the video exists in /Volumes/{CONFIG['CATALOG']}/{CONFIG['SCHEMA']}/videos/.", reset_stats, reset_events, gr.update(interactive=False), gr.update(interactive=False)


def load_processed_video():
    global current_processed_video
    reset_stats, reset_events = reset_realtime_statistics()
    
    empty_events = create_dataframe("events")
    empty_stats = create_dataframe("aggregated")
    empty_stats.loc[0] = ["No data collected", "This is a pre-processed video"]
    
    try:
        temp_path, success_message = _download_video_from_databricks("processed_video.mp4")
        current_processed_video = temp_path
        
        status_message = f"✅ {success_message}\n📹 Video loaded in output player.\n📊 Real-time statistics reset.\nℹ️ Note: This video contains no real-time data collection - it's a pre-processed sample."
        
        return None, status_message, temp_path, empty_stats, empty_events, gr.update(interactive=True), gr.update(interactive=False)
        
    except Exception as e:
        empty_stats.loc[0] = ["Error", f"Failed to load video: {str(e)}"]
        return None, f"Processed video not found. Volume download failed: {str(e)}. Please check if the video exists in /Volumes/{CONFIG['CATALOG']}/{CONFIG['SCHEMA']}/videos/processed_video.mp4.", None, empty_stats, empty_events, gr.update(interactive=False), gr.update(interactive=False)


def clear_tables():
    global current_processed_video
    
    if current_processed_video and os.path.exists(current_processed_video):
        try:
            os.unlink(current_processed_video)
        except Exception:
            pass
    
    current_processed_video = None
    empty_stats, empty_events = reset_realtime_statistics()
    
    status_message = "✅ Results cleared successfully! All real-time data has been reset. Upload a new video to start processing."
    
    return (
        empty_stats,
        empty_events,
        gr.update(interactive=False),
        gr.update(interactive=False),
        status_message,
        None,
        None,
        None
    )


def cleanup_temp_files():
    global current_processed_video
    
    if current_processed_video and os.path.exists(current_processed_video):
        try:
            os.unlink(current_processed_video)
        except Exception:
            pass
    
    current_processed_video = None


def get_query_result(statement_id):
    try:
        w = WorkspaceClient(config=cfg)
        result = w.statement_execution.get_statement(statement_id)
        return pd.DataFrame(
            result.result.data_array, 
            columns=[i.name for i in result.manifest.schema.columns]
        )
    except Exception:
        return pd.DataFrame()

def process_genie_response(response):
    messages = []
    
    for i, attachment in enumerate(response.attachments):
        if hasattr(attachment, 'text') and attachment.text:
            message = ChatMessage(role="assistant", content=attachment.text.content)
            messages.append(message)
        elif hasattr(attachment, 'query') and attachment.query:
            try:
                data = get_query_result(response.query_result.statement_id)
                content = attachment.query.description
                if not data.empty:
                    content += f"\n\n**Query Result:**\n{data.to_string()}"
                if attachment.query.query:
                    content += f"\n\n**Generated SQL:**\n```sql\n{attachment.query.query}\n```"
                
                message = ChatMessage(role="assistant", content=content)
                messages.append(message)
            except Exception as e:
                message = ChatMessage(role="assistant", content=f"Query execution failed: {str(e)}")
                messages.append(message)
    
    if not messages:
        messages.append(ChatMessage(role="assistant", content="I received your message but couldn't process the response properly."))
    
    return messages


def chat_with_genie(message, history):
    global conversation_id
    
    if not message.strip():
        return history, ""
    
    try:
        if cfg is None:
            error_msg = "Databricks credentials not configured. Please set DATABRICKS_TOKEN or DATABRICKS_USERNAME environment variable."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        w = WorkspaceClient(config=cfg)
        history.append({"role": "user", "content": message})
        
        if conversation_id:
            conversation = w.genie.create_message_and_wait(
                CONFIG['GENIE_SPACE_ID'], conversation_id, message
            )
        else:
            conversation = w.genie.start_conversation_and_wait(CONFIG['GENIE_SPACE_ID'], message)
            conversation_id = conversation.conversation_id
        
        messages = process_genie_response(conversation)
        
        for i, msg in enumerate(messages):
            if msg.role == "assistant":
                if not isinstance(msg, ChatMessage):
                    continue
                if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                    continue
                
                if not isinstance(msg.content, str):
                    msg.content = str(msg.content)
                
                msg.content = sanitize_message_content(msg.content)
                history.append({"role": msg.role, "content": msg.content})
        
        for i, msg in enumerate(history):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                history = [m for m in history if isinstance(m, dict) and 'role' in m and 'content' in m]
                break
        
        return history, ""
        
    except Exception as e:
        error_msg = f"Error communicating with Genie: {str(e)}"
        history.append({"role": "assistant", "content": sanitize_message_content(error_msg)})
        
        for i, msg in enumerate(history):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                history = [m for m in history if isinstance(m, dict) and 'role' in m and 'content' in m]
                break
        
        return history, ""


def clear_chat():
    return [], ""

def reset_conversation():
    global conversation_id
    conversation_id = None
    return [], ""

def sanitize_message_content(content):
    if not isinstance(content, str):
        content = str(content)
    
    content = content.replace('\x00', '')
    content = content.strip()
    
    return content

def has_realtime_data():
    return not speed_estimator.realtime_events.empty

def append_to_databricks_with_status():
    if not has_realtime_data():
        return "No real-time data available. Please process a video first to collect data.", gr.update(interactive=False), update_databricks_status()
    
    success, message = speed_estimator.append_to_databricks()
    
    if success:
        clear_chart_cache()
        return message, gr.update(interactive=False), update_databricks_status()
    else:
        return message, gr.update(interactive=False), update_databricks_status()



def delete_databricks_tables():
    """Delete Databricks tables and return status message"""
    try:
        if cfg is None:
            return "❌ No Databricks connection available. Please configure your credentials first."
        
        is_connected, message = test_databricks_connection()
        if not is_connected:
            return f"❌ Cannot connect to Databricks: {message}"
        conn = get_databricks_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {CONFIG['EVENTS_TABLE']}")
                cursor.execute(f"DROP TABLE IF EXISTS {CONFIG['STATS_TABLE']}")
        except Exception as e:
            return f"❌ Error executing DROP TABLE commands: {str(e)}"
        finally:
            if conn:
                conn.close()
        
        clear_chart_cache()
        
        return "✅ Successfully deleted all Databricks tables (traffic_events and traffic_aggregated). Tables will be recreated when new data is inserted."
        
    except Exception as e:
        return f"❌ Error deleting tables: {str(e)}"

def send_example_question(question, history):
    if not question.strip():
        return history, ""
    
    return chat_with_genie(question, history)

def ask_average_speed(history):
    return send_example_question("What's the average speed of all vehicles detected?", history)

def ask_speed_violations(history):
    return send_example_question("How many speed violations occurred and what's the violation rate?", history)

def ask_vehicle_types(history):
    return send_example_question("What's the distribution of vehicle types (cars vs trucks)?", history)

def ask_fastest_vehicle(history):
    return send_example_question("Show me the fastest vehicle detected and its details", history)

def _create_empty_chart(message: str):
    return go.Figure().add_annotation(
        text=message, 
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=14)
    )

def _get_events_data(use_cache=True):
    global _cached_events_data, _cache_timestamp
    
    if use_cache and is_cache_valid():
        return _cached_events_data
    
    try:
        events_df, _ = read_tables()
        _cached_events_data = events_df
        _cache_timestamp = get_cache_key()
        return events_df
    except Exception:
        return pd.DataFrame()

def speed_distribution_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data in traffic_events table<br>Process a video and append to Databricks to see charts")
    
    vehicle_speeds = events_df.groupby('car_id', observed=True)['estimated_speed_kmh'].max()
    fig = px.histogram(
        x=vehicle_speeds.values, nbins=20, title="Speed Distribution of Detected Vehicles",
        labels={'x': 'Speed (km/h)', 'y': 'Number of Vehicles'}, color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False, height=400, title_x=0.5)
    return fig

def vehicle_type_pie_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data in traffic_events table<br>Process a video and append to Databricks to see charts")
    
    vehicle_types = events_df.groupby('car_id', observed=True)['vehicle_type'].first()
    type_counts = vehicle_types.value_counts()
    
    fig = px.pie(
        values=type_counts.values, names=type_counts.index, title="Vehicle Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=400, title_x=0.5)
    return fig


def speed_over_time_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data available in traffic_events table")
    
    avg_speed_per_frame = events_df.groupby('frame_id', observed=True)['estimated_speed_kmh'].mean().reset_index()
    
    fig = px.line(
        avg_speed_per_frame, x='frame_id', y='estimated_speed_kmh', title="Average Speed Over Time",
        labels={'frame_id': 'Frame Number', 'estimated_speed_kmh': 'Average Speed (km/h)'}
    )
    fig.update_layout(height=400, title_x=0.5)
    return fig

def speed_violations_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data available in traffic_events table")
    
    vehicle_speeds = events_df.groupby('car_id', observed=True)['estimated_speed_kmh'].max()
    vehicle_types = events_df.groupby('car_id', observed=True)['vehicle_type'].first()
    
    vehicle_data = pd.DataFrame({
        'vehicle_id': vehicle_speeds.index,
        'max_speed': vehicle_speeds.values,
        'vehicle_type': vehicle_types.values,
        'speed_category': pd.cut(vehicle_speeds.values, 
                               bins=[0, 80, 100, 130, float('inf')], 
                               labels=['Slow (0-80)', 'Normal (80-100)', 'Fast (100-130)', 'Speeding (>130)'])
    })
    
    category_counts = vehicle_data.groupby(['speed_category', 'vehicle_type'], observed=True).size().reset_index(name='count')
    
    fig = px.bar(
        category_counts,
        x='speed_category',
        y='count',
        color='vehicle_type',
        title="Speed Violations by Vehicle Type",
        labels={'speed_category': 'Speed Category', 'count': 'Number of Vehicles'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, title_x=0.5)
    return fig


def traffic_flow_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data available in traffic_events table")
    
    vehicles_per_frame = events_df.groupby('frame_id', observed=True).size().reset_index(name='vehicle_count')
    
    fig = px.line(
        vehicles_per_frame,
        x='frame_id',
        y='vehicle_count',
        title="Traffic Flow Over Time",
        labels={'frame_id': 'Frame Number', 'vehicle_count': 'Number of Vehicles'},
        color_discrete_sequence=['#ff7f0e']
    )
    fig.update_layout(height=400, title_x=0.5)
    return fig

def vehicle_speed_comparison(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data available in traffic_events table")
    
    vehicle_speeds = events_df.groupby('car_id', observed=True)['estimated_speed_kmh'].max()
    vehicle_types = events_df.groupby('car_id', observed=True)['vehicle_type'].first()
    
    speed_data = pd.DataFrame({
        'vehicle_type': vehicle_types.values,
        'max_speed': vehicle_speeds.values
    })
    
    fig = px.box(
        speed_data,
        x='vehicle_type',
        y='max_speed',
        title="Speed Distribution by Vehicle Type",
        labels={'vehicle_type': 'Vehicle Type', 'max_speed': 'Maximum Speed (km/h)'},
        color='vehicle_type',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_layout(height=400, title_x=0.5, showlegend=False)
    return fig


def direction_metrics_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data in traffic_events table<br>Process a video and append to Databricks to see charts")
    
    vehicle_directions = events_df.groupby('car_id', observed=True)['direction'].last().value_counts()
    
    fig = px.pie(
        values=vehicle_directions.values,
        names=vehicle_directions.index,
        title="Traffic Direction Distribution",
        color_discrete_map={
            'in': '#2E8B57',
            'out': '#FF6347',
            'stationary': '#FFD700',
            'unknown': '#808080'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, title_x=0.5)
    return fig


def direction_over_time_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data in traffic_events table<br>Process a video and append to Databricks to see charts")
    
    if 'direction' not in events_df.columns:
        return _create_empty_chart("Direction column not found in data<br>This may be due to older data format")
    
    if 'frame_id' not in events_df.columns:
        return _create_empty_chart("Frame ID column not found in data<br>This may be due to data format issues")
    
    valid_directions = events_df['direction'].notna() & (events_df['direction'] != '') & (events_df['direction'] != 'unknown')
    if not valid_directions.any():
        return _create_empty_chart("No valid direction data found<br>All direction values are null or unknown")
    
    events_df['frame_interval'] = (events_df['frame_id'] / 30).round().astype(int) * 30
    direction_counts = events_df.groupby(['frame_interval', 'direction'], observed=True).size().reset_index(name='count')
    
    if direction_counts.empty:
        return _create_empty_chart("No direction data available<br>Check if direction column exists and has valid values")
    
    fig = px.line(
        direction_counts,
        x='frame_interval',
        y='count',
        color='direction',
        title="Traffic Direction Flow Over Time",
        labels={'frame_interval': 'Frame Number', 'count': 'Number of Vehicles'},
        color_discrete_map={
            'in': '#2E8B57',
            'out': '#FF6347',
            'stationary': '#FFD700',
            'unknown': '#808080'
        }
    )
    fig.update_layout(height=400, title_x=0.5)
    return fig

def traffic_forecast_chart(events_df=None):
    if events_df is None:
        events_df = _get_events_data()
    
    if events_df.empty:
        return _create_empty_chart("No data in traffic_events table<br>Process a video and append to Databricks to see charts")
    
    if 'timestamp' not in events_df.columns:
        return _create_empty_chart("Timestamp column not found in data<br>This may be due to data format issues")
    
    valid_timestamps = events_df['timestamp'].notna() & (events_df['timestamp'] > 0)
    if not valid_timestamps.any():
        return _create_empty_chart("No valid timestamp data found<br>All timestamp values are null or zero")
    
    events_df['time_minutes'] = events_df['frame_id']
    traffic_flow = events_df.groupby('time_minutes', observed=True).size().reset_index(name='vehicle_count')
    
    if len(traffic_flow) < 3:
        return _create_empty_chart("Insufficient data for forecasting<br>Need at least 3 data points")
    
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    X = traffic_flow['time_minutes'].values.reshape(-1, 1)
    y = traffic_flow['vehicle_count'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    historical_data = traffic_flow.copy()
    historical_data['type'] = 'Historical'
    
    last_time = traffic_flow['time_minutes'].max()
    total_input_time = last_time - traffic_flow['time_minutes'].min()
    forecast_duration_pct = 0.25
    forecast_duration = max(1, int(total_input_time * forecast_duration_pct))
    forecast_times = np.arange(last_time + 1, last_time + forecast_duration + 1).reshape(-1, 1)
    forecast_values = model.predict(forecast_times)
    
    forecast_data = pd.DataFrame({
        'time_minutes': forecast_times.flatten(),
        'vehicle_count': forecast_values,
        'type': 'Forecast'
    })
    
    combined_data = pd.concat([historical_data, forecast_data], ignore_index=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data['time_minutes'],
        y=historical_data['vehicle_count'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['time_minutes'],
        y=forecast_data['vehicle_count'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    fig.add_vline(
        x=last_time + 0.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"Traffic Flow Forecast (Next {forecast_duration} Minutes - {forecast_duration_pct*100}% of Input Time)",
        xaxis_title="Time (minutes)",
        yaxis_title="Number of Vehicles",
        height=400,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Trend: y = {slope:.2f}x + {intercept:.2f}<br>R² = {r2:.3f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    return fig

def load_charts():
    events_df = _get_events_data(use_cache=True)
    
    return (
        speed_distribution_chart(events_df),
        vehicle_type_pie_chart(events_df),
        speed_over_time_chart(events_df),
        speed_violations_chart(events_df),
        traffic_flow_chart(events_df),
        vehicle_speed_comparison(events_df),
        direction_metrics_chart(events_df),
        direction_over_time_chart(events_df),
        traffic_forecast_chart(events_df)
    )

def load_raw_data():
    events_df, _ = get_latest_table_data()
    
    if events_df.empty:
        events_df = create_dataframe("events")
        stats_df = create_dataframe("aggregated")
        stats_df.loc[0] = ["No data in tables", "Tables are empty"]
    else:
        stats_df = speed_estimator._calculate_stats(events_df)
        
        conn = get_databricks_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(f"DELETE FROM {CONFIG['STATS_TABLE']}")
                
                success, message = batch_insert_to_table(CONFIG['STATS_TABLE'], stats_df, conn, "aggregated_stats")
            except Exception as e:
                pass
            finally:
                conn.close()
    
    return stats_df, events_df

def clear_chart_cache():
    global _cached_events_data, _cache_timestamp
    _cached_events_data = None
    _cache_timestamp = 0

def create_interface():
    with gr.Blocks(title="Vehicle Speed Estimation", css="""
        .refresh_status_btn {
            width: 40px !important;
            min-width: 40px !important;
        }
        .databricks-status {
            font-weight: bold;
            font-size: 12px;
        }
    """) as interface:
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# 🚗 Vehicle Speed Estimation App")
                gr.Markdown("📹 Upload a video to detect vehicles and estimate their speeds in real-time.")
                gr.Markdown("[Based on Roboflow Supervision speed estimation example using Yolo](https://github.com/roboflow/sports/tree/main/examples/soccer)")
            with gr.Column(scale=1, min_width=200):
                with gr.Row():
                    databricks_status_btn = gr.Button(
                        value="🔴 Testing...", 
                        variant="secondary", 
                        size="sm",
                        interactive=False,
                        elem_classes="databricks-status"
                    )
                    refresh_status_btn = gr.Button(
                        value="🔄", 
                        variant="secondary", 
                        size="sm",
                        elem_id="refresh_status_btn"
                    )
        
        with gr.Tab("🎬 Video Processing & Stats"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(
                        label="📹 Input Video", 
                        height=300,
                        format="mp4"
                    )
                    
                    confidence_threshold = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                        label="Confidence Threshold",
                        visible=False
                    )
                    iou_threshold = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                        label="IOU Threshold",
                        visible=False
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("▶️ Process Video", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop Processing", variant="stop")
                    
                    with gr.Row():
                        load_sample_btn = gr.Button("📤 Sample Video", variant="secondary")
                        load_processed_btn = gr.Button("📥 Sample Processed Video", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear Results", variant="secondary")
                    
                    
                    with gr.Row():
                        append_databricks_btn = gr.Button("💾 Save data to Databricks", variant="primary", interactive=False)
                        show_video_btn = gr.Button("🎬 Show Processed Video", variant="secondary", interactive=False)
                    
                    status_text = gr.Textbox(label="📊 Status", interactive=False)
                
                with gr.Column():
                    current_frame = gr.Image(
                        label="🖼️ Current Frame Being Processed", 
                        height=300,
                        show_label=True
                    )
                    
                    output_video = gr.Video(
                        label="🎬 Processed Video", 
                        height=300,
                        format="mp4"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Real-time Aggregated Statistics")
                    gr.Markdown("Live analysis of detected vehicles and traffic patterns during processing")
                    stats_table = gr.Dataframe(
                        headers=get_table_schema("aggregated").get("columns", []),
                        datatype=[get_table_schema("aggregated").get("dtypes", {})[col] for col in get_table_schema("aggregated").get("columns", [])],
                        label="📈 Real-time Analysis Results",
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Column():
                    gr.Markdown("### 🚗 Real-time Events Data")
                    gr.Markdown("Live frame-by-frame data for each detected vehicle")
                    events_table = gr.Dataframe(
                        headers=get_table_schema("events").get("columns", []),
                        datatype=[get_table_schema("events").get("dtypes", {})[col] for col in get_table_schema("events").get("columns", [])],
                        label="🚗 Live Vehicle Detection Events",
                        interactive=False,
                        wrap=True
                    )
        
        with gr.Tab("📋 Raw Data", id="raw_data_tab") as raw_data_tab:
            gr.Markdown("### 📋 Raw Data from Databricks")
            gr.Markdown("Data stored in Databricks tables - events and aggregated statistics")
            
            
            with gr.Row():
                gr.Markdown("#### 📊 traffic_aggregated Table")
            with gr.Row():
                raw_stats_table = gr.Dataframe(
                    headers=get_table_schema("aggregated").get("columns", []),
                    datatype=[get_table_schema("aggregated").get("dtypes", {})[col] for col in get_table_schema("aggregated").get("columns", [])],
                    label="📈 Aggregated Statistics",
                    interactive=False,
                    wrap=True
                )
            
            with gr.Row():
                gr.Markdown("#### 🚗 traffic_events Table")
            with gr.Row():
                raw_events_table = gr.Dataframe(
                    headers=get_table_schema("events").get("columns", []),
                    datatype=[get_table_schema("events").get("dtypes", {})[col] for col in get_table_schema("events").get("columns", [])],
                    label="🚗 Vehicle Events",
                    interactive=False,
                    wrap=True
                )
        
        with gr.Tab("📊 Data Visualization", id="data_viz_tab") as data_viz_tab:
            gr.Markdown("### 📈 Traffic Analysis Charts")
            gr.Markdown("Interactive visualizations from Databricks traffic_events and traffic_aggregated tables")
            
            
            with gr.Row():
                with gr.Column():
                    speed_dist_chart = gr.Plot(label="Speed Distribution")
                    vehicle_type_chart = gr.Plot(label="Vehicle Type Distribution")
            
            with gr.Row():
                with gr.Column():
                    speed_time_chart = gr.Plot(label="Speed Over Time")
                    violations_chart = gr.Plot(label="Speed Violations Analysis")
            
            with gr.Row():
                with gr.Column():
                    traffic_flow_chart = gr.Plot(label="Traffic Flow Over Time")
                    speed_comparison_chart = gr.Plot(label="Speed Comparison by Vehicle Type")
            
            with gr.Row():
                with gr.Column():
                    direction_dist_chart = gr.Plot(label="Traffic Direction Distribution")
                    direction_flow_chart = gr.Plot(label="Direction Flow Over Time")
            
            with gr.Row():
                with gr.Column():
                    forecast_chart = gr.Plot(label="Traffic Flow Forecast")
        
        with gr.Tab("🤖 AI Assistant"):
            gr.Markdown("### 🤖 Genie AI Assistant")
            gr.Markdown("Ask questions about your traffic analysis data using natural language")
            
            chatbot = gr.Chatbot(
                label="💬 Chat with Genie",
                height=400,
                show_label=True,
                container=True,
                type="messages"
            )
            
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question about your traffic data...",
                    label="Message",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary", scale=1)
            
            gr.Markdown("**💡 Quick Questions:**")
            with gr.Row():
                example_btn1 = gr.Button("🚗 Average Speed", variant="secondary", size="sm")
                example_btn2 = gr.Button("⚡ Speed Violations", variant="secondary", size="sm")
                example_btn3 = gr.Button("📊 Vehicle Types", variant="secondary", size="sm")
                example_btn4 = gr.Button("🏃 Fastest Vehicle", variant="secondary", size="sm")
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Conversation", variant="stop", size="sm")
        
        with gr.Tab("⚙️ Config"):
            gr.Markdown("### ⚙️ Configuration & Maintenance")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**⚠️ Warning:** These operations will permanently delete data from Databricks tables.")
                    
                    with gr.Row():
                        delete_tables_btn = gr.Button("🗑️ Delete All Tables", variant="stop", size="lg")
                    
                    config_status = gr.Textbox(
                        label="📊 Status", 
                        interactive=False,
                        value="Ready to perform database operations"
                    )
        
        process_btn.click(
            fn=process_video,
            inputs=[input_video, confidence_threshold, iou_threshold],
            outputs=[stats_table, status_text, current_frame, events_table, show_video_btn, append_databricks_btn]
        )
        
        stop_btn.click(
            fn=stop_processing,
            outputs=[status_text]
        )
        
        load_sample_btn.click(
            fn=load_sample_video,
            outputs=[input_video, status_text, stats_table, events_table, show_video_btn, append_databricks_btn]
        )
        
        load_processed_btn.click(
            fn=load_processed_video,
            outputs=[input_video, status_text, output_video, stats_table, events_table, show_video_btn, append_databricks_btn]
        )
        
        clear_btn.click(
            fn=clear_tables,
            outputs=[stats_table, events_table, show_video_btn, append_databricks_btn, status_text, input_video, output_video, current_frame]
        )
        
        
        
        append_databricks_btn.click(
            fn=append_to_databricks_with_status,
            outputs=[status_text, show_video_btn, databricks_status_btn]
        )
        
        
        delete_tables_btn.click(
            fn=delete_databricks_tables,
            outputs=[config_status]
        )
        
        show_video_btn.click(
            fn=show_processed_video,
            outputs=[output_video, status_text]
        )
        
        send_btn.click(
            fn=chat_with_genie,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            fn=chat_with_genie,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot, chat_input]
        )
        
        example_btn1.click(
            fn=ask_average_speed,
            inputs=[chatbot],
            outputs=[chatbot, chat_input]
        )
        
        example_btn2.click(
            fn=ask_speed_violations,
            inputs=[chatbot],
            outputs=[chatbot, chat_input]
        )
        
        example_btn3.click(
            fn=ask_vehicle_types,
            inputs=[chatbot],
            outputs=[chatbot, chat_input]
        )
        
        example_btn4.click(
            fn=ask_fastest_vehicle,
            inputs=[chatbot],
            outputs=[chatbot, chat_input]
        )
        
        reset_btn.click(
            fn=reset_conversation,
            outputs=[chatbot, chat_input]
        )
        
        interface.load(
            fn=load_raw_data,
            outputs=[raw_stats_table, raw_events_table]
        )
        
        interface.load(
            fn=load_charts,
            outputs=[
                speed_dist_chart,
                vehicle_type_chart,
                speed_time_chart,
                violations_chart,
                traffic_flow_chart,
                speed_comparison_chart,
                direction_dist_chart,
                direction_flow_chart,
                forecast_chart
            ]
        )
        
        interface.load(
            fn=update_databricks_status,
            outputs=[databricks_status_btn]
        )
        
        
        refresh_status_btn.click(
            fn=update_databricks_status,
            outputs=[databricks_status_btn]
        )
        
        raw_data_tab.select(
            fn=load_raw_data,
            outputs=[raw_stats_table, raw_events_table]
        )
        
        data_viz_tab.select(
            fn=load_charts,
            outputs=[
                speed_dist_chart,
                vehicle_type_chart,
                speed_time_chart,
                violations_chart,
                traffic_flow_chart,
                speed_comparison_chart,
                direction_dist_chart,
                direction_flow_chart,
                forecast_chart
            ]
        )
        
        interface.cleanup = cleanup_temp_files
        
    return interface

demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        share=False,
        debug=False,
        show_error=True
    )
