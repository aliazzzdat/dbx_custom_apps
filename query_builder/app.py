import gradio as gr
import pandas as pd
from databricks import sql
import os
from typing import List, Optional, Tuple
import json
import re

class DatabricksQueryBuilder:
    def __init__(self):
        self.connection = None
        self.cursor = None
        
    def connect(self, server_hostname: str, http_path: str, access_token: str) -> bool:
        try:
            self.connection = sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=access_token
            )
            self.cursor = self.connection.cursor()
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_catalogs(self) -> List[str]:
        try:
            if self.cursor is None:
                print("Error fetching catalogs: No database connection")
                return []
            self.cursor.execute("SHOW CATALOGS")
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error fetching catalogs: {e}")
            return []
    
    def get_schemas(self, catalog: str) -> List[str]:
        try:
            if self.cursor is None:
                print("Error fetching schemas: No database connection")
                return []
            self.cursor.execute(f"SHOW SCHEMAS IN {catalog}")
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error fetching schemas: {e}")
            return []
    
    def get_tables(self, catalog: str, schema: str) -> List[str]:
        try:
            if self.cursor is None:
                print("Error fetching tables: No database connection")
                return []
            self.cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
            results = self.cursor.fetchall()
            return [row[1] for row in results]
        except Exception as e:
            print(f"Error fetching tables: {e}")
            return []
    
    def get_columns(self, catalog: str, schema: str, table: str) -> List[str]:
        try:
            if self.cursor is None:
                print("Error fetching columns: No database connection")
                return []
            self.cursor.execute(f"DESCRIBE {catalog}.{schema}.{table}")
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error fetching columns: {e}")
            return []
    
    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            if self.cursor is None:
                print("Error executing query: No database connection")
                return pd.DataFrame({"Error": ["No database connection"]})
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame({"Error": [str(e)]})

db_query_builder = DatabricksQueryBuilder()

def sanitize_sql_input(input_text: str) -> str:
    if not input_text:
        return ""
    
    dangerous_patterns = [
        r'--.*',
        r'/\*.*?\*/',
        r';\s*$',
        r'union\s+select',
        r'drop\s+table',
        r'delete\s+from',
        r'insert\s+into',
        r'update\s+set',
        r'exec\s*\(',
        r'execute\s*\('
    ]
    
    sanitized = input_text.strip()
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    sanitized = sanitized.replace("'", "''")
    
    sanitized = re.sub(r';\s*$', '', sanitized)
    
    return sanitized.strip()

def validate_column_name(column_name: str) -> bool:
    if not column_name or not column_name.strip():
        return False
    
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_.]*$'
    return bool(re.match(pattern, column_name.strip()))

SERVER_HOSTNAME = os.getenv("DATABRICKS_HOST") or "e2-demo-field-eng.cloud.databricks.com"
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH") or "/sql/1.0/warehouses/862f1d757f0424f7"
ACCESS_TOKEN = os.getenv("DATABRICKS_TOKEN") 

def initialize_connection():
    success = db_query_builder.connect(SERVER_HOSTNAME, HTTP_PATH, ACCESS_TOKEN)
    if success:
        catalogs = db_query_builder.get_catalogs()
        return (gr.update(choices=catalogs, value=None), 
                gr.update(choices=catalogs, value=None),
                f"Connected successfully! Found {len(catalogs)} catalogs.")
    else:
        return (gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None),
                "Failed to connect. Please check your credentials.")

def update_schemas(catalog: str):
    if not catalog:
        return gr.update(choices=[])
    
    schemas = db_query_builder.get_schemas(catalog)
    return gr.update(choices=schemas)

def update_tables(catalog: str, schema: str):
    if not catalog or not schema:
        return gr.update(choices=[])
    
    tables = db_query_builder.get_tables(catalog, schema)
    return gr.update(choices=tables)

def update_columns(catalog: str, schema: str, table: str):
    if not catalog or not schema or not table:
        return gr.update(choices=[], value=[])
    
    columns = db_query_builder.get_columns(catalog, schema, table)
    return gr.update(choices=columns, value=[])

def update_join_schemas(join_catalog: str):
    if not join_catalog:
        return gr.update(choices=[])
    
    schemas = db_query_builder.get_schemas(join_catalog)
    return gr.update(choices=schemas)

def update_join_tables(join_catalog: str, join_schema: str):
    if not join_catalog or not join_schema:
        return gr.update(choices=[])
    
    tables = db_query_builder.get_tables(join_catalog, join_schema)
    return gr.update(choices=tables)

def get_catalogs_for_joins():
    return db_query_builder.get_catalogs()

def get_schemas_for_join(join_catalog: str):
    if not join_catalog:
        return []
    return db_query_builder.get_schemas(join_catalog)

def get_tables_for_join(join_catalog: str, join_schema: str):
    if not join_catalog or not join_schema:
        return []
    return db_query_builder.get_tables(join_catalog, join_schema)

def get_columns_for_join_table(join_catalog: str, join_schema: str, join_table: str):
    if not join_catalog or not join_schema or not join_table:
        return []
    return db_query_builder.get_columns(join_catalog, join_schema, join_table)

def get_columns_for_base_table(catalog: str, schema: str, table: str):
    if not catalog or not schema or not table:
        return []
    return db_query_builder.get_columns(catalog, schema, table)

def update_base_join_columns(catalog: str, schema: str, table: str):
    if catalog and schema and table:
        columns = get_columns_for_base_table(catalog, schema, table)
        return gr.update(choices=columns)
    return gr.update(choices=[])

def update_all_base_join_columns(catalog: str, schema: str, table: str):
    return catalog, schema, table

def get_base_columns_for_joins(catalog: str, schema: str, table: str):
    if catalog and schema and table:
        return get_columns_for_base_table(catalog, schema, table)
    return []

def get_columns_for_dropdown(catalog: str, schema: str, table: str):
    if not catalog or not schema or not table:
        return []
    return db_query_builder.get_columns(catalog, schema, table)

def update_filter_columns(catalog: str, schema: str, table: str):
    if catalog and schema and table:
        columns = get_columns_for_base_table(catalog, schema, table)
        return gr.update(choices=columns)
    return gr.update(choices=[])

def update_order_by_columns(catalog: str, schema: str, table: str):
    if catalog and schema and table:
        columns = get_columns_for_base_table(catalog, schema, table)
        return gr.update(choices=columns, value=[])
    return gr.update(choices=[], value=[])

def update_order_by_directions(selected_columns: List[str]):
    if not selected_columns:
        return gr.update(choices=[], value=[])
    
    direction_choices = []
    for i, col in enumerate(selected_columns):
        direction_choices.extend([f"{col} ASC", f"{col} DESC"])
    
    return gr.update(choices=direction_choices, value=[])

def generate_order_by_clause(selected_columns: List[str], selected_directions: List[str], catalog: str = "", schema: str = "", table: str = ""):
    if not selected_columns:
        return ""
    
    order_parts = []
    for col in selected_columns:
        direction = "ASC"
        for dir_choice in selected_directions:
            if dir_choice.startswith(f"{col} "):
                direction = dir_choice.split(" ", 1)[1]
                break
        
        # Use fully qualified column name if catalog, schema, and table are provided
        if catalog and schema and table:
            qualified_col = f"{catalog}.{schema}.{table}.{col}"
        else:
            qualified_col = col
            
        order_parts.append(f"{qualified_col} {direction}")
    
    return ", ".join(order_parts)


def generate_sql_query(catalog: str, schema: str, table: str, 
                      base_columns: List[str] = None,
                      custom_columns: List[str] = None,
                      joins: List[dict] = None,
                      filters: List[dict] = None,
                      aggregations: List[dict] = None,
                      order_by_columns: List[str] = None,
                      order_by_directions: List[str] = None,
                      limit_rows: int = 100):
    if not catalog or not schema or not table:
        return "Please select a table and at least one column."
    
    all_columns = []
    
    if base_columns and len(base_columns) > 0:
        # Add fully qualified column names with catalog.schema.table prefix
        qualified_columns = [f"{catalog}.{schema}.{table}.{col}" for col in base_columns]
        all_columns.extend(qualified_columns)
    else:
        all_columns.append("*")
    
    if custom_columns:
        valid_custom_columns = [sanitize_sql_input(col) for col in custom_columns if col and col.strip()]
        all_columns.extend(valid_custom_columns)
    
    group_by_columns = []
    if aggregations:
        for agg in aggregations:
            if agg.get('group_by_columns'):
                for col in agg['group_by_columns']:
                    if col and col.strip() and col not in group_by_columns:
                        group_by_columns.append(f"{catalog}.{schema}.{table}.{col.strip()}")
            
            if (agg.get('aggregated_column') and agg.get('aggregated_type') and
                agg.get('aggregated_column').strip() and agg.get('aggregated_type').strip()):
                col = agg['aggregated_column'].strip()
                agg_type = agg['aggregated_type'].strip()
                
                if agg_type == "COUNT":
                    if col == "*":
                        agg_expr = "COUNT(*)"
                    else:
                        agg_expr = f"COUNT({catalog}.{schema}.{table}.{col})"
                else:
                    agg_expr = f"{agg_type}({catalog}.{schema}.{table}.{col})"
                
                all_columns.append(agg_expr)
    
    if not all_columns:
        return "Please select at least one column or add custom columns."
    
    columns_str = ", ".join(all_columns)
    query = f"SELECT {columns_str}\nFROM {catalog}.{schema}.{table}"
    
    if joins:
        for join in joins:
            if (join.get('join_type') and join.get('join_catalog') and 
                join.get('join_schema') and join.get('join_table') and join.get('join_condition') and
                join.get('join_type').strip() and join.get('join_catalog').strip() and 
                join.get('join_schema').strip() and join.get('join_table').strip() and 
                join.get('join_condition').strip()):
                query += f"\n{join['join_type']} {join['join_catalog']}.{join['join_schema']}.{join['join_table']} ON {join['join_condition']}"
    
    where_conditions = []
    if filters:
        for filter_cond in filters:
            if (filter_cond.get('filter_column') and filter_cond.get('filter_type') and 
                filter_cond.get('filter_column').strip() and filter_cond.get('filter_type').strip()):
                col = sanitize_sql_input(filter_cond['filter_column'])
                op = filter_cond['filter_type'].strip()
                
                if filter_cond.get('filter_second_column') and filter_cond.get('filter_second_column').strip():
                    second_col = sanitize_sql_input(filter_cond['filter_second_column'])
                    where_conditions.append(f"{catalog}.{schema}.{table}.{col} {op} {catalog}.{schema}.{table}.{second_col}")
                elif filter_cond.get('filter_value') and filter_cond.get('filter_value').strip():
                    val = sanitize_sql_input(filter_cond['filter_value'])
                    
                    if op == "LIKE":
                        where_conditions.append(f"{catalog}.{schema}.{table}.{col} {op} '{val}'")
                    elif op == "IN":
                        values = [v.strip() for v in val.split(',') if v.strip()]
                        if values:
                            formatted_values = "', '".join(values)
                            where_conditions.append(f"{catalog}.{schema}.{table}.{col} {op} ('{formatted_values}')")
                    else:
                        if op in ["<", ">", ">=", "<="] and val.replace('.', '').replace('-', '').isdigit():
                            where_conditions.append(f"{catalog}.{schema}.{table}.{col} {op} {val}")
                        else:
                            where_conditions.append(f"{catalog}.{schema}.{table}.{col} {op} '{val}'")
    
    if where_conditions:
        query += f"\nWHERE {' AND '.join(where_conditions)}"
    
    if group_by_columns:
        query += f"\nGROUP BY {', '.join(group_by_columns)}"
    
    if order_by_columns and len(order_by_columns) > 0:
        order_by_clause = generate_order_by_clause(order_by_columns, order_by_directions or [], catalog, schema, table)
        if order_by_clause:
            query += f"\nORDER BY {order_by_clause}"
    
    if limit_rows and limit_rows > 0:
        query += f"\nLIMIT {int(limit_rows)}"
    
    return query

def execute_sql_query(query: str):
    if not query or query.startswith("Please select"):
        return pd.DataFrame({"Message": ["No valid query to execute"]})
    
    return db_query_builder.execute_query(query)

def enable_query_config(catalog: str, schema: str, table: str, columns: List[str]):
    is_enabled = bool(catalog and schema and table)
    
    if is_enabled:
        return (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                gr.update(open=False), gr.update(open=False), gr.update(open=False), gr.update(open=False),
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))
    else:
        return (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                gr.update(open=False), gr.update(open=False), gr.update(open=False), gr.update(open=False),
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))

def clear_all_configuration():
    return (
        gr.update(value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=[]),
        gr.update(choices=[], value=[]),
        gr.update(choices=[], value=[]),
        gr.update(value=100),
        [],
        [],
        [],
        [],
        0,
        0,
        0,
        0,
        ("", "", ""),
        "",
        gr.update(value=pd.DataFrame()),
        gr.update(open=False),
        gr.update(open=False),
        gr.update(open=False),
        gr.update(open=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )



with gr.Blocks(title="Databricks Query Builder") as app:
    gr.Markdown("# Databricks SQL Query Builder")
    gr.Markdown("Build queries interactively.")
    
    connection_status = gr.Textbox(label="Connection Status", interactive=False, value="Initializing connection...")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ## 📋 Usage Instructions
            
            **Please follow this order for best results:**
            
            1. **Select a catalog** from the dropdown
            2. **Select a schema** from the dropdown  
            3. **Select a table** from the dropdown
            4. **Add the numbers** of custom columns, joins, filters, and aggregations needed
            5. **Fill them at the same time** - adding and filling one by one may cause issues
            6. **If you click remove, it will reset the sql query** - please update all other compenents that have not been removed
            
            **Important Notes:**
            - The SQL query updates in real-time as you make changes
            - All user inputs are sanitized to prevent SQL injection attacks
            - You can add columns that equal other columns within the same table
            - Use the accordion sections to configure your query components
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ## ✅ TODO & Features
            
            **Current Features:**
            - ✅ Real-time SQL query generation
            - ✅ Dynamic column selection
            - ✅ Multiple joins, filters, and aggregations
            
            **Planned Features:**
            - 🔄 Enhanced column selection from joins and custom columns in joins and filters and aggregations
            - 🔄 Boolean management with NOT, AND & OR statements in filters
            - 🔄 Better error handling and validation
            - 🔄 Add create table button
            - 🔄 Mange Union of Tables
            - 🔄 Fix input filters value depending on type of input filters (int or string) in where statements
            """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Table Structure")
            
            catalog_dropdown = gr.Dropdown(
                label="Catalog",
                choices=[],
                interactive=True,
                info="Select a catalog"
            )
            
            schema_dropdown = gr.Dropdown(
                label="Schema",
                choices=[],
                interactive=True,
                info="Select a schema"
            )
            
            table_dropdown = gr.Dropdown(
                label="Table",
                choices=[],
                interactive=True,
                info="Select a table"
            )
            
            columns_multiselect = gr.Dropdown(
                label="Columns",
                choices=[],
                multiselect=True,
                interactive=True,
                info="Select columns (all selected by default)"
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**ORDER BY**")
                    order_by_columns = gr.Dropdown(
                        label="Select Columns to Order By",
                        choices=[],
                        multiselect=True,
                        interactive=False,
                        info="Select columns to order by (none selected by default)"
                    )
                    order_by_directions = gr.CheckboxGroup(
                        label="Order Direction",
                        choices=[],
                        interactive=False,
                        info="Select ASC/DESC for each column (select in same order as columns above)"
                    )
                with gr.Column(scale=1):
                    limit_rows = gr.Number(
                        label="LIMIT",
                        value=100,
                        info="Maximum number of rows to return",
                        interactive=False
                    )
        
        with gr.Column(scale=1):
            gr.Markdown("## Query Configuration")
            
            custom_columns_data = gr.State([])
            joins_data = gr.State([])
            filters_data = gr.State([])
            aggregations_data = gr.State([])
            current_table_state = gr.State(("", "", ""))
            
            with gr.Accordion("Custom Columns", open=False) as custom_columns_accordion:
                gr.Markdown("Add custom SQL expressions, calculations, or aliases. You can create complex expressions using columns from the selected table.")
                custom_columns_count = gr.State(0)
                add_custom_columns_btn = gr.Button("Add Custom Column", interactive=False)
                remove_custom_columns_btn = gr.Button("Remove Custom Column", interactive=False)
                add_custom_columns_btn.click(lambda x: x + 1, custom_columns_count, custom_columns_count)
                
                def remove_custom_column_with_clear(current_count):
                    new_count = current_count - 1 if current_count > 0 else 0
                    return new_count, []
                
                remove_custom_columns_btn.click(
                    remove_custom_column_with_clear, 
                    inputs=[custom_columns_count], 
                    outputs=[custom_columns_count, custom_columns_data]
                )
                
                with gr.Blocks():

                    @gr.render(inputs=[custom_columns_count, current_table_state])
                    def render_custom_columns(count, table_info):
                        custom_columns_boxes = []
                        
                        if table_info and isinstance(table_info, (tuple, list)) and len(table_info) == 3:
                            catalog, schema, table = table_info
                        else:
                            catalog, schema, table = "", "", ""
                        
                        available_columns = get_columns_for_dropdown(catalog, schema, table) if catalog and schema and table else []
                        
                        for i in range(count):
                            custom_columns_box = gr.Textbox(
                                key=f"custom_expression_{i}",
                                label=f"Custom Expression {i+1}",
                                placeholder="e.g., column1 * 2 as doubled, CONCAT(col1, col2) as combined",
                                interactive=True,
                                info="Custom SQL expression"
                            )
                            
                            custom_columns_boxes.append(custom_columns_box)

                        def custom_columns_update_query(*args):
                            custom_cols = []
                            if len(args) >= 1:
                                for i in range(0, len(args), 1):
                                    if i < len(args):
                                        expression = args[i]
                                        
                                        if expression and expression.strip():
                                            custom_cols.append(expression.strip())
                            
                            return custom_cols
                        
                        if custom_columns_boxes:
                            for custom_box in custom_columns_boxes:
                                def make_custom_update_handler(components):
                                    def update_custom_columns_and_query(*args):
                                        custom_cols = custom_columns_update_query(*args)
                                        return custom_cols
                                    return update_custom_columns_and_query
                                
                                custom_box.change(
                                    fn=make_custom_update_handler(custom_columns_boxes),
                                    inputs=custom_columns_boxes,
                                    outputs=[custom_columns_data]
                                )

            
            with gr.Accordion("Joins", open=False) as joins_accordion:
                gr.Markdown("Add multiple joins to your query")
                joins_count_state = gr.State(0)
                add_join_btn = gr.Button("Add Join", interactive=False)
                remove_join_btn = gr.Button("Remove Join", interactive=False)
                
                def add_join_with_update(current_count):
                    return current_count + 1
                
                add_join_btn.click(add_join_with_update, inputs=[joins_count_state], outputs=joins_count_state)
                
                def remove_join_with_clear(current_count):
                    new_count = current_count - 1 if current_count > 0 else 0
                    return new_count, []
                
                remove_join_btn.click(
                    remove_join_with_clear, 
                    inputs=[joins_count_state], 
                    outputs=[joins_count_state, joins_data]
                )
                
                with gr.Blocks():

                    @gr.render(inputs=[joins_count_state, current_table_state])
                    def render_joins(count, table_info):
                        join_components = []
                        catalogs = get_catalogs_for_joins()
                        
                        if table_info and isinstance(table_info, (tuple, list)) and len(table_info) == 3:
                            catalog, schema, table = table_info
                        else:
                            catalog, schema, table = "", "", ""
                        
                        for i in range(count):
                            base_columns = get_base_columns_for_joins(catalog, schema, table)
                            with gr.Row():
                                join_type = gr.Dropdown(
                                    key=f"join_type_{i}",
                                    label=f"Join Type {i+1}",
                                    choices=["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN"],
                                    value="INNER JOIN",
                                    interactive=True
                                )
                                join_catalog = gr.Dropdown(
                                    key=f"join_catalog_{i}",
                                    label=f"Join Catalog {i+1}",
                                    choices=catalogs,
                                    value=None,
                                    interactive=True
                                )
                                join_schema = gr.Dropdown(
                                    key=f"join_schema_{i}",
                                    label=f"Join Schema {i+1}",
                                    choices=[],
                                    interactive=True
                                )
                                join_table = gr.Dropdown(
                                    key=f"join_table_{i}",
                                    label=f"Join Table {i+1}",
                                    choices=[],
                                    interactive=True
                                )
                            
                            with gr.Row():
                                base_join_column = gr.Dropdown(
                                    key=f"base_join_column_{i}",
                                    label=f"Base Table Column {i+1}",
                                    choices=base_columns,
                                    interactive=True,
                                    info="Column from the main table"
                                )
                                join_column = gr.Dropdown(
                                    key=f"join_column_{i}",
                                    label=f"Join Table Column {i+1}",
                                    choices=[],
                                    interactive=True,
                                    info="Column from the join table"
                                )
                            
                            with gr.Row():
                                join_operator = gr.Dropdown(
                                    key=f"join_operator_{i}",
                                    label=f"Join Operator {i+1}",
                                    choices=["=", "<", ">", "!=", ">=", "<=", "LIKE", "IN"],
                                    value="=",
                                    interactive=True,
                                    info="Comparison operator"
                                )
                                join_value = gr.Textbox(
                                    key=f"join_value_{i}",
                                    label=f"Join Value {i+1}",
                                    placeholder="Enter value for comparison",
                                    interactive=True,
                                    info="Value to compare against (leave empty for column-to-column comparison)"
                                )
                            join_components.extend([join_type, join_catalog, join_schema, join_table, base_join_column, join_column, join_operator, join_value])
                            
                            def make_join_schema_updater(join_catalog_idx):
                                def update_join_schema(selected_catalog):
                                    if selected_catalog:
                                        schemas = get_schemas_for_join(selected_catalog)
                                        return gr.update(choices=schemas)
                                    return gr.update(choices=[])
                                return update_join_schema
                            
                            def make_join_table_updater(join_catalog_idx, join_schema_idx):
                                def update_join_table(selected_catalog, selected_schema):
                                    if selected_catalog and selected_schema:
                                        tables = get_tables_for_join(selected_catalog, selected_schema)
                                        return gr.update(choices=tables)
                                    return gr.update(choices=[])
                                return update_join_table
                            
                            def make_join_column_updater(join_catalog_idx, join_schema_idx, join_table_idx):
                                def update_join_column(selected_catalog, selected_schema, selected_table):
                                    if selected_catalog and selected_schema and selected_table:
                                        columns = get_columns_for_join_table(selected_catalog, selected_schema, selected_table)
                                        return gr.update(choices=columns)
                                    return gr.update(choices=[])
                                return update_join_column
                            
                            def make_base_column_updater(join_idx):
                                def update_base_column():
                                    current_catalog = catalog_dropdown.value
                                    current_schema = schema_dropdown.value
                                    current_table = table_dropdown.value
                                    if current_catalog and current_schema and current_table:
                                        columns = get_columns_for_base_table(current_catalog, current_schema, current_table)
                                        return gr.update(choices=columns)
                                    return gr.update(choices=[])
                                return update_base_column
                            
                            def make_join_condition_updater(join_idx):
                                def update_join_condition(base_col, join_col, operator, value):
                                    pass
                                return update_join_condition
                            
                            join_catalog.change(
                                fn=make_join_schema_updater(i),
                                inputs=[join_catalog],
                                outputs=[join_schema]
                            )
                            
                            join_schema.change(
                                fn=make_join_table_updater(i, i),
                                inputs=[join_catalog, join_schema],
                                outputs=[join_table]
                            )
                            
                            join_table.change(
                                fn=make_join_column_updater(i, i, i),
                                inputs=[join_catalog, join_schema, join_table],
                                outputs=[join_column]
                            )
                            
                            
                            
                            
                            base_join_column.change(
                                fn=make_join_condition_updater(i),
                                inputs=[base_join_column, join_column, join_operator, join_value],
                                outputs=[]
                            )
                            
                            join_column.change(
                                fn=make_join_condition_updater(i),
                                inputs=[base_join_column, join_column, join_operator, join_value],
                                outputs=[]
                            )
                            
                            join_operator.change(
                                fn=make_join_condition_updater(i),
                                inputs=[base_join_column, join_column, join_operator, join_value],
                                outputs=[]
                            )
                            
                            join_value.change(
                                fn=make_join_condition_updater(i),
                                inputs=[base_join_column, join_column, join_operator, join_value],
                                outputs=[]
                            )

                        def joins_update_query(*args):
                            joins_list = []
                            if len(args) >= 8:
                                for i in range(0, len(args), 8):
                                    if i + 7 < len(args):
                                        join_type, join_catalog, join_schema, join_table, base_join_column, join_column, join_operator, join_value = args[i:i+8]
                                        if (join_type and join_catalog and join_schema and join_table and 
                                            base_join_column and join_column and join_operator and
                                            join_type.strip() and join_catalog.strip() and join_schema.strip() and 
                                            join_table.strip() and base_join_column.strip() and 
                                            join_column.strip() and join_operator.strip()):
                                            
                                            if table_info and isinstance(table_info, (tuple, list)) and len(table_info) == 3:
                                                current_catalog, current_schema, current_table = table_info
                                            else:
                                                current_catalog, current_schema, current_table = "", "", ""
                                            
                                            if current_catalog and current_schema and current_table:
                                                base_table_name = f"{current_catalog}.{current_schema}.{current_table}"
                                                join_table_name = f"{join_catalog}.{join_schema}.{join_table}"
                                                
                                                if join_value and join_value.strip():
                                                    if join_operator in ["LIKE", "IN"]:
                                                        join_condition = f"{base_table_name}.{base_join_column} {join_operator} '{join_value}'"
                                                    else:
                                                        join_condition = f"{base_table_name}.{base_join_column} {join_operator} '{join_value}'"
                                                else:
                                                    join_condition = f"{base_table_name}.{base_join_column} {join_operator} {join_table_name}.{join_column}"
                                                
                                                joins_list.append({
                                                    'join_type': join_type.strip(),
                                                    'join_catalog': join_catalog.strip(),
                                                    'join_schema': join_schema.strip(),
                                                    'join_table': join_table.strip(),
                                                    'join_condition': join_condition
                                                })
                            
                            return joins_list
                        
                        if join_components:
                            for join_component in join_components:
                                def make_join_update_handler(components):
                                    def update_joins_and_query(*args):
                                        joins_list = joins_update_query(*args)
                                        return joins_list
                                    return update_joins_and_query
                                
                                join_component.change(
                                    fn=make_join_update_handler(join_components),
                                    inputs=join_components,
                                    outputs=[joins_data]
                                )

            
            with gr.Accordion("Filters", open=False) as filters_accordion:
                gr.Markdown("Add multiple filter conditions to your query. You can compare columns against values or other columns in the same table.")
                filters_count = gr.State(0)
                add_filter_btn = gr.Button("Add Filter", interactive=False)
                remove_filter_btn = gr.Button("Remove Filter", interactive=False)
                
                def add_filter_with_update(current_count):
                    return current_count + 1
                
                add_filter_btn.click(add_filter_with_update, inputs=[filters_count], outputs=filters_count)
                
                def remove_filter_with_clear(current_count):
                    new_count = current_count - 1 if current_count > 0 else 0
                    return new_count, []
                
                remove_filter_btn.click(
                    remove_filter_with_clear, 
                    inputs=[filters_count], 
                    outputs=[filters_count, filters_data]
                )
                
                with gr.Blocks():

                    @gr.render(inputs=[filters_count, current_table_state])
                    def render_filters(count, table_info):
                        filter_components = []
                        
                        if table_info and isinstance(table_info, (tuple, list)) and len(table_info) == 3:
                            catalog, schema, table = table_info
                        else:
                            catalog, schema, table = "", "", ""
                        
                        for i in range(count):
                            filter_columns = get_columns_for_dropdown(catalog, schema, table)
                            with gr.Row():
                                filter_column = gr.Dropdown(
                                    key=f"filter_column_{i}",
                                    label=f"Filter Column {i+1}",
                                    choices=filter_columns,
                                    interactive=True,
                                    info="Select column to filter"
                                )
                                filter_type = gr.Dropdown(
                                    key=f"filter_type_{i}",
                                    label=f"Filter Type {i+1}",
                                    choices=["=", "<", ">", ">=", "<=", "!=", "LIKE", "IN"],
                                    value="=",
                                    interactive=True,
                                    info="Select comparison operator"
                                )
                                filter_second_column = gr.Dropdown(
                                    key=f"filter_second_column_{i}",
                                    label=f"Second Column for Comparison {i+1}",
                                    choices=filter_columns,
                                    value=None,
                                    interactive=True,
                                    info="Select second column for comparison (leave empty to use value below)"
                                )
                                filter_value = gr.Textbox(
                                    key=f"filter_value_{i}",
                                    label=f"Filter Value {i+1}",
                                    placeholder="Enter filter value",
                                    interactive=True,
                                    info="Enter the value to compare against (ignored if second column is selected)"
                                )
                            filter_components.extend([filter_column, filter_type, filter_second_column, filter_value])
                            
                            columns = get_columns_for_base_table(catalog, schema, table) if catalog and schema and table else []
                            

                        def filters_update_query(*args):
                            filters_list = []
                            if len(args) >= 4:
                                for i in range(0, len(args), 4):
                                    if i + 3 < len(args):
                                        filter_column, filter_type, filter_second_column, filter_value = args[i:i+4]
                                        if (filter_column and filter_type and
                                            filter_column.strip() and filter_type.strip()):
                                            
                                            if filter_second_column and filter_second_column.strip():
                                                filters_list.append({
                                                    'filter_column': filter_column.strip(),
                                                    'filter_type': filter_type.strip(),
                                                    'filter_second_column': filter_second_column.strip(),
                                                    'filter_value': None
                                                })
                                            elif filter_value and filter_value.strip():
                                                filters_list.append({
                                                    'filter_column': filter_column.strip(),
                                                    'filter_type': filter_type.strip(),
                                                    'filter_second_column': None,
                                                    'filter_value': filter_value.strip()
                                                })
                            
                            return filters_list
                        
                        if filter_components:
                            for filter_component in filter_components:
                                def make_filter_update_handler(components):
                                    def update_filters_and_query(*args):
                                        filters_list = filters_update_query(*args)
                                        return filters_list
                                    return update_filters_and_query
                                
                                filter_component.change(
                                    fn=make_filter_update_handler(filter_components),
                                    inputs=filter_components,
                                    outputs=[filters_data]
                                )

                
            
            with gr.Accordion("Aggregation", open=False) as aggregation_accordion:
                gr.Markdown("Add aggregation functions to your query")
                aggregation_count = gr.State(0)
                add_aggregation_btn = gr.Button("Add Aggregation", interactive=False)
                remove_aggregation_btn = gr.Button("Remove Aggregation", interactive=False)
                
                def add_aggregation_with_update(current_count):
                    return current_count + 1
                
                add_aggregation_btn.click(add_aggregation_with_update, inputs=[aggregation_count], outputs=aggregation_count)
                
                def remove_aggregation_with_clear(current_count):
                    new_count = current_count - 1 if current_count > 0 else 0
                    return new_count, []
                
                remove_aggregation_btn.click(
                    remove_aggregation_with_clear, 
                    inputs=[aggregation_count], 
                    outputs=[aggregation_count, aggregations_data]
                )
                
                with gr.Blocks():

                    @gr.render(inputs=[aggregation_count, current_table_state])
                    def render_aggregations(count, table_info):
                        aggregation_components = []
                        
                        if table_info and isinstance(table_info, (tuple, list)) and len(table_info) == 3:
                            catalog, schema, table = table_info
                        else:
                            catalog, schema, table = "", "", ""
                        
                        agg_columns = get_columns_for_dropdown(catalog, schema, table)
                        
                        if count >= 1:
                            aggregation_group_by_columns = gr.Dropdown(
                                key="aggregation_group_by_columns",
                                label="GROUP BY Columns",
                                choices=agg_columns,
                                multiselect=True,
                                interactive=True,
                                info="Select base columns to group by"
                            )
                            aggregation_components.append(aggregation_group_by_columns)
                            
                            
                        
                        for i in range(count):
                            agg_columns = get_columns_for_dropdown(catalog, schema, table)
                            with gr.Row():
                                aggregation_aggregated_column = gr.Dropdown(
                                    key=f"aggregation_aggregated_column_{i}",
                                    label=f"Aggregated Column {i+1}",
                                    choices=agg_columns,
                                    interactive=True,
                                    info="Select column to aggregate"
                                )
                                aggregation_aggregated_type = gr.Dropdown(
                                    key=f"aggregation_aggregated_type_{i}",
                                    label=f"Aggregation Type {i+1}",
                                    choices=["COUNT", "MIN", "MAX", "FIRST", "LAST", "AVG", "CONCAT_WS"],
                                    value="COUNT",
                                    interactive=True,
                                    info="Select aggregation function"
                                )
                            
                            aggregation_components.extend([aggregation_aggregated_column, aggregation_aggregated_type])
                            
                            

                        def aggregations_update_query(*args):
                            aggregations_list = []
                            
                            if len(args) >= 1:
                                group_by_columns = args[0] if args[0] else []
                                valid_group_by_columns = [col.strip() for col in group_by_columns if col and col.strip()]
                                
                                if len(args) >= 3:
                                    for i in range(1, len(args), 2):
                                        if i + 1 < len(args):
                                            aggregated_col, agg_type = args[i:i+2]
                                            if (aggregated_col and agg_type and 
                                                aggregated_col.strip() and agg_type.strip()):
                                                aggregations_list.append({
                                                    'group_by_columns': valid_group_by_columns,
                                                    'aggregated_column': aggregated_col.strip(),
                                                    'aggregated_type': agg_type.strip()
                                                })
                            
                            return aggregations_list
                        
                        if aggregation_components:
                            for agg_component in aggregation_components:
                                def make_agg_update_handler(components):
                                    def update_aggregations_and_query(*args):
                                        aggregations_list = aggregations_update_query(*args)
                                        return aggregations_list
                                    return update_aggregations_and_query
                                
                                agg_component.change(
                                    fn=make_agg_update_handler(aggregation_components),
                                    inputs=aggregation_components,
                                    outputs=[aggregations_data]
                                )

                
            
    
    gr.Markdown("## Generated SQL Query")
    with gr.Row():
        sql_query = gr.Textbox(
            label="SQL Query",
            lines=4,
            interactive=False,
            info="Generated query will appear here"
        )
        execute_btn = gr.Button("Execute Query", variant="secondary")
    
    with gr.Row():
        clear_btn = gr.Button("Clear All Configuration", variant="stop", size="sm")
    
    results_df = gr.Dataframe(
        label="Query Results",
        interactive=False,
        wrap=True
    )
    

    app.load(
        fn=initialize_connection,
        inputs=[],
        outputs=[catalog_dropdown, catalog_dropdown, connection_status]
    )
    
    catalog_dropdown.change(
        fn=update_schemas,
        inputs=[catalog_dropdown],
        outputs=[schema_dropdown]
    )
    
    schema_dropdown.change(
        fn=update_tables,
        inputs=[catalog_dropdown, schema_dropdown],
        outputs=[table_dropdown]
    )
    
    table_dropdown.change(
        fn=update_columns,
        inputs=[catalog_dropdown, schema_dropdown, table_dropdown],
        outputs=[columns_multiselect]
    )
    
    table_dropdown.change(
        fn=update_order_by_columns,
        inputs=[catalog_dropdown, schema_dropdown, table_dropdown],
        outputs=[order_by_columns]
    )
    
    order_by_columns.change(
        fn=update_order_by_directions,
        inputs=[order_by_columns],
        outputs=[order_by_directions]
    )
    
    table_dropdown.change(
        fn=enable_query_config,
        inputs=[catalog_dropdown, schema_dropdown, table_dropdown, columns_multiselect],
        outputs=[order_by_columns, order_by_directions, limit_rows,
                custom_columns_accordion, joins_accordion, filters_accordion, aggregation_accordion,
                add_custom_columns_btn, remove_custom_columns_btn,
                add_join_btn, remove_join_btn,
                add_filter_btn, remove_filter_btn,
                add_aggregation_btn, remove_aggregation_btn]
    )
    
    def update_table_state(catalog, schema, table):
        return (catalog, schema, table)
    
    table_dropdown.change(
        fn=update_table_state,
        inputs=[catalog_dropdown, schema_dropdown, table_dropdown],
        outputs=[current_table_state]
    )
    
    columns_multiselect.change(
        fn=enable_query_config,
        inputs=[catalog_dropdown, schema_dropdown, table_dropdown, columns_multiselect],
        outputs=[order_by_columns, order_by_directions, limit_rows,
                custom_columns_accordion, joins_accordion, filters_accordion, aggregation_accordion,
                add_custom_columns_btn, remove_custom_columns_btn,
                add_join_btn, remove_join_btn,
                add_filter_btn, remove_filter_btn,
                add_aggregation_btn, remove_aggregation_btn]
    )
    
    def update_query_full(catalog, schema, table, columns, order_by_columns, order_by_directions, limit_rows,
                         custom_columns_data, joins_data, filters_data, aggregations_data):
        custom_cols = custom_columns_data if custom_columns_data else []
        joins = joins_data if joins_data else []
        filters = filters_data if filters_data else []
        aggregations = aggregations_data if aggregations_data else []
        
        return generate_sql_query(
            catalog, schema, table,
            base_columns=columns,
            custom_columns=custom_cols,
            joins=joins,
            filters=filters,
            aggregations=aggregations,
            order_by_columns=order_by_columns,
            order_by_directions=order_by_directions,
            limit_rows=limit_rows
        )
    
    query_inputs = [catalog_dropdown, schema_dropdown, table_dropdown, columns_multiselect,
                   order_by_columns, order_by_directions, limit_rows, custom_columns_data, joins_data, filters_data, aggregations_data]
    
    for component in query_inputs:
        component.change(
            fn=update_query_full,
            inputs=query_inputs,
            outputs=[sql_query]
        )
    
    custom_columns_data.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    joins_data.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    filters_data.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    aggregations_data.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    custom_columns_count.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    joins_count_state.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    filters_count.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    aggregation_count.change(
        fn=update_query_full,
        inputs=query_inputs,
        outputs=[sql_query]
    )
    
    execute_btn.click(
        fn=execute_sql_query,
        inputs=[sql_query],
        outputs=[results_df]
    )
    
    clear_btn.click(
        fn=clear_all_configuration,
        inputs=[],
        outputs=[
            catalog_dropdown, schema_dropdown, table_dropdown, columns_multiselect,
            order_by_columns, order_by_directions, limit_rows,
            custom_columns_data, joins_data, filters_data, aggregations_data,
            custom_columns_count, joins_count_state, filters_count, aggregation_count,
            current_table_state,
            sql_query, results_df,
            custom_columns_accordion, joins_accordion, filters_accordion, aggregation_accordion,
            add_custom_columns_btn, remove_custom_columns_btn,
            add_join_btn, remove_join_btn,
            add_filter_btn, remove_filter_btn,
            add_aggregation_btn, remove_aggregation_btn
        ]
    )
    

if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
