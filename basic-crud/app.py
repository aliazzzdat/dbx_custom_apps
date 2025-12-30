import gradio as gr
import pandas as pd
import os
from databricks import sql
from databricks.sdk.core import Config

# Environment configuration
DEFAULT_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

def get_connection(http_path):
    """Get or create a cached connection to Databricks"""
    cfg = Config()  # Set the DATABRICKS_HOST environment variable when running locally
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{http_path}",
        credentials_provider=lambda: cfg.authenticate,
    )

def read_table(table_name, conn):
    """Read all records from the specified table"""
    try:
        with conn.cursor() as cursor:
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            result = cursor.fetchall_arrow().to_pandas()
            return result
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def get_table_schema(table_name, conn):
    """Get column names and types from the table"""
    try:
        with conn.cursor() as cursor:
            query = f"DESCRIBE TABLE {table_name}"
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [row[0] for row in result if row[0] not in ['# Partition Information', '# col_name']]
            return columns
    except Exception as e:
        return []

def insert_record(table_name, conn, columns, values):
    """Insert a new record into the table"""
    try:
        with conn.cursor() as cursor:
            # Build INSERT with named parameters
            column_names = ', '.join(columns)
            placeholders = ', '.join([f':{col}' for col in columns])
            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            
            # Create parameters dictionary
            params = {col: val for col, val in zip(columns, values)}
            
            cursor.execute(query, params)
            conn.commit()  # Commit the transaction
            return "Record inserted successfully!"
    except Exception as e:
        conn.rollback()  # Rollback on error
        return f"Error inserting record: {str(e)}"

def update_record(table_name, conn, columns, values, primary_key_col, primary_key_val):
    """Update an existing record in the table"""
    try:
        with conn.cursor() as cursor:
            # Build SET clause with named parameters
            set_clause = ', '.join([f"{col} = :{col}" for col in columns])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key_col} = :where_id"
            
            # Create parameters dictionary
            params = {col: val for col, val in zip(columns, values)}
            params['where_id'] = primary_key_val
            
            cursor.execute(query, params)
            conn.commit()  # Commit the transaction
            return "Record updated successfully!"
    except Exception as e:
        conn.rollback()  # Rollback on error
        return f"Error updating record: {str(e)}"

def delete_record(table_name, conn, primary_key_col, primary_key_val):
    """Delete a record from the table"""
    try:
        with conn.cursor() as cursor:
            # Use named parameter for DELETE
            query = f"DELETE FROM {table_name} WHERE {primary_key_col} = :id"
            params = {'id': primary_key_val}
            
            cursor.execute(query, params)
            conn.commit()  # Commit the transaction
            return "Record deleted successfully!"
    except Exception as e:
        conn.rollback()  # Rollback on error
        return f"Error deleting record: {str(e)}"

def load_data(table_name):
    """Load data from Databricks table"""
    if not DEFAULT_HTTP_PATH:
        return pd.DataFrame({"Error": ["DATABRICKS_HTTP_PATH environment variable not set"]}), gr.update(visible=False), gr.update(visible=False), [], None
    
    if not table_name:
        return pd.DataFrame({"Message": ["Please enter Table Name"]}), gr.update(visible=False), gr.update(visible=False), [], None
    
    try:
        conn = get_connection(DEFAULT_HTTP_PATH)
        df = read_table(table_name, conn)
        columns = get_table_schema(table_name, conn)
        
        if df.empty or "Error" in df.columns:
            return df, gr.update(visible=False), gr.update(visible=False), [], None
        
        return df, gr.update(visible=True), gr.update(visible=True), columns, None
    except Exception as e:
        error_df = pd.DataFrame({"Error": [str(e)]})
        return error_df, gr.update(visible=False), gr.update(visible=False), [], None

def create_new_record(table_name, columns, *field_values):
    """Create a new record with the provided field values"""
    if not table_name or not columns:
        return gr.update(value="Please load a table first", visible=True), None, None, gr.update(visible=True)
    
    try:
        conn = get_connection(DEFAULT_HTTP_PATH)
        # Filter out empty values
        values = [v if v != "" else None for v in field_values[:len(columns)]]
        message = insert_record(table_name, conn, columns, values)
        df = read_table(table_name, conn)
        return gr.update(value=message, visible=True), df, None, gr.update(visible=False)
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), None, None, gr.update(visible=True)

def on_row_select(evt: gr.SelectData):
    """Capture the selected row index"""
    return evt.index[0]

def prepare_edit(table_name, columns, selected_row):
    """Prepare the edit modal with selected row data"""
    if not columns:
        return [gr.update() for _ in range(10)] + [gr.update(value="Please load a table first", visible=True), gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)]
    
    if selected_row is None:
        return [gr.update() for _ in range(10)] + [gr.update(value="Please select a row from the table first", visible=True), gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)]
    
    try:
        conn = get_connection(DEFAULT_HTTP_PATH)
        df = read_table(table_name, conn)
        
        if selected_row >= len(df):
            return [gr.update() for _ in range(10)] + [gr.update(value="Invalid row selection", visible=True), gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)]
        
        row_data = df.iloc[selected_row]
        updates = []
        
        # Show all columns (up to 10) and allow editing all of them
        for i, col in enumerate(columns[:10]):
            if i < len(columns):
                updates.append(gr.update(value=str(row_data[col]), visible=True, label=col))
            else:
                updates.append(gr.update(visible=False))
        
        # Fill remaining slots if less than 10 columns
        while len(updates) < 10:
            updates.append(gr.update(visible=False))
        
        # Store the original primary key value (first column) to identify the record
        original_primary_key = str(row_data[columns[0]])
        
        updates.append(gr.update(value="", visible=False))  # message - hidden initially
        updates.append(gr.update(visible=True))  # edit_modal visibility
        updates.append(original_primary_key)  # original primary key value
        updates.append(gr.update(visible=False))  # create_modal visibility - close it
        updates.append(gr.update(visible=False))  # delete_modal visibility - close it
        
        return updates
    except Exception as e:
        return [gr.update() for _ in range(10)] + [gr.update(value=f"Error: {str(e)}", visible=True), gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)]

def save_edit(table_name, columns, original_pk_value, *field_values):
    """Save the edited record - allows updating ALL columns including primary key"""
    if not table_name or not columns:
        return gr.update(value="Please load a table first", visible=True), None, gr.update(visible=True), None
    
    try:
        conn = get_connection(DEFAULT_HTTP_PATH)
        # Use the original primary key value to identify the record
        primary_key_col = columns[0]
        
        # Update ALL columns (including primary key if it was changed)
        values = [v if v != "" else None for v in field_values[:len(columns)]]
        
        message = update_record(table_name, conn, columns, values, primary_key_col, original_pk_value)
        df = read_table(table_name, conn)
        return gr.update(value=message, visible=True), df, gr.update(visible=False), None
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), None, gr.update(visible=True), None

def prepare_delete(table_name, columns, selected_row):
    """Prepare the delete modal with selected row data"""
    if not columns:
        return "Please load a table first", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    
    if selected_row is None:
        return "Please select a row from the table first", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    
    try:
        conn = get_connection(DEFAULT_HTTP_PATH)
        df = read_table(table_name, conn)
        
        if selected_row >= len(df):
            return "Invalid row selection", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
        
        row_data = df.iloc[selected_row]
        primary_key_col = columns[0]
        primary_key_val = row_data[primary_key_col]
        
        summary = f"Are you sure you want to delete the record with {primary_key_col} = {primary_key_val}?"
        
        return summary, gr.update(visible=True), str(primary_key_val), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return f"Error: {str(e)}", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)

def confirm_delete(table_name, columns, primary_key_val):
    """Confirm and execute the delete operation"""
    if not table_name or not columns:
        return gr.update(value="Please load a table first", visible=True), None, gr.update(visible=True), None
    
    try:
        conn = get_connection(DEFAULT_HTTP_PATH)
        primary_key_col = columns[0]
        
        message = delete_record(table_name, conn, primary_key_col, primary_key_val)
        df = read_table(table_name, conn)
        return gr.update(value=message, visible=True), df, gr.update(visible=False), None
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), None, gr.update(visible=True), None

# Build the Gradio interface
with gr.Blocks(title="Databricks CRUD App") as demo:
    gr.Markdown("# 🗄️ Databricks CRUD Application")
    gr.Markdown("Manage your Databricks tables with Create, Read, Update, and Delete operations")
    
    # Connection configuration
    with gr.Row():
        table_name_input = gr.Textbox(
            label="Unity Catalog Table Name",
            placeholder="ali_azzouz.airport.airports",
            value="ali_azzouz.airport.airports",
            scale=3
        )
        load_btn = gr.Button("Load Table", variant="primary", scale=1)
    
    # State to store table columns and selected row
    columns_state = gr.State([])
    selected_row_state = gr.State(None)
    
    # Status message
    status_msg = gr.Textbox(label="Status", interactive=False, visible=False)
    
    # Data display
    data_display = gr.Dataframe(
        label="Table Data",
        interactive=False,
        wrap=True
    )
    
    # Action buttons
    with gr.Row(visible=False) as action_row:
        create_btn = gr.Button("➕ Create New Record", variant="primary")
        edit_btn = gr.Button("✏️ Edit Selected", variant="secondary")
        delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
    
    instructions_text = gr.Markdown("*To edit or delete: click on a row in the table above, then click the corresponding button*", visible=False)
    
    # ===== CREATE MODAL =====
    with gr.Column(visible=False) as create_modal:
        with gr.Group():
            gr.Markdown("## ➕ Create New Record")
            gr.Markdown("Fill in the fields below to create a new record")
            create_fields = []
            for i in range(10):  # Support up to 10 columns
                field = gr.Textbox(label=f"Field {i+1}", visible=False)
                create_fields.append(field)
            
            create_status = gr.Textbox(label="Status", interactive=False, visible=False)
            
            with gr.Row():
                create_save_btn = gr.Button("💾 Save New Record", variant="primary", scale=1)
                create_cancel_btn = gr.Button("❌ Cancel", variant="secondary", scale=1)
    
    # ===== EDIT MODAL =====
    with gr.Column(visible=False) as edit_modal:
        with gr.Group():
            gr.Markdown("## ✏️ Edit Record")
            gr.Markdown("Modify the fields below to update the record")
            edit_fields = []
            for i in range(10):  # Support up to 10 columns
                field = gr.Textbox(label=f"Field {i+1}", visible=False)
                edit_fields.append(field)
            
            edit_status = gr.Textbox(label="Status", interactive=False, visible=False)
            edit_original_pk_state = gr.State("")  # Store original primary key value
            
            with gr.Row():
                edit_save_btn = gr.Button("💾 Save Changes", variant="primary", scale=1)
                edit_cancel_btn = gr.Button("❌ Cancel", variant="secondary", scale=1)
    
    # ===== DELETE MODAL =====
    with gr.Column(visible=False) as delete_modal:
        with gr.Group():
            gr.Markdown("## 🗑️ Delete Record")
            gr.Markdown("⚠️ **Warning:** This action cannot be undone!")
            delete_summary = gr.Textbox(label="Confirmation", interactive=False)
            delete_key_state = gr.State("")
            
            delete_status = gr.Textbox(label="Status", interactive=False, visible=False)
            
            with gr.Row():
                delete_confirm_btn = gr.Button("🗑️ Confirm Delete", variant="stop", scale=1)
                delete_cancel_btn = gr.Button("❌ Cancel", variant="secondary", scale=1)
    
    # Event handlers
    def show_create_modal(columns):
        """Show the Create modal with empty fields"""
        if not columns:
            return [gr.update(visible=False) for _ in range(10)] + [gr.update(value="Please load a table first", visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
        
        updates = []
        for i, col in enumerate(columns[:10]):
            updates.append(gr.update(visible=True, label=col, value=""))
        
        while len(updates) < 10:
            updates.append(gr.update(visible=False))
        
        updates.append(gr.update(value="", visible=False))  # status - hidden initially
        updates.append(gr.update(visible=True))  # create_modal visibility
        updates.append(gr.update(visible=False))  # edit_modal visibility - close it
        updates.append(gr.update(visible=False))  # delete_modal visibility - close it
        
        return updates
    
    load_btn.click(
        fn=load_data,
        inputs=[table_name_input],
        outputs=[data_display, action_row, instructions_text, columns_state, selected_row_state]
    )
    
    # Handle row selection in dataframe
    data_display.select(
        fn=on_row_select,
        outputs=[selected_row_state]
    )
    
    create_btn.click(
        fn=show_create_modal,
        inputs=[columns_state],
        outputs=create_fields + [create_status, create_modal, edit_modal, delete_modal]
    )
    
    create_save_btn.click(
        fn=create_new_record,
        inputs=[table_name_input, columns_state] + create_fields,
        outputs=[create_status, data_display, selected_row_state, create_modal]
    )
    
    create_cancel_btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=[create_modal]
    )
    
    edit_btn.click(
        fn=prepare_edit,
        inputs=[table_name_input, columns_state, selected_row_state],
        outputs=edit_fields + [edit_status, edit_modal, edit_original_pk_state, create_modal, delete_modal]
    )
    
    edit_save_btn.click(
        fn=save_edit,
        inputs=[table_name_input, columns_state, edit_original_pk_state] + edit_fields,
        outputs=[edit_status, data_display, edit_modal, selected_row_state]
    )
    
    edit_cancel_btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=[edit_modal]
    )
    
    delete_btn.click(
        fn=prepare_delete,
        inputs=[table_name_input, columns_state, selected_row_state],
        outputs=[delete_summary, delete_modal, delete_key_state, delete_status, create_modal, edit_modal]
    )
    
    delete_confirm_btn.click(
        fn=confirm_delete,
        inputs=[table_name_input, columns_state, delete_key_state],
        outputs=[delete_status, data_display, delete_modal, selected_row_state]
    )
    
    delete_cancel_btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=[delete_modal]
    )

if __name__ == "__main__":
    demo.launch()

