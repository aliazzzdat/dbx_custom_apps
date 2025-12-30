import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks import sql
from databricks.sdk.core import Config
from databricks.sdk import WorkspaceClient
from gradio import ChatMessage
from datetime import datetime
import os

# Initialize Databricks configuration
cfg = Config()
w = WorkspaceClient()

# Configuration
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01efe16a65e21836acefb797ae6a8fe4")
TABLE_NAME = "ali_azzouz.retail.churn_prediction"
DEFAULT_WAREHOUSE_ID = os.getenv("DATABRICKS_HTTP_PATH", "")

# Global connection cache
connection_cache = {}

def get_connection(warehouse_id):
    """Get or create a cached connection to Databricks"""
    if warehouse_id not in connection_cache:
        connection_cache[warehouse_id] = sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate,
        )
    return connection_cache[warehouse_id]

def read_table(table_name):
    """Read data from Databricks table"""
    try:
        conn = get_connection(DEFAULT_WAREHOUSE_ID)
        with conn.cursor() as cursor:
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        return f"Error reading table: {str(e)}"

def execute_query(query):
    """Execute a SQL query"""
    try:
        conn = get_connection(DEFAULT_WAREHOUSE_ID)
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
    except Exception as e:
        return f"Error executing query: {str(e)}"

# ==================== CRUD HELPER FUNCTIONS ====================

def get_table_schema(table_name):
    """Get column names from the table"""
    try:
        conn = get_connection(DEFAULT_WAREHOUSE_ID)
        with conn.cursor() as cursor:
            query = f"DESCRIBE TABLE {table_name}"
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [row[0] for row in result if row[0] not in ['# Partition Information', '# col_name', '']]
            return columns
    except Exception as e:
        return []

def insert_record(table_name, columns, values):
    """Insert a new record into the table"""
    try:
        conn = get_connection(DEFAULT_WAREHOUSE_ID)
        with conn.cursor() as cursor:
            # Build INSERT with named parameters
            column_names = ', '.join(columns)
            placeholders = ', '.join([f':{col}' for col in columns])
            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            
            # Create parameters dictionary
            params = {col: val if val != "" else None for col, val in zip(columns, values)}
            
            cursor.execute(query, params)
        return True, "Record inserted successfully!"
    except Exception as e:
        return False, f"Error inserting record: {str(e)}"

def update_record(table_name, columns, values, primary_key_col, primary_key_val):
    """Update an existing record in the table"""
    try:
        conn = get_connection(DEFAULT_WAREHOUSE_ID)
        with conn.cursor() as cursor:
            # Build SET clause with named parameters
            set_clause = ', '.join([f"{col} = :{col}" for col in columns])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key_col} = :where_id"
            
            # Create parameters dictionary
            params = {col: val if val != "" else None for col, val in zip(columns, values)}
            params['where_id'] = primary_key_val
            
            cursor.execute(query, params)
        return True, "Record updated successfully!"
    except Exception as e:
        return False, f"Error updating record: {str(e)}"

def delete_record(table_name, primary_key_col, primary_key_val):
    """Delete a record from the table"""
    try:
        conn = get_connection(DEFAULT_WAREHOUSE_ID)
        with conn.cursor() as cursor:
            # Use named parameter for DELETE
            query = f"DELETE FROM {table_name} WHERE {primary_key_col} = :id"
            params = {'id': primary_key_val}
            
            cursor.execute(query, params)
        return True, "Record deleted successfully!"
    except Exception as e:
        return False, f"Error deleting record: {str(e)}"

# ==================== TAB 1: KPI VISUALIZATIONS ====================

def create_kpi_dashboard():
    """Create comprehensive KPI dashboard for marketing"""
    try:
        df = read_table(TABLE_NAME)
        
        if isinstance(df, str):  # Error message
            return df, None, None, None, None, None
        
        # Calculate KPIs
        total_users = len(df)
        churned_users = df['churn'].sum()
        churn_rate = (churned_users / total_users * 100) if total_users > 0 else 0
        active_users = total_users - churned_users
        
        # KPI Metrics HTML
        kpi_html = f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3 style="margin: 0;">Total Users</h3>
                <h1 style="margin: 10px 0;">{total_users:,}</h1>
            </div>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3 style="margin: 0;">Churned Users</h3>
                <h1 style="margin: 10px 0;">{churned_users:,}</h1>
            </div>
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3 style="margin: 0;">Active Users</h3>
                <h1 style="margin: 10px 0;">{active_users:,}</h1>
            </div>
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3 style="margin: 0;">Churn Rate</h3>
                <h1 style="margin: 10px 0;">{churn_rate:.2f}%</h1>
            </div>
        </div>
        """
        
        # 1. Churn Distribution Pie Chart
        churn_dist = df['churn'].value_counts().reset_index()
        churn_dist.columns = ['Status', 'Count']
        churn_dist['Status'] = churn_dist['Status'].map({0: 'Active', 1: 'Churned'})
        fig_churn = px.pie(churn_dist, values='Count', names='Status', 
                           title='Churn Distribution',
                           color='Status',
                           color_discrete_map={'Active': '#00f2fe', 'Churned': '#f5576c'})
        
        # 2. Churn by Channel
        churn_by_channel = df.groupby('canal')['churn'].agg(['sum', 'count']).reset_index()
        churn_by_channel['churn_rate'] = (churn_by_channel['sum'] / churn_by_channel['count'] * 100)
        fig_channel = px.bar(churn_by_channel, x='canal', y='churn_rate',
                            title='Churn Rate by Channel (%)',
                            labels={'churn_rate': 'Churn Rate (%)', 'canal': 'Channel'},
                            color='churn_rate',
                            color_continuous_scale='Reds')
        
        # 3. Churn by Country
        churn_by_country = df.groupby('country')['churn'].agg(['sum', 'count']).reset_index()
        churn_by_country['churn_rate'] = (churn_by_country['sum'] / churn_by_country['count'] * 100)
        churn_by_country = churn_by_country.sort_values('churn_rate', ascending=False).head(10)
        fig_country = px.bar(churn_by_country, x='country', y='churn_rate',
                            title='Top 10 Countries by Churn Rate (%)',
                            labels={'churn_rate': 'Churn Rate (%)', 'country': 'Country'},
                            color='churn_rate',
                            color_continuous_scale='RdYlGn_r')
        
        # 4. Order Count vs Churn
        fig_orders = px.scatter(df, x='order_count', y='total_amount', 
                               color='churn',
                               title='Order Count vs Total Amount (Colored by Churn)',
                               labels={'order_count': 'Order Count', 'total_amount': 'Total Amount'},
                               color_discrete_map={0: '#00f2fe', 1: '#f5576c'},
                               opacity=0.6)
        
        # 5. Days Since Last Activity Distribution
        fig_activity = px.histogram(df, x='days_since_last_activity', 
                                   color='churn',
                                   title='Days Since Last Activity Distribution',
                                   labels={'days_since_last_activity': 'Days Since Last Activity'},
                                   color_discrete_map={0: '#00f2fe', 1: '#f5576c'},
                                   barmode='overlay',
                                   opacity=0.7)
        
        return kpi_html, fig_churn, fig_channel, fig_country, fig_orders, fig_activity
        
    except Exception as e:
        error_msg = f"Error creating dashboard: {str(e)}"
        return error_msg, None, None, None, None, None

# ==================== TAB 2: TABLE VIEWER WITH CRUD ====================

def load_table_data():
    """Load data from the churn prediction table"""
    try:
        df = read_table(TABLE_NAME)
        
        if isinstance(df, str):  # Error message
            return df, gr.update(visible=False), gr.update(visible=False), [], None
        
        columns = get_table_schema(TABLE_NAME)
        
        return df, gr.update(visible=True), gr.update(visible=True), columns, None
    except Exception as e:
        return f"Error loading table: {str(e)}", gr.update(visible=False), gr.update(visible=False), [], None

def search_table(search_term):
    """Search and filter table data"""
    try:
        df = read_table(TABLE_NAME)
        
        if isinstance(df, str):  # Error message
            return df
        
        if search_term:
            # Search across string columns
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            df = df[mask]
        
        return df
    except Exception as e:
        return f"Error searching table: {str(e)}"

def on_row_select(evt: gr.SelectData):
    """Capture the selected row index"""
    return evt.index[0]

def show_create_modal(columns):
    """Show the Create modal with empty fields"""
    if not columns:
        return [gr.update(visible=False) for _ in range(15)] + [
            gr.update(value="Table not loaded", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        ]
    
    updates = []
    for i, col in enumerate(columns[:15]):
        updates.append(gr.update(visible=True, label=col, value=""))
    
    while len(updates) < 15:
        updates.append(gr.update(visible=False))
    
    updates.append(gr.update(value="", visible=False))  # status
    updates.append(gr.update(visible=True))  # create_modal visibility
    updates.append(gr.update(visible=False))  # edit_modal visibility
    updates.append(gr.update(visible=False))  # delete_modal visibility
    
    return updates

def create_new_record(columns, *field_values):
    """Create a new record with the provided field values"""
    if not columns:
        return gr.update(value="Please load table first", visible=True), None, None, gr.update(visible=True)
    
    try:
        values = list(field_values[:len(columns)])
        success, message = insert_record(TABLE_NAME, columns, values)
        
        if success:
            df = read_table(TABLE_NAME)
            return gr.update(value=message, visible=True), df, None, gr.update(visible=False)
        else:
            return gr.update(value=message, visible=True), None, None, gr.update(visible=True)
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), None, None, gr.update(visible=True)

def prepare_edit(columns, selected_row):
    """Prepare the edit modal with selected row data"""
    if not columns:
        return [gr.update() for _ in range(15)] + [
            gr.update(value="Please load table first", visible=True),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        ]
    
    if selected_row is None:
        return [gr.update() for _ in range(15)] + [
            gr.update(value="Please select a row from the table first", visible=True),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        ]
    
    try:
        df = read_table(TABLE_NAME)
        
        if isinstance(df, str) or selected_row >= len(df):
            return [gr.update() for _ in range(15)] + [
                gr.update(value="Invalid row selection", visible=True),
                gr.update(visible=False),
                "",
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        
        row_data = df.iloc[selected_row]
        updates = []
        
        for i, col in enumerate(columns[:15]):
            if i < len(columns):
                updates.append(gr.update(value=str(row_data[col]), visible=True, label=col))
            else:
                updates.append(gr.update(visible=False))
        
        while len(updates) < 15:
            updates.append(gr.update(visible=False))
        
        original_primary_key = str(row_data[columns[0]])
        
        updates.append(gr.update(value="", visible=False))  # message
        updates.append(gr.update(visible=True))  # edit_modal visibility
        updates.append(original_primary_key)  # original primary key value
        updates.append(gr.update(visible=False))  # create_modal visibility
        updates.append(gr.update(visible=False))  # delete_modal visibility
        
        return updates
    except Exception as e:
        return [gr.update() for _ in range(15)] + [
            gr.update(value=f"Error: {str(e)}", visible=True),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
            gr.update(visible=False)
        ]

def save_edit(columns, original_pk_value, *field_values):
    """Save the edited record"""
    if not columns:
        return gr.update(value="Please load table first", visible=True), None, gr.update(visible=True), None
    
    try:
        primary_key_col = columns[0]
        values = list(field_values[:len(columns)])
        
        success, message = update_record(TABLE_NAME, columns, values, primary_key_col, original_pk_value)
        
        if success:
            df = read_table(TABLE_NAME)
            return gr.update(value=message, visible=True), df, gr.update(visible=False), None
        else:
            return gr.update(value=message, visible=True), None, gr.update(visible=True), None
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), None, gr.update(visible=True), None

def prepare_delete(columns, selected_row):
    """Prepare the delete modal with selected row data"""
    if not columns:
        return "Please load table first", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    
    if selected_row is None:
        return "Please select a row from the table first", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    
    try:
        df = read_table(TABLE_NAME)
        
        if isinstance(df, str) or selected_row >= len(df):
            return "Invalid row selection", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
        
        row_data = df.iloc[selected_row]
        primary_key_col = columns[0]
        primary_key_val = row_data[primary_key_col]
        
        summary = f"Are you sure you want to delete the record with {primary_key_col} = {primary_key_val}?"
        
        return summary, gr.update(visible=True), str(primary_key_val), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return f"Error: {str(e)}", gr.update(visible=False), "", gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)

def confirm_delete(columns, primary_key_val):
    """Confirm and execute the delete operation"""
    if not columns:
        return gr.update(value="Please load table first", visible=True), None, gr.update(visible=True), None
    
    try:
        primary_key_col = columns[0]
        
        success, message = delete_record(TABLE_NAME, primary_key_col, primary_key_val)
        
        if success:
            df = read_table(TABLE_NAME)
            return gr.update(value=message, visible=True), df, gr.update(visible=False), None
        else:
            return gr.update(value=message, visible=True), None, gr.update(visible=True), None
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), None, gr.update(visible=True), None

# ==================== TAB 3: GENIE CHATBOT ====================

# Sample questions for Head of Marketing
SAMPLE_QUESTIONS = [
    "What is the churn rate by country and which countries have the highest churn?",
    "Show me the correlation between order count and churn probability for customers who haven't made a purchase in the last 30 days",
    "What is the average total amount spent by churned vs active customers segmented by age group and channel?"
]

# Conversation state
conversation_id = None

def get_query_result(statement_id):
    """Get query result from Genie"""
    try:
        result = w.statement_execution.get_statement(statement_id)
        return pd.DataFrame(
            result.result.data_array, 
            columns=[i.name for i in result.manifest.schema.columns]
        )
    except Exception:
        return pd.DataFrame()

def process_genie_response(response):
    """Process and format Genie response"""
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

def sanitize_message_content(content):
    """Sanitize message content to remove invalid characters"""
    if not isinstance(content, str):
        content = str(content)
    
    content = content.replace('\x00', '')
    content = content.strip()
    
    return content

def chat_with_genie(message, history):
    """Chat with Genie AI"""
    global conversation_id
    
    if not message.strip():
        return history, ""
    
    try:
        if cfg is None:
            error_msg = "Databricks credentials not configured. Please set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        history.append({"role": "user", "content": message})
        
        # Query Genie
        if conversation_id:
            conversation = w.genie.create_message_and_wait(
                GENIE_SPACE_ID, conversation_id, message
            )
        else:
            conversation = w.genie.start_conversation_and_wait(GENIE_SPACE_ID, message)
            conversation_id = conversation.conversation_id
        
        # Process response
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
        
        # Ensure all messages are valid dictionaries
        for i, msg in enumerate(history):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                history = [m for m in history if isinstance(m, dict) and 'role' in m and 'content' in m]
                break
        
        return history, ""
        
    except Exception as e:
        error_msg = f"Error communicating with Genie: {str(e)}"
        history.append({"role": "assistant", "content": sanitize_message_content(error_msg)})
        
        # Ensure all messages are valid dictionaries
        for i, msg in enumerate(history):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                history = [m for m in history if isinstance(m, dict) and 'role' in m and 'content' in m]
                break
        
        return history, ""

def use_sample_question(question):
    """Use a sample question"""
    return question

def reset_conversation():
    """Reset conversation state"""
    global conversation_id
    conversation_id = None
    return [], ""

# ==================== TAB 4: MARKETING ACTIONS/CAMPAIGNS ====================

def trigger_campaign(campaign_type, target_segment, channel, message):
    """Trigger a fake marketing campaign"""
    try:
        # Get target customers based on segment
        if target_segment == "High Risk Churners":
            query = f"""
            SELECT user_id, email, firstname, lastname, churn 
            FROM {TABLE_NAME} 
            WHERE churn = 1 
            LIMIT 100
            """
        elif target_segment == "Inactive Users (30+ days)":
            query = f"""
            SELECT user_id, email, firstname, lastname, days_since_last_activity 
            FROM {TABLE_NAME} 
            WHERE days_since_last_activity >= 30 
            LIMIT 100
            """
        elif target_segment == "Low Order Count":
            query = f"""
            SELECT user_id, email, firstname, lastname, order_count 
            FROM {TABLE_NAME} 
            WHERE order_count < 3 
            LIMIT 100
            """
        else:  # All Customers
            query = f"""
            SELECT user_id, email, firstname, lastname 
            FROM {TABLE_NAME} 
            LIMIT 100
            """
        
        df = execute_query(query)
        
        if isinstance(df, str):  # Error
            return df, None
        
        # Create campaign summary
        summary = f"""
        ## 🚀 Campaign Launched Successfully!
        
        **Campaign Type:** {campaign_type}
        **Target Segment:** {target_segment}
        **Channel:** {channel}
        **Message:** {message}
        
        **Targets:** {len(df)} customers
        **Status:** ✅ Active
        **Launch Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ### Next Steps:
        - Monitor engagement metrics in the next 24-48 hours
        - Track conversion rates and customer responses
        - Adjust messaging based on performance
        """
        
        return summary, df
        
    except Exception as e:
        return f"Error launching campaign: {str(e)}", None

# ==================== TAB 5: PERSONALIZED MESSAGES ====================

def generate_personalized_message(user_id):
    """Generate personalized message for a specific customer"""
    try:
        # Get customer data
        query = f"""
        SELECT 
            user_id,
            firstname,
            platform,
            country,
            gender,
            total_amount,
            order_count,
            creation_date,
            last_transaction,
            days_since_last_activity
        FROM {TABLE_NAME}
        WHERE user_id = '{user_id}'
        """
        
        df = execute_query(query)
        
        if isinstance(df, str) or df.empty:
            return "Customer not found", None
        
        customer = df.iloc[0]
        
        # Generate marketing copy using the SQL function
        message_query = f"""
        SELECT generate_marketing_copy(
            '{customer['user_id']}',
            '{customer['firstname']}',
            '{customer['platform']}',
            '{customer['country']}',
            {customer['gender']},
            {customer['total_amount']},
            {customer['order_count']},
            '{customer['creation_date']}',
            '{customer['last_transaction']}',
            {customer['days_since_last_activity']},
            churn_predictor('{customer['user_id']}')
        ) as marketing_message
        FROM {TABLE_NAME}
        WHERE user_id = '{customer['user_id']}'
        LIMIT 1
        """
        
        message_df = execute_query(message_query)
        
        if isinstance(message_df, str):
            return f"Error generating message: {message_df}", df
        
        marketing_message = message_df.iloc[0]['marketing_message'] if not message_df.empty else "No message generated"
        
        result = f"""
        ## 📧 Personalized Marketing Message
        
        **Customer:** {customer['firstname']}
        **User ID:** {customer['user_id']}
        **Country:** {customer['country']}
        **Platform:** {customer['platform']}
        **Total Orders:** {customer['order_count']}
        **Total Spend:** ${customer['total_amount']}
        **Days Inactive:** {customer['days_since_last_activity']}
        
        ---
        
        ### Generated Message:
        
        {marketing_message}
        """
        
        return result, df
        
    except Exception as e:
        return f"Error generating personalized message: {str(e)}", None

def search_churned_customers(limit=20):
    """Get list of churned customers"""
    try:
        query = f"""
        SELECT user_id, firstname, lastname, email, 
               order_count, total_amount, days_since_last_activity
        FROM {TABLE_NAME}
        WHERE churn = 1
        LIMIT {limit}
        """
        
        df = execute_query(query)
        return df
        
    except Exception as e:
        return f"Error fetching churned customers: {str(e)}"

# ==================== GRADIO APP ====================

def create_app():
    """Create the main Gradio application"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Marketing Analytics Dashboard") as app:
        gr.Markdown("""
        # 📊 Marketing Analytics & Churn Prediction Dashboard
        **Head of Data Marketing Portal** - Retail Churn Prediction & Customer Engagement
        """)
        
        with gr.Tabs() as tabs:
            # ========== TAB 1: KPI DASHBOARD ==========
            with gr.Tab("📈 KPI Dashboard"):
                gr.Markdown("### Real-time Marketing KPIs and Churn Analytics")
                
                refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
                
                kpi_metrics = gr.HTML()
                
                with gr.Row():
                    churn_pie = gr.Plot(label="Churn Distribution")
                    channel_bar = gr.Plot(label="Churn by Channel")
                
                with gr.Row():
                    country_bar = gr.Plot(label="Top Countries by Churn")
                    orders_scatter = gr.Plot(label="Orders vs Amount")
                
                activity_hist = gr.Plot(label="Days Since Last Activity")
                
                refresh_btn.click(
                    fn=create_kpi_dashboard,
                    inputs=[],
                    outputs=[kpi_metrics, churn_pie, channel_bar, country_bar, orders_scatter, activity_hist]
                )
            
            # ========== TAB 2: DATA EXPLORER WITH CRUD ==========
            with gr.Tab("🔍 Data Explorer & CRUD"):
                gr.Markdown("### Browse, Search, and Manage Customer Data")
                
                # State to store table columns and selected row
                columns_state = gr.State([])
                selected_row_state = gr.State(None)
                
                with gr.Row():
                    load_table_btn = gr.Button("📋 Load Table", variant="primary", scale=1)
                    search_input = gr.Textbox(
                        label="Search",
                        placeholder="Enter search term (email, name, country, etc.)",
                        scale=3
                    )
                    search_btn = gr.Button("🔍 Search", scale=1)
                
                # Status message
                status_msg = gr.Textbox(label="Status", interactive=False, visible=False)
                
                # Data display
                table_output = gr.Dataframe(
                    label="Customer Data",
                    wrap=True,
                    interactive=False
                )
                
                # Action buttons
                with gr.Row(visible=False) as action_row:
                    create_btn = gr.Button("➕ Create New Record", variant="primary")
                    edit_btn = gr.Button("✏️ Edit Selected", variant="secondary")
                    delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                
                instructions_text = gr.Markdown(
                    "*To edit or delete: click on a row in the table above, then click the corresponding button*",
                    visible=False
                )
                
                # ===== CREATE MODAL =====
                with gr.Column(visible=False) as create_modal:
                    with gr.Group():
                        gr.Markdown("## ➕ Create New Customer Record")
                        gr.Markdown("Fill in the fields below to create a new customer record")
                        create_fields = []
                        for i in range(15):  # Support up to 15 columns
                            field = gr.Textbox(label=f"Field {i+1}", visible=False)
                            create_fields.append(field)
                        
                        create_status = gr.Textbox(label="Status", interactive=False, visible=False)
                        
                        with gr.Row():
                            create_save_btn = gr.Button("💾 Save New Record", variant="primary", scale=1)
                            create_cancel_btn = gr.Button("❌ Cancel", variant="secondary", scale=1)
                
                # ===== EDIT MODAL =====
                with gr.Column(visible=False) as edit_modal:
                    with gr.Group():
                        gr.Markdown("## ✏️ Edit Customer Record")
                        gr.Markdown("Modify the fields below to update the customer record")
                        edit_fields = []
                        for i in range(15):  # Support up to 15 columns
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
                        gr.Markdown("## 🗑️ Delete Customer Record")
                        gr.Markdown("⚠️ **Warning:** This action cannot be undone!")
                        delete_summary = gr.Textbox(label="Confirmation", interactive=False)
                        delete_key_state = gr.State("")
                        
                        delete_status = gr.Textbox(label="Status", interactive=False, visible=False)
                        
                        with gr.Row():
                            delete_confirm_btn = gr.Button("🗑️ Confirm Delete", variant="stop", scale=1)
                            delete_cancel_btn = gr.Button("❌ Cancel", variant="secondary", scale=1)
                
                # Event handlers for CRUD operations
                load_table_btn.click(
                    fn=load_table_data,
                    outputs=[table_output, action_row, instructions_text, columns_state, selected_row_state]
                )
                
                search_btn.click(
                    fn=search_table,
                    inputs=[search_input],
                    outputs=table_output
                )
                
                table_output.select(
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
                    inputs=[columns_state] + create_fields,
                    outputs=[create_status, table_output, selected_row_state, create_modal]
                )
                
                create_cancel_btn.click(
                    fn=lambda: gr.update(visible=False),
                    outputs=[create_modal]
                )
                
                edit_btn.click(
                    fn=prepare_edit,
                    inputs=[columns_state, selected_row_state],
                    outputs=edit_fields + [edit_status, edit_modal, edit_original_pk_state, create_modal, delete_modal]
                )
                
                edit_save_btn.click(
                    fn=save_edit,
                    inputs=[columns_state, edit_original_pk_state] + edit_fields,
                    outputs=[edit_status, table_output, edit_modal, selected_row_state]
                )
                
                edit_cancel_btn.click(
                    fn=lambda: gr.update(visible=False),
                    outputs=[edit_modal]
                )
                
                delete_btn.click(
                    fn=prepare_delete,
                    inputs=[columns_state, selected_row_state],
                    outputs=[delete_summary, delete_modal, delete_key_state, delete_status, create_modal, edit_modal]
                )
                
                delete_confirm_btn.click(
                    fn=confirm_delete,
                    inputs=[columns_state, delete_key_state],
                    outputs=[delete_status, table_output, delete_modal, selected_row_state]
                )
                
                delete_cancel_btn.click(
                    fn=lambda: gr.update(visible=False),
                    outputs=[delete_modal]
                )
            
            # ========== TAB 3: GENIE CHATBOT ==========
            with gr.Tab("🤖 Genie AI Assistant"):
                gr.Markdown("### Ask Questions About Your Data")
                
                gr.Markdown("**Sample Questions for Marketing Analysis:**")
                
                with gr.Row():
                    for i, q in enumerate(SAMPLE_QUESTIONS):
                        sample_btn = gr.Button(f"💡 Sample {i+1}", scale=1)
                
                chatbot = gr.Chatbot(
                    label="Genie Chat",
                    height=400,
                    type="messages"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your customer data...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Conversation")
                
                # Sample question buttons
                for i, q in enumerate(SAMPLE_QUESTIONS):
                    sample_btn = gr.Button(f"💡 Sample {i+1}", visible=False)
                    sample_btn.click(
                        fn=lambda q=q: q,
                        outputs=msg_input
                    )
                
                # Set up sample question buttons properly
                sample_btns = []
                with gr.Row():
                    for i, q in enumerate(SAMPLE_QUESTIONS):
                        btn = gr.Button(f"📌 Q{i+1}", scale=1)
                        sample_btns.append(btn)
                        btn.click(fn=lambda q=q: q, outputs=msg_input)
                
                send_btn.click(
                    fn=chat_with_genie,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )
                
                msg_input.submit(
                    fn=chat_with_genie,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )
                
                clear_btn.click(
                    fn=reset_conversation,
                    outputs=[chatbot, msg_input]
                )
            
            # ========== TAB 4: MARKETING CAMPAIGNS ==========
            with gr.Tab("🎯 Marketing Campaigns"):
                gr.Markdown("### Launch Retention Campaigns & Actions")
                
                with gr.Row():
                    campaign_type = gr.Dropdown(
                        label="Campaign Type",
                        choices=[
                            "Email Campaign",
                            "SMS Campaign",
                            "Push Notification",
                            "Discount Offer",
                            "Loyalty Reward",
                            "Re-engagement Campaign"
                        ],
                        value="Email Campaign"
                    )
                    
                    target_segment = gr.Dropdown(
                        label="Target Segment",
                        choices=[
                            "High Risk Churners",
                            "Inactive Users (30+ days)",
                            "Low Order Count",
                            "All Customers"
                        ],
                        value="High Risk Churners"
                    )
                
                with gr.Row():
                    channel = gr.Dropdown(
                        label="Channel",
                        choices=["Email", "SMS", "Push", "In-App"],
                        value="Email"
                    )
                    
                campaign_message = gr.Textbox(
                    label="Campaign Message",
                    placeholder="Enter your marketing message...",
                    lines=3,
                    value="We miss you! Come back and enjoy 20% off your next purchase."
                )
                
                launch_btn = gr.Button("🚀 Launch Campaign", variant="primary", size="lg")
                
                campaign_summary = gr.Markdown()
                campaign_targets = gr.Dataframe(label="Target Customers")
                
                launch_btn.click(
                    fn=trigger_campaign,
                    inputs=[campaign_type, target_segment, channel, campaign_message],
                    outputs=[campaign_summary, campaign_targets]
                )
            
            # ========== TAB 5: PERSONALIZED MESSAGES ==========
            with gr.Tab("✉️ Personalized Messages"):
                gr.Markdown("### Generate AI-Powered Personalized Messages for Churned Customers")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Select a Customer:**")
                        
                        load_customers_btn = gr.Button("📋 Load Churned Customers")
                        
                        churned_customers = gr.Dataframe(
                            label="Churned Customers",
                            interactive=False,
                            wrap=True
                        )
                        
                        user_id_input = gr.Textbox(
                            label="User ID",
                            placeholder="Enter or select User ID from table above"
                        )
                        
                        generate_btn = gr.Button("✨ Generate Message", variant="primary")
                    
                    with gr.Column(scale=2):
                        message_output = gr.Markdown(label="Generated Message")
                        customer_details = gr.Dataframe(label="Customer Details")
                
                load_customers_btn.click(
                    fn=search_churned_customers,
                    inputs=[],
                    outputs=churned_customers
                )
                
                generate_btn.click(
                    fn=generate_personalized_message,
                    inputs=[user_id_input],
                    outputs=[message_output, customer_details]
                )
        
        gr.Markdown("""
        ---
        *Dashboard powered by Gradio & Databricks Genie AI*
        """)
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()

