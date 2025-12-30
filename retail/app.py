import gradio as gr
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config

# Configuration
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/862f1d757f0424f7"  # Replace with your actual HTTP path
TABLE_NAME = "ali_azzouz.retail.churn_prediction"

cfg = Config()  # Set the DATABRICKS_HOST environment variable when running locally

# Global connection (initialized once)
_connection = None

def get_connection():
    """Get or create a cached Databricks connection."""
    global _connection
    if _connection is None:
        _connection = sql.connect(
            server_hostname=cfg.host,
            http_path=DATABRICKS_HTTP_PATH,
            credentials_provider=lambda: cfg.authenticate,
        )
    return _connection

def read_churn_customers():
    """Read churn prediction data from Databricks table."""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = f"SELECT * FROM {TABLE_NAME}"
            cursor.execute(query)
            df = cursor.fetchall_arrow().to_pandas()
        return df, f"✅ Successfully retrieved {len(df)} customers from churn prediction table"
    except Exception as e:
        return pd.DataFrame(), f"❌ Error reading data: {str(e)}"

def generate_marketing_copy(customer_id):
    """Generate personalized marketing copy for a customer."""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = f"SELECT * FROM ali_azzouz.retail.generate_marketing_copy_for_id('{customer_id}')"
            cursor.execute(query)
            results = cursor.fetchall_arrow().to_pandas()
            
            if results.empty:
                return "⚠️ No marketing copy generated"
            
            # Format the results nicely
            output = ["📧 Marketing Copy Generated:\n"]
            for idx, row in results.iterrows():
                output.append("=" * 60)
                for col in results.columns:
                    output.append(f"{col}: {row[col]}")
                output.append("")
            
            return "\n".join(output)
    except Exception as e:
        return f"❌ Error generating marketing copy: {str(e)}"

def trigger_campaign(customer_id):
    """Trigger marketing campaign for a single customer."""
    if not customer_id or not customer_id.strip():
        return "⚠️ Please select a customer ID"
    
    customer_id = customer_id.strip()
    result = f"Customer ID: {customer_id}\n{'='*60}\n"
    marketing_copy = generate_marketing_copy(customer_id)
    result += marketing_copy
    
    return result

def send_message(customer_id, marketing_copy):
    """Simulate sending the marketing message to the customer."""
    if not customer_id or not customer_id.strip():
        return "⚠️ Please generate a marketing campaign first"
    
    if not marketing_copy or "⚠️" in marketing_copy or "❌" in marketing_copy:
        return "⚠️ Please generate a valid marketing campaign before sending"
    
    return f"✅ Marketing message successfully sent to customer: {customer_id}\n\n📧 Message delivered via email and SMS\n🕐 Sent at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"

def load_and_display_data():
    """Load churn customers and return both dataframe and status message."""
    df, status = read_churn_customers()
    return df, status

# Create Gradio Interface
with gr.Blocks(title="Churn Prevention Marketing Campaign", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎯 Customer Retention Marketing Campaign
        ### Proactive engagement for at-risk customers
        
        This app connects to your Databricks churn prediction model to identify at-risk customers 
        and generate personalized marketing campaigns to re-engage them.
        """
    )
    
    # Load data automatically on startup
    initial_df, initial_status = load_and_display_data()
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📊 At-Risk Customers")
            status_text = gr.Textbox(
                label="Status", 
                value=initial_status,
                interactive=False
            )
            churn_data = gr.Dataframe(
                label="Churn Prediction Data",
                value=initial_df,
                interactive=False,
                wrap=True
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📧 Marketing Campaign Generator")
            gr.Markdown(
                """
                Select a customer ID from the table above to generate personalized marketing copy.
                
                **Example:** `2d17d7cd-38ae-440d-8485-34ce4f8f3b46`
                """
            )
            customer_id_input = gr.Textbox(
                label="Customer ID",
                placeholder="Enter a single customer ID",
                lines=1
            )
            generate_btn = gr.Button("✨ Generate Marketing Campaign", variant="primary", size="lg")
            campaign_output = gr.Textbox(
                label="Generated Marketing Copy",
                lines=12,
                interactive=False
            )
            
            with gr.Row():
                send_btn = gr.Button("📤 Send Message", variant="secondary", size="lg")
            
            send_status = gr.Textbox(
                label="Delivery Status",
                lines=4,
                interactive=False
            )
    
    gr.Markdown(
        """
        ---
        **Configuration:**
        - **Table:** `ali_azzouz.retail.churn_prediction`
        - **HTTP Path:** `/sql/1.0/warehouses/862f1d757f0424f7`
        
        💡 **Tip:** Copy a customer ID from the table above, generate the marketing campaign, 
        and click "Send Message" to deliver the retention campaign.
        """
    )
    
    # Event handlers
    generate_btn.click(
        fn=trigger_campaign,
        inputs=[customer_id_input],
        outputs=[campaign_output]
    )
    
    send_btn.click(
        fn=send_message,
        inputs=[customer_id_input, campaign_output],
        outputs=[send_status]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()

