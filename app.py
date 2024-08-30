import gradio as gr
from databricks import sql


# TODO
# retrive dynamically all columns names from created colummns for filter + agg
# --retrive dynamically all columns names from joined colummns for filter + agg
# --sql query engine create + select
# --hide outputs
# propose more transformations/agg and select data_type to propose relevant functions ?
# avoid sql injection in custom
# --column to select and sorting
# --clear button
# add union, except
# --add try except at each query
# put cursor as variable instead of global

TEST = False
CURSOR = None


def set_connection(db_host, db_token, db_warehouse_id):
    global CURSOR
    if TEST: 
        CURSOR = True
        return True
    elif all(v is not None for v in [db_host, db_token, db_warehouse_id]):
        try:
            connection = sql.connect(
                server_hostname=db_host,
                http_path=f"/sql/1.0/warehouses/{db_warehouse_id}",
                access_token=db_token
            )
            CURSOR = connection.cursor()
            return True
        except Exception as e:
            print(e)
            print(db_host, db_token, db_warehouse_id)
            CURSOR = None
            return False


def get_catalogs_names():
    if TEST:
        catalogs = ['catalog1', 'catalog2']
    else:
        if CURSOR is not None:
            try:
                CURSOR.execute("SHOW CATALOGS;")
                df = CURSOR.fetchall_arrow().to_pandas()
                catalogs = df['catalog'].tolist()
            except Exception as e:
                print(e)
                catalogs = ['error']
        else:
            catalogs = ['error']
    return gr.update(choices=catalogs, value=None)


def get_schemas_names(catalog):
    if TEST:
        schemas = ['schemas1', 'schemas2', 'schemas3']
    else:
        if CURSOR is not None:
            try:
                CURSOR.execute(f"SHOW SCHEMAS IN {catalog};")
                df = CURSOR.fetchall_arrow().to_pandas()
                schemas = df['databaseName'].tolist()
            except Exception as e:
                print(e)
                schemas = ['error']
        else:
            schemas = ['error']
    schemas = list(map(lambda x: f"{catalog}.{x}", schemas))
    return gr.update(choices=schemas, value=None)


def get_tables_names(schema):
    if TEST:
        tables = ['tablesA', 'tablesB', 'tablesC']
    else:
        if CURSOR is not None:
            try:
                CURSOR.execute(f"SHOW TABLES IN {schema};")
                df = CURSOR.fetchall_arrow().to_pandas()
                tables = df['tableName'].tolist()
            except Exception as e:
                print(e)
                tables = ['error']
        else:
            tables = ['error']
    tables = list(map(lambda x: f"{schema}.{x}", tables))
    return gr.update(choices=tables, value=None)


def get_columns_names(table):
    if TEST:
        columns = ['columns0', 'columns1', 'columns2']
    else:
        if CURSOR is not None:
            try:
                CURSOR.execute(f"DESCRIBE {table};")
                df = CURSOR.fetchall_arrow().to_pandas()
                columns = df['col_name'].tolist()
            except Exception as e:
                print(e)
                columns = ['error']
        else:
            columns = ['error']
    columns = list(map(lambda x: f"{table}.{x}", columns))
    return gr.update(choices=columns, value=None)


def build_sql_query(main_df, joins, filters, aggregations_keys, aggregations_metrics, customs, selected_columns, sorting):
    query = f"SELECT "

    if selected_columns:
        query += f"{', '.join(selected_columns)}"
    elif aggregations_keys:
        query += f"{aggregations_keys}"
        query += f"{aggregations_metrics}"
    else:
        query += "*"

    if customs:
        query += f", {customs}"

    query += f"\nFROM {main_df}"

    if joins:
        query += f"\n{joins}"

    if filters:
        query += f"\nWHERE {filters}"

    if aggregations_keys:
        query += f"\nGROUP BY {aggregations_keys}"

    if sorting:
        query += f"\nORDER BY {sorting}"

    return query


def create_table(main_df, joins, filters, aggregations, customs):
    query = "CREATE TABLE APP_TABLE AS "
    query += build_sql_query(main_df, joins, filters, aggregations, customs)
    return query

def run_sql_query(query):
    CURSOR.execute(query)
    return CURSOR.fetchall_arrow().to_pandas()

def return_joins_results(*args):
    joins_string = ""
    for i in range(0, len(args), 4):
        joins_string += f"{args[i]} JOIN {args[i+1]
                                          } on {args[i+3]} = {args[i+2]} \n"
    return joins_string

def return_customs_results(*args):
    return ", ".join(args)


def return_filters_results(*args):
    filters_string = ""
    filters_string += f"({args[0]} ({args[1]} {args[2]} {args[3]})) "
    for i in range(4, len(args), 5):
        filters_string += f"{args[i]} ({args[i+1]
                                        } ({args[i+2]} {args[i+3]} {args[i+4]})) "
    return filters_string


def return_aggregations_keys_results(*args):
    return ", ".join(args[0])


def return_aggregations_metrics_results(*args):
    aggregations_metrics_string = ""
    for i in range(0, len(args), 2):
        aggregations_metrics_string += f", {args[i+1]}({args[i]})"
    return aggregations_metrics_string


def get_all_columns(main_table, joins_output):
    all_columns = []
    if main_table:
        all_columns.extend(get_columns_names(main_table)['choices'])

    if joins_output:
        joined_tables = [j.split()[2] for j in joins_output.split('\n') if j]
        for table in joined_tables:
            all_columns.extend(get_columns_names(table)['choices'])

    return list(set(all_columns))  # Remove duplicates


with gr.Blocks() as demo:

    with gr.Row():
        databricks_token = gr.Textbox(
            label="DATABRICKS_TOKEN", type="password")
        databricks_host = gr.Textbox(label="DATABRICKS_HOST")
        databricks_warehouse_id = gr.Textbox(label="DATABRICKS_WAREHOUSE_ID")
        check_connection = gr.Checkbox(label="Connected", interactive=False, value=False)
        #check_connection = gr.Textbox(label="Connected", interactive=False)
        connect = gr.Button("Connect")

        connect.click(set_connection, [
                      databricks_host, databricks_token, databricks_warehouse_id], check_connection)

    with gr.Row():

        catalogs_list = gr.Dropdown(
            choices=[], label="Catalogs list", interactive=True)
        check_connection.change(fn=get_catalogs_names, outputs=catalogs_list)

        schemas_list = gr.Dropdown(
            choices=[], label="Schemas list", interactive=True)
        tables_list = gr.Dropdown(
            choices=[], label="Tables list", interactive=True)
        selected_columns = gr.Dropdown(
            choices=[], label="Select columns to display", multiselect=True, interactive=True)
        sorting_column = gr.Dropdown(
            choices=[], label="Select column for sorting", interactive=True)
        sorting_order = gr.Radio(
            choices=["ASC", "DESC"], label="Sorting order", interactive=True)
        sorting_output = gr.Textbox(label="Sorting output", visible=TEST)

        catalogs_list.change(fn=get_schemas_names,
                             inputs=catalogs_list, outputs=schemas_list)
        schemas_list.change(fn=get_tables_names,
                            inputs=schemas_list, outputs=tables_list)
        tables_list.change(get_columns_names,
                           inputs=tables_list, outputs=selected_columns)

        build_sql_btn = gr.Button("Build SQL Query")
        sql_output = gr.Textbox(label="Generated SQL Query")
        run_sql_btn = gr.Button("View Query")
        clear_btn = gr.Button("Clear", variant="stop")

        def update_sorting_column(columns):
            return gr.update(choices=columns)

        selected_columns.change(update_sorting_column,
                                inputs=selected_columns, outputs=sorting_column)

        def update_sorting_output(column, order):
            if column and order:
                return f"{column} {order}"
            return None

        sorting_column.change(update_sorting_output, inputs=[
                              sorting_column, sorting_order], outputs=sorting_output)
        sorting_order.change(update_sorting_output, inputs=[
                             sorting_column, sorting_order], outputs=sorting_output)

    # joins
    with gr.Row():

        joins_count = gr.State(0)
        joins_add_btn = gr.Button("Add join")
        joins_add_btn.click(lambda x: x + 1, joins_count, joins_count)
        joins_remove_btn = gr.Button("Remove join")
        joins_remove_btn.click(
            lambda x: 0 if x == 0 else x - 1, joins_count, joins_count)
        joins_output = gr.Textbox(label="joins merged output", visible=TEST)
        joins_remove_btn.click(lambda x: None if x <
                               2 else x, joins_count, joins_output)

    @gr.render(inputs=joins_count)
    def render_joins(count):
        joins_boxes = []
        for i in range(count):

            with gr.Row():
                main_col = gr.Dropdown(
                    label="Main Table Column", interactive=True)
                tables_list.change(get_columns_names,
                                   inputs=tables_list, outputs=main_col)

                catalogs_join_list = gr.Dropdown(choices=get_catalogs_names(
                ), label="Catalogs list table to join", interactive=True)
                schemas_join_list = gr.Dropdown(
                    choices=[], label="Schemas list table to join", interactive=True)
                tables_join_list = gr.Dropdown(
                    choices=[], label="Tables list table to join", interactive=True)
                columns_join_list = gr.Dropdown(
                    choices=[], label="Columns list table to join", interactive=True)
                join_type = gr.Dropdown(
                    choices=["INNER", "LEFT", "RIGHT", "OUTER"], label="join Type")

                catalogs_join_list.change(
                    fn=get_schemas_names, inputs=catalogs_join_list, outputs=schemas_join_list)
                schemas_join_list.change(
                    fn=get_tables_names, inputs=schemas_join_list, outputs=tables_join_list)
                tables_join_list.change(
                    get_columns_names, inputs=tables_join_list, outputs=columns_join_list)

                joins_boxes.append(join_type)
                joins_boxes.append(tables_join_list)
                joins_boxes.append(main_col)
                joins_boxes.append(columns_join_list)

                join_type.change(return_joins_results,
                                 inputs=joins_boxes, outputs=[joins_output])
                tables_join_list.change(
                    return_joins_results, inputs=joins_boxes, outputs=joins_output)
                main_col.change(return_joins_results,
                                inputs=joins_boxes, outputs=[joins_output])
                columns_join_list.change(
                    return_joins_results, inputs=joins_boxes, outputs=joins_output)

    # customs
    with gr.Row():
        customs_count = gr.State(0)
        customs_add_btn = gr.Button("Add custom")
        customs_add_btn.click(lambda x: x + 1, customs_count, customs_count)
        customs_remove_btn = gr.Button("Remove custom")
        customs_remove_btn.click(
            lambda x: 0 if x == 0 else x - 1, customs_count, customs_count)
        customs_output = gr.Textbox(
            label="Customs merged output", visible=TEST)
        customs_remove_btn.click(lambda x: None if x <
                                 2 else x, customs_count, customs_output)

    @gr.render(inputs=customs_count)
    def render_customs(count):
        customs_boxes = []
        for i in range(count):

            with gr.Row():
                txt_box = gr.Textbox(
                    label="Custom merged output ((column1 + column2) as total)")
                customs_boxes.append(txt_box)

                txt_box.change(return_customs_results,
                               inputs=customs_boxes, outputs=[customs_output])

    # filters
    with gr.Row():
        filters_count = gr.State(0)
        filters_add_btn = gr.Button("Add filter")
        filters_add_btn.click(lambda x: x + 1, filters_count, filters_count)
        filters_remove_btn = gr.Button("Remove filter")
        filters_remove_btn.click(
            lambda x: 0 if x == 0 else x - 1, filters_count, filters_count)
        filters_output = gr.Textbox(
            label="Filters merged output", visible=TEST)
        filters_remove_btn.click(lambda x: None if x <
                                 2 else x, filters_count, filters_output)

    @gr.render(inputs=filters_count)
    def render_filters(count):
        filter_boxes = []
        for i in range(count):

            with gr.Row():
                if i > 0:
                    bool_col = gr.Dropdown(
                        label="Boolean Filter", choices=['AND', 'OR'])
                    filter_boxes.append(bool_col)
                    bool_col.change(return_filters_results,
                                    inputs=filter_boxes, outputs=[filters_output])
                not_col = gr.Dropdown(label="NOT filter", choices=['', 'NOT'])
                main_col = gr.Dropdown(
                    label="Main Table Column Filter", interactive=True)
                operator_col = gr.Dropdown(label="Operator Filter", choices=[
                                           '=', '!=', '>', '<', '>=', '<=', 'IN', 'LIKE'])
                value_col = gr.Textbox(label="Filters value")

                def update_filter_columns(main_table, joins):
                    all_columns = get_all_columns(main_table, joins)
                    return gr.update(choices=all_columns)

                tables_list.change(update_filter_columns,
                                   inputs=[tables_list, joins_output], outputs=main_col)
                joins_output.change(update_filter_columns,
                                    inputs=[tables_list, joins_output], outputs=main_col)

                filter_boxes.append(not_col)
                filter_boxes.append(main_col)
                filter_boxes.append(operator_col)
                filter_boxes.append(value_col)

                not_col.change(return_filters_results,
                               inputs=filter_boxes, outputs=[filters_output])
                main_col.change(return_filters_results,
                                inputs=filter_boxes, outputs=[filters_output])
                operator_col.change(
                    return_filters_results, inputs=filter_boxes, outputs=[filters_output])
                value_col.change(return_filters_results,
                                 inputs=filter_boxes, outputs=[filters_output])

    # aggregations
    with gr.Row():
        aggregations_count = gr.State(0)
        aggregations_add_btn = gr.Button("Add aggregation")
        aggregations_add_btn.click(
            lambda x: x + 1, aggregations_count, aggregations_count)
        aggregations_remove_btn = gr.Button("Remove aggregation")
        aggregations_remove_btn.click(
            lambda x: 0 if x == 0 else x - 1, aggregations_count, aggregations_count)
        aggregations_metrics_output = gr.Textbox(
            label="Aggregations metrics merged output", visible=TEST)
        aggregations_keys_output = gr.Textbox(
            label="Aggregations keys merged output", visible=TEST)
        aggregations_remove_btn.click(
            lambda x: None if x < 2 else x, aggregations_count, aggregations_keys_output)
        aggregations_remove_btn.click(
            lambda x: None if x < 2 else x, aggregations_count, aggregations_metrics_output)

    @gr.render(inputs=aggregations_count)
    def render_aggregations(count):
        aggregations_boxes = []

        for i in range(count):
            if i == 0:
                with gr.Row():
                    key_col = gr.Dropdown(
                        label="Keys aggregation", multiselect=True)

                    def update_aggregation_columns(main_table, joins):
                        all_columns = get_all_columns(main_table, joins)
                        return gr.update(choices=all_columns)

                    tables_list.change(update_aggregation_columns,
                                       inputs=[tables_list, joins_output], outputs=key_col)
                    joins_output.change(update_aggregation_columns,
                                        inputs=[tables_list, joins_output], outputs=key_col)

                    key_col.change(return_aggregations_keys_results,
                                   inputs=key_col, outputs=aggregations_keys_output)

            with gr.Row():
                agg_col = gr.Dropdown(label="Agg aggregation")
                func_col = gr.Dropdown(label="Func aggregation", choices=[
                                       "MEAN", "COUNT", "SUM", "MIN", "MAX", "FIRST", "LAST", "CONCAT_WS"])

                tables_list.change(update_aggregation_columns,
                                   inputs=[tables_list, joins_output], outputs=agg_col)
                joins_output.change(update_aggregation_columns,
                                    inputs=[tables_list, joins_output], outputs=agg_col)
                tables_list.change(get_columns_names,
                                   inputs=tables_list, outputs=agg_col)

                aggregations_boxes.append(agg_col)
                aggregations_boxes.append(func_col)

                agg_col.change(return_aggregations_metrics_results,
                               inputs=aggregations_boxes, outputs=aggregations_metrics_output)
                func_col.change(return_aggregations_metrics_results,
                                inputs=aggregations_boxes, outputs=aggregations_metrics_output)

    query_result = gr.Dataframe()

    build_sql_btn.click(
        build_sql_query,
        inputs=[tables_list, joins_output, filters_output, aggregations_keys_output,
                aggregations_metrics_output, customs_output, selected_columns, sorting_output],
        outputs=sql_output
    )

    run_sql_btn.click(
        run_sql_query,
        inputs=sql_output,
        outputs=query_result
    )
 
    arrays_list = [selected_columns]
    counters_list = [joins_count, customs_count,
                     filters_count, aggregations_count]
    components_list = [catalogs_list, schemas_list, tables_list, sorting_column, sorting_order, sorting_output,
                       sql_output, joins_output, customs_output, filters_output, aggregations_metrics_output, aggregations_keys_output, query_result]

    def reset_all_arrays():
        return [gr.update(value=[]) for _ in range(len(arrays_list))]
    
    def reset_all_counters():
        return [gr.update(value=0) for _ in range(len(counters_list))]

    def reset_all_components():
        return [gr.update(value=None) for _ in range(len(components_list))]

    clear_btn.click(
        reset_all_arrays,
        outputs=arrays_list
    )

    clear_btn.click(
        reset_all_counters,
        outputs=counters_list
    )

    clear_btn.click(
        reset_all_components,
        outputs=components_list
    )

demo.launch()
