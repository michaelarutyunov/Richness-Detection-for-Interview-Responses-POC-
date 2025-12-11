            with gr.Row():
                download_graph_btn = gr.Button("📥 Download Graph Data (JSON)", size="sm")
                graph_data_file = gr.File(
                    label="Graph Data File",
                    visible=True,
                )
                graph_summary = gr.Textbox(
                    label="Graph Summary",
                    value="No graph data yet",
                    interactive=False,
                    lines=3
                )