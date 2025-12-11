# COMPLETE CHANGES NEEDED for replacing "Refresh Graph Data" with "Download Graph Data (JSON)"

# 1. UI COMPONENT CHANGES (in _build_graph_tab method)
# Replace:
#     with gr.Row():
#         refresh_btn = gr.Button("🔄 Refresh Graph Data", size="sm")
#         graph_summary = gr.Textbox(...)

# With:
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

# 2. RETURN STATEMENT CHANGE (in _build_graph_tab method)
# Change:
#     return nodes_table, edges_table, refresh_btn, graph_summary
# To:
    return nodes_table, edges_table, download_graph_btn, graph_data_file, graph_summary

# 3. NEW METHOD NEEDED (add this new method to the InterviewUI class)
    async def export_graph_data_json(self) -> Optional[str]:
        """Export graph data (nodes and edges) as JSON file."""
        try:
            if not self.current_controller or not self.current_session_id:
                logger.warning("Graph data export attempted with no active session")
                return None

            logger.info(f"Exporting graph data for session {self.current_session_id}")
            
            import json
            import tempfile
            
            # Get the raw graph data
            graph = self.current_controller.graph
            
            # Prepare nodes data with full details
            nodes_export = []
            for node in graph.nodes.values():
                nodes_export.append({
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "timestamp": node.timestamp.isoformat(),
                    "is_ambiguous": node.is_ambiguous,
                    "metadata": node.metadata
                })
            
            # Prepare edges data with full details
            edges_export = []
            for edge in graph.edges.values():
                source_node = graph.get_node(edge.source_id)
                target_node = graph.get_node(edge.target_id)
                
                edges_export.append({
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "source_label": source_node.label if source_node else edge.source_id[:8],
                    "target_label": target_node.label if target_node else edge.target_id[:8],
                    "relation_type": edge.relation_type,
                    "metadata": edge.metadata
                })
            
            # Create export data structure
            export_data = {
                "session_id": self.current_session_id,
                "export_timestamp": datetime.now().isoformat(),
                "graph_metadata": {
                    "total_nodes": len(graph.nodes),
                    "total_edges": len(graph.edges),
                    "concept_name": self.current_controller.config.concept_name
                },
                "nodes": nodes_export,
                "edges": edges_export
            }
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_graph_data.json",
                delete=False,
                prefix=f"graph_{self.current_session_id}_",
            ) as temp_file:
                json.dump(export_data, temp_file, indent=2)
            
            logger.info(f"Graph data export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Graph data export failed: {e}")
            return None

# 4. EVENT HANDLER WIRING CHANGE (in _wire_event_handlers method)
# Find this section in the method signature:
#     def _wire_event_handlers(
#         self,
#         start_btn, submit_btn, clear_btn, refresh_btn,
#         export_json_btn, export_transcript_btn,
#         ...
#     ):

# Change to:
    def _wire_event_handlers(
        self,
        start_btn, submit_btn, clear_btn, download_graph_btn,
        export_json_btn, export_transcript_btn,
        ...
    ):

# 5. EVENT HANDLER CHANGE (in _wire_event_handlers method)
# Replace the refresh button handler:
#     # Refresh graph data
#     refresh_btn.click(
#         fn=lambda: self._refresh_graph_data() if self.current_controller else ([], [], "No active session"),
#         outputs=[nodes_table, edges_table, graph_summary]
#     )

# With the new download handler:
    # Download graph data
    download_graph_btn.click(
        fn=self.export_graph_data_json,
        outputs=[graph_data_file]
    )

# 6. INTERFACE BUILDING CHANGE (in build_interface method)
# Find this line:
#     nodes_table, edges_table, refresh_btn, graph_summary = self._build_graph_tab()

# Change to:
    nodes_table, edges_table, download_graph_btn, graph_data_file, graph_summary = self._build_graph_tab()

# 7. WIRE EVENT HANDLERS CALL CHANGE (in build_interface method)
# Find this call:
#     self._wire_event_handlers(
#         start_btn, submit_btn, clear_btn, refresh_btn,
#         export_json_btn, export_transcript_btn,
#         ...
#     )

# Change to:
    self._wire_event_handlers(
        start_btn, submit_btn, clear_btn, download_graph_btn,
        export_json_btn, export_transcript_btn,
        ...
    )