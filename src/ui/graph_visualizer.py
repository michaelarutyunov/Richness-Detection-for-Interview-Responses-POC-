"""
Graph Visualizer for interview system.

Converts NetworkX graphs to interactive Plotly visualizations.
"""

import networkx as nx
import plotly.graph_objects as go

from src.core.interview_graph import InterviewGraph


def create_plotly_graph(graph: InterviewGraph) -> go.Figure:
    """
    Create interactive Plotly visualization of interview graph.

    Features:
    - Nodes colored by type (attribute, value, consequence, etc.)
    - Node size by richness weight + visit count
    - Edges with arrows
    - Hover info: label, type, quotes, visit count
    - Layout: spring_layout

    Args:
        graph: Interview graph to visualize

    Returns:
        Plotly Figure object
    """
    if graph.node_count == 0:
        # Return empty placeholder
        return go.Figure().add_annotation(
            text="No graph data yet", showarrow=False, font={"size": 20}
        )

    # Get NetworkX graph
    g = graph.graph

    # Calculate layout
    pos = nx.spring_layout(g, k=1.5, iterations=50, seed=42)

    # Define color map for node types
    color_map = {
        "attribute": "#3498db",  # Blue
        "functional_consequence": "#2ecc71",  # Green
        "psychosocial_consequence": "#9b59b6",  # Purple
        "value": "#e74c3c",  # Red
        "instrumental_value": "#f39c12",  # Orange
        "terminal_value": "#e67e22",  # Dark Orange
    }

    # Create edge traces
    edge_traces = []
    for edge in g.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line={"width": 1, "color": "#888"},
            hoverinfo="none",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_hover = []

    for node_id in g.nodes():
        x, y = pos[node_id]
        node_data = g.nodes[node_id]["data"]

        node_x.append(x)
        node_y.append(y)

        # Color by type
        node_colors.append(color_map.get(node_data.type, "#95a5a6"))

        # Size by richness weight + visit count
        base_size = graph.schema.get_richness_weight(node_data.type) * 10
        visit_bonus = node_data.visit_count * 5
        node_sizes.append(base_size + visit_bonus + 10)

        # Label
        node_text.append(node_data.label)

        # Hover info
        quotes_preview = "<br>".join(node_data.source_quotes[:2])
        if len(node_data.source_quotes) > 2:
            quotes_preview += f"<br>... +{len(node_data.source_quotes) - 2} more"

        hover_info = (
            f"<b>{node_data.label}</b><br>"
            f"Type: {node_data.type}<br>"
            f"Visits: {node_data.visit_count}<br>"
            f"Turn: {node_data.creation_turn}<br>"
            f"<br><i>Quotes:</i><br>{quotes_preview}"
        )
        node_hover.append(hover_info)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker={"size": node_sizes, "color": node_colors, "line": {"width": 2, "color": "white"}},
        text=node_text,
        textposition="top center",
        textfont={"size": 10},
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title="Interview Knowledge Graph",
        showlegend=False,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        height=600,
        plot_bgcolor="#f8f9fa",
    )

    return fig
