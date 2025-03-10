"""
Basic visualization functionality for document data.
"""
import re
from typing import Dict, List, Any, Optional, Union

import streamlit as st
import pandas as pd
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import rgb2hex

class Visualizer:
    """Visualize extracted data from documents."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def visualize_entity_network(self, 
                                entities: Dict[str, List[Dict[str, Any]]],
                                max_entities: int = 20) -> plt.Figure:
        """Create a network graph of entities and their relationships.
        
        Args:
            entities: Dictionary of entities by type
            max_entities: Maximum number of entities to include
            
        Returns:
            Matplotlib figure object
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes for people and organizations
        people = entities.get("people", [])[:max_entities//2]
        organizations = entities.get("organizations", [])[:max_entities//2]
        
        # Add people nodes
        for person in people:
            name = person.get("name", "Unknown")
            title = person.get("title", "")
            G.add_node(name, type="person", title=title)
        
        # Add organization nodes
        for org in organizations:
            name = org.get("name", "Unknown")
            org_type = org.get("type", "")
            G.add_node(name, type="organization", org_type=org_type)
        
        # Create edges based on context mentions
        for person in people:
            person_name = person.get("name", "")
            context = person.get("context", "").lower()
            
            for org in organizations:
                org_name = org.get("name", "")
                # If organization is mentioned in person's context or vice versa
                if org_name.lower() in context or person_name.lower() in org.get("context", "").lower():
                    G.add_edge(person_name, org_name, weight=1)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Define node colors by type
        color_map = []
        for node in G:
            if G.nodes[node].get("type") == "person":
                color_map.append("skyblue")
            else:
                color_map.append("lightgreen")
        
        # Calculate node size based on connections
        node_size = [300 * (1 + G.degree(node)) for node in G]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_size, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Entity Relationship Network")
        plt.axis("off")
        
        fig = plt.gcf()
        return fig
    
    def visualize_data_points(self, 
                             data_points: Dict[str, List[Dict[str, Any]]],
                             category: str = "percentages") -> alt.Chart:
        """Create a bar chart of extracted data points.
        
        Args:
            data_points: Dictionary of data points by category
            category: Category of data points to visualize
            
        Returns:
            Altair chart object
        """
        if category not in data_points or not data_points[category]:
            return None
        
        # Prepare data
        data = data_points[category]
        df = pd.DataFrame(data)
        
        # Extract numerical values if available
        if category == "percentages" and "percentage" in df.columns:
            value_col = "percentage"
        elif category == "monetary_values" and "amount" in df.columns:
            value_col = "amount"
        elif category == "quantities" and "quantity" in df.columns:
            value_col = "quantity"
        else:
            value_col = None
        
        # If no numerical values available, try to extract from text
        if value_col is None and "value" in df.columns:
            # Try to extract numerical values from text
            df["numerical_value"] = df["value"].apply(self._extract_number)
            if not df["numerical_value"].isna().all():
                value_col = "numerical_value"
        
        # If still no values, return None
        if value_col is None:
            return None
        
        # Create context label for x-axis
        if "context" in df.columns:
            df["label"] = df["context"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        else:
            df["label"] = df["value"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        
        # Sort by value
        df = df.sort_values(by=value_col, ascending=False).head(10)
        
        # Create chart
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("label:N", title="Item", sort="-y"),
            y=alt.Y(f"{value_col}:Q", title=category.replace("_", " ").title())
        ).properties(
            title=f"Top {category.replace('_', ' ').title()} Mentioned",
            width=600,
            height=400
        )
        
        return chart
    
    def visualize_timeline(self, dates: List[Dict[str, str]]) -> alt.Chart:
        """Create a timeline visualization of date entities.
        
        Args:
            dates: List of date entities with context
            
        Returns:
            Altair chart object
        """
        if not dates:
            return None
        
        # Parse dates from text
        events = []
        for date_item in dates:
            date_text = date_item.get("date", "") or date_item.get("value", "")
            context = date_item.get("context", "")
            
            # Try to extract a year at minimum
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_text)
            if year_match:
                year = int(year_match.group(1))
                events.append({
                    "year": year,
                    "date": date_text,
                    "event": context[:50] + ("..." if len(context) > 50 else "")
                })
        
        if not events:
            return None
        
        # Create dataframe
        df = pd.DataFrame(events).sort_values("year")
        
        # Create chart
        chart = alt.Chart(df).mark_circle(size=100).encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("event:N", title=None, axis=alt.Axis(labelLimit=200)),
            tooltip=["date", "event"]
        ).properties(
            title="Timeline of Events",
            width=700,
            height=300 + 20 * min(len(df), 10)
        )
        
        # Add a rule mark for the years
        timeline = alt.Chart(df).mark_rule(color="gray", opacity=0.5).encode(
            x="year:Q"
        )
        
        return chart + timeline
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a numerical value from text.
        
        Args:
            text: Text containing a number
            
        Returns:
            Extracted number or None
        """
        if not text:
            return None
            
        # Try to find a number in the text
        match = re.search(r'[-+]?\d*\.\d+|\d+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
                
        return None


# Helper functions for Streamlit visualization

def display_entity_network(entities: Dict[str, List[Dict[str, Any]]]) -> None:
    """Display entity network visualization in Streamlit.
    
    Args:
        entities: Dictionary of entities by type
    """
    visualizer = Visualizer()
    fig = visualizer.visualize_entity_network(entities)
    st.pyplot(fig)

def display_data_charts(data_points: Dict[str, List[Dict[str, Any]]]) -> None:
    """Display charts for different data point categories in Streamlit.
    
    Args:
        data_points: Dictionary of data points by category
    """
    visualizer = Visualizer()
    
    # Display charts for different categories
    categories = ["percentages", "monetary_values", "quantities", "statistics"]
    for category in categories:
        if category in data_points and data_points[category]:
            chart = visualizer.visualize_data_points(data_points, category)
            if chart:
                st.subheader(f"{category.replace('_', ' ').title()}")
                st.altair_chart(chart, use_container_width=True)

def display_timeline(dates: List[Dict[str, str]]) -> None:
    """Display timeline visualization in Streamlit.
    
    Args:
        dates: List of date entities with context
    """
    visualizer = Visualizer()
    chart = visualizer.visualize_timeline(dates)
    if chart:
        st.subheader("Timeline")
        st.altair_chart(chart, use_container_width=True)
