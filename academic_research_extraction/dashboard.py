"""
Streamlit dashboard for academic research extraction system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import networkx as nx
import pyvis
from pyvis.network import Network
import io


def load_data(output_dir: str = "outputs"):
    """Load all data from the output directory."""
    output_path = Path(output_dir)
    
    data = {}
    
    # Load papers
    papers_path = output_path / "papers_index.parquet"
    if papers_path.exists():
        data["papers"] = pd.read_parquet(papers_path)
    else:
        data["papers"] = pd.DataFrame()
    
    # Load classifications
    labels_path = output_path / "labels_multilabel.parquet"
    if labels_path.exists():
        data["classifications"] = pd.read_parquet(labels_path)
    else:
        data["classifications"] = pd.DataFrame()
    
    # Load topics summary
    topics_path = output_path / "topics_summary.json"
    if topics_path.exists():
        with open(topics_path, 'r') as f:
            data["topics"] = json.load(f)
    else:
        data["topics"] = {}
    
    # Load entities
    entities_path = output_path / "entities.parquet"
    if entities_path.exists():
        data["entities"] = pd.read_parquet(entities_path)
    else:
        data["entities"] = pd.DataFrame()
    
    # Load relations
    relations_path = output_path / "relations.parquet"
    if relations_path.exists():
        data["relations"] = pd.read_parquet(relations_path)
    else:
        data["relations"] = pd.DataFrame()
    
    # Load trend analysis
    trends_path = output_path / "trend_analysis.csv"
    if trends_path.exists():
        data["trends"] = pd.read_csv(trends_path)
    else:
        data["trends"] = pd.DataFrame()
    
    return data


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Academic Research Extraction Dashboard",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Academic Research Extraction Dashboard")
    st.markdown("Explore truffle biology and ecology research papers")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    if data["papers"].empty:
        st.error("No data found. Please run the pipeline first.")
        st.info("Run: `python main.py --config configs/sources.yaml --output outputs`")
        return
    
    # Sidebar
    st.sidebar.title("Filters")
    
    # Year filter
    if not data["papers"].empty and "year" in data["papers"].columns:
        years = sorted(data["papers"]["year"].dropna().unique())
        if years:
            year_range = st.sidebar.slider(
                "Year Range",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years)))
            )
        else:
            year_range = None
    else:
        year_range = None
    
    # Topic filter
    if not data["classifications"].empty:
        # Get available topics
        topic_cols = [col for col in data["classifications"].columns if col.startswith("label_")]
        if topic_cols:
            selected_topics = st.sidebar.multiselect(
                "Topics",
                [col.replace("label_", "") for col in topic_cols],
                default=[]
            )
        else:
            selected_topics = []
    else:
        selected_topics = []
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Papers", "🏷️ Classifications", "🌐 Knowledge Graph", "📈 Trends"
    ])
    
    with tab1:
        st.header("Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", len(data["papers"]))
        
        with col2:
            if not data["papers"].empty and "oa_status" in data["papers"].columns:
                oa_count = len(data["papers"][data["papers"]["oa_status"] == "open"])
                st.metric("Open Access", f"{oa_count} ({oa_count/len(data["papers"])*100:.1f}%)")
            else:
                st.metric("Open Access", "N/A")
        
        with col3:
            if not data["entities"].empty:
                st.metric("Entities", len(data["entities"]))
            else:
                st.metric("Entities", "N/A")
        
        with col4:
            if not data["relations"].empty:
                st.metric("Relations", len(data["relations"]))
            else:
                st.metric("Relations", "N/A")
        
        # Publication trends
        if not data["trends"].empty:
            st.subheader("Publication Trends")
            
            # Filter trends by year range
            if year_range:
                filtered_trends = data["trends"][
                    (data["trends"]["year"] >= year_range[0]) & 
                    (data["trends"]["year"] <= year_range[1])
                ]
            else:
                filtered_trends = data["trends"]
            
            # Overall trends
            overall_trends = filtered_trends[filtered_trends["topic"] == "overall"]
            if not overall_trends.empty:
                fig = px.line(
                    overall_trends, 
                    x="year", 
                    y="count",
                    title="Publication Count Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Open access trends
            if not overall_trends.empty and "oa_share" in overall_trends.columns:
                fig = px.line(
                    overall_trends, 
                    x="year", 
                    y="oa_share",
                    title="Open Access Share Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Papers")
        
        # Search
        search_term = st.text_input("Search papers", placeholder="Enter title, author, or keyword...")
        
        # Filter papers
        filtered_papers = data["papers"].copy()
        
        if search_term:
            mask = (
                filtered_papers["title"].str.contains(search_term, case=False, na=False) |
                filtered_papers["abstract"].str.contains(search_term, case=False, na=False)
            )
            filtered_papers = filtered_papers[mask]
        
        if year_range:
            filtered_papers = filtered_papers[
                (filtered_papers["year"] >= year_range[0]) & 
                (filtered_papers["year"] <= year_range[1])
            ]
        
        # Display papers
        st.write(f"Showing {len(filtered_papers)} papers")
        
        for idx, row in filtered_papers.iterrows():
            with st.expander(f"{row['title']} ({row.get('year', 'N/A')})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Authors:** {', '.join(eval(row.get('authors', '[]')) if isinstance(row.get('authors'), str) else row.get('authors', []))}")
                    st.write(f"**Journal:** {row.get('journal', 'N/A')}")
                    st.write(f"**Citations:** {row.get('citations', 0)}")
                    st.write(f"**Open Access:** {row.get('oa_status', 'Unknown')}")
                    
                    if pd.notna(row.get('abstract')):
                        st.write(f"**Abstract:** {row['abstract'][:500]}...")
                
                with col2:
                    if pd.notna(row.get('best_pdf_url')):
                        st.link_button("PDF", row['best_pdf_url'])
                    if pd.notna(row.get('best_html_url')):
                        st.link_button("HTML", row['best_html_url'])
    
    with tab3:
        st.header("Classifications")
        
        if not data["classifications"].empty:
            # Topic distribution
            topic_cols = [col for col in data["classifications"].columns if col.startswith("label_")]
            if topic_cols:
                st.subheader("Topic Distribution")
                
                # Calculate topic counts
                topic_counts = {}
                for col in topic_cols:
                    topic_name = col.replace("label_", "")
                    count = len(data["classifications"][data["classifications"][col] > 0.3])
                    topic_counts[topic_name] = count
                
                # Create pie chart
                fig = px.pie(
                    values=list(topic_counts.values()),
                    names=list(topic_counts.keys()),
                    title="Papers by Topic"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Topic details
                st.subheader("Topic Details")
                topic_df = pd.DataFrame(list(topic_counts.items()), columns=["Topic", "Count"])
                st.dataframe(topic_df, use_container_width=True)
        else:
            st.info("No classification data available")
    
    with tab4:
        st.header("Knowledge Graph")
        
        if not data["entities"].empty and not data["relations"].empty:
            # Entity types
            st.subheader("Entity Types")
            entity_types = data["entities"]["label"].value_counts()
            fig = px.bar(
                x=entity_types.index,
                y=entity_types.values,
                title="Entity Types"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Relation types
            st.subheader("Relation Types")
            relation_types = data["relations"]["predicate"].value_counts()
            fig = px.bar(
                x=relation_types.index,
                y=relation_types.values,
                title="Relation Types"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interactive graph
            st.subheader("Interactive Knowledge Graph")
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for _, entity in data["entities"].iterrows():
                G.add_node(
                    entity["text"],
                    label=entity["label"],
                    confidence=entity["confidence"]
                )
            
            # Add edges
            for _, relation in data["relations"].iterrows():
                G.add_edge(
                    relation["subject"],
                    relation["object"],
                    label=relation["predicate"]
                )
            
            # Convert to PyVis
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            net.from_nx(G)
            
            # Generate HTML
            html = net.generate_html()
            
            # Display
            st.components.v1.html(html, height=600)
        else:
            st.info("No knowledge graph data available")
    
    with tab5:
        st.header("Trends")
        
        if not data["trends"].empty:
            # Topic trends over time
            st.subheader("Topic Trends Over Time")
            
            # Filter by selected topics
            if selected_topics:
                topic_trends = data["trends"][data["trends"]["topic"].isin(selected_topics)]
            else:
                topic_trends = data["trends"][data["trends"]["topic"] != "overall"]
            
            if not topic_trends.empty:
                fig = px.line(
                    topic_trends,
                    x="year",
                    y="count",
                    color="topic",
                    title="Topic Trends Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Geographic trends
            st.subheader("Geographic Distribution")
            
            # Extract geographic data
            geo_data = data["trends"][data["trends"]["topic"] == "geographic"]
            if not geo_data.empty:
                # This would need more sophisticated geographic analysis
                st.info("Geographic analysis requires additional processing")
        else:
            st.info("No trend data available")


if __name__ == "__main__":
    main()
