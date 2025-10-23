"""
Trend analysis for academic research papers.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..data_schema import PaperMetadata, ClassificationLabels, TrendAnalysis


class TrendAnalyzer:
    """Analyze trends in academic research papers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trend analyzer with configuration."""
        self.config = config
        self.taxonomy_labels = config.get("classification", {}).get("taxonomy_labels", [])
        
    def analyze_trends(
        self, 
        papers: List[PaperMetadata], 
        classifications: List[ClassificationLabels]
    ) -> List[TrendAnalysis]:
        """Analyze trends in papers over time."""
        logger.info("Analyzing trends in academic papers...")
        
        # Create DataFrame for analysis
        df = self._create_analysis_dataframe(papers, classifications)
        
        # Analyze yearly trends
        yearly_trends = self._analyze_yearly_trends(df)
        
        # Analyze topic trends
        topic_trends = self._analyze_topic_trends(df)
        
        # Analyze geographic trends
        geographic_trends = self._analyze_geographic_trends(df)
        
        # Combine all trends
        all_trends = yearly_trends + topic_trends + geographic_trends
        
        logger.info(f"Generated {len(all_trends)} trend analyses")
        return all_trends
    
    def _create_analysis_dataframe(
        self, 
        papers: List[PaperMetadata], 
        classifications: List[ClassificationLabels]
    ) -> pd.DataFrame:
        """Create DataFrame for trend analysis."""
        data = []
        
        # Create classification lookup
        classification_lookup = {c.paper_id: c for c in classifications}
        
        for paper in papers:
            row = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "journal": paper.journal,
                "citations": paper.citations,
                "oa_status": paper.oa_status.value,
                "language": paper.language.value if paper.language else "unknown",
                "affiliations": paper.affiliations,
                "subjects": paper.subjects
            }
            
            # Add classification labels
            if paper.paper_id in classification_lookup:
                classification = classification_lookup[paper.paper_id]
                for label in self.taxonomy_labels:
                    row[f"label_{label}"] = classification.labels.get(label, 0.0)
            else:
                for label in self.taxonomy_labels:
                    row[f"label_{label}"] = 0.0
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _analyze_yearly_trends(self, df: pd.DataFrame) -> List[TrendAnalysis]:
        """Analyze yearly publication trends."""
        trends = []
        
        # Filter papers with valid years
        valid_years = df[df["year"].notna() & (df["year"] >= 1950) & (df["year"] <= 2024)]
        
        if valid_years.empty:
            return trends
        
        # Overall yearly trends
        yearly_counts = valid_years.groupby("year").size()
        
        for year, count in yearly_counts.items():
            # Calculate OA share for this year
            year_data = valid_years[valid_years["year"] == year]
            oa_count = len(year_data[year_data["oa_status"] == "open"])
            oa_share = oa_count / len(year_data) if len(year_data) > 0 else 0.0
            
            # Get top papers for this year
            top_papers = year_data.nlargest(5, "citations")["paper_id"].tolist()
            
            trend = TrendAnalysis(
                year=int(year),
                topic="overall",
                count=int(count),
                oa_share=oa_share,
                top_papers=top_papers,
                geographic_coverage={}
            )
            trends.append(trend)
        
        return trends
    
    def _analyze_topic_trends(self, df: pd.DataFrame) -> List[TrendAnalysis]:
        """Analyze trends by topic/label."""
        trends = []
        
        # Filter papers with valid years
        valid_years = df[df["year"].notna() & (df["year"] >= 1950) & (df["year"] <= 2024)]
        
        if valid_years.empty:
            return trends
        
        # Analyze each taxonomy label
        for label in self.taxonomy_labels:
            label_col = f"label_{label}"
            if label_col not in valid_years.columns:
                continue
            
            # Get papers with this label (confidence > 0.3)
            label_papers = valid_years[valid_years[label_col] > 0.3]
            
            if label_papers.empty:
                continue
            
            # Group by year
            yearly_label_counts = label_papers.groupby("year").size()
            
            for year, count in yearly_label_counts.items():
                # Calculate OA share for this label/year
                year_data = label_papers[label_papers["year"] == year]
                oa_count = len(year_data[year_data["oa_status"] == "open"])
                oa_share = oa_count / len(year_data) if len(year_data) > 0 else 0.0
                
                # Get top papers for this label/year
                top_papers = year_data.nlargest(5, "citations")["paper_id"].tolist()
                
                trend = TrendAnalysis(
                    year=int(year),
                    topic=label,
                    count=int(count),
                    oa_share=oa_share,
                    top_papers=top_papers,
                    geographic_coverage={}
                )
                trends.append(trend)
        
        return trends
    
    def _analyze_geographic_trends(self, df: pd.DataFrame) -> List[TrendAnalysis]:
        """Analyze geographic distribution trends."""
        trends = []
        
        # Extract countries from affiliations
        df["countries"] = df["affiliations"].apply(self._extract_countries)
        
        # Filter papers with valid years and countries
        valid_data = df[
            df["year"].notna() & 
            (df["year"] >= 1950) & 
            (df["year"] <= 2024) & 
            df["countries"].apply(lambda x: len(x) > 0)
        ]
        
        if valid_data.empty:
            return trends
        
        # Analyze by year
        yearly_data = valid_data.groupby("year")
        
        for year, year_df in yearly_data:
            # Count papers by country
            country_counts = {}
            for countries in year_df["countries"]:
                for country in countries:
                    country_counts[country] = country_counts.get(country, 0) + 1
            
            # Calculate OA share
            oa_count = len(year_df[year_df["oa_status"] == "open"])
            oa_share = oa_count / len(year_df) if len(year_df) > 0 else 0.0
            
            # Get top papers
            top_papers = year_df.nlargest(5, "citations")["paper_id"].tolist()
            
            trend = TrendAnalysis(
                year=int(year),
                topic="geographic",
                count=len(year_df),
                oa_share=oa_share,
                top_papers=top_papers,
                geographic_coverage=country_counts
            )
            trends.append(trend)
        
        return trends
    
    def _extract_countries(self, affiliations: List[str]) -> List[str]:
        """Extract countries from affiliation strings."""
        if not affiliations:
            return []
        
        # Simple country extraction (can be improved with geocoding)
        countries = []
        country_keywords = [
            "USA", "United States", "America",
            "UK", "United Kingdom", "Britain",
            "France", "Germany", "Italy", "Spain",
            "China", "Japan", "India", "Australia",
            "Canada", "Brazil", "Mexico"
        ]
        
        for affiliation in affiliations:
            if not affiliation:
                continue
            affiliation_lower = affiliation.lower()
            for country in country_keywords:
                if country.lower() in affiliation_lower:
                    countries.append(country)
        
        return list(set(countries))
    
    def generate_evidence_heatmaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate evidence heatmaps for key relationships."""
        logger.info("Generating evidence heatmaps...")
        
        heatmaps = {}
        
        # Species vs Soil Properties heatmap
        species_soil_heatmap = self._create_species_soil_heatmap(df)
        if not species_soil_heatmap.empty:
            heatmaps["species_soil"] = species_soil_heatmap.to_dict()
        
        # Species vs Host Trees heatmap
        species_host_heatmap = self._create_species_host_heatmap(df)
        if not species_host_heatmap.empty:
            heatmaps["species_host"] = species_host_heatmap.to_dict()
        
        # Year vs Topic heatmap
        year_topic_heatmap = self._create_year_topic_heatmap(df)
        if not year_topic_heatmap.empty:
            heatmaps["year_topic"] = year_topic_heatmap.to_dict()
        
        return heatmaps
    
    def _create_species_soil_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create heatmap of species vs soil properties."""
        # This would require more sophisticated entity extraction
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _create_species_host_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create heatmap of species vs host trees."""
        # This would require more sophisticated entity extraction
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _create_year_topic_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create heatmap of year vs topic trends."""
        # Filter valid years
        valid_years = df[df["year"].notna() & (df["year"] >= 1950) & (df["year"] <= 2024)]
        
        if valid_years.empty:
            return pd.DataFrame()
        
        # Create pivot table
        heatmap_data = []
        for label in self.taxonomy_labels:
            label_col = f"label_{label}"
            if label_col not in valid_years.columns:
                continue
            
            # Get papers with this label
            label_papers = valid_years[valid_years[label_col] > 0.3]
            
            if label_papers.empty:
                continue
            
            # Count by year
            yearly_counts = label_papers.groupby("year").size()
            
            for year, count in yearly_counts.items():
                heatmap_data.append({
                    "year": int(year),
                    "topic": label,
                    "count": int(count)
                })
        
        if not heatmap_data:
            return pd.DataFrame()
        
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_table = heatmap_df.pivot_table(
            index="topic", 
            columns="year", 
            values="count", 
            fill_value=0
        )
        
        return pivot_table
    
    def save_trend_analysis(self, trends: List[TrendAnalysis], output_path: Path):
        """Save trend analysis to CSV file."""
        data = []
        for trend in trends:
            data.append({
                "year": trend.year,
                "topic": trend.topic,
                "count": trend.count,
                "oa_share": trend.oa_share,
                "top_papers": ",".join(trend.top_papers),
                "geographic_coverage": str(trend.geographic_coverage)
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = output_path / "trend_analysis.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved trend analysis to {csv_path}")
    
    def create_visualizations(self, trends: List[TrendAnalysis], output_path: Path):
        """Create trend visualization plots."""
        logger.info("Creating trend visualizations...")
        
        # Convert trends to DataFrame
        data = []
        for trend in trends:
            data.append({
                "year": trend.year,
                "topic": trend.topic,
                "count": trend.count,
                "oa_share": trend.oa_share
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("No trend data to visualize")
            return
        
        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall publication trends
        overall_trends = df[df["topic"] == "overall"]
        if not overall_trends.empty:
            axes[0, 0].plot(overall_trends["year"], overall_trends["count"], marker="o")
            axes[0, 0].set_title("Overall Publication Trends")
            axes[0, 0].set_xlabel("Year")
            axes[0, 0].set_ylabel("Number of Papers")
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Open Access trends
        if not overall_trends.empty:
            axes[0, 1].plot(overall_trends["year"], overall_trends["oa_share"] * 100, marker="s", color="green")
            axes[0, 1].set_title("Open Access Share Over Time")
            axes[0, 1].set_xlabel("Year")
            axes[0, 1].set_ylabel("OA Share (%)")
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Topic trends (top 5 topics)
        topic_trends = df[df["topic"] != "overall"]
        if not topic_trends.empty:
            top_topics = topic_trends.groupby("topic")["count"].sum().nlargest(5).index
            
            for topic in top_topics:
                topic_data = topic_trends[topic_trends["topic"] == topic]
                axes[1, 0].plot(topic_data["year"], topic_data["count"], marker="o", label=topic)
            
            axes[1, 0].set_title("Top 5 Topic Trends")
            axes[1, 0].set_xlabel("Year")
            axes[1, 0].set_ylabel("Number of Papers")
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Topic distribution (pie chart)
        if not topic_trends.empty:
            topic_counts = topic_trends.groupby("topic")["count"].sum()
            axes[1, 1].pie(topic_counts.values, labels=topic_counts.index, autopct="%1.1f%%")
            axes[1, 1].set_title("Topic Distribution")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "trend_analysis_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved trend visualizations to {plot_path}")


def analyze_trends_batch(
    papers: List[PaperMetadata],
    classifications: List[ClassificationLabels],
    config: Dict[str, Any],
    output_dir: str = "outputs"
) -> List[TrendAnalysis]:
    """Convenience function to analyze trends."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TrendAnalyzer(config)
    
    # Analyze trends
    trends = analyzer.analyze_trends(papers, classifications)
    
    # Save results
    analyzer.save_trend_analysis(trends, output_path)
    
    # Create visualizations
    analyzer.create_visualizations(trends, output_path)
    
    return trends
