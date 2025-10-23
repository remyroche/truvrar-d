"""
Mapping and visualization tools for truffle habitat data.
"""
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


class MappingTools:
    """Tools for creating maps and visualizations of truffle habitat data."""
    
    def __init__(self):
        self.color_palette = {
            'Tuber melanosporum': '#8B0000',  # Dark red
            'Tuber magnatum': '#FFD700',      # Gold
            'Tuber aestivum': '#228B22',      # Forest green
            'Tuber borchii': '#FFA500',       # Orange
            'Tuber brumale': '#4169E1',       # Royal blue
            'Tuber mesentericum': '#9370DB',  # Medium purple
            'Tuber macrosporum': '#DC143C',   # Crimson
            'Tuber indicum': '#FF6347',       # Tomato
            'Tuber himalayense': '#20B2AA',   # Light sea green
            'Tuber oregonense': '#32CD32',    # Lime green
            'Tuber gibbosum': '#8B4513',      # Saddle brown
            'Tuber canaliculatum': '#FF1493'  # Deep pink
        }
        
    def create_species_distribution_map(self, data: pd.DataFrame, 
                                      save_path: Optional[Path] = None) -> folium.Map:
        """Create an interactive map showing species distribution."""
        if data.empty:
            logger.warning("No data to map")
            return None
            
        # Calculate map center
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB Dark_Matter').add_to(m)
        
        # Add species markers
        for _, row in data.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
                
            species = row['species']
            color = self.color_palette.get(species, '#808080')
            
            # Create popup content
            popup_content = f"""
            <b>Species:</b> {species}<br>
            <b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
            <b>Date:</b> {row.get('event_date', 'Unknown')}<br>
            <b>Source:</b> {row.get('source', 'Unknown')}
            """
            
            if 'soil_pH' in row and not pd.isna(row['soil_pH']):
                popup_content += f"<br><b>Soil pH:</b> {row['soil_pH']:.2f}"
                
            if 'mean_annual_temp_C' in row and not pd.isna(row['mean_annual_temp_C']):
                popup_content += f"<br><b>Mean Annual Temp:</b> {row['mean_annual_temp_C']:.1f}°C"
                
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=folium.Popup(popup_content, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
            
        # Add legend
        self._add_species_legend(m, data['species'].unique())
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save map
        if save_path:
            m.save(str(save_path))
            logger.info(f"Species distribution map saved to {save_path}")
            
        return m
        
    def create_environmental_map(self, data: pd.DataFrame, 
                               variable: str, 
                               save_path: Optional[Path] = None) -> folium.Map:
        """Create a map colored by environmental variable values."""
        if data.empty or variable not in data.columns:
            logger.warning(f"No data or variable {variable} not found")
            return None
            
        # Filter data with valid values
        valid_data = data.dropna(subset=[variable, 'latitude', 'longitude'])
        
        if valid_data.empty:
            logger.warning(f"No valid data for variable {variable}")
            return None
            
        # Calculate map center
        center_lat = valid_data['latitude'].mean()
        center_lon = valid_data['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Create color scale
        values = valid_data[variable].values
        min_val, max_val = values.min(), values.max()
        
        # Add markers with color based on variable value
        for _, row in valid_data.iterrows():
            value = row[variable]
            
            # Normalize value to 0-1 for color mapping
            normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            # Create color (red to blue gradient)
            color = self._value_to_color(normalized_value)
            
            # Create popup content
            popup_content = f"""
            <b>Species:</b> {row['species']}<br>
            <b>{variable}:</b> {value:.2f}<br>
            <b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_content, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
            
        # Add color bar
        self._add_color_bar(m, variable, min_val, max_val)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        if save_path:
            m.save(str(save_path))
            logger.info(f"Environmental map for {variable} saved to {save_path}")
            
        return m
        
    def create_correlation_heatmap(self, data: pd.DataFrame, 
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """Create a correlation heatmap of environmental variables."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to environmental variables
        env_cols = [col for col in numeric_cols if any(prefix in col for prefix in ['soil_', 'climate_'])]
        
        if not env_cols:
            logger.warning("No environmental variables found for correlation analysis")
            return None
            
        # Calculate correlation matrix
        corr_matrix = data[env_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        ax.set_title('Environmental Variables Correlation Matrix', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()
            
        return fig
        
    def create_species_environmental_plots(self, data: pd.DataFrame, 
                                         variables: List[str],
                                         save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """Create box plots showing environmental variables by species."""
        if data.empty:
            logger.warning("No data for plotting")
            return []
            
        figures = []
        
        for variable in variables:
            if variable not in data.columns:
                continue
                
            # Filter data with valid values
            valid_data = data.dropna(subset=[variable, 'species'])
            
            if valid_data.empty:
                continue
                
            # Create box plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get species order by median value
            species_order = valid_data.groupby('species')[variable].median().sort_values().index
            
            # Create box plot
            sns.boxplot(
                data=valid_data,
                x='species',
                y=variable,
                order=species_order,
                ax=ax
            )
            
            ax.set_title(f'{variable} by Species', fontsize=14)
            ax.set_xlabel('Species', fontsize=12)
            ax.set_ylabel(variable, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            figures.append(fig)
            
            # Save plot
            if save_dir:
                save_path = save_dir / f"{variable}_by_species.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
                
        return figures
        
    def create_habitat_suitability_map(self, data: pd.DataFrame, 
                                     suitability_scores: np.ndarray,
                                     save_path: Optional[Path] = None) -> folium.Map:
        """Create a map colored by habitat suitability scores."""
        if data.empty or len(suitability_scores) != len(data):
            logger.warning("Data and suitability scores length mismatch")
            return None
            
        # Calculate map center
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add markers with color based on suitability score
        for i, (_, row) in enumerate(data.iterrows()):
            score = suitability_scores[i]
            
            # Create color based on suitability score (0-1)
            color = self._suitability_to_color(score)
            
            # Create popup content
            popup_content = f"""
            <b>Species:</b> {row['species']}<br>
            <b>Suitability Score:</b> {score:.3f}<br>
            <b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_content, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
            
        # Add suitability legend
        self._add_suitability_legend(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        if save_path:
            m.save(str(save_path))
            logger.info(f"Habitat suitability map saved to {save_path}")
            
        return m
        
    def _add_species_legend(self, m: folium.Map, species: List[str]):
        """Add species legend to map."""
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Species Legend</b></p>
        """
        
        for species_name in species:
            color = self.color_palette.get(species_name, '#808080')
            legend_html += f"""
            <p><i class="fa fa-circle" style="color:{color}"></i> {species_name}</p>
            """
            
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))
        
    def _add_color_bar(self, m: folium.Map, variable: str, min_val: float, max_val: float):
        """Add color bar to map."""
        color_bar_html = f"""
        <div style="position: fixed; 
                    top: 50px; right: 50px; width: 150px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>{variable}</b></p>
        <p>Min: {min_val:.2f}</p>
        <p>Max: {max_val:.2f}</p>
        <div style="width: 100%; height: 20px; background: linear-gradient(to right, #ff0000, #0000ff);"></div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(color_bar_html))
        
    def _add_suitability_legend(self, m: folium.Map):
        """Add suitability legend to map."""
        legend_html = """
        <div style="position: fixed; 
                    top: 50px; right: 50px; width: 150px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>Habitat Suitability</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> Low (0-0.33)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Medium (0.33-0.66)</p>
        <p><i class="fa fa-circle" style="color:green"></i> High (0.66-1.0)</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
    def _value_to_color(self, normalized_value: float) -> str:
        """Convert normalized value to color."""
        # Red to blue gradient
        if normalized_value < 0.5:
            # Red to yellow
            red = 255
            green = int(255 * normalized_value * 2)
            blue = 0
        else:
            # Yellow to blue
            red = int(255 * (2 - normalized_value * 2))
            green = 255
            blue = int(255 * (normalized_value - 0.5) * 2)
            
        return f"#{red:02x}{green:02x}{blue:02x}"
        
    def _suitability_to_color(self, score: float) -> str:
        """Convert suitability score to color."""
        if score < 0.33:
            return 'red'
        elif score < 0.66:
            return 'yellow'
        else:
            return 'green'