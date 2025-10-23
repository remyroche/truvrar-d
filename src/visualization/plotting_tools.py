"""
Additional plotting tools for truffle habitat analysis.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PlottingTools:
    """Additional plotting tools for habitat analysis."""
    
    def __init__(self):
        self.color_palette = sns.color_palette("husl", 12)
        
    def create_environmental_distribution_plots(self, data: pd.DataFrame, 
                                              variables: List[str],
                                              save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """Create distribution plots for environmental variables."""
        if data.empty:
            logger.warning("No data for plotting")
            return []
            
        figures = []
        
        for variable in variables:
            if variable not in data.columns:
                continue
                
            # Filter data with valid values
            valid_data = data.dropna(subset=[variable])
            
            if valid_data.empty:
                continue
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Distribution Analysis: {variable}', fontsize=16)
            
            # Histogram
            axes[0, 0].hist(valid_data[variable], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Histogram')
            axes[0, 0].set_xlabel(variable)
            axes[0, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[0, 1].boxplot(valid_data[variable])
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].set_ylabel(variable)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(valid_data[variable], dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            
            # Density plot
            axes[1, 1].hist(valid_data[variable], bins=30, density=True, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('Density Plot')
            axes[1, 1].set_xlabel(variable)
            axes[1, 1].set_ylabel('Density')
            
            plt.tight_layout()
            figures.append(fig)
            
            # Save plot
            if save_dir:
                save_path = save_dir / f"{variable}_distribution.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Distribution plot saved to {save_path}")
                
        return figures
        
    def create_species_comparison_plots(self, data: pd.DataFrame, 
                                      variables: List[str],
                                      save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """Create comparison plots between species."""
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
                
            # Create violin plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            species_list = valid_data['species'].unique()
            colors = self.color_palette[:len(species_list)]
            
            sns.violinplot(
                data=valid_data,
                x='species',
                y=variable,
                ax=ax,
                palette=colors
            )
            
            ax.set_title(f'{variable} Distribution by Species', fontsize=14)
            ax.set_xlabel('Species', fontsize=12)
            ax.set_ylabel(variable, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            figures.append(fig)
            
            # Save plot
            if save_dir:
                save_path = save_dir / f"{variable}_species_comparison.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Species comparison plot saved to {save_path}")
                
        return figures
        
    def create_interactive_3d_plot(self, data: pd.DataFrame, 
                                 x_var: str, y_var: str, z_var: str,
                                 color_var: str = 'species',
                                 save_path: Optional[Path] = None) -> go.Figure:
        """Create interactive 3D scatter plot."""
        if data.empty:
            logger.warning("No data for plotting")
            return None
            
        # Filter data with valid values
        valid_data = data.dropna(subset=[x_var, y_var, z_var, color_var])
        
        if valid_data.empty:
            logger.warning("No valid data for 3D plot")
            return None
            
        # Create 3D scatter plot
        fig = px.scatter_3d(
            valid_data,
            x=x_var,
            y=y_var,
            z=z_var,
            color=color_var,
            title=f'3D Scatter Plot: {x_var} vs {y_var} vs {z_var}',
            labels={x_var: x_var, y_var: y_var, z_var: z_var}
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title=x_var,
                yaxis_title=y_var,
                zaxis_title=z_var
            ),
            width=800,
            height=600
        )
        
        # Save plot
        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive 3D plot saved to {save_path}")
            
        return fig
        
    def create_environmental_heatmap(self, data: pd.DataFrame, 
                                   variables: List[str],
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """Create heatmap of environmental variables by species."""
        if data.empty:
            logger.warning("No data for plotting")
            return None
            
        # Filter data with valid values
        valid_data = data.dropna(subset=variables + ['species'])
        
        if valid_data.empty:
            logger.warning("No valid data for heatmap")
            return None
            
        # Calculate mean values by species
        species_means = valid_data.groupby('species')[variables].mean()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize data for better visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(species_means)
        
        # Create heatmap
        sns.heatmap(
            normalized_data,
            xticklabels=variables,
            yticklabels=species_means.index,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            ax=ax
        )
        
        ax.set_title('Environmental Variables by Species (Normalized)', fontsize=14)
        ax.set_xlabel('Environmental Variables', fontsize=12)
        ax.set_ylabel('Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Environmental heatmap saved to {save_path}")
            
        return fig
        
    def create_temporal_analysis_plots(self, data: pd.DataFrame, 
                                     save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """Create temporal analysis plots."""
        if data.empty or 'month' not in data.columns:
            logger.warning("No temporal data for plotting")
            return []
            
        figures = []
        
        # Monthly distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        monthly_counts = data['month'].value_counts().sort_index()
        monthly_counts.plot(kind='bar', ax=ax, color='skyblue')
        
        ax.set_title('Truffle Occurrences by Month', fontsize=14)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Occurrences', fontsize=12)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.tight_layout()
        figures.append(fig)
        
        # Save plot
        if save_dir:
            save_path = save_dir / "monthly_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monthly distribution plot saved to {save_path}")
            
        # Species-specific monthly patterns
        if 'species' in data.columns:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            for species in data['species'].unique():
                species_data = data[data['species'] == species]
                monthly_counts = species_data['month'].value_counts().sort_index()
                
                ax.plot(monthly_counts.index, monthly_counts.values, 
                       marker='o', label=species, linewidth=2)
                
            ax.set_title('Monthly Occurrence Patterns by Species', fontsize=14)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Number of Occurrences', fontsize=12)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures.append(fig)
            
            # Save plot
            if save_dir:
                save_path = save_dir / "species_monthly_patterns.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Species monthly patterns plot saved to {save_path}")
                
        return figures
        
    def create_geographic_distribution_plots(self, data: pd.DataFrame, 
                                           save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """Create geographic distribution plots."""
        if data.empty:
            logger.warning("No data for plotting")
            return []
            
        figures = []
        
        # Latitude distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latitude histogram
        ax1.hist(data['latitude'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Latitude Distribution', fontsize=14)
        ax1.set_xlabel('Latitude', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.axvline(data['latitude'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {data["latitude"].mean():.2f}')
        ax1.legend()
        
        # Longitude histogram
        ax2.hist(data['longitude'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Longitude Distribution', fontsize=14)
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.axvline(data['longitude'].mean(), color='red', linestyle='--',
                   label=f'Mean: {data["longitude"].mean():.2f}')
        ax2.legend()
        
        plt.tight_layout()
        figures.append(fig)
        
        # Save plot
        if save_dir:
            save_path = save_dir / "geographic_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Geographic distribution plot saved to {save_path}")
            
        # Species-specific geographic distribution
        if 'species' in data.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, species in enumerate(data['species'].unique()):
                species_data = data[data['species'] == species]
                ax.scatter(species_data['longitude'], species_data['latitude'], 
                          label=species, alpha=0.6, s=50, color=self.color_palette[i])
                
            ax.set_title('Geographic Distribution by Species', fontsize=14)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures.append(fig)
            
            # Save plot
            if save_dir:
                save_path = save_dir / "species_geographic_distribution.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Species geographic distribution plot saved to {save_path}")
                
        return figures