"""
Database schema for microbiome-environmental data.

This module defines the database schema for storing and querying
microbiome data integrated with environmental layers.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class MicrobiomeStudy(Base):
    """Table for microbiome studies from EBI Metagenomics."""
    __tablename__ = 'microbiome_studies'
    
    id = Column(Integer, primary_key=True)
    study_id = Column(String(50), unique=True, nullable=False, index=True)
    study_accession = Column(String(50), unique=True, nullable=False)
    study_title = Column(Text)
    study_description = Column(Text)
    biome = Column(String(100))
    biome_category = Column(String(100))
    environment_biome = Column(String(100))
    environment_feature = Column(String(100))
    environment_material = Column(String(100))
    host = Column(String(100))
    host_taxonomy_id = Column(String(50))
    sample_count = Column(Integer)
    run_count = Column(Integer)
    analysis_completed = Column(Boolean)
    publication_date = Column(DateTime)
    submission_date = Column(DateTime)
    center_name = Column(String(100))
    source_link = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    samples = relationship("MicrobiomeSample", back_populates="study")


class MicrobiomeSample(Base):
    """Table for microbiome samples."""
    __tablename__ = 'microbiome_samples'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(String(50), unique=True, nullable=False, index=True)
    study_id = Column(String(50), ForeignKey('microbiome_studies.study_id'), nullable=False)
    sample_accession = Column(String(50))
    sample_name = Column(String(200))
    sample_description = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    environment_biome = Column(String(100))
    environment_feature = Column(String(100))
    environment_material = Column(String(100))
    host = Column(String(100))
    host_taxonomy_id = Column(String(50))
    collection_date = Column(DateTime)
    geographic_location = Column(String(200))
    depth = Column(Float)
    elevation = Column(Float)
    temperature = Column(Float)
    ph = Column(Float)
    salinity = Column(Float)
    nitrogen = Column(Float)
    phosphorus = Column(Float)
    carbon = Column(Float)
    run_count = Column(Integer)
    analysis_completed = Column(Boolean)
    has_coordinates = Column(Boolean)
    host_standardized = Column(String(100))
    fruiting_evidence = Column(Boolean)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    study = relationship("MicrobiomeStudy", back_populates="samples")
    abundance_data = relationship("AbundanceData", back_populates="sample")
    environmental_data = relationship("EnvironmentalData", back_populates="sample")


class AbundanceData(Base):
    """Table for processed abundance data."""
    __tablename__ = 'abundance_data'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(String(50), ForeignKey('microbiome_samples.sample_id'), nullable=False)
    taxonomic_level = Column(String(20), nullable=False)  # family, genus, species
    taxon_name = Column(String(200), nullable=False)
    abundance = Column(Float)
    relative_abundance = Column(Float)
    created_at = Column(DateTime)
    
    # Relationships
    sample = relationship("MicrobiomeSample", back_populates="abundance_data")


class EnvironmentalData(Base):
    """Table for environmental data (soil, climate, geology)."""
    __tablename__ = 'environmental_data'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(String(50), ForeignKey('microbiome_samples.sample_id'), nullable=False)
    
    # Soil data (from SoilGrids)
    soil_ph = Column(Float)
    soil_organic_carbon = Column(Float)
    soil_nitrogen = Column(Float)
    soil_phosphorus = Column(Float)
    soil_calcium_carbonate = Column(Float)
    sand_content = Column(Float)
    silt_content = Column(Float)
    clay_content = Column(Float)
    bulk_density = Column(Float)
    
    # Climate data (from WorldClim)
    mean_annual_temperature = Column(Float)
    mean_annual_precipitation = Column(Float)
    temperature_seasonality = Column(Float)
    precipitation_seasonality = Column(Float)
    max_temperature_warmest_month = Column(Float)
    min_temperature_coldest_month = Column(Float)
    precipitation_wettest_month = Column(Float)
    precipitation_driest_month = Column(Float)
    
    # Geological data (from GLiM)
    glim_rock_type = Column(String(100))
    glim_rock_type_code = Column(Integer)
    glim_confidence = Column(String(20))
    glim_area_km2 = Column(Float)
    
    # Derived environmental variables
    ph_combined = Column(Float)
    temp_combined = Column(Float)
    ph_category = Column(String(20))  # acidic, neutral, alkaline, highly_alkaline
    temp_category = Column(String(20))  # cold, cool, warm, hot
    soil_texture = Column(String(50))  # clay, silt, sand, loam, etc.
    geological_ph_preference = Column(String(50))
    
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    sample = relationship("MicrobiomeSample", back_populates="environmental_data")


class MicrobiomeSummary(Base):
    """Table for microbiome summary data."""
    __tablename__ = 'microbiome_summary'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(String(50), ForeignKey('microbiome_samples.sample_id'), nullable=False)
    
    # Microbiome composition
    bacterial_families = Column(JSON)  # JSON list of bacterial families
    fungal_guilds = Column(JSON)  # JSON list of fungal guilds
    key_taxa = Column(JSON)  # JSON list of key taxa
    
    # Diversity metrics
    bacterial_family_count = Column(Integer)
    fungal_guild_count = Column(Integer)
    key_taxa_count = Column(Integer)
    microbiome_diversity = Column(String(20))  # low, medium, high, very_high
    
    # Environmental-microbiome compatibility
    env_microbiome_compatibility = Column(Integer)  # Compatibility score
    
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    sample = relationship("MicrobiomeSample")


class TruffleSpecies(Base):
    """Table for truffle species information."""
    __tablename__ = 'truffle_species'
    
    id = Column(Integer, primary_key=True)
    species_name = Column(String(100), unique=True, nullable=False)
    scientific_name = Column(String(200), unique=True, nullable=False)
    common_name = Column(String(200))
    family = Column(String(100))
    genus = Column(String(100))
    species = Column(String(100))
    description = Column(Text)
    habitat_preferences = Column(JSON)
    environmental_requirements = Column(JSON)
    economic_value = Column(String(50))  # high, medium, low
    conservation_status = Column(String(50))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class HostTree(Base):
    """Table for host tree species."""
    __tablename__ = 'host_trees'
    
    id = Column(Integer, primary_key=True)
    species_name = Column(String(100), unique=True, nullable=False)
    scientific_name = Column(String(200), unique=True, nullable=False)
    common_name = Column(String(200))
    family = Column(String(100))
    genus = Column(String(100))
    species = Column(String(100))
    description = Column(Text)
    mycorrhizal_type = Column(String(50))  # ectomycorrhizal, arbuscular, etc.
    soil_preferences = Column(JSON)
    climate_preferences = Column(JSON)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class MicrobiomeEnvironmentLink(Base):
    """Table linking microbiome data with environmental conditions."""
    __tablename__ = 'microbiome_environment_links'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(String(50), ForeignKey('microbiome_samples.sample_id'), nullable=False)
    truffle_species_id = Column(Integer, ForeignKey('truffle_species.id'))
    host_tree_id = Column(Integer, ForeignKey('host_trees.id'))
    
    # Environmental conditions
    ph_range = Column(String(20))  # e.g., "6.5-7.5"
    temperature_range = Column(String(20))  # e.g., "10-20"
    soil_type = Column(String(50))
    rock_type = Column(String(100))
    
    # Microbiome characteristics
    dominant_bacterial_families = Column(JSON)
    dominant_fungal_guilds = Column(JSON)
    key_helper_bacteria = Column(JSON)
    
    # Fruiting success indicators
    fruiting_observed = Column(Boolean)
    fruiting_frequency = Column(String(20))  # rare, occasional, frequent, abundant
    fruiting_quality = Column(String(20))  # poor, fair, good, excellent
    
    # Metadata
    confidence_score = Column(Float)  # 0-1 confidence in the link
    data_quality = Column(String(20))  # low, medium, high
    source_references = Column(JSON)  # List of source references
    
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    sample = relationship("MicrobiomeSample")
    truffle_species = relationship("TruffleSpecies")
    host_tree = relationship("HostTree")


class DataSource(Base):
    """Table for tracking data sources."""
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String(100), unique=True, nullable=False)
    source_type = Column(String(50))  # api, file, database, etc.
    source_url = Column(Text)
    description = Column(Text)
    last_updated = Column(DateTime)
    data_quality = Column(String(20))  # low, medium, high
    coverage_area = Column(String(100))
    temporal_coverage = Column(String(100))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class ProcessingLog(Base):
    """Table for tracking data processing operations."""
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    operation_type = Column(String(50), nullable=False)  # collect, process, merge, etc.
    source_name = Column(String(100))
    records_processed = Column(Integer)
    records_successful = Column(Integer)
    records_failed = Column(Integer)
    error_messages = Column(JSON)
    processing_time_seconds = Column(Float)
    status = Column(String(20))  # success, partial, failed
    created_at = Column(DateTime)


# Database schema definition
def create_schema():
    """Create the database schema."""
    return Base.metadata


# Example queries
class MicrobiomeQueries:
    """Example queries for microbiome-environmental data."""
    
    @staticmethod
    def get_samples_by_location(session, lat_min, lat_max, lon_min, lon_max):
        """Get samples within a geographic bounding box."""
        return session.query(MicrobiomeSample).filter(
            MicrobiomeSample.latitude >= lat_min,
            MicrobiomeSample.latitude <= lat_max,
            MicrobiomeSample.longitude >= lon_min,
            MicrobiomeSample.longitude <= lon_max
        ).all()
    
    @staticmethod
    def get_samples_by_environmental_conditions(session, ph_min=None, ph_max=None, 
                                              temp_min=None, temp_max=None,
                                              soil_type=None, rock_type=None):
        """Get samples matching environmental conditions."""
        query = session.query(MicrobiomeSample).join(EnvironmentalData)
        
        if ph_min is not None:
            query = query.filter(EnvironmentalData.ph_combined >= ph_min)
        if ph_max is not None:
            query = query.filter(EnvironmentalData.ph_combined <= ph_max)
        if temp_min is not None:
            query = query.filter(EnvironmentalData.temp_combined >= temp_min)
        if temp_max is not None:
            query = query.filter(EnvironmentalData.temp_combined <= temp_max)
        if soil_type is not None:
            query = query.filter(EnvironmentalData.soil_texture == soil_type)
        if rock_type is not None:
            query = query.filter(EnvironmentalData.glim_rock_type == rock_type)
        
        return query.all()
    
    @staticmethod
    def get_microbiome_diversity_by_environment(session):
        """Get microbiome diversity grouped by environmental conditions."""
        return session.query(
            EnvironmentalData.ph_category,
            EnvironmentalData.temp_category,
            EnvironmentalData.soil_texture,
            MicrobiomeSummary.microbiome_diversity,
            func.count(MicrobiomeSummary.id).label('sample_count')
        ).join(MicrobiomeSample).join(MicrobiomeSummary).group_by(
            EnvironmentalData.ph_category,
            EnvironmentalData.temp_category,
            EnvironmentalData.soil_texture,
            MicrobiomeSummary.microbiome_diversity
        ).all()
    
    @staticmethod
    def get_fruiting_success_by_microbiome(session):
        """Get fruiting success rates by microbiome characteristics."""
        return session.query(
            MicrobiomeSummary.microbiome_diversity,
            MicrobiomeSummary.env_microbiome_compatibility,
            func.avg(MicrobiomeSample.fruiting_evidence.cast(Integer)).label('fruiting_rate')
        ).join(MicrobiomeSample).group_by(
            MicrobiomeSummary.microbiome_diversity,
            MicrobiomeSummary.env_microbiome_compatibility
        ).all()
