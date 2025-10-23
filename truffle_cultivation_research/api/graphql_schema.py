"""
GraphQL schema for truffle cultivation knowledge graph and simulation API
"""

from graphene import ObjectType, String, Int, Float, List, Field, Schema, Mutation, InputObjectType
from graphene.types.datetime import DateTime
from typing import Dict, Any
import json

# Input Types
class FungusInput(InputObjectType):
    species = String(required=True)
    strain = String()
    genotype = String()
    mating_type = String()
    culture_history = String()

class HostTreeInput(InputObjectType):
    species = String(required=True)
    age = Float()
    rootstock = String()
    root_architecture = String()

class EnvironmentInput(InputObjectType):
    ph = Float()
    electrical_conductivity = Float()
    dissolved_oxygen = Float()
    temperature = Float()
    humidity = Float()
    co2_level = Float()
    light_spectrum = String()
    flow_rate = Float()

class ExperimentInput(InputObjectType):
    name = String(required=True)
    description = String()
    design = String()
    replicates = Int()
    start_date = DateTime()
    end_date = DateTime()

# Output Types
class Uncertainty(ObjectType):
    mean = Float()
    standard_deviation = Float()
    sample_size = Int()
    confidence = Float()

class Fungus(ObjectType):
    id = String()
    species = String()
    strain = String()
    genotype = String()
    mating_type = String()
    culture_history = String()
    created_at = DateTime()
    updated_at = DateTime()

class HostTree(ObjectType):
    id = String()
    species = String()
    age = Float()
    rootstock = String()
    root_architecture = String()
    created_at = DateTime()
    updated_at = DateTime()

class Mycorrhiza(ObjectType):
    id = String()
    association_type = String()
    formation_date = DateTime()
    status = String()
    colonization_percent = Float()
    uncertainty = Field(Uncertainty)
    fungus = Field(Fungus)
    host_tree = Field(HostTree)

class Environment(ObjectType):
    id = String()
    ph = Float()
    electrical_conductivity = Float()
    dissolved_oxygen = Float()
    temperature = Float()
    humidity = Float()
    co2_level = Float()
    light_spectrum = String()
    flow_rate = Float()
    created_at = DateTime()
    updated_at = DateTime()

class NutrientRecipe(ObjectType):
    id = String()
    name = String()
    description = String()
    ph = Float()
    electrical_conductivity = Float()
    macro_nutrients = String()  # JSON string
    micro_nutrients = String()  # JSON string
    chelators = List(String)
    carbon_sources = List(String)
    created_at = DateTime()
    updated_at = DateTime()

class Protocol(ObjectType):
    id = String()
    name = String()
    description = String()
    inoculation_method = String()
    sterilization_method = String()
    biofilm_type = String()
    hydro_module = String()
    duration = Int()
    temperature = Float()
    humidity = Float()
    created_at = DateTime()
    updated_at = DateTime()

class Experiment(ObjectType):
    id = String()
    name = String()
    description = String()
    design = String()
    replicates = Int()
    start_date = DateTime()
    end_date = DateTime()
    status = String()
    created_at = DateTime()
    updated_at = DateTime()

class Outcome(ObjectType):
    id = String()
    colonization_percent = Float()
    hyphal_density = Float()
    primordia_count = Int()
    yield = Float()
    measurement_date = DateTime()
    measurement_method = String()
    uncertainty = Field(Uncertainty)
    experiment = Field(Experiment)

class Evidence(ObjectType):
    id = String()
    evidence_code = String()
    description = String()
    source = String()
    reliability = Float()
    created_at = DateTime()
    updated_at = DateTime()

class SimulationResult(ObjectType):
    id = String()
    name = String()
    config = String()  # JSON string
    final_state = String()  # JSON string
    statistics = String()  # JSON string
    created_at = DateTime()
    updated_at = DateTime()

class NetworkMetrics(ObjectType):
    num_nodes = Int()
    num_edges = Int()
    density = Float()
    average_clustering = Float()
    average_shortest_path_length = Float()

class HyphalGrowthStats(ObjectType):
    total_tips = Int()
    active_tips = Int()
    total_length = Float()
    branching_events = Int()
    anastomosis_events = Int()
    network_metrics = Field(NetworkMetrics)

class SimulationState(ObjectType):
    time = Float()
    step = Int()
    hyphal_density = List(List(List(Float)))
    nutrient_concentrations = String()  # JSON string
    environmental_conditions = String()  # JSON string
    control_actions = String()  # JSON string
    performance_metrics = String()  # JSON string
    abm_stats = Field(HyphalGrowthStats)

# Query Types
class Query(ObjectType):
    # Fungi queries
    fungi = List(Fungus, species=String(), strain=String())
    fungus = Field(Fungus, id=String(required=True))
    
    # Host tree queries
    host_trees = List(HostTree, species=String())
    host_tree = Field(HostTree, id=String(required=True))
    
    # Mycorrhiza queries
    mycorrhizae = List(Mycorrhiza, fungus_id=String(), host_id=String())
    mycorrhiza = Field(Mycorrhiza, id=String(required=True))
    
    # Environment queries
    environments = List(Environment, ph_min=Float(), ph_max=Float(), 
                       ec_min=Float(), ec_max=Float())
    environment = Field(Environment, id=String(required=True))
    
    # Nutrient recipe queries
    nutrient_recipes = List(NutrientRecipe, name=String())
    nutrient_recipe = Field(NutrientRecipe, id=String(required=True))
    
    # Protocol queries
    protocols = List(Protocol, name=String(), inoculation_method=String())
    protocol = Field(Protocol, id=String(required=True))
    
    # Experiment queries
    experiments = List(Experiment, status=String(), start_date=DateTime(), end_date=DateTime())
    experiment = Field(Experiment, id=String(required=True))
    
    # Outcome queries
    outcomes = List(Outcome, experiment_id=String(), min_colonization=Float())
    outcome = Field(Outcome, id=String(required=True))
    
    # Evidence queries
    evidence = List(Evidence, evidence_code=String())
    
    # Simulation queries
    simulations = List(SimulationResult, name=String())
    simulation = Field(SimulationResult, id=String(required=True))
    
    # Complex queries
    best_colonization_protocols = List(Protocol, 
                                      fungus_species=String(), 
                                      host_species=String(),
                                      max_ph=Float(),
                                      min_colonization=Float())
    
    similar_nutrient_recipes = List(NutrientRecipe, 
                                   recipe_id=String(required=True),
                                   max_ec=Float(),
                                   similarity_threshold=Float())
    
    def resolve_fungi(self, info, species=None, strain=None):
        # This would query the actual database
        # For now, return mock data
        return [
            Fungus(
                id="fungus_001",
                species="Tuber melanosporum",
                strain="TME-001",
                genotype="MAT1-1",
                mating_type="MAT1-1",
                culture_history="Isolated from natural truffle ground in Perigord, France"
            )
        ]
    
    def resolve_fungus(self, info, id):
        # This would query the actual database
        return Fungus(
            id=id,
            species="Tuber melanosporum",
            strain="TME-001",
            genotype="MAT1-1",
            mating_type="MAT1-1",
            culture_history="Isolated from natural truffle ground in Perigord, France"
        )
    
    def resolve_host_trees(self, info, species=None):
        # This would query the actual database
        return [
            HostTree(
                id="host_001",
                species="Quercus ilex",
                age=2.5,
                rootstock="Q. ilex seedling",
                root_architecture="Taproot with lateral branches"
            )
        ]
    
    def resolve_host_tree(self, info, id):
        # This would query the actual database
        return HostTree(
            id=id,
            species="Quercus ilex",
            age=2.5,
            rootstock="Q. ilex seedling",
            root_architecture="Taproot with lateral branches"
        )
    
    def resolve_mycorrhizae(self, info, fungus_id=None, host_id=None):
        # This would query the actual database
        return [
            Mycorrhiza(
                id="mycorrhiza_001",
                association_type="ectomycorrhiza",
                formation_date="2023-03-15",
                status="active",
                colonization_percent=85.3,
                uncertainty=Uncertainty(
                    mean=85.3,
                    standard_deviation=3.2,
                    sample_size=15,
                    confidence=0.95
                ),
                fungus=Fungus(
                    id="fungus_001",
                    species="Tuber melanosporum",
                    strain="TME-001",
                    genotype="MAT1-1",
                    mating_type="MAT1-1",
                    culture_history="Isolated from natural truffle ground in Perigord, France"
                ),
                host_tree=HostTree(
                    id="host_001",
                    species="Quercus ilex",
                    age=2.5,
                    rootstock="Q. ilex seedling",
                    root_architecture="Taproot with lateral branches"
                )
            )
        ]
    
    def resolve_mycorrhiza(self, info, id):
        # This would query the actual database
        return Mycorrhiza(
            id=id,
            association_type="ectomycorrhiza",
            formation_date="2023-03-15",
            status="active",
            colonization_percent=85.3,
            uncertainty=Uncertainty(
                mean=85.3,
                standard_deviation=3.2,
                sample_size=15,
                confidence=0.95
            ),
            fungus=Fungus(
                id="fungus_001",
                species="Tuber melanosporum",
                strain="TME-001",
                genotype="MAT1-1",
                mating_type="MAT1-1",
                culture_history="Isolated from natural truffle ground in Perigord, France"
            ),
            host_tree=HostTree(
                id="host_001",
                species="Quercus ilex",
                age=2.5,
                rootstock="Q. ilex seedling",
                root_architecture="Taproot with lateral branches"
            )
        )
    
    def resolve_environments(self, info, ph_min=None, ph_max=None, ec_min=None, ec_max=None):
        # This would query the actual database with filters
        return [
            Environment(
                id="env_001",
                ph=6.2,
                electrical_conductivity=1.2,
                dissolved_oxygen=8.5,
                temperature=22.0,
                humidity=75.0,
                co2_level=400,
                light_spectrum="blue:red:far-red = 1:2:1",
                flow_rate=0.5
            )
        ]
    
    def resolve_environment(self, info, id):
        # This would query the actual database
        return Environment(
            id=id,
            ph=6.2,
            electrical_conductivity=1.2,
            dissolved_oxygen=8.5,
            temperature=22.0,
            humidity=75.0,
            co2_level=400,
            light_spectrum="blue:red:far-red = 1:2:1",
            flow_rate=0.5
        )
    
    def resolve_nutrient_recipes(self, info, name=None):
        # This would query the actual database
        return [
            NutrientRecipe(
                id="recipe_001",
                name="Truffle Base Medium",
                description="Optimized nutrient solution for truffle cultivation",
                ph=6.2,
                electrical_conductivity=1.2,
                macro_nutrients=json.dumps({
                    "nitrogen": 150,
                    "phosphorus": 50,
                    "potassium": 200,
                    "calcium": 200,
                    "magnesium": 50
                }),
                micro_nutrients=json.dumps({
                    "iron": 2.5,
                    "manganese": 0.5,
                    "zinc": 0.1,
                    "copper": 0.05,
                    "boron": 0.1,
                    "molybdenum": 0.01
                }),
                chelators=["EDTA", "DTPA"],
                carbon_sources=["sucrose", "glucose"]
            )
        ]
    
    def resolve_nutrient_recipe(self, info, id):
        # This would query the actual database
        return NutrientRecipe(
            id=id,
            name="Truffle Base Medium",
            description="Optimized nutrient solution for truffle cultivation",
            ph=6.2,
            electrical_conductivity=1.2,
            macro_nutrients=json.dumps({
                "nitrogen": 150,
                "phosphorus": 50,
                "potassium": 200,
                "calcium": 200,
                "magnesium": 50
            }),
            micro_nutrients=json.dumps({
                "iron": 2.5,
                "manganese": 0.5,
                "zinc": 0.1,
                "copper": 0.05,
                "boron": 0.1,
                "molybdenum": 0.01
            }),
            chelators=["EDTA", "DTPA"],
            carbon_sources=["sucrose", "glucose"]
        )
    
    def resolve_protocols(self, info, name=None, inoculation_method=None):
        # This would query the actual database
        return [
            Protocol(
                id="protocol_001",
                name="Hydroponic Inoculation Protocol v2.1",
                description="Step-by-step procedure for mycorrhizal inoculation in hydroponic systems",
                inoculation_method="mycelial_mat",
                sterilization_method="UV-C + H2O2",
                biofilm_type="agarose_scaffold",
                hydro_module="ebb_and_flow",
                duration=42,
                temperature=22.0,
                humidity=75.0
            )
        ]
    
    def resolve_protocol(self, info, id):
        # This would query the actual database
        return Protocol(
            id=id,
            name="Hydroponic Inoculation Protocol v2.1",
            description="Step-by-step procedure for mycorrhizal inoculation in hydroponic systems",
            inoculation_method="mycelial_mat",
            sterilization_method="UV-C + H2O2",
            biofilm_type="agarose_scaffold",
            hydro_module="ebb_and_flow",
            duration=42,
            temperature=22.0,
            humidity=75.0
        )
    
    def resolve_experiments(self, info, status=None, start_date=None, end_date=None):
        # This would query the actual database
        return [
            Experiment(
                id="exp_001",
                name="Tuber melanosporum colonization under low pH",
                description="Testing colonization efficiency of T. melanosporum on Q. ilex under acidic conditions",
                design="randomized_controlled_trial",
                replicates=15,
                start_date="2023-03-01",
                end_date="2023-04-15",
                status="completed"
            )
        ]
    
    def resolve_experiment(self, info, id):
        # This would query the actual database
        return Experiment(
            id=id,
            name="Tuber melanosporum colonization under low pH",
            description="Testing colonization efficiency of T. melanosporum on Q. ilex under acidic conditions",
            design="randomized_controlled_trial",
            replicates=15,
            start_date="2023-03-01",
            end_date="2023-04-15",
            status="completed"
        )
    
    def resolve_outcomes(self, info, experiment_id=None, min_colonization=None):
        # This would query the actual database
        return [
            Outcome(
                id="outcome_001",
                colonization_percent=85.3,
                hyphal_density=12.7,
                primordia_count=3,
                yield=0.0,
                measurement_date="2023-04-15",
                measurement_method="microscopy",
                uncertainty=Uncertainty(
                    mean=85.3,
                    standard_deviation=3.2,
                    sample_size=15,
                    confidence=0.95
                ),
                experiment=Experiment(
                    id="exp_001",
                    name="Tuber melanosporum colonization under low pH",
                    description="Testing colonization efficiency of T. melanosporum on Q. ilex under acidic conditions",
                    design="randomized_controlled_trial",
                    replicates=15,
                    start_date="2023-03-01",
                    end_date="2023-04-15",
                    status="completed"
                )
            )
        ]
    
    def resolve_outcome(self, info, id):
        # This would query the actual database
        return Outcome(
            id=id,
            colonization_percent=85.3,
            hyphal_density=12.7,
            primordia_count=3,
            yield=0.0,
            measurement_date="2023-04-15",
            measurement_method="microscopy",
            uncertainty=Uncertainty(
                mean=85.3,
                standard_deviation=3.2,
                sample_size=15,
                confidence=0.95
            ),
            experiment=Experiment(
                id="exp_001",
                name="Tuber melanosporum colonization under low pH",
                description="Testing colonization efficiency of T. melanosporum on Q. ilex under acidic conditions",
                design="randomized_controlled_trial",
                replicates=15,
                start_date="2023-03-01",
                end_date="2023-04-15",
                status="completed"
            )
        )
    
    def resolve_evidence(self, info, evidence_code=None):
        # This would query the actual database
        return [
            Evidence(
                id="evidence_001",
                evidence_code="in_planta",
                description="Microscopic analysis of root cross-sections",
                source="Laboratory microscopy analysis",
                reliability=0.95
            )
        ]
    
    def resolve_simulations(self, info, name=None):
        # This would query the actual database
        return [
            SimulationResult(
                id="sim_001",
                name="Basic Truffle Cultivation Simulation",
                config=json.dumps({
                    "total_time": 24.0,
                    "dt": 0.1,
                    "grid_size": [100, 100, 50],
                    "grid_spacing": 10.0
                }),
                final_state=json.dumps({
                    "time": 24.0,
                    "hyphal_density": [[[0.1, 0.2], [0.3, 0.4]]],
                    "nutrient_concentrations": {"nitrate": [[[0.05, 0.08], [0.06, 0.07]]]}
                }),
                statistics=json.dumps({
                    "total_steps": 240,
                    "simulation_time": 24.0,
                    "hyphal_tips_created": 150,
                    "hyphal_tips_active": 120
                })
            )
        ]
    
    def resolve_simulation(self, info, id):
        # This would query the actual database
        return SimulationResult(
            id=id,
            name="Basic Truffle Cultivation Simulation",
            config=json.dumps({
                "total_time": 24.0,
                "dt": 0.1,
                "grid_size": [100, 100, 50],
                "grid_spacing": 10.0
            }),
            final_state=json.dumps({
                "time": 24.0,
                "hyphal_density": [[[0.1, 0.2], [0.3, 0.4]]],
                "nutrient_concentrations": {"nitrate": [[[0.05, 0.08], [0.06, 0.07]]]}
            }),
            statistics=json.dumps({
                "total_steps": 240,
                "simulation_time": 24.0,
                "hyphal_tips_created": 150,
                "hyphal_tips_active": 120
            })
        )
    
    def resolve_best_colonization_protocols(self, info, fungus_species=None, host_species=None, 
                                          max_ph=None, min_colonization=None):
        # This would run a complex query combining multiple tables
        # For now, return mock data
        return [
            Protocol(
                id="protocol_001",
                name="Hydroponic Inoculation Protocol v2.1",
                description="Step-by-step procedure for mycorrhizal inoculation in hydroponic systems",
                inoculation_method="mycelial_mat",
                sterilization_method="UV-C + H2O2",
                biofilm_type="agarose_scaffold",
                hydro_module="ebb_and_flow",
                duration=42,
                temperature=22.0,
                humidity=75.0
            )
        ]
    
    def resolve_similar_nutrient_recipes(self, info, recipe_id, max_ec=None, similarity_threshold=None):
        # This would run a similarity search query
        # For now, return mock data
        return [
            NutrientRecipe(
                id="recipe_002",
                name="Similar Truffle Medium",
                description="Similar nutrient solution with lower EC",
                ph=6.0,
                electrical_conductivity=1.0,
                macro_nutrients=json.dumps({
                    "nitrogen": 120,
                    "phosphorus": 40,
                    "potassium": 180,
                    "calcium": 180,
                    "magnesium": 45
                }),
                micro_nutrients=json.dumps({
                    "iron": 2.0,
                    "manganese": 0.4,
                    "zinc": 0.08,
                    "copper": 0.04,
                    "boron": 0.08,
                    "molybdenum": 0.008
                }),
                chelators=["EDTA"],
                carbon_sources=["sucrose"]
            )
        ]

# Mutation Types
class CreateFungus(Mutation):
    class Arguments:
        input = FungusInput(required=True)
    
    Output = Fungus
    
    def mutate(self, info, input):
        # This would create a new fungus in the database
        return Fungus(
            id="fungus_new",
            species=input.species,
            strain=input.strain,
            genotype=input.genotype,
            mating_type=input.mating_type,
            culture_history=input.culture_history
        )

class CreateHostTree(Mutation):
    class Arguments:
        input = HostTreeInput(required=True)
    
    Output = HostTree
    
    def mutate(self, info, input):
        # This would create a new host tree in the database
        return HostTree(
            id="host_new",
            species=input.species,
            age=input.age,
            rootstock=input.rootstock,
            root_architecture=input.root_architecture
        )

class CreateExperiment(Mutation):
    class Arguments:
        input = ExperimentInput(required=True)
    
    Output = Experiment
    
    def mutate(self, info, input):
        # This would create a new experiment in the database
        return Experiment(
            id="exp_new",
            name=input.name,
            description=input.description,
            design=input.design,
            replicates=input.replicates,
            start_date=input.start_date,
            end_date=input.end_date,
            status="planned"
        )

class Mutations(ObjectType):
    create_fungus = CreateFungus.Field()
    create_host_tree = CreateHostTree.Field()
    create_experiment = CreateExperiment.Field()

# Create schema
schema = Schema(query=Query, mutation=Mutations)