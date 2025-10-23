// Neo4j Property Graph Schema for Truffle Cultivation Knowledge Graph
// This mirrors the RDF/OWL schema for fast exploration and path queries

// Create constraints and indexes
CREATE CONSTRAINT fungus_id IF NOT EXISTS FOR (f:Fungus) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT host_tree_id IF NOT EXISTS FOR (h:HostTree) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT mycorrhiza_id IF NOT EXISTS FOR (m:Mycorrhiza) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT nutrient_recipe_id IF NOT EXISTS FOR (n:NutrientRecipe) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT environment_id IF NOT EXISTS FOR (e:Environment) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT protocol_id IF NOT EXISTS FOR (p:Protocol) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT experiment_id IF NOT EXISTS FOR (e:Experiment) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT outcome_id IF NOT EXISTS FOR (o:Outcome) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE;

// Create indexes for common query patterns
CREATE INDEX fungus_species IF NOT EXISTS FOR (f:Fungus) ON (f.species);
CREATE INDEX host_species IF NOT EXISTS FOR (h:HostTree) ON (h.species);
CREATE INDEX environment_ph IF NOT EXISTS FOR (e:Environment) ON (e.pH);
CREATE INDEX environment_ec IF NOT EXISTS FOR (e:Environment) ON (e.electricalConductivity);
CREATE INDEX outcome_colonization IF NOT EXISTS FOR (o:Outcome) ON (o.colonizationPercent);
CREATE INDEX experiment_date IF NOT EXISTS FOR (e:Experiment) ON (e.startDate);

// Node labels and relationships
// Fungus nodes
CREATE (f:Fungus {
    id: 'fungus_001',
    species: 'Tuber melanosporum',
    strain: 'TME-001',
    genotype: 'MAT1-1',
    matingType: 'MAT1-1',
    cultureHistory: 'Isolated from natural truffle ground in Perigord, France',
    createdAt: datetime(),
    updatedAt: datetime()
});

// Host Tree nodes
CREATE (h:HostTree {
    id: 'host_001',
    species: 'Quercus ilex',
    age: 2.5,
    rootstock: 'Q. ilex seedling',
    rootArchitecture: 'Taproot with lateral branches',
    createdAt: datetime(),
    updatedAt: datetime()
});

// Mycorrhiza association
CREATE (m:Mycorrhiza {
    id: 'mycorrhiza_001',
    associationType: 'ectomycorrhiza',
    formationDate: date('2023-03-15'),
    status: 'active',
    colonizationPercent: 85.3,
    createdAt: datetime(),
    updatedAt: datetime()
});

// Environment conditions
CREATE (e:Environment {
    id: 'env_001',
    pH: 6.2,
    electricalConductivity: 1.2,
    dissolvedOxygen: 8.5,
    temperature: 22.0,
    humidity: 75.0,
    co2Level: 400,
    lightSpectrum: 'blue:red:far-red = 1:2:1',
    flowRate: 0.5,
    createdAt: datetime(),
    updatedAt: datetime()
});

// Nutrient Recipe
CREATE (nr:NutrientRecipe {
    id: 'recipe_001',
    name: 'Truffle Base Medium',
    description: 'Optimized nutrient solution for truffle cultivation',
    pH: 6.2,
    electricalConductivity: 1.2,
    macroNutrients: {
        nitrogen: 150,
        phosphorus: 50,
        potassium: 200,
        calcium: 200,
        magnesium: 50
    },
    microNutrients: {
        iron: 2.5,
        manganese: 0.5,
        zinc: 0.1,
        copper: 0.05,
        boron: 0.1,
        molybdenum: 0.01
    },
    chelators: ['EDTA', 'DTPA'],
    carbonSources: ['sucrose', 'glucose'],
    createdAt: datetime(),
    updatedAt: datetime()
});

// Protocol
CREATE (p:Protocol {
    id: 'protocol_001',
    name: 'Hydroponic Inoculation Protocol v2.1',
    description: 'Step-by-step procedure for mycorrhizal inoculation in hydroponic systems',
    inoculationMethod: 'mycelial_mat',
    sterilizationMethod: 'UV-C + H2O2',
    biofilmType: 'agarose_scaffold',
    hydroModule: 'ebb_and_flow',
    duration: 42,
    temperature: 22.0,
    humidity: 75.0,
    createdAt: datetime(),
    updatedAt: datetime()
});

// Experiment
CREATE (exp:Experiment {
    id: 'exp_001',
    name: 'Tuber melanosporum colonization under low pH',
    description: 'Testing colonization efficiency of T. melanosporum on Q. ilex under acidic conditions',
    design: 'randomized_controlled_trial',
    replicates: 15,
    startDate: date('2023-03-01'),
    endDate: date('2023-04-15'),
    status: 'completed',
    createdAt: datetime(),
    updatedAt: datetime()
});

// Outcome
CREATE (o:Outcome {
    id: 'outcome_001',
    colonizationPercent: 85.3,
    hyphalDensity: 12.7,
    primordiaCount: 3,
    yield: 0.0,
    measurementDate: date('2023-04-15'),
    measurementMethod: 'microscopy',
    uncertainty: {
        mean: 85.3,
        standardDeviation: 3.2,
        sampleSize: 15,
        confidence: 0.95
    },
    createdAt: datetime(),
    updatedAt: datetime()
});

// Evidence
CREATE (ev:Evidence {
    id: 'evidence_001',
    evidenceCode: 'in_planta',
    description: 'Microscopic analysis of root cross-sections',
    source: 'Laboratory microscopy analysis',
    reliability: 0.95,
    createdAt: datetime(),
    updatedAt: datetime()
});

// Relationships
CREATE (f)-[:FORMS_MYCORRHIZA_WITH {
    associationType: 'ectomycorrhiza',
    formationDate: date('2023-03-15'),
    status: 'active'
}]->(h);

CREATE (m)-[:OBSERVED_UNDER {
    observationDate: date('2023-03-15'),
    duration: 42
}]->(e);

CREATE (m)-[:OF_FUNGUS]->(f);
CREATE (m)-[:WITH_HOST]->(h);

CREATE (nr)-[:HAS_NUTRIENT {
    concentration: 150,
    unit: 'mg/L'
}]->(n:Nutrient {
    id: 'nutrient_001',
    name: 'Nitrate',
    chemicalFormula: 'NO3-',
    chebiId: 'CHEBI:16301'
});

CREATE (exp)-[:USES]->(p);
CREATE (exp)-[:USES]->(nr);
CREATE (exp)-[:USES]->(e);

CREATE (exp)-[:HAS_OUTCOME]->(o);

CREATE (o)-[:SUPPORTED_BY {
    confidence: 0.95,
    evidenceDate: date('2023-04-15')
}]->(ev);

// Provenance relationships
CREATE (ev)-[:GENERATED_BY]->(prov:Provenance {
    id: 'prov_001',
    agent: 'Dr. Maria Rodriguez',
    activity: 'Microscopy Analysis',
    timestamp: datetime('2023-04-15T14:30:00'),
    device: 'Olympus BX51',
    calibration: 'Standardized protocol v1.2'
});

// Additional useful relationships for path queries
CREATE (f)-[:PREFERS_ENVIRONMENT]->(e);
CREATE (h)-[:COMPATIBLE_WITH]->(f);
CREATE (p)-[:OPTIMIZED_FOR]->(f);
CREATE (p)-[:OPTIMIZED_FOR]->(h);

// Create some additional sample data for testing
CREATE (f2:Fungus {
    id: 'fungus_002',
    species: 'Tuber aestivum',
    strain: 'TAE-001',
    genotype: 'MAT1-2',
    matingType: 'MAT1-2',
    cultureHistory: 'Isolated from commercial truffle orchard in Spain'
});

CREATE (h2:HostTree {
    id: 'host_002',
    species: 'Corylus avellana',
    age: 1.8,
    rootstock: 'C. avellana seedling',
    rootArchitecture: 'Fibrous root system'
});

CREATE (e2:Environment {
    id: 'env_002',
    pH: 7.1,
    electricalConductivity: 1.8,
    dissolvedOxygen: 7.2,
    temperature: 24.0,
    humidity: 80.0,
    co2Level: 450,
    lightSpectrum: 'blue:red:far-red = 1:1:2',
    flowRate: 0.8
});

CREATE (f2)-[:FORMS_MYCORRHIZA_WITH {
    associationType: 'ectomycorrhiza',
    formationDate: date('2023-02-20'),
    status: 'active'
}]->(h2);

CREATE (f2)-[:PREFERS_ENVIRONMENT]->(e2);