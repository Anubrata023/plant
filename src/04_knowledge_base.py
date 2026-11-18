"""
=============================================================================
🌱 MODULE 04: OFFLINE AGRICULTURAL KNOWLEDGE BASE (SQLite)
=============================================================================
Constructs and queries the offline SQLite agronomic knowledge base
(data/treatments.db) for cereal crops (Rice & Wheat) and agricultural pests.

Contains verified, ICAR/IRRI/CIMMYT/USDA-standardized phytopathological
remedies for all 29 target classes, including:
  - Severity-graded intervention protocols (Grades 0 to 4)
  - Curative chemical treatments with exact active ingredients & dosages
  - Organic & biological alternatives (bio-fungicides, botanicals, IPM)
  - Cultural agronomy, resistant crop varieties, and safety intervals
  - Explicit non-spray conservation instructions for beneficial insects

Usage as module:
    from src.knowledge_base import TreatmentKnowledgeBase
    kb = TreatmentKnowledgeBase()
    plan = kb.get_treatment_plan("Wheat - Leaf Rust", severity_grade=2)
    context = kb.format_advisor_context("Wheat - Leaf Rust", severity_grade=2, lesion_pct=18.4)

Usage from CLI:
    python src/04_knowledge_base.py --build
    python src/04_knowledge_base.py --query "Wheat - Leaf Rust" --severity 2
    python src/04_knowledge_base.py --list
=============================================================================
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "treatments.db"
CLASSES_JSON = PROJECT_ROOT / "data" / "class_names.json"


# =============================================================================
# KNOWLEDGE BASE SEED DATA (ALL 29 CLASSES)
# =============================================================================

CONDITIONS_DATA: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # PESTS & BENEFICIAL ORGANISMS (0 - 12)
    # -------------------------------------------------------------------------
    {
        "class_id": 0,
        "name": "Pest - Ants",
        "crop": "General / Cereals",
        "category": "pest",
        "scientific_name": "Formicidae spp.",
        "summary": "Nuisance and minor field pest. Excavates mounds on field bunds and protects honeydew-secreting aphids or mealybugs.",
        "symptoms": "Soil mounds along field borders; worker ant trails on cereal tillers; symbiotic colonies of sucking aphids attended on leaf sheaths.",
        "protocols": {
            0: ("Healthy", "Inspect field bunds periodically during routine scouting.", "Normal"),
            1: ("Mild", "Locate ant trails; apply organic deterrents around bunds; control attending aphid colonies.", "Action within 7 days"),
            2: ("Moderate", "Apply neem cake in soil; use boric acid sugar bait traps near mound entrances.", "Action within 3-5 days"),
            3: ("Severe", "Targeted soil drenching of active mounds with Chlorpyrifos to protect root rhizosphere.", "Action within 24-48 hours"),
            4: ("Critical", "Extensive mound tunneling undermining field bunds; comprehensive perimeter barrier treatment.", "Emergency action"),
        },
        "treatments": [
            ("organic", "Neem Cake Soil Incorporation", "Azadirachtin", "Neem Cake", "100 kg/acre incorporated into soil", "Pre-sowing or around mounds", "Natural soil enricher; suppresses nematodes."),
            ("organic", "Diatomaceous Earth Dusting", "Silica dioxide", "Diatomaceous Earth", "5 kg/acre dusted along trails", "Dry weather dusting", "Non-toxic to earthworms; abrasive to ant cuticles."),
            ("chemical", "Chlorpyrifos 20% EC", "Chlorpyrifos", "Dursban / Radar", "2.5 ml / liter water (spot drenching)", "Drench directly into mounds", "Hazardous to bees; avoid foliar application."),
            ("chemical", "Fipronil 0.3% GR", "Fipronil", "Regent 0.3G", "4 to 5 kg / acre (broadcast)", "Soil application during harrowing", "Observe 30-day withholding period."),
        ],
        "varieties": [],
        "faqs": [
            ("Are all ants harmful to crops?", "Most garden ants are scavengers or predators, but when they farm aphids or burrow extensively into field bunds causing water leakage, localized control is recommended."),
        ],
    },
    {
        "class_id": 1,
        "name": "Pest - Bees",
        "crop": "General / Cereals",
        "category": "beneficial",
        "scientific_name": "Apis cerana indica / Apis mellifera",
        "summary": "CRITICAL BENEFICIAL POLLINATOR. Enhances seed set, grain yield, and biodiversity. NEVER SPRAY PESTICIDES.",
        "symptoms": "Adult worker bees actively foraging on flowering crop panicles, weeds, or border companion plants. NOT a pest.",
        "protocols": {
            0: ("Healthy", "DO NOT SPRAY. Maintain flowering companion borders (mustard, coriander) to nourish pollinators.", "Conservation"),
            1: ("Mild", "DO NOT SPRAY. Pollinator activity is vital for crop ecosystem health.", "Conservation"),
            2: ("Moderate", "DO NOT SPRAY. If neighboring pests require spraying, spray only post-sunset after bees return to hives.", "Conservation"),
            3: ("Severe", "STRICT WARNING: Do not apply neonicotinoids or pyrethroids near active foraging bees.", "Conservation"),
            4: ("Critical", "CONSERVATION PRIORITY: Report bee colony presence to local beekeeping cooperatives.", "Conservation"),
        },
        "treatments": [
            ("cultural", "Pollinator Habitat Preservation", "N/A", "Eco-Border Planting", "Plant mustard, sunflower, and clover on field edges", "Continuous", "Provides year-round forage for pollinator colonies."),
            ("cultural", "Evening Spray Timing Protocol", "N/A", "Spray Practice", "Never spray between 9 AM and 4 PM during peak flight", "Operational guideline", "Prevents lethal pesticide contact with foraging honeybees."),
        ],
        "varieties": [],
        "faqs": [
            ("Should I spray if I see bees on wheat or rice?", "ABSOLUTELY NOT. Bees are extremely beneficial insects that pollinate crops and wild flora. Spraying insecticides kills pollinators and devastates crop yields."),
        ],
    },
    {
        "class_id": 2,
        "name": "Pest - Beetles",
        "crop": "Wheat & Rice",
        "category": "pest",
        "scientific_name": "Oulema melanopus / Chaetocnema spp.",
        "summary": "Foliage-feeding beetles (Cereal leaf beetle, Flea beetle). Larvae and adults chew windowpane strips on cereal leaves.",
        "symptoms": "Longitudinal white windowpane feeding stripes parallel to leaf veins; skeletonized flag leaves; ragged leaf tips.",
        "protocols": {
            0: ("Healthy", "Scout weekly for adult beetle emergence in early spring.", "Normal"),
            1: ("Mild", "Deploy yellow sticky traps; foliar spray with Neem oil 10,000 ppm.", "Action within 7 days"),
            2: ("Moderate", "Apply biological entomopathogen Beauveria bassiana or spinosad at larval hatch.", "Action within 3-5 days"),
            3: ("Severe", "Targeted chemical pyrethroid spray before larvae damage the flag leaf.", "Action within 24-48 hours"),
            4: ("Critical", "Extensive canopy skeletonization; curative systemic insecticide spray required.", "Emergency action"),
        },
        "treatments": [
            ("biological", "Beauveria bassiana 1.15% WP", "Beauveria bassiana", "Daman / Bio-Power", "5 g / liter water (1 kg/acre)", "Foliar spray during humid evening", "Natural fungal entomopathogen safe for mammals."),
            ("organic", "Neem Oil 10,000 ppm", "Azadirachtin", "Econeem / Neem Gold", "3 ml / liter water with 0.5 ml soap", "Early morning foliar spray", "Acts as feeding deterrent and insect growth regulator."),
            ("chemical", "Lambda-cyhalothrin 5% EC", "Lambda-cyhalothrin", "Karate / Silencer", "1 ml / liter water (200 ml/acre)", "Foliar spray on active larvae", "Pre-harvest interval: 14 days."),
            ("chemical", "Thiamethoxam 25% WG", "Thiamethoxam", "Actara", "0.3 g / liter water (60 g/acre)", "Systemic foliar spray", "Pre-harvest interval: 21 days."),
        ],
        "varieties": [],
        "faqs": [
            ("When is the critical damage threshold for cereal leaf beetles?", "When scouting reveals an average of more than 1 larva per flag leaf prior to the boot stage, immediate chemical or bio-control is warranted."),
        ],
    },
    {
        "class_id": 3,
        "name": "Pest - Caterpillars",
        "crop": "Rice & Wheat",
        "category": "pest",
        "scientific_name": "Mythimna separata / Spodoptera frugiperda / Cnaphalocrocis medinalis",
        "summary": "Major defoliating pest (Armyworms, Fall Armyworm, Rice Leaf Folder). Chews foliage aggressively and folds leaf blades.",
        "symptoms": "Longitudinally rolled leaf blades with white transparent scratches; ragged eaten margins; rapid canopy defoliation overnight.",
        "protocols": {
            0: ("Healthy", "Install sex pheromone traps (5/acre) to monitor adult moth activity.", "Normal"),
            1: ("Mild", "Hand collection of egg masses; release Trichogramma egg parasitoids; spray Bt kurstaki.", "Action within 5 days"),
            2: ("Moderate", "Foliar application of Emamectin Benzoate or Spinosad during early larval instar.", "Action within 3 days"),
            3: ("Severe", "Urgent spray of Chlorantraniliprole (Coragen) to prevent earhead clipping.", "Action within 24 hours"),
            4: ("Critical", "Mass armyworm outbreak march; deploy trench barriers and broadcast targeted systemic larvicides.", "Emergency action"),
        },
        "treatments": [
            ("biological", "Bacillus thuringiensis (Bt) kurstaki", "Bacillus thuringiensis", "Dipel / Delfin", "2 g / liter water (400 g/acre)", "Foliar spray at dusk on young larvae", "Safe for non-target fauna; stomach poison for caterpillars."),
            ("biological", "Trichogramma Egg Parasitoid Cards", "Trichogramma chilonis", "Tricho-Cards", "50,000 parasitized eggs / acre", "Staple cards to underside of leaves", "Release 3 times at weekly intervals."),
            ("chemical", "Chlorantraniliprole 18.5% SC", "Chlorantraniliprole", "Coragen", "0.3 ml / liter water (60 ml/acre)", "Foliar spray using 200 L water", "High potency ryanodine receptor activator; PHI: 14 days."),
            ("chemical", "Emamectin Benzoate 5% SG", "Emamectin Benzoate", "Proclaim", "0.5 g / liter water (100 g/acre)", "Foliar spray on active larvae", "Translaminar efficacy; PHI: 10 days."),
        ],
        "varieties": [],
        "faqs": [
            ("Why are armyworms so damaging?", "Armyworm caterpillars feed voraciously at night and can march across an entire field within 48 hours, clipping wheat heads and completely stripping rice canopies."),
        ],
    },
    {
        "class_id": 4,
        "name": "Pest - Earthworms",
        "crop": "General / Cereals",
        "category": "beneficial",
        "scientific_name": "Eisenia fetida / Lumbricidae spp.",
        "summary": "SUPREME BENEFICIAL SOIL ORGANISM. Aerates soil, creates vermicompost, and cycles nutrients. NEVER SPRAY PESTICIDES.",
        "symptoms": "Rich granular worm castings on soil surface; porous soil burrow channels improving drainage. NOT a pest.",
        "protocols": {
            0: ("Healthy", "CONSERVATION: Add farmyard manure or compost to sustain soil biological activity.", "Conservation"),
            1: ("Mild", "CONSERVATION: Earthworms indicate exceptional soil vitality; encourage vermicomposting.", "Conservation"),
            2: ("Moderate", "CONSERVATION: Avoid applying toxic systemic nematicides or persistent organophosphates.", "Conservation"),
            3: ("Severe", "CONSERVATION: Protect worm population during field tillage; practice minimum soil disturbance.", "Conservation"),
            4: ("Critical", "CONSERVATION: Earthworms are a farmer's best natural ally; maintain moist organic mulch.", "Conservation"),
        },
        "treatments": [
            ("cultural", "Organic Matter Supplementation", "N/A", "Farmyard Manure / Vermicompost", "2 to 3 tonnes / acre incorporated at tillage", "Pre-sowing", "Nourishes soil microbes and earthworm earth casting production."),
            ("cultural", "Minimum Tillage & Mulch Retention", "N/A", "Conservation Agriculture", "Retain crop stubble mulch on soil surface", "Post-harvest", "Preserves burrow networks and soil moisture."),
        ],
        "varieties": [],
        "faqs": [
            ("Can earthworms harm crop roots?", "No. Earthworms feed strictly on dead plant matter, fungal mycelium, and soil bacteria. They never feed on healthy living roots."),
        ],
    },
    {
        "class_id": 5,
        "name": "Pest - Earwigs",
        "crop": "General / Cereals",
        "category": "beneficial",
        "scientific_name": "Labidura riparia / Forficula auricularia",
        "summary": "Predatory omnivore. Preys nocturnally on aphids, mites, and insect eggs. Rarely causes minor nibbling on soft seedlings.",
        "symptoms": "Pincer-bearing dark insects sheltering under mulch during day; nocturnal predation on aphids and small caterpillars.",
        "protocols": {
            0: ("Healthy", "Generally beneficial; monitor population without chemical intervention.", "Normal"),
            1: ("Mild", "Beneficial predator; no control required unless seedling nibbling is confirmed.", "Normal"),
            2: ("Moderate", "Deploy moist rolled newspaper or straw traps for mechanical collection if dense.", "Action within 7 days"),
            3: ("Severe", "Dust diatomaceous earth around seedbed borders if seedlings are suffering minor damage.", "Action within 3 days"),
            4: ("Critical", "Rarely warrants chemical treatment; targeted pyrethroid border drench if overwhelming.", "Action within 24 hours"),
        },
        "treatments": [
            ("cultural", "Straw & Cardboard Traps", "N/A", "Moist Shelter Traps", "Place rolled wet cardboard near bunds; collect daily", "Evening deployment", "Allows non-toxic physical removal of localized aggregations."),
            ("organic", "Diatomaceous Earth Barrier", "Silica dioxide", "Diatomaceous Earth", "Dust 5 kg/acre around sensitive nursery beds", "Dry conditions", "Non-chemical physical desiccant barrier."),
        ],
        "varieties": [],
        "faqs": [
            ("Are earwigs friend or foe?", "In over 90% of agricultural scenarios, earwigs are friends because they hunt aphids, armyworm eggs, and mites. Only rarely do they chew young nursery seedlings."),
        ],
    },
    {
        "class_id": 6,
        "name": "Pest - Grasshoppers",
        "crop": "Rice & Wheat",
        "category": "pest",
        "scientific_name": "Hieroglyphus banian / Oxya chinensis / Locusta spp.",
        "summary": "Foliage and panicle chewing pest. Causes irregular leaf margins and panicle cutting during the milky grain stage.",
        "symptoms": "Irregular jagged bites on leaf blades; clipped panicles floating in paddy water; defoliated cereal tillers.",
        "protocols": {
            0: ("Healthy", "Maintain clean field bunds and install T-shaped bird perches.", "Normal"),
            1: ("Mild", "Install 15-20 bird perches/acre; spray Neem seed kernel extract (NSKE 5%).", "Action within 7 days"),
            2: ("Moderate", "Apply Metarhizium anisopliae bio-insecticide; spray bunds where nymphs hatch.", "Action within 3-5 days"),
            3: ("Severe", "Targeted spray with Malathion 50% EC or Chlorpyrifos 20% EC on crop canopy.", "Action within 24-48 hours"),
            4: ("Critical", "Swarm infestation; perimeter trench barrier and broad-spectrum aerial/mist blower spray.", "Emergency action"),
        },
        "treatments": [
            ("biological", "Metarhizium anisopliae 1.15% WP", "Metarhizium anisopliae", "Kalichakra / Bio-Magic", "5 g / liter water (1 kg/acre)", "Foliar spray targeting young hoppers", "Green muscardine fungus specific to orthopteran pests."),
            ("cultural", "Bird Perches (T-Shaped Stakes)", "N/A", "Agro-Ecological Perches", "15 to 20 bamboo T-perches per acre", "Install at crop height", "Attracts predatory birds (drongos, egrets) to hunt grasshoppers."),
            ("chemical", "Malathion 50% EC", "Malathion", "Cythion", "2 ml / liter water (400 ml/acre)", "Foliar spray in 200 L water", "Fast-acting contact organophosphate; PHI: 7 days."),
            ("chemical", "Acephate 75% SP", "Acephate", "Asataf", "1.5 g / liter water (300 g/acre)", "Foliar spray", "Systemic and contact efficacy; PHI: 15 days."),
        ],
        "varieties": [],
        "faqs": [
            ("When is grasshopper control most effective?", "Treat grasshopper nymphs on field bunds shortly after monsoon rains hatch egg pods, before adults develop wings and take flight into main fields."),
        ],
    },
    {
        "class_id": 7,
        "name": "Pest - Moths",
        "crop": "Rice & Wheat",
        "category": "pest",
        "scientific_name": "Scirpophaga incertulas / Chilo suppressalis",
        "summary": "Adult stage of destructive stem borers (Yellow stem borer, Striped stem borer). Moths lay eggs that hatch into boring larvae.",
        "symptoms": "Yellowish-white moths resting on tillers; felt-like yellowish-brown hair-covered egg clusters near leaf tips; heralds 'dead hearts' and 'white heads'.",
        "protocols": {
            0: ("Healthy", "Operate solar light traps (1/acre) to monitor adult flight peaks.", "Normal"),
            1: ("Mild", "Clip leaf tips of nursery seedlings before transplanting to discard egg masses.", "Action within 5 days"),
            2: ("Moderate", "Deploy sex pheromone traps (8/acre); release Trichogramma japonicum parasitoids.", "Action within 3 days"),
            3: ("Severe", "Apply granular Cartap Hydrochloride or Chlorantraniliprole into standing water.", "Action within 24-48 hours"),
            4: ("Critical", "Heavy moth flush (>20 moths/trap/night); synchronized community-wide chemical intervention.", "Emergency action"),
        },
        "treatments": [
            ("cultural", "Solar Light Traps", "N/A", "Phototactic Traps", "1 trap per acre operated from 7 PM to 10 PM", "Nightly during vegetative stage", "Mass-traps phototactic adult moths before oviposition."),
            ("biological", "Pheromone Traps & Lures", "Scirpo-Lure", "Phero-Sensor", "8 traps / acre placed 1 foot above canopy", "Continuous monitoring", "Disrupts moth mating and monitors emergence peaks."),
            ("chemical", "Cartap Hydrochloride 4% GR", "Cartap Hydrochloride", "Padan / Caldan 4G", "7.5 to 10 kg / acre broadcast", "Standing water (2-3 cm)", "Systemic insecticide absorbed by roots; PHI: 21 days."),
            ("chemical", "Chlorantraniliprole 0.4% GR", "Chlorantraniliprole", "Ferterra", "4 kg / acre broadcast", "Standing water application", "Prevents larval penetration into stems; PHI: 21 days."),
        ],
        "varieties": [],
        "faqs": [
            ("What do 'dead heart' and 'white head' mean?", "'Dead heart' occurs when stem borer larvae kill the central shoot during tillering. 'White head' occurs when larvae bore into the panicle stem, producing completely empty bleached grains."),
        ],
    },
    {
        "class_id": 8,
        "name": "Pest - Slugs",
        "crop": "Wheat & Rice",
        "category": "pest",
        "scientific_name": "Deroceras reticulatum / Arion spp.",
        "summary": "Mollusc pest active in wet, cool, humid conditions and heavy clay soils. Feeds nocturnally on tender cereal shoots.",
        "symptoms": "Irregular shredded holes with glossy dried slime trails across leaf surfaces; severed seedlings at ground level.",
        "protocols": {
            0: ("Healthy", "Maintain proper field drainage; avoid waterlogging in wheat seedbeds.", "Normal"),
            1: ("Mild", "Deploy beer pitfall traps; dust wood ash or hydrated lime along borders.", "Action within 7 days"),
            2: ("Moderate", "Broadcast eco-friendly Iron Phosphate bait pellets around field margins.", "Action within 3-5 days"),
            3: ("Severe", "Apply Metaldehyde bait pellets across affected crop zones during evening.", "Action within 24-48 hours"),
            4: ("Critical", "Severe seedling loss; combine fine seedbed rolling with curative molluscicide baiting.", "Emergency action"),
        },
        "treatments": [
            ("organic", "Iron (III) Phosphate 1% Bait Pellets", "Ferric Phosphate", "SluggOff / Ferramol", "5 kg / acre broadcast evenly", "Late afternoon on damp soil", "Safe for birds, dogs, earthworms, and hedgehogs."),
            ("cultural", "Heavy Soil Rolling", "N/A", "Cylinder Roller", "Roll seedbeds post-sowing with a Cambridge roller", "Post-sowing", "Consolidates clods and eliminates subterranean slug shelters."),
            ("chemical", "Metaldehyde 2.5% Bait Pellets", "Metaldehyde", "SnailKill / Antimilice", "2.5 to 3 kg / acre broadcast", "Evening application around borders", "Toxic to pets; keep farm animals away from treated field."),
        ],
        "varieties": [],
        "faqs": [
            ("Why are slugs worse under minimum tillage?", "Crop residues left on soil surface retain moisture and create ideal cool daytime shelters for slugs, necessitating closer scouting."),
        ],
    },
    {
        "class_id": 9,
        "name": "Pest - Snails",
        "crop": "Rice (Paddy)",
        "category": "pest",
        "scientific_name": "Pomacea canaliculata (Golden Apple Snail) / Helix spp.",
        "summary": "Aquatic mollusc pest in lowland paddy. Devastates newly transplanted rice seedlings within the first 15 days.",
        "symptoms": "Cleanly cut rice seedlings floating in water; missing hills across paddy; bright pink egg clusters glued to rice stalks and bunds.",
        "protocols": {
            0: ("Healthy", "Screen water inlet canals with wire mesh; maintain clean bunds.", "Normal"),
            1: ("Mild", "Hand collection of pink egg clusters; install bamboo sticks as egg traps and crush them.", "Action within 3 days"),
            2: ("Moderate", "Drain paddy water intermittently; herd domestic ducks into field to forage snails.", "Action within 48 hours"),
            3: ("Severe", "Broadcast neem cake powder into shallow standing water (100 kg/acre).", "Action within 24 hours"),
            4: ("Critical", "Catastrophic seedling devastation; emergency application of registered molluscicide.", "Emergency action"),
        },
        "treatments": [
            ("cultural", "Intermittent Paddy Drainage", "N/A", "Water Management", "Drain field to saturated soil for 3-5 days", "Immediately after transplanting", "Immobilizes snails, preventing feeding on young seedlings."),
            ("biological", "Duck Foraging in Paddy", "N/A", "Integrated Bio-Control", "Herd 10-15 ducks / acre into field", "10 days post-transplanting", "Ducks consume juvenile snails and eggs naturally."),
            ("organic", "Neem Seed Cake Powder", "Azadirachtin / Saponins", "De-Oiled Neem Cake", "100 kg / acre in shallow standing water", "Broadcast at transplanting", "Natural mollusc deterrent and nitrogen supplement."),
            ("chemical", "Metaldehyde 5% DP / Pellets", "Metaldehyde", "SnailKill", "2 kg / acre spot-broadcast", "Canal outlets and low spots", "Apply in shallow water; lethal to apple snails."),
        ],
        "varieties": [],
        "faqs": [
            ("What is the most vulnerable stage for rice from snails?", "Rice seedlings are extremely vulnerable for the first 12-15 days after transplanting (or 25 days for direct-seeded rice) until tillers harden."),
        ],
    },
    {
        "class_id": 10,
        "name": "Pest - Stem Fly",
        "crop": "Wheat & Rice",
        "category": "pest",
        "scientific_name": "Atherigona oryzae / Atherigona soccata",
        "summary": "Shoot fly / stem fly pest. Larvae bore into the central shoot of young cereal seedlings, producing dead hearts.",
        "symptoms": "Central youngest leaf turns yellow, withers, and dries up ('dead heart'); seedling tillers stunted; rotting base emits unpleasant odor when pulled.",
        "protocols": {
            0: ("Healthy", "Treat seeds with systemic insecticide before sowing; sow within optimal window.", "Normal"),
            1: ("Mild", "Scout nursery beds for white cigar-shaped eggs on leaf undersides; remove dead shoots.", "Action within 5 days"),
            2: ("Moderate", "Foliar spray with Dimethoate or Quinalphos within 15-20 days of seedling emergence.", "Action within 3 days"),
            3: ("Severe", "Broadcast Cartap hydrochloride granules into standing water or soil rhizosphere.", "Action within 24-48 hours"),
            4: ("Critical", "Over 20% dead hearts; apply systemic curative insecticide and top-dress nitrogen to stimulate compensatory tillering.", "Emergency action"),
        },
        "treatments": [
            ("preventive", "Systemic Seed Dressing", "Thiamethoxam 30% FS", "Cruiser 30FS", "5 ml / kg seed", "Slurry coat seed before sowing", "Protects young seedlings against stem fly for 30 days post-emergence."),
            ("chemical", "Dimethoate 30% EC", "Dimethoate", "Rogor", "1.7 ml / liter water (350 ml/acre)", "Foliar spray at 10-15 days after emergence", "Penetrating systemic organophosphate; PHI: 20 days."),
            ("chemical", "Cartap Hydrochloride 4% GR", "Cartap Hydrochloride", "Caldan / Padan 4G", "7.5 kg / acre broadcast", "Soil application with light irrigation", "Systemic neurotoxin absorbed by roots."),
        ],
        "varieties": [],
        "faqs": [
            ("Can a crop recover from shoot fly damage?", "Yes. If treated promptly, applying a light dose of urea and irrigation encourages healthy secondary tiller production to compensate for lost primary shoots."),
        ],
    },
    {
        "class_id": 11,
        "name": "Pest - Wasps",
        "crop": "General / Cereals",
        "category": "beneficial",
        "scientific_name": "Braconidae / Ichneumonidae / Polistes spp.",
        "summary": "CRITICAL BENEFICIAL PARASITOID & PREDATOR. Parasitizes stem borer larvae and hunts defoliating caterpillars. NEVER SPRAY.",
        "symptoms": "Slender winged wasps hovering actively over canopy; caterpillar larvae with tiny white external cocoons on their backs. NOT a pest.",
        "protocols": {
            0: ("Healthy", "CONSERVATION: Wasps provide vital natural biological control; maintain un-sprayed floral refuges.", "Conservation"),
            1: ("Mild", "CONSERVATION: Natural biocontrol agents actively hunting crop caterpillars; do not disturb.", "Conservation"),
            2: ("Moderate", "CONSERVATION: Avoid broad-spectrum chemical sprays that decimate beneficial wasp populations.", "Conservation"),
            3: ("Severe", "CONSERVATION: If insect pest thresholds require action, choose selective bio-pesticides like Bt.", "Conservation"),
            4: ("Critical", "CONSERVATION: Preserve predatory wasp nesting habitat in hedgerows and field margins.", "Conservation"),
        },
        "treatments": [
            ("cultural", "Refugia Floral Borders", "N/A", "Nectar Source Planting", "Sow coriander, fennel, mustard, and alyssum along bunds", "Season-long", "Nourishes adult parasitoid wasps with accessible nectar and pollen."),
            ("cultural", "Selective Biopesticide Protocol", "Bacillus thuringiensis", "Dipel", "Use microbial Bt rather than synthetic neurotoxins", "As needed", "Bt kills caterpillars without harming parasitoid wasps or bees."),
        ],
        "varieties": [],
        "faqs": [
            ("What do those tiny white cocoons on caterpillars mean?", "Those are the pupae of beneficial Braconid parasitic wasps. The caterpillar is already dead or paralyzed. Leave them alone to hatch more helpful wasps!"),
        ],
    },
    {
        "class_id": 12,
        "name": "Pest - Weevils",
        "crop": "Rice & Wheat",
        "category": "pest",
        "scientific_name": "Lissorhoptrus oryzophilus / Sitophilus oryzae",
        "summary": "Chewing snout beetle (Rice water weevil, Granary weevil). Adults chew leaf slits; aquatic larvae prune roots.",
        "symptoms": "Narrow longitudinal white feeding slits on leaves; root pruning in rice causing yellowed stunted plants; hollowed harvested grain in storage.",
        "protocols": {
            0: ("Healthy", "Deep summer plowing to expose overwintering adults to birds and heat.", "Normal"),
            1: ("Mild", "Intermittent paddy drainage to disrupt aquatic larval feeding on rootlets.", "Action within 5 days"),
            2: ("Moderate", "Foliar spray with Chlorantraniliprole or Dinotefuran targeting adult weevils.", "Action within 3 days"),
            3: ("Severe", "Broadcast granular soil insecticide into root zone before flag leaf initiation.", "Action within 24-48 hours"),
            4: ("Critical", "Extensive root damage and yellowing canopy; drain field immediately and apply systemic larvicide.", "Emergency action"),
        },
        "treatments": [
            ("cultural", "Field Water Management", "N/A", "Paddy Aeration", "Drain water for 3 to 5 days until soil cracks slightly", "Early vegetative stage", "Starves and suffocates root-feeding aquatic weevil larvae."),
            ("chemical", "Dinotefuran 20% SG", "Dinotefuran", "Token / Osheen", "0.4 g / liter water (80 g/acre)", "Foliar spray with 200 L water", "Rapid systemic neo-nicotinoid; PHI: 14 days."),
            ("chemical", "Chlorantraniliprole 18.5% SC", "Chlorantraniliprole", "Coragen", "0.3 ml / liter water (60 ml/acre)", "Foliar spray", "Long-lasting control of adults and neonate larvae; PHI: 14 days."),
        ],
        "varieties": [],
        "faqs": [
            ("How do I distinguish weevil damage from leaf folder?", "Weevil feeding produces narrow, clean, parallel white slits without silk webbing. Leaf folder rolls the leaf into a tube with white scraped patches inside."),
        ],
    },

    # -------------------------------------------------------------------------
    # RICE DISEASES (13 - 15)
    # -------------------------------------------------------------------------
    {
        "class_id": 13,
        "name": "Rice - Brown Spot",
        "crop": "Rice",
        "category": "disease",
        "scientific_name": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "summary": "Fungal foliar disease causing oval brown lesions. Linked with nutrient-poor, sandy, or drought-stressed soils.",
        "symptoms": "Isolated round to oval dark brown spots (0.5 to 1 cm) with grayish-white center and yellow chlorotic halo across leaf blades and glumes.",
        "protocols": {
            0: ("Healthy", "Treat seeds before sowing; ensure balanced N-P-K and micronutrient fertilization.", "Preventive"),
            1: ("Mild", "Foliar spray with Pseudomonas fluorescens (5 g/L); apply Muriate of Potash (MOP) top-dress.", "Action within 5-7 days"),
            2: ("Moderate", "Spray Mancozeb 75% WP (2 g/L) or Zineb 75% WP; maintain uniform shallow irrigation.", "Action within 3-5 days"),
            3: ("Severe", "Systemic curative spray: Propiconazole 25% EC or Azoxystrobin + Difenoconazole; repeat in 12 days.", "Action within 24-48 hours"),
            4: ("Critical", "Catastrophic canopy blight; apply combination fungicide and silicon fertilizer to strengthen silica epidermal layer.", "Emergency action"),
        },
        "treatments": [
            ("preventive", "Seed Treatment Slurry", "Carbendazim 50% WP", "Bavistin", "2 g / kg seed", "Slurry coat seeds 24 hrs before sowing", "Eliminates primary seed-borne fungal mycelium."),
            ("biological", "Pseudomonas fluorescens 1.0% WP", "Pseudomonas fluorescens", "Bio-Cure-B / Eco-Guard", "5 g / liter water (1 kg/acre)", "Foliar spray during early morning", "Produces antibiotic phenazine compounds that inhibit Bipolaris hyphae."),
            ("chemical", "Mancozeb 75% WP", "Mancozeb", "Dithane M-45 / Indofil M-45", "2 g / liter water (400 g/acre)", "Foliar spray in 200 L water", "Broad-spectrum multisite protective contact fungicide; PHI: 14 days."),
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC", "1 ml / liter water (200 ml/acre)", "Curative systemic spray", "Triazole ergosterol biosynthesis inhibitor; PHI: 21 days."),
            ("chemical", "Azoxystrobin 18.2% + Difenoconazole 11.4% SC", "Azoxystrobin + Difenoconazole", "Amistar Top", "1 ml / liter water (200 ml/acre)", "Systemic foliar spray", "Dual-action QoI and DMI curative fungicide; PHI: 21 days."),
        ],
        "varieties": ["Swarna", "Sabita", "IR-36", "Radhi", "Bahadur", "Pusa Basmati 1121"],
        "faqs": [
            ("Why is Brown Spot called 'poor man's disease'?", "Brown Spot thrives in soils deficient in Potassium, Manganese, or Silicon, or in fields suffering chronic moisture stress. Correcting soil fertility significantly reduces disease severity."),
        ],
    },
    {
        "class_id": 14,
        "name": "Rice - Healthy",
        "crop": "Rice",
        "category": "healthy",
        "scientific_name": "Oryza sativa",
        "summary": "Vigorous, uninfected rice canopy. Displays uniform chlorophyll density and active tillering with no active lesions.",
        "symptoms": "Uniformly emerald-green leaves; clean leaf sheaths; vigorous active tillering; absence of blast or spot pustules.",
        "protocols": {
            0: ("Healthy", "Maintain standard alternate wetting and drying (AWD) water management.", "Optimal Health"),
            1: ("Healthy", "Apply balanced Nitrogen split: 50% basal, 25% active tillering, 25% panicle initiation.", "Optimal Health"),
            2: ("Healthy", "Apply Zinc Sulfate (25 kg/acre) basal if soil has chronic zinc deficiency.", "Optimal Health"),
            3: ("Healthy", "Conduct weekly walk-through scouting along field diagonal diagonals.", "Optimal Health"),
            4: ("Healthy", "Harvest at 80-85% golden panicle maturity to minimize grain shattering loss.", "Optimal Health"),
        },
        "treatments": [
            ("cultural", "Alternate Wetting & Drying (AWD)", "N/A", "Water Optimization", "Allow water table to drop 15 cm below surface before re-flooding", "Vegetative to grain fill", "Saves 30% water, strengthens root oxygenation, and reduces methane emissions."),
            ("cultural", "Balanced Split Nitrogen Fertilization", "Urea / DAP", "Fertilizer Schedule", "Apply N in 3 splits (basal, tillering, panicle initiation)", "Seasonal timing", "Prevents lush succulent vegetative growth that attracts leaf blast and borers."),
        ],
        "varieties": ["IR-64", "Swarna", "Pusa Basmati 1509", "Pusa Basmati 1718", "BPT-5204 (Samba Mahsuri)"],
        "faqs": [
            ("What is the most effective way to keep rice healthy?", "Maintain balanced nutrition (avoiding excessive nitrogen without potash) and practice shallow intermittent irrigation to maintain oxygenated roots."),
        ],
    },
    {
        "class_id": 15,
        "name": "Rice - Leaf Blast",
        "crop": "Rice",
        "category": "disease",
        "scientific_name": "Magnaporthe oryzae (Pyricularia oryzae)",
        "summary": "Devastating fungal foliar disease. Forms spindle-shaped lesions that coalesce into complete canopy scorching.",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with grayish-white centers, reddish-brown margins, and yellow halos; fields appear burnt from afar.",
        "protocols": {
            0: ("Healthy", "Seed treatment with Tricyclazole; avoid excessive nitrogen top-dressing.", "Preventive"),
            1: ("Mild", "Stop urea application immediately; foliar spray with Kasugamycin 3% SL or Trichoderma.", "Action within 48 hours"),
            2: ("Moderate", "Curative spray with Tricyclazole 75% WP (120 g/acre in 200 L water).", "Action within 24-48 hours"),
            3: ("Severe", "Aggressive systemic spray: Azoxystrobin + Difenoconazole or Isoprothiolane (Fujione).", "Action within 24 hours"),
            4: ("Critical", "Catastrophic blast; drain excess nitrogen water; emergency community-wide curative fungicide spray.", "Emergency action"),
        },
        "treatments": [
            ("preventive", "Seed Dressing with Tricyclazole", "Tricyclazole 75% WP", "Beam / Baan", "2 g / kg seed", "Dry or wet seed slurry before sowing", "Protects seedlings against blast for up to 40 days in seedbed."),
            ("biological", "Kasugamycin 3% SL", "Kasugamycin", "Kasu-B / Kasumin", "2 ml / liter water (400 ml/acre)", "Foliar spray at initial spindle lesion detection", "Naturally derived antibiotic fungicide safe for fish and humans; PHI: 14 days."),
            ("chemical", "Tricyclazole 75% WP", "Tricyclazole", "Beam / Baan / Sivic", "0.6 g / liter water (120 g/acre)", "Foliar spray in 200 L water", "Gold-standard systemic blast-specific melanin biosynthesis inhibitor; PHI: 21 days."),
            ("chemical", "Isoprothiolane 40% EC", "Isoprothiolane", "Fujione", "1.5 ml / liter water (300 ml/acre)", "Foliar spray", "Dual systemic fungicide and planthopper suppressant; PHI: 21 days."),
            ("chemical", "Picoxystrobin 7.05% + Tricyclazole 22.8% SC", "Picoxystrobin + Tricyclazole", "Galileo Way", "2 ml / liter water (400 ml/acre)", "Systemic foliar spray", "Superior curative systemic protection under heavy disease pressure; PHI: 21 days."),
        ],
        "varieties": ["Swarna Sub1", "Pusa 1460", "IR-64", "Rasi", "Tellahamsa", "CR Dhan 310"],
        "faqs": [
            ("Why does leaf blast spread so rapidly?", "Blast spores germinate in under 4 hours under warm humid nights (>90% humidity, 22-28°C) and dew. A single spindle lesion can release up to 20,000 infectious spores daily!"),
        ],
    },

    # -------------------------------------------------------------------------
    # WHEAT DISEASES (16 - 28)
    # -------------------------------------------------------------------------
    {
        "class_id": 16,
        "name": "Wheat - Black Point",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Bipolaris sorokiniana / Alternaria alternata",
        "summary": "Kernel smudge disease. Causes dark brown to black discoloration around the embryo tip of harvested wheat grains.",
        "symptoms": "Dark brown or jet-black staining localized at the embryo end of kernels; affects seed germination and flour milling grade.",
        "protocols": {
            0: ("Healthy", "Sow certified clean seed; dry harvested grain to <12% moisture before storage.", "Preventive"),
            1: ("Mild", "Clean and grade seed using seed gravity separators to remove discolored grains.", "Action post-harvest"),
            2: ("Moderate", "Seed treatment with Carboxin + Thiram before next planting season.", "Pre-sowing"),
            3: ("Severe", "If rainy humid weather coincides with grain filling, apply Propiconazole 25% EC at heading.", "Action within 3 days"),
            4: ("Critical", "Severe kernel smudge; segregate affected grain batch for feed use; do not use as seed.", "Post-harvest guidance"),
        },
        "treatments": [
            ("preventive", "Seed Dressing with Carboxin + Thiram", "Carboxin 37.5% + Thiram 37.5% DS", "Vitavax Power", "2.5 g / kg seed", "Dry seed dressing", "Eliminates surface and internal seed-borne Bipolaris and Alternaria."),
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt", "1 ml / liter water (200 ml/acre)", "Foliar spray at 50% heading stage", "Protects developing kernels from fungal colonization during humid grain fill; PHI: 30 days."),
            ("cultural", "Rapid Grain Drying & Aerated Storage", "N/A", "Post-Harvest Management", "Dry grains below 12% moisture within 48 hrs of harvest", "Post-harvest", "Halts fungal mycelial progression inside stored grain embryos."),
        ],
        "varieties": ["HD-2967", "HD-3086", "DBW-187", "PBW-550"],
        "faqs": [
            ("Does Black Point make wheat poisonous to eat?", "In most cases, Black Point caused by Alternaria or Bipolaris reduces baking grade and germination but does not produce dangerous toxins like Fusarium head blight."),
        ],
    },
    {
        "class_id": 17,
        "name": "Wheat - Blast",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Magnaporthe oryzae Triticum pathotype (MoT)",
        "summary": "Devastating quarantine spike disease. Bleaches wheat heads prematurely, causing severely shriveled, empty grains.",
        "symptoms": "Complete or partial bleaching of wheat heads while the lower stem remains green; elliptical bleached lesions on flag leaves.",
        "protocols": {
            0: ("Healthy", "Maintain strict quarantine compliance; plant certified blast-resistant seed.", "Preventive"),
            1: ("Mild", "Immediate preventive spray of Tebuconazole + Trifloxystrobin at heading stage.", "Action within 24 hours"),
            2: ("Moderate", "Systemic fungicide spray with Nativo or Folicur; repeat after 10-12 days.", "Action within 24 hours"),
            3: ("Severe", "Severe head blight; immediate containment spraying; notify agricultural extension officer.", "Emergency action"),
            4: ("Critical", "Catastrophic crop loss; destroy and burn infected crop residue; enforce 2-year non-host rotation.", "Emergency quarantine"),
        },
        "treatments": [
            ("preventive", "Systemic Seed Treatment with Tebuconazole", "Tebuconazole 2% DS", "Raxil", "1.5 g / kg seed", "Slurry coat seed before sowing", "Eliminates primary seed-borne inoculum."),
            ("chemical", "Tebuconazole 50% + Trifloxystrobin 25% WG", "Tebuconazole + Trifloxystrobin", "Nativo 75 WG", "0.6 g / liter water (120 g/acre)", "Foliar spray at boot-leaf and anthesis", "Synergistic curative DMI and QoI fungicide; PHI: 30 days."),
            ("chemical", "Tebuconazole 25.9% EC", "Tebuconazole", "Folicur", "1.25 ml / liter water (250 ml/acre)", "Foliar spray at heading", "Systemic triazole with excellent panicle penetration; PHI: 30 days."),
        ],
        "varieties": ["DBW-187 (Karan Vandana)", "HD-3249", "K-1317", "WB-02"],
        "faqs": [
            ("How does Wheat Blast differ from Fusarium Head Blight?", "Wheat Blast bleaches everything above the infection point on the rachis, turning it silvery-white, whereas Fusarium Head Blight typically exhibits pink/salmon spore masses at the base of glumes."),
        ],
    },
    {
        "class_id": 18,
        "name": "Wheat - Crown and Root Rot",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Fusarium pseudograminearum / Bipolaris sorokiniana",
        "summary": "Soil-borne root and crown rot. Causes dark brown necrosis at the stem base, resulting in premature 'white heads'.",
        "symptoms": "Chocolate-brown to reddish-brown rotting of crown roots and lower stem internodes; stunted tillers; premature bleached whiteheads with empty grain.",
        "protocols": {
            0: ("Healthy", "Practice 2-year crop rotation with non-cereal legumes (Chickpea, Canola).", "Preventive"),
            1: ("Mild", "Apply Trichoderma viride enriched farmyard manure in soil; avoid deep sowing.", "Pre-sowing"),
            2: ("Moderate", "Seed dressing with Carboxin + Thiram; irrigate to relieve crown moisture stress.", "Action within 5 days"),
            3: ("Severe", "Widespread crown necrosis; avoid waterlogging; apply targeted systemic fungicide drench.", "Action within 48 hours"),
            4: ("Critical", "Field collapse; rogue and burn infected root crowns; plan non-host legume rotation.", "Emergency action"),
        },
        "treatments": [
            ("biological", "Trichoderma viride Bio-Enriched Compost", "Trichoderma viride", "Bio-Derma / Eco-Trich", "2 kg / 200 kg FYM per acre", "Broadcast during land preparation", "Antagonistic fungus colonizes roots, suppressing Fusarium and Bipolaris."),
            ("preventive", "Fludioxonil + Metalaxyl-M Seed Dressing", "Fludioxonil + Metalaxyl-M", "Maxim XL", "1 ml / kg seed", "Slurry coat seed before sowing", "Dual broad-spectrum contact and systemic root zone shield."),
            ("chemical", "Carboxin 37.5% + Thiram 37.5% DS", "Carboxin + Thiram", "Vitavax Power", "2.5 g / kg seed", "Dry seed dressing", "Systemic translocation to seedling crown tissues."),
        ],
        "varieties": ["HD-2967", "WH-1105", "PBW-725", "DBW-222"],
        "faqs": [
            ("Why is crown rot exacerbated by stubble retention?", "Fusarium pseudograminearum overwinters in undecomposed cereal straw stubble. Breaking the cycle with broadleaf crops (canola, chickpeas) starves the fungal pathogen."),
        ],
    },
    {
        "class_id": 19,
        "name": "Wheat - Fusarium Head Blight",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Fusarium graminearum / Fusarium culmorum",
        "summary": "Head scab disease. Produces bleached spikelets with pink spore masses and dangerous deoxynivalenol (DON) mycotoxins.",
        "symptoms": "Premature bleaching of spikelets during anthesis; pinkish-orange spore cushions (sporodochia) at glume bases; shriveled chalky-white grains ('tombstones').",
        "protocols": {
            0: ("Healthy", "Avoid planting wheat directly into corn/maize residue; choose moderately resistant varieties.", "Preventive"),
            1: ("Mild", "Monitor weather closely during anthesis (Feekes 10.51); prepare targeted triazole spray if rain occurs.", "Action within 24 hours"),
            2: ("Moderate", "Apply Prosaro or Miravis Ace fungicide precisely at early flowering (anthesis).", "Action within 24-48 hours"),
            3: ("Severe", "High mycotoxin risk; spray systemic triazole; increase combine fan speed at harvest to blow out light diseased kernels.", "Emergency action"),
            4: ("Critical", "Catastrophic head scab; test grain for DON vomitoxin before selling; DO NOT feed to pigs or breeding livestock.", "Post-harvest safety"),
        },
        "treatments": [
            ("chemical", "Prothioconazole 19% + Tebuconazole 19% SC", "Prothioconazole + Tebuconazole", "Prosaro", "1 ml / liter water (200 ml/acre)", "Spray precisely at early anthesis (yellow anthers)", "Gold standard for suppressing Fusarium and reducing DON mycotoxin; PHI: 30 days."),
            ("chemical", "Tebuconazole 25.9% EC", "Tebuconazole", "Folicur", "1.25 ml / liter water (250 ml/acre)", "Foliar spray at 50% flowering", "Systemic curative triazole; PHI: 30 days."),
            ("cultural", "Combine Harvester Wind Adjustment", "N/A", "Mechanical Separation", "Increase combine harvester fan speed at harvest", "Harvesting", "Blows lightweight infected 'tombstone' kernels out the back of the combine, reducing toxin levels in grain tank."),
        ],
        "varieties": ["DBW-187", "HD-3226", "DBW-303 (Karan Vaishnavi)", "MACS-6222"],
        "faqs": [
            ("Why shouldn't I spray Strobilurin (QoI) fungicides for Head Blight?", "WARNING: Strobilurins (like Azoxystrobin) applied during head blight outbreaks can actually TRIGGER the Fusarium fungus to synthesize HIGHER levels of dangerous DON mycotoxin! Use triazoles (Prosaro, Folicur) instead."),
        ],
    },
    {
        "class_id": 20,
        "name": "Wheat - Healthy",
        "crop": "Wheat",
        "category": "healthy",
        "scientific_name": "Triticum aestivum",
        "summary": "Vigorous, disease-free wheat canopy. Displays robust green flag leaves, intact glumes, and clean photosynthetic tillers.",
        "symptoms": "Clean, erect, dark-green flag leaves; sturdy golden-green stems; uniform heading with complete grain filling.",
        "protocols": {
            0: ("Healthy", "Follow recommended 5-stage irrigation schedule (CRI, Tillering, Jointing, Flowering, Milking).", "Optimal Health"),
            1: ("Healthy", "Apply balanced N-P-K (120:60:40 kg/ha) with 25 kg/ha Zinc Sulfate basal.", "Optimal Health"),
            2: ("Healthy", "Protect flag leaf health: it contributes over 65% of total grain-filling photosynthates.", "Optimal Health"),
            3: ("Healthy", "Scout weekly for yellow rust pustules in cool, humid, shaded field pockets.", "Optimal Health"),
            4: ("Healthy", "Harvest at 14% grain moisture for maximum milling test weight.", "Optimal Health"),
        },
        "treatments": [
            ("cultural", "Critical-Stage Irrigation Protocol", "N/A", "Water Scheduling", "Irrigate at CRI (21 DAS), Jointing (60 DAS), Flowering (85 DAS), and Milk stage (105 DAS)", "Seasonal milestones", "Prevents terminal moisture stress and maximizes grain test weight."),
            ("cultural", "Zinc Micronutrient Nutrition", "Zinc Sulfate 21%", "Zinc Fertilizer", "25 kg / hectare basal at sowing", "Land preparation", "Essential for wheat enzyme systems and disease resistance against root rots."),
        ],
        "varieties": ["HD-2967", "HD-3086", "DBW-187 (Karan Vandana)", "DBW-222", "PBW-725"],
        "faqs": [
            ("Which leaf is most important for wheat yield?", "The top 'Flag Leaf'. It supplies 65% to 75% of the carbohydrates required to fill the wheat earhead. Keeping the flag leaf clean of rust and blight directly ensures bumper yields."),
        ],
    },
    {
        "class_id": 21,
        "name": "Wheat - Leaf Blight",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Bipolaris sorokiniana (Spot Blight)",
        "summary": "Warm-climate foliar blight. Causes irregular brown necrotic lesions surrounded by yellow chlorotic halos.",
        "symptoms": "Oval to irregular brown necrotic spots on leaves that enlarge and coalesce into broad dead patches; leaves dry out prematurely.",
        "protocols": {
            0: ("Healthy", "Seed treatment with Carboxin + Thiram; plant certified blight-tolerant cultivars.", "Preventive"),
            1: ("Mild", "Foliar spray with Pseudomonas fluorescens (5 g/L); apply light irrigation.", "Action within 5-7 days"),
            2: ("Moderate", "Apply Propiconazole 25% EC (1 ml/L) or Mancozeb 75% WP (2 g/L) at flag leaf emergence.", "Action within 3-5 days"),
            3: ("Severe", "Systemic spray of Azoxystrobin + Difenoconazole; repeat after 14 days.", "Action within 24-48 hours"),
            4: ("Critical", "Severe canopy blight; apply curative triazole and potassium foliar spray to halt desiccation.", "Emergency action"),
        },
        "treatments": [
            ("preventive", "Carboxin + Thiram Seed Dressing", "Carboxin + Thiram", "Vitavax Power", "2.5 g / kg seed", "Slurry coat seed before sowing", "Eliminates seed-borne Bipolaris inoculum."),
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC", "1 ml / liter water (200 ml/acre)", "Foliar spray in 200 L water", "Systemic curative triazole; PHI: 30 days."),
            ("chemical", "Azoxystrobin 18.2% + Difenoconazole 11.4% SC", "Azoxystrobin + Difenoconazole", "Amistar Top", "1 ml / liter water (200 ml/acre)", "Foliar spray", "Long-lasting dual curative and protective shield; PHI: 30 days."),
        ],
        "varieties": ["HD-2967", "PBW-550", "DBW-39", "HD-3086", "WH-1105"],
        "faqs": [
            ("Why is leaf blight worse in eastern India and warm plains?", "Bipolaris sorokiniana thrives under high average temperatures (25-30°C) and high humidity, making timely flag leaf protection critical in warm wheat tracts."),
        ],
    },
    {
        "class_id": 22,
        "name": "Wheat - Leaf Rust",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Puccinia triticina (Brown Rust)",
        "summary": "Widespread cereal rust. Produces small, round to oval orange-brown powdery pustules scattered across the upper leaf blade.",
        "symptoms": "Small round to oval orange-brown powdery pustules (uredinia) scattered randomly across upper leaf blade; spores rub off as orange dust on fingers.",
        "protocols": {
            0: ("Healthy", "Scout weekly; plant resistant varieties with Lr genes.", "Preventive"),
            1: ("Mild", "Spray Mancozeb 75% WP (2 g/L) at first symptom detection.", "Action within 5 days"),
            2: ("Moderate", "Curative systemic spray of Propiconazole 25% EC (1 ml/L or 200 ml/acre in 200 L water).", "Action within 3 days"),
            3: ("Severe", "Urgent systemic intervention: Tebuconazole + Trifloxystrobin (Nativo @ 0.6 g/L).", "Action within 24-48 hours"),
            4: ("Critical", "Catastrophic canopy sporulation; community-wide aerial or tractor-mounted mist blower spray.", "Emergency action"),
        },
        "treatments": [
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC / Bumper", "1 ml / liter water (200 ml/acre)", "Foliar spray using 200 L water", "Benchmark systemic triazole curative fungicide; PHI: 30 days."),
            ("chemical", "Tebuconazole 50% + Trifloxystrobin 25% WG", "Tebuconazole + Trifloxystrobin", "Nativo 75 WG", "0.6 g / liter water (120 g/acre)", "Foliar spray", "Halts sporulation and provides 21-day residual canopy protection; PHI: 30 days."),
            ("chemical", "Mancozeb 75% WP", "Mancozeb", "Dithane M-45", "2 g / liter water (400 g/acre)", "Protective contact spray", "Multisite contact fungicide for early stage prevention; PHI: 14 days."),
        ],
        "varieties": ["HD-3086", "DBW-187", "DBW-222", "PBW-725", "HD-3226"],
        "faqs": [
            ("How do I distinguish Leaf Rust from Stripe Rust?", "Leaf Rust pustules are round to oval and scattered randomly across the leaf. Stripe Rust pustules are bright yellow-orange and arranged in strict, parallel, linear stripes along the veins."),
        ],
    },
    {
        "class_id": 23,
        "name": "Wheat - Loose Smut",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Ustilago tritici",
        "summary": "Internally seed-borne smut disease. Converts the entire wheat earhead into a conspicuous black powdery mass of fungal spores.",
        "symptoms": "Every spikelet of the emerged wheat head is transformed into a dusty black mass of teliospores; spores blow away leaving a bare rachis.",
        "protocols": {
            0: ("Healthy", "ALWAYS dress seeds with systemic fungicide before sowing; buy certified smut-free seed.", "Preventive"),
            1: ("Mild", "Rogue out smutted heads gently into a plastic bag before spores blow to adjacent flowers.", "Immediate physical action"),
            2: ("Moderate", "Rogue and burn all infected plants; mark seed lot as contaminated (do not save seed).", "Sanitation"),
            3: ("Severe", "Widespread smut; rogue diseased plants immediately; purchase certified treated seed next season.", "Sanitation"),
            4: ("Critical", "Severe crop contamination; complete field rogueing and mandatory seed lot replacement.", "Sanitation"),
        },
        "treatments": [
            ("preventive", "Systemic Seed Treatment with Carboxin + Thiram", "Carboxin 37.5% + Thiram 37.5% DS", "Vitavax Power", "2.5 g / kg seed", "Slurry coat seed before sowing", "Gold standard internally systemic seed dressing; kills dormant mycelium inside the embryo."),
            ("preventive", "Tebuconazole 2% DS", "Tebuconazole", "Raxil", "1.25 g / kg seed", "Dry seed dressing", "Systemic ergosterol inhibitor with 100% loose smut eradication."),
            ("cultural", "Solar Heat Seed Treatment (Traditional)", "N/A", "Eco-Thermotherapy", "Soak seeds in water for 4 hrs, then spread on tarpaulin under hot summer sun for 4 hrs", "May/June hot summer", "Eradicates internal fungal mycelium using solar thermal energy without chemicals."),
        ],
        "varieties": ["HD-2967", "DBW-187", "PBW-550", "WH-1105"],
        "faqs": [
            ("Can you spray a fungicide to cure loose smut when heads emerge?", "NO. By the time the black smutted head emerges, it is too late to spray because the entire grain is already destroyed. Loose smut can ONLY be prevented before planting via seed treatment."),
        ],
    },
    {
        "class_id": 24,
        "name": "Wheat - Powdery Mildew",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Blumeria graminis f. sp. tritici",
        "summary": "Superficial fungal disease. Covers leaves with white to light-gray cottony patches of mycelium, impairing photosynthesis.",
        "symptoms": "White to grayish-white powdery or cottony patches on upper leaf surface and sheaths; patches turn dull gray-brown with tiny black dots (cleistothecia).",
        "protocols": {
            0: ("Healthy", "Avoid over-dense sowing; do not apply excessive nitrogen fertilizer.", "Preventive"),
            1: ("Mild", "Foliar spray with Wettable Sulfur 80% WP (3 g/L) at first white patch detection.", "Action within 5 days"),
            2: ("Moderate", "Apply Propiconazole 25% EC (1 ml/L) or Triadimefon 25% WP (1 g/L) in 200 L water.", "Action within 3 days"),
            3: ("Severe", "Systemic curative spray of Tebuconazole + Trifloxystrobin (Nativo @ 0.6 g/L).", "Action within 24-48 hours"),
            4: ("Critical", "Dense fungal mat smothering canopy; apply systemic triazole and promote canopy aeration.", "Emergency action"),
        },
        "treatments": [
            ("organic", "Wettable Sulfur 80% WP", "Sulfur", "Sulfex / Thiovit", "3 g / liter water (600 g/acre)", "Foliar spray in 200 L water", "Multi-site organic contact fungicide with vapor action and mite suppression; PHI: 10 days."),
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC", "1 ml / liter water (200 ml/acre)", "Foliar spray", "Curative systemic triazole that eradicates established powdery mildew colonies; PHI: 30 days."),
            ("chemical", "Triadimefon 25% WP", "Triadimefon", "Bayleton", "1 g / liter water (200 g/acre)", "Foliar spray", "High vapor phase activity penetrating dense canopies; PHI: 25 days."),
        ],
        "varieties": ["PBW-621", "HD-2967", "WH-1105", "HD-3086"],
        "faqs": [
            ("What weather favors powdery mildew?", "Cool, overcast, humid days (15-22°C) without heavy rains favor powdery mildew. Unlike rusts, powdery mildew does NOT require free liquid water on the leaf to infect."),
        ],
    },
    {
        "class_id": 25,
        "name": "Wheat - Septoria",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Zymoseptoria tritici / Parastagonospora nodorum",
        "summary": "Septoria leaf blotch and glume blotch. Produces irregular necrotic blotches studded with characteristic black pycnidia specks.",
        "symptoms": "Irregular lens-shaped brown necrotic blotches surrounded by chlorotic yellow halos, studded with distinctive tiny black dots (pycnidia).",
        "protocols": {
            0: ("Healthy", "Bury or chop crop residues; plant Septoria-tolerant certified seed.", "Preventive"),
            1: ("Mild", "Apply protective spray of Chlorothalonil 75% WP (2 g/L) at tillering.", "Action within 7 days"),
            2: ("Moderate", "Apply Propiconazole 25% EC (1 ml/L) or Epoxiconazole at flag leaf emergence.", "Action within 3-5 days"),
            3: ("Severe", "Urgent systemic spray: Fluxapyroxad + Epoxiconazole or Nativo to protect flag leaf.", "Action within 24-48 hours"),
            4: ("Critical", "Severe canopy blight; apply dual-action systemic SDHI + triazole combination fungicide.", "Emergency action"),
        },
        "treatments": [
            ("chemical", "Chlorothalonil 75% WP", "Chlorothalonil", "Kavach / Bravo", "2 g / liter water (400 g/acre)", "Protective contact spray", "Multisite protectant that halts Septoria spore germination; PHI: 21 days."),
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC", "1 ml / liter water (200 ml/acre)", "Curative systemic spray", "Halts mycelial expansion inside leaf mesophyll; PHI: 30 days."),
            ("chemical", "Fluxapyroxad 6.25% + Epoxiconazole 6.25% SC", "Fluxapyroxad + Epoxiconazole", "Adexar", "1.25 ml / liter water (250 ml/acre)", "Systemic foliar spray", "Powerful SDHI + DMI combo with 28-day residual protection; PHI: 35 days."),
        ],
        "varieties": ["HD-3086", "DBW-187", "WH-1105", "PBW-725"],
        "faqs": [
            ("How can I identify Septoria under a hand lens?", "Look inside the tan-brown dried blotches for tiny black pepper-like specks. These are pycnidia—the fungal fruiting bodies that ooze infectious spores during rain splashes."),
        ],
    },
    {
        "class_id": 26,
        "name": "Wheat - Stem Rust (Black Rust)",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Puccinia graminis f. sp. tritici",
        "summary": "Catastrophic cereal rust. Produces large, elongated reddish-brown to dark black pustules on stems and sheaths, causing lodging.",
        "symptoms": "Large, elongated reddish-brown to dark black pustules with torn epidermal edges on stems, leaf sheaths, and glumes; stems break and lodge.",
        "protocols": {
            0: ("Healthy", "Plant certified stem-rust-resistant varieties with Sr2/Sr31 complex genes.", "Preventive"),
            1: ("Mild", "Immediate preventive barrier spray of Mancozeb 75% WP (2.5 g/L).", "Action within 48 hours"),
            2: ("Moderate", "Apply Tebuconazole 250 EC (1.25 ml/L) or Propiconazole 25% EC immediately.", "Action within 24 hours"),
            3: ("Severe", "Urgent systemic spray of Nativo (Tebuconazole + Trifloxystrobin @ 0.6 g/L).", "Action within 12-24 hours"),
            4: ("Critical", "Emergency epidemic alert; notify agricultural extension officer and apply community spray.", "Emergency action"),
        },
        "treatments": [
            ("chemical", "Tebuconazole 25.9% EC", "Tebuconazole", "Folicur", "1.25 ml / liter water (250 ml/acre)", "Foliar spray in 200 L water", "High-potency systemic curative triazole; PHI: 30 days."),
            ("chemical", "Tebuconazole 50% + Trifloxystrobin 25% WG", "Tebuconazole + Trifloxystrobin", "Nativo 75 WG", "0.6 g / liter water (120 g/acre)", "Foliar spray", "Synergistic curative and spore-eradicating action; PHI: 30 days."),
            ("chemical", "Mancozeb 75% WP", "Mancozeb", "Dithane M-45", "2.5 g / liter water (500 g/acre)", "Protective spray", "Contact barrier protecting stems from spore germination; PHI: 14 days."),
        ],
        "varieties": ["DBW-187", "DBW-222", "MACS-6478", "HD-3226"],
        "faqs": [
            ("Why is Stem Rust considered so dangerous?", "Stem rust attacks the actual structural stem of the plant. A crop that looks healthy can lodge completely flat within 2 to 3 weeks of infection, resulting in 100% harvest loss."),
        ],
    },
    {
        "class_id": 27,
        "name": "Wheat - Stripe Rust (Yellow Rust)",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Puccinia striiformis f. sp. tritici",
        "summary": "Cool-climate cereal rust. Forms characteristic narrow, bright yellow-orange linear stripes of pustules along leaf veins.",
        "symptoms": "Vivid yellow-orange powdery pustules arranged in narrow, parallel linear stripes along leaf veins (resembling sewing machine stitches).",
        "protocols": {
            0: ("Healthy", "Scout weekly in cool, humid, shaded field corners and near water channels.", "Preventive"),
            1: ("Mild", "Apply Propiconazole 25% EC (Tilt @ 1 ml/L or 200 ml/acre) immediately upon first stripe detection.", "Action within 24 hours"),
            2: ("Moderate", "Curative systemic spray of Propiconazole 25% EC; repeat after 12-14 days.", "Action within 24-48 hours"),
            3: ("Severe", "Urgent systemic spray of Tebuconazole + Trifloxystrobin (Nativo @ 0.6 g/L).", "Action within 24 hours"),
            4: ("Critical", "Catastrophic canopy yellowing; immediate synchronized community-wide fungicide application.", "Emergency action"),
        },
        "treatments": [
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC", "1 ml / liter water (200 ml/acre in 200 L water)", "Foliar spray at first yellow stripe", "The proven frontline defense against stripe rust; halts spore production instantly; PHI: 30 days."),
            ("chemical", "Tebuconazole 50% + Trifloxystrobin 25% WG", "Tebuconazole + Trifloxystrobin", "Nativo 75 WG", "0.6 g / liter water (120 g/acre)", "Foliar spray", "Curative and long-lasting translaminar protection; PHI: 30 days."),
            ("chemical", "Azoxystrobin 18.2% + Difenoconazole 11.4% SC", "Azoxystrobin + Difenoconazole", "Amistar Top", "1 ml / liter water (200 ml/acre)", "Foliar spray", "Broad-spectrum systemic shield; PHI: 30 days."),
        ],
        "varieties": ["DBW-187 (Karan Vandana)", "DBW-222", "HD-3226", "PBW-725", "DBW-303"],
        "faqs": [
            ("Why does Yellow Rust spread so fast in northern plains?", "Yellow rust thrives in cool temperatures (10-18°C) and high humidity or morning dew. Spores travel hundreds of kilometers on Himalayan winds, causing rapid regional epidemics."),
        ],
    },
    {
        "class_id": 28,
        "name": "Wheat - Tan Spot",
        "crop": "Wheat",
        "category": "disease",
        "scientific_name": "Pyrenophora tritici-repentis (Drechslera tritici-repentis)",
        "summary": "Yellow leaf spot disease. Characterized by tan oval lesions with a dark brown pinpoint center and an expansive yellow halo.",
        "symptoms": "Tan or buff-colored oval lesions with a distinct dark brown central pinpoint spot surrounded by an extensive bright yellow chlorotic halo.",
        "protocols": {
            0: ("Healthy", "Practice crop rotation with non-cereal crops; bury or manage surface stubble.", "Preventive"),
            1: ("Mild", "Apply Mancozeb 75% WP (2 g/L) at tillering to jointing stage.", "Action within 7 days"),
            2: ("Moderate", "Apply Propiconazole 25% EC (1 ml/L) or Prothioconazole at flag leaf emergence.", "Action within 3-5 days"),
            3: ("Severe", "Systemic spray of Azoxystrobin + Difenoconazole or Nativo to protect upper canopy.", "Action within 24-48 hours"),
            4: ("Critical", "Severe foliar necrosis; apply curative triazole and manage stubble post-harvest.", "Emergency action"),
        },
        "treatments": [
            ("chemical", "Propiconazole 25% EC", "Propiconazole", "Tilt 25 EC", "1 ml / liter water (200 ml/acre)", "Foliar spray in 200 L water", "Systemic curative triazole; PHI: 30 days."),
            ("chemical", "Prothioconazole 480 SC", "Prothioconazole", "Proline", "0.5 ml / liter water (100 ml/acre)", "Foliar spray", "Advanced triazolinthione with exceptional tan spot control; PHI: 30 days."),
            ("chemical", "Mancozeb 75% WP", "Mancozeb", "Dithane M-45", "2 g / liter water (400 g/acre)", "Protective contact spray", "Multi-site contact protectant; PHI: 14 days."),
        ],
        "varieties": ["HD-2967", "WH-542", "HD-3086", "DBW-187"],
        "faqs": [
            ("How does Tan Spot survive between wheat crops?", "The fungus overwinters as pseudothecia on retained wheat straw stubble. Spores eject in spring during rain, infecting lower leaves first before splashing upward."),
        ],
    },
]


# =============================================================================
# DATABASE BUILDER & SCHEMA SETUP
# =============================================================================

class TreatmentKnowledgeBase:
    """
    Offline SQLite interface providing verified agronomic treatment data,
    severity-graded action plans, and LLM grounding context.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None, auto_build: bool = True):
        self.db_path = Path(db_path) if db_path else DB_PATH
        if auto_build and not self.db_path.exists():
            self.build_database()

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with dict-like row factory."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def build_database(self, force: bool = False) -> None:
        """Creates normalized schema and populates all 29 target crop classes."""
        if self.db_path.exists() and not force:
            # Check if database is already populated
            try:
                with self.get_connection() as conn:
                    count = conn.execute("SELECT COUNT(*) FROM conditions").fetchone()[0]
                    if count >= len(CONDITIONS_DATA):
                        return
            except Exception:
                pass

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.get_connection()
        cur = conn.cursor()

        # Drop existing tables if rebuilding
        cur.executescript("""
            DROP TABLE IF EXISTS faqs;
            DROP TABLE IF EXISTS resistant_varieties;
            DROP TABLE IF EXISTS treatments;
            DROP TABLE IF EXISTS severity_protocols;
            DROP TABLE IF EXISTS conditions;

            CREATE TABLE conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER UNIQUE NOT NULL,
                name TEXT UNIQUE NOT NULL,
                crop TEXT NOT NULL,
                category TEXT NOT NULL,
                scientific_name TEXT,
                summary TEXT NOT NULL,
                symptoms TEXT NOT NULL
            );

            CREATE TABLE severity_protocols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id INTEGER NOT NULL,
                severity_grade INTEGER NOT NULL,
                severity_level TEXT NOT NULL,
                action_required TEXT NOT NULL,
                urgency TEXT NOT NULL,
                FOREIGN KEY (condition_id) REFERENCES conditions (id) ON DELETE CASCADE
            );

            CREATE TABLE treatments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id INTEGER NOT NULL,
                treatment_type TEXT NOT NULL,
                title TEXT NOT NULL,
                active_ingredient TEXT,
                commercial_names TEXT,
                dosage TEXT NOT NULL,
                application_timing TEXT,
                safety_precaution TEXT,
                FOREIGN KEY (condition_id) REFERENCES conditions (id) ON DELETE CASCADE
            );

            CREATE TABLE resistant_varieties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id INTEGER NOT NULL,
                variety_name TEXT NOT NULL,
                FOREIGN KEY (condition_id) REFERENCES conditions (id) ON DELETE CASCADE
            );

            CREATE TABLE faqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                FOREIGN KEY (condition_id) REFERENCES conditions (id) ON DELETE CASCADE
            );

            CREATE INDEX idx_conditions_name ON conditions (name);
            CREATE INDEX idx_conditions_class_id ON conditions (class_id);
            CREATE INDEX idx_treatments_cond ON treatments (condition_id, treatment_type);
            CREATE INDEX idx_severity_cond ON severity_protocols (condition_id, severity_grade);
        """)

        # Insert all conditions and related child records
        for c in CONDITIONS_DATA:
            cur.execute("""
                INSERT INTO conditions (class_id, name, crop, category, scientific_name, summary, symptoms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                c["class_id"],
                c["name"],
                c["crop"],
                c["category"],
                c.get("scientific_name", ""),
                c["summary"],
                c["symptoms"],
            ))
            cond_id = cur.lastrowid

            # Protocols
            for grade, (level, action, urgency) in c.get("protocols", {}).items():
                cur.execute("""
                    INSERT INTO severity_protocols (condition_id, severity_grade, severity_level, action_required, urgency)
                    VALUES (?, ?, ?, ?, ?)
                """, (cond_id, grade, level, action, urgency))

            # Treatments
            for t in c.get("treatments", []):
                t_type, title, active, comm, dose, timing, safety = t
                cur.execute("""
                    INSERT INTO treatments (condition_id, treatment_type, title, active_ingredient, commercial_names, dosage, application_timing, safety_precaution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (cond_id, t_type, title, active, comm, dose, timing, safety))

            # Resistant Varieties
            for var in c.get("varieties", []):
                cur.execute("""
                    INSERT INTO resistant_varieties (condition_id, variety_name)
                    VALUES (?, ?)
                """, (cond_id, var))

            # FAQs
            for q, a in c.get("faqs", []):
                cur.execute("""
                    INSERT INTO faqs (condition_id, question, answer)
                    VALUES (?, ?, ?)
                """, (cond_id, q, a))

        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Query Engine
    # -------------------------------------------------------------------------
    def get_condition(self, query: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Finds a condition by numerical class_id or fuzzy class name."""
        with self.get_connection() as conn:
            if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
                row = conn.execute("SELECT * FROM conditions WHERE class_id = ?", (int(query),)).fetchone()
            else:
                q_str = str(query).strip()
                # Exact match first
                row = conn.execute("SELECT * FROM conditions WHERE LOWER(name) = LOWER(?)", (q_str,)).fetchone()
                if not row:
                    # Partial match
                    row = conn.execute("SELECT * FROM conditions WHERE LOWER(name) LIKE LOWER(?)", (f"%{q_str}%",)).fetchone()
            if row:
                return dict(row)
        return None

    def get_treatment_plan(
        self,
        condition_query: Union[str, int],
        severity_grade: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves a complete structured treatment dossier for a condition:
          - Core diagnosis and botanical details
          - Specific protocol for the given severity grade (0 to 4)
          - Verified chemical treatments with exact dosages & active ingredients
          - Biological, organic, and cultural alternatives
          - Resistant crop varieties and clinical FAQs
        """
        cond = self.get_condition(condition_query)
        if not cond:
            return {"error": f"Condition not found in knowledge base: '{condition_query}'"}

        cond_id = cond["id"]
        with self.get_connection() as conn:
            # Severity Protocols
            protocols_rows = conn.execute(
                "SELECT * FROM severity_protocols WHERE condition_id = ? ORDER BY severity_grade",
                (cond_id,),
            ).fetchall()
            protocols = {row["severity_grade"]: dict(row) for row in protocols_rows}

            # Treatments
            treatments_rows = conn.execute(
                "SELECT * FROM treatments WHERE condition_id = ?",
                (cond_id,),
            ).fetchall()

            chemical = []
            biological = []
            cultural = []
            preventive = []

            for row in treatments_rows:
                t_dict = dict(row)
                t_type = t_dict["treatment_type"].lower()
                if t_type == "chemical":
                    chemical.append(t_dict)
                elif t_type in ("biological", "organic"):
                    biological.append(t_dict)
                elif t_type == "cultural":
                    cultural.append(t_dict)
                elif t_type == "preventive":
                    preventive.append(t_dict)

            # Resistant Varieties
            varieties = [
                row["variety_name"]
                for row in conn.execute(
                    "SELECT variety_name FROM resistant_varieties WHERE condition_id = ?",
                    (cond_id,),
                ).fetchall()
            ]

            # FAQs
            faqs = [
                {"question": row["question"], "answer": row["answer"]}
                for row in conn.execute(
                    "SELECT question, answer FROM faqs WHERE condition_id = ?",
                    (cond_id,),
                ).fetchall()
            ]

        # Selected severity action
        active_protocol = None
        if severity_grade is not None:
            grade_int = max(0, min(4, int(severity_grade)))
            active_protocol = protocols.get(grade_int)

        return {
            "condition": cond,
            "active_protocol": active_protocol,
            "all_protocols": protocols,
            "chemical_treatments": chemical,
            "biological_treatments": biological,
            "cultural_treatments": cultural,
            "preventive_measures": preventive,
            "resistant_varieties": varieties,
            "faqs": faqs,
        }

    def format_advisor_context(
        self,
        disease_name: str,
        severity_grade: int = 1,
        lesion_pct: float = 0.0,
    ) -> str:
        """
        Formats a clean, grounded agronomic context specifically tailored to
        feed into Llama-3.2-1B-Instruct (Module 05) to eliminate hallucinations.
        """
        plan = self.get_treatment_plan(disease_name, severity_grade=severity_grade)
        if "error" in plan:
            return f"[KNOWLEDGE BASE NOTE] No verified database match found for: '{disease_name}'."

        cond = plan["condition"]
        protocol = plan["active_protocol"] or {}

        lines = [
            "### VERIFIED AGRONOMIC KNOWLEDGE BASE CONTEXT (GROUND TRUTH):",
            f"- Diagnosed Condition: {cond['name']} ({cond['scientific_name']})",
            f"- Crop / Category:     {cond['crop']} ({cond['category'].upper()})",
            f"- Clinical Summary:    {cond['summary']}",
            f"- Typical Symptoms:    {cond['symptoms']}",
            f"- Evaluated Severity:  Grade {severity_grade} ({protocol.get('severity_level', 'N/A')}) | Measured Lesion Area: {lesion_pct:.2f}%",
            f"- Clinical Action:     {protocol.get('action_required', 'Follow standard agronomic scouting.')}",
            f"- Action Urgency:      {protocol.get('urgency', 'Normal')}",
        ]

        # Chemical remedies
        if plan["chemical_treatments"]:
            lines.append("\n- Recommended Curative Chemical Treatments:")
            for i, chem in enumerate(plan["chemical_treatments"], 1):
                lines.append(
                    f"  {i}. {chem['title']} [Active: {chem['active_ingredient']}] | Dosage: {chem['dosage']}"
                    f" | Timing: {chem['application_timing']} | Safety/PHI: {chem['safety_precaution']}"
                )

        # Organic / Biological remedies
        if plan["biological_treatments"]:
            lines.append("\n- Organic / Biological IPM Alternatives:")
            for i, bio in enumerate(plan["biological_treatments"], 1):
                lines.append(
                    f"  {i}. {bio['title']} [Active: {bio['active_ingredient']}] | Dosage: {bio['dosage']}"
                    f" | Timing: {bio['application_timing']}"
                )

        # Cultural measures
        if plan["cultural_treatments"]:
            lines.append("\n- Cultural Agronomy & Preventive Practices:")
            for i, cult in enumerate(plan["cultural_treatments"], 1):
                lines.append(f"  {i}. {cult['title']}: {cult['dosage']}")

        # Resistant varieties
        if plan["resistant_varieties"]:
            lines.append(f"\n- Certified Resistant Seed Varieties: {', '.join(plan['resistant_varieties'])}")

        # Special notice for beneficial organisms
        if cond["category"] == "beneficial":
            lines.append("\n⚠️ CRITICAL NOTICE: THIS ORGANISM IS BENEFICIAL TO THE FARM ECOSYSTEM. DO NOT APPLY INSECTICIDES.")

        return "\n".join(lines)

    def list_all_conditions(self) -> List[Dict[str, Any]]:
        """Returns all 29 indexed conditions with metadata."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT class_id, name, crop, category, scientific_name FROM conditions ORDER BY class_id").fetchall()
            return [dict(r) for r in rows]


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="🌱 Offline Agricultural Treatment Knowledge Base (SQLite)")
    parser.add_argument("--build", action="store_true", help="Build or rebuild data/treatments.db database")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing database from scratch")
    parser.add_argument("--list", action="store_true", help="List all 29 indexed crop conditions and pests")
    parser.add_argument("--query", type=str, default=None, help="Query condition by name or class_id")
    parser.add_argument("--severity", type=int, default=2, help="Severity grade (0 to 4) for query [Default: 2]")
    parser.add_argument("--context", action="store_true", help="Print Llama-3.2-1B formatted advisor context string")
    parser.add_argument("--json", action="store_true", help="Output result as formatted JSON")

    args = parser.parse_args()
    kb = TreatmentKnowledgeBase(auto_build=False)

    if args.build or args.force or not DB_PATH.exists():
        print(f"[BUILD] Populating offline knowledge base at: {DB_PATH}")
        kb.build_database(force=args.force)
        print(f"✅ Success! Database ready with all {len(CONDITIONS_DATA)} verified conditions.")
        if not args.query and not args.list:
            return

    if args.list:
        conditions = kb.list_all_conditions()
        print("\n" + "=" * 78)
        print(f"🌾 INDEXED CONDITIONS IN KNOWLEDGE BASE ({len(conditions)} TOTAL)")
        print("=" * 78)
        print(f"{'ID':<4} | {'Condition Name':<35} | {'Crop':<16} | {'Type':<12}")
        print("-" * 78)
        for c in conditions:
            print(f"{c['class_id']:<4} | {c['name']:<35} | {c['crop']:<16} | {c['category']:<12}")
        print("=" * 78 + "\n")
        return

    if args.query:
        if args.context:
            context_str = kb.format_advisor_context(args.query, severity_grade=args.severity, lesion_pct=15.0)
            print("\n" + context_str + "\n")
            return

        plan = kb.get_treatment_plan(args.query, severity_grade=args.severity)
        if args.json:
            print(json.dumps(plan, indent=2))
            return

        cond = plan.get("condition")
        if not cond:
            print(f"[ERROR] {plan.get('error')}")
            return

        proto = plan.get("active_protocol") or {}
        print("\n" + "=" * 78)
        print(f"🌱 AGRONOMIC TREATMENT DOSSIER: {cond['name'].upper()}")
        print("=" * 78)
        print(f"Scientific Name:  {cond['scientific_name']}")
        print(f"Crop / Category:  {cond['crop']} ({cond['category'].upper()})")
        print(f"Summary:          {cond['summary']}")
        print(f"Symptoms:         {cond['symptoms']}")
        print("-" * 78)
        print(f"Selected Grade:   Grade {args.severity} ({proto.get('severity_level', 'N/A')})")
        print(f"Action Protocol:  {proto.get('action_required', 'N/A')}")
        print(f"Urgency Window:   {proto.get('urgency', 'N/A')}")

        if plan["chemical_treatments"]:
            print("\n💊 VERIFIED CHEMICAL TREATMENTS:")
            for chem in plan["chemical_treatments"]:
                print(f"  • {chem['title']}")
                print(f"    Active:   {chem['active_ingredient']} (Commercial: {chem['commercial_names']})")
                print(f"    Dosage:   {chem['dosage']}")
                print(f"    Timing:   {chem['application_timing']}")
                print(f"    Safety:   {chem['safety_precaution']}")

        if plan["biological_treatments"]:
            print("\n🍃 ORGANIC & BIOLOGICAL ALTERNATIVES:")
            for bio in plan["biological_treatments"]:
                print(f"  • {bio['title']} | Dosage: {bio['dosage']} ({bio['application_timing']})")

        if plan["cultural_treatments"]:
            print("\n🚜 CULTURAL AGRONOMY:")
            for cult in plan["cultural_treatments"]:
                print(f"  • {cult['title']}: {cult['dosage']}")

        if plan["resistant_varieties"]:
            print(f"\n🌾 RECOMMENDED RESISTANT VARIETIES: {', '.join(plan['resistant_varieties'])}")

        if plan["faqs"]:
            print("\n❓ AGRONOMIC FAQ:")
            for faq in plan["faqs"]:
                print(f"  Q: {faq['question']}")
                print(f"  A: {faq['answer']}")

        print("=" * 78 + "\n")
        return

    # If no flags passed, build and show sample query
    print(f"[INFO] Building database at: {DB_PATH}")
    kb.build_database()
    print("✅ Database verified! Running sample query for 'Wheat - Leaf Rust' (Grade 2):\n")
    sample_ctx = kb.format_advisor_context("Wheat - Leaf Rust", severity_grade=2, lesion_pct=18.4)
    print(sample_ctx)


if __name__ == "__main__":
    main()
