# config.py

TARGET_NAMES = [
    "ex_max", "em_max", "ext_coeff", "qy", "brightness", "stroke_shift", "pka", "aromaticity", "instability_index"
    
    
]

SPECTRAL_TARGETS = ["ex_max", "em_max", "ext_coeff", "qy", "brightness", "stroke_shift", "pka"]

# List of all classification tasks
CLASSIFICATION_TASKS = [
    "oligomerization",   # mono, di, tetramer, wd
    "switch_type",       # p, s, c, mp, o
    "lifetime",          # low / med / high (or 2-3 bins)
    "maturation"         # fast / slow / unknown
]

# Number of discrete classes for each classification task
CLASSIFICATION_DIMS = {
    "oligomerization": 5,  # e.g. m, d, t, wd
    "switch_type": 5,      # e.g. p, s, c, mp, o
    "lifetime": 3,         # bin continuous lifetime (e.g. short/medium/long)
    "maturation": 3        # fast vs. slow (or expressed vs not)
}

CLASSIFICATION_LOSS_WEIGHTS = {
    "oligomerization": 1.0,
    "switch_type": 1.0,
    "lifetime": 0.5,
    "maturation": 0.5
}

CLASSIFICATION_NAMES = {
    "ex_max": 7,  # UV, Blue, Green, Yellow, Orange, Red, Far-Red
    "em_max": 7,
    "ext_coeff": 3,
    "qy": 3,
    "brightness": 4,  # Dim, Moderate, Bright, Very Bright
    "stroke_shift": 3  # Small, Moderate, Large
}

NUM_SPECTRAL = len(SPECTRAL_TARGETS)


NUM_OUTPUTS = len(TARGET_NAMES)

ERROR_BINS = {
    "ex_max": [10, 40, 80],
    "em_max": [10, 40, 80],
    "ext_coeff": [10000, 20000, 50000],
    "qy": [0.05, 0.15, 0.30],
    "brightness": [5, 30],
    "stroke_shift": [ 10, 30, 80]
    }
    
NUM_CLASSES = {t: len(edges) + 1 for t, edges in ERROR_BINS.items()}


#"hydrophobicity", "aromaticity", "molecular_weight", "isoelectric_point", "instability_index", "stroke_shift", , "isoelectric_point", "instability_index"
#, "aromaticity", "isoelectric_point", "hydrophobicity", "isoelectric_point"