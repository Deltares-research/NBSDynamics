import json

schema = {
    "Spartina": {
        "ColStart": "2000-04-01",
        "ColEnd": "2000-05-31",
        "random": 7,
        "mud_colonization": [0.0, 0.0],
        "fl_dr": 0.005,
        "Maximum age": 20,
        "Number LifeStages": 2,
        "initial root length": 0.05,
        "initial shoot length": 0.015,
        "initial diameter": 0.003,
        "start growth period": "2000-04-01",
        "end growth period": "2000-10-31",
        "start winter period": "2000-11-30",
        "maximum plant height": [0.8, 1.3],
        "maximum diameter": [0.003, 0.005],
        "maximum root length": [0.2, 1],
        "maximum years in LifeStage": [1, 19],
        "numStem": [700, 700],  # 3.5. number of stems per m2
        "iniCol_frac": 0.6,  # 3.6. initial colonization fraction (0-1)
        "Cd": [1.1, 1.15],  # 3.7. drag coefficient
        "desMort_thres": [400, 400],  # 3.9. dessication mortality threshold
        "desMort_slope": [0.75, 0.75],  # 3.10. dessication mortality slope
        "floMort_thres": [0.4, 0.4],  # 3.11. flooding mortality threshold
        "floMort_slope": [0.25, 0.25],  # 3.12. flooding mortality slope
        "vel_thres": [0.15, 0.25],  # 3.13. flow velocity threshold
        "vel_slope": [3, 3],  # 3.14. flow velocity slope
        "maxH_winter": [0.4, 0.4],  # 3.15  max height during winter time
    },
    "Salicornia": {
        "ColStart": "2000-02-15",
        "ColEnd": "2000-04-30",
        "random": 20,
        "mud_colonization": [0.0, 0.0],
        "fl_dr": 0.005,
        "Maximum age": 1,
        "Number LifeStages": 1,
        "initial root length": 0.15,
        "initial shoot length": 0.05,
        "initial diameter": 0.01,
        "start growth period": "2000-02-15",
        "end growth period": "2000-10-15",
        "start winter period": "2000-11-01",
        "maximum plant height": [0.4, 0],
        "maximum diameter": [0.015, 0],
        "maximum root length": [0.05, 0],
        "maximum years in LifeStage": [1, 0],
        "numStem": [190, 0],  # 3.5. number of stems per m2
        "iniCol_frac": 0.2,  # 3.6. initial colonization fraction (0-1)
        "Cd": [0.7, 0],  # 3.7. drag coefficient
        "desMort_thres": [400, 1],  # 3.9. dessication mortality threshold
        "desMort_slope": [0.75, 1],  # 3.10. dessication mortality slope
        "floMort_thres": [0.5, 1],  # 3.11. flooding mortality threshold
        "floMort_slope": [0.12, 1],  # 3.12. flooding mortality slope
        "vel_thres": [0.15, 1],  # 3.13. flow velocity threshold
        "vel_slope": [3, 1],  # 3.14. flow velocity slope
        "maxH_winter": [0.0, 0.0],  # 3.15  max height during winter time
    },
    "Puccinellia": {
        "ColStart": "2000-03-01",
        "ColEnd": "2000-04-30",
        "random": 7,
        "mud_colonization": [0.0, 0.0],
        "fl_dr": 0.005,
        "Maximum age": 20,
        "Number LifeStages": 2,
        "initial root length": 0.02,
        "initial shoot length": 0.05,
        "initial diameter": 0.004,
        "start growth period": "2000-03-01",
        "end growth period": "2000-11-15",
        "start winter period": "2000-11-30",
        "maximum plant height": [0.2, 0.35],
        "maximum diameter": [0.004, 0.005],
        "maximum root length": [0.15, 0.15],
        "maximum years in LifeStage": [1, 19],
        "numStem": [6500, 6500],  # 3.5. number of stems per m2
        "iniCol_frac": 0.3,  # 3.6. initial colonization fraction (0-1)
        "Cd": [0.7, 0.7],  # 3.7. drag coefficient
        "desMort_thres": [400, 400],  # 3.9. dessication mortality threshold
        "desMort_slope": [0.75, 0.75],  # 3.10. dessication mortality slope
        "floMort_thres": [0.35, 0.35],  # 3.11. flooding mortality threshold
        "floMort_slope": [0.4, 0.4],  # 3.12. flooding mortality slope
        "vel_thres": [0.25, 0.5],  # 3.13. flow velocity threshold
        "vel_slope": [3, 3],  # 3.14. flow velocity slope
        "maxH_winter": [0.2, 0.2],  # 3.15  max height during winter time
    },
}

with open("veg_constants.json", "w") as write_file:
    json.dump(schema, write_file, indent=4)
