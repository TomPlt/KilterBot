# KilterBot

Use some emulator for android i.e. [Geny Cloud](https://cloud.geny.io/).
Login and pull the database of Kilter app via `kilterproblems/Login.ps1` (Login is needed as only than the db is visible) <br>

## Example Climb 
![kilterimage](data/pngs/example_output.png)

## Hold Distributions Depending on Difficulty
![heatmap](data/pngs/kilter_heatmap.png)

## Generating Graphs out of Beta Videos of the Climbs 
Using motion tracking with mediapipe in `climber.py` and save landmarks in `data/specific_landmarks_sequence.json` <br>
![climber](data/pngs/climber.gif) <br>
comparing them to the static landmarks of the holds `data/holds.json` gives the squence information

## Mapping Landmarks to Kilterholds 
![climber](data/pngs/hold_matching.png) <br>

## Graph Visualization
![swooped](data/pngs/swooped.png) <br>
after using lm detection we can represent the graph differently, which directly relates moves to edges of the graph <br>
![Alt text](data/pngs/lm_graph.png)
### Technical Stuff 
run `mlflow server --backend-store-uri file:///mlruns --no-serve-artifacts` 
