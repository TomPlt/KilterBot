# KilterBot

Use some emulator for android i.e. [Geny Cloud](https://cloud.geny.io/).
Login and pull the database of Kilter app via `kilterproblems/Login.ps1` (Login is needed as only than the db is visible) <br>

## Example Climb 
![kilterimage](data/pngs/example_output.png)

## Generating Graphs out of the beta videos of the climbs 
Using motion tracking with mediapipe in `climber.py` and save landmarks in `data/specific_landmarks_sequence.json` <br>
![climber](climber.gif) <br>
comparing them to the static landmarks of the holds `data/holds.json` gives the squence information

## Mapping Landmarks to Kilterholds 
![climber](data/pngs/hold_matching.png)