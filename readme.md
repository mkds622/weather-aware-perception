# Weather-Aware Perception: CARLA vs RADIATE

## Overview
This project evaluates whether weather-dependent signal exists in simulated (CARLA) and real-world (RADIATE) sensor data using LiDAR, radar, and camera inputs.

### Key Findings
- CARLA radar and LiDAR show no meaningful weather-dependent signal
- Camera dominates classification in simulation
- Real-world radar (RADIATE) exhibits clear weather signal

---

## Project Structure

configs/        → Simulation configs  
scripts/        → Data collection + weather runs  
train_*.py      → Training pipelines  
models/         → Trained models (minimal)  
results/        → Output screenshots  

---

## Dataset Structure

### CARLA
raw/weather_dataset_extended_2/run_x/{clear,fog,rain}/ego/
  - camera_front/*.png  
  - lidar/*.bin  
  - radar_front/*.npy  
  - radar_back/*.npy  

### RADIATE
raw/radiate/{train,val}/{clear,rain,fog}/Navtech_Polar/*.png

---

## Results

| Setup | Validation Accuracy |
|------|-------------------|
| Camera (CARLA) | ~1.00 |
| Radar (CARLA) | ~0.33 |
| LiDAR (CARLA) | ~0.33 |
| Fusion (CARLA) | ~0.90 |
| Radar (RADIATE) | ~0.66 |

---

## Visual Evidence

### LiDAR (CARLA)
![LiDAR Comparison](results/carla/lidar_pc.png)

### Camera-only Performance
![Camera](results/carla/camera_only/result.png)

### Radar-only Performance
![Radar](results/carla/radar_only_extended_2/result.png)

### Fusion Performance
![Fusion](results/carla/fusion_rc_extended_2/result.png)

### RADIATE Performance
![Fusion](results/radiate/radiate_extensive/result.png)

---

## Running Experiments

python train_camera_only.py  
python train_radar_only.py  
python train_mlp.py  
python train_fusion_mlp.py  
python train_radiate.py  

---

## Notes
- Models use statistical feature representations
- Raw datasets not included due to size
- Results reproducible given dataset structure

---

## Report
Final report.pdf
