# Traffic Semantic Graphs

<img src="figures/project_overview.png" alt="project overview">

## 📁 Directory Overview

```plaintext
TRAFFIC_SEMANTIC_GRAPHS/
├── data/                            # Local-only: raw and processed data (not included in repo)
│   ├── distributions/               # Data distributions
│   ├── graphical/                   # Graph files for L2D and NuPlan datasets
│   ├── graphical_final/             # Final graphical data after processing and filtering
│   ├── processed/                   # Processed tabular data
│   ├── processed_frames/            # Processed image frames (e.g., YOLO, depth outputs)
│   ├── raw/                         # Raw inputs (images, tabular)
│   ├── semantic_tags/               # Semi-manually-generated semantic tags
│   └── temporary_data/              # Temporary data storage
├── figures/                         # Project figures and visualizations
│   ├── project_overview.pdf
│   └── project_overview.png
├── ml-depth-pro/                    # Machine learning depth prediction project
│   └── checkpoints/                 # Checkpoints for ML models
├── nuplan-devkit/                   # NuPlan development kit
├── scripts/                         # Scripts for data processing and visualization
│   ├── 1A_l2d_processing.py         # L2D data processing script
│   ├── 1B_nup_processing.py         # NuPlan data processing script
│   ├── 1C_final_processing.py       # Final processing script
│   └── scene_visualizer.py          # Scene visualization script
├── src/                             # Main source code
│   ├── data_processing/             # Modules for data loading and processing
│   │   ├── filtering.py
│   │   ├── final_post_processing.py
│   │   ├── l2d_generate_graphs.py
│   │   ├── l2d_lane_processing.py
│   │   ├── l2d_load_data.py
│   │   ├── l2d_process_frames.py
│   │   ├── l2d_process_pqts.py
│   │   ├── l2d_process_tags.py
│   │   ├── nup_load_data.py
│   │   └── nup_process_jsons.py
│   ├── risk_analysis.py             # Risk analysis module
│   ├── utils.py                     # Utility functions
│   └── visualizations.py            # Visualization functions
└── README.md                        # This file

```
**Note:** The actual data files are not included in the repository due to storage limitations.

## Python Environment Set-up
*Tested on python 3.12*
1. Run `pip install -r requirements.txt`
2. Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` >> for alternatives check out this link: [PyTorch Start Locally](https://pytorch.org/get-started/locally/)
3. Run `python -c "import torch; print('CUDA is available.' if torch.cuda.is_available() else 'CUDA is not available.')"` >> see if cuda is available on your device
4. Run `pip install torch-geometric`

## Phase 1: Data Processing

This phase involves processing the raw data from the L2D and NuPlan datasets into a format suitable for risk analysis.

### 1A: L2D Data Processing

The `1A_l2d_processing.py` script processes the L2D dataset.

**Usage:**
```bash
python -m scripts.1A_l2d_processing [-h] [--min_ep MIN_EP] [--max_ep MAX_EP] [--download] [--process_tabular] [--add_tags] [--process_frames] [--process_lanes] [--generate_graphs] [--all]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--min_ep MIN_EP`: Minimum episode number to process.
- `--max_ep MAX_EP`: Maximum episode number to process.
- `--download`: Run data download step.
- `--process_tabular`: Run tabular data processing step.
- `--add_tags`: Run tag processing step.
- `--process_frames`: Run frame processing step.
- `--process_lanes`: Run lane processing step.
- `--generate_graphs`: Run graph generation step.
- `--all`: Run all steps (default if no flags are set).

To run the entire L2D processing pipeline, use the following command:
```bash
python -m scripts.1A_l2d_processing --all
```

### 1B: NuPlan Data Processing

The `1B_nup_processing.py` script processes the NuPlan dataset.

**Usage:**
```bash
python -m scripts.1B_nup_processing [-h] [--city CITY] [--file_min FILE_MIN] [--file_max FILE_MAX] [--episodes EPISODES [EPISODES ...]] [--load] [--latlon] [--weather] [--weather_codes] [--temporal] [--tags]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--city CITY`: City to process (boston or pittsburgh).
- `--file_min FILE_MIN`: Minimum DB file index to process (inclusive).
- `--file_max FILE_MAX`: Maximum DB file index to process (exclusive). Use 'none' for all after file_min.
- `--episodes EPISODES [EPISODES ...]`: Episodes to Process.
- `--load`: Run only the data loading step.
- `--latlon`: Run only the lat/lon addition step.
- `--weather`: Run only the weather enrichment step.
- `--weather_codes`: Run only the weather code replacement step.
- `--temporal`: Run only the temporal feature addition step.
- `--tags`: Run only the tag extraction step.

To run the NuPlan processing for a specific city (e.g., Boston) and a specific step (e.g., tags), use the following command:
```bash
python -m scripts.1B_nup_processing --city boston --tags
```

To run all steps, you can omit the step-specific flags:
```bash
python -m scripts.1B_nup_processing --city boston
```

### 1C: Final Post-Processing

The `1C_final_processing.py` script performs final post-processing on the generated graph data.

**Usage:**
```bash
python -m scripts.1C_final_processing [-h] [--process_l2d] [--process_nuplan_boston] [--process_nuplan_pittsburgh]
```

**Arguments:**
- `-h, --help`: show this help message and exit
- `--process_l2d`: Run L2D final processing
- `--process_nuplan_boston`: Run nuPlan Boston final processing
- `--process_nuplan_pittsburgh`: Run nuPlan Pittsburgh final processing

To run the final processing for a specific dataset, use the corresponding flag. For example, to process the L2D dataset:
```bash
python -m scripts.1C_final_processing --process_l2d
```
