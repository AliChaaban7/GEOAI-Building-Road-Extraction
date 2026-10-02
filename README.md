<div align="center">

🛰️ Deep Learning-Based Extraction of Buildings & Roads

from Multi-Source Remote Sensing Imagery

Master Thesis · GeoAI · Remote Sensing · Deep Learning · GIS

<br>

A complete GeoAI framework for extracting buildings and roads from aerial, satellite, and drone imagery — from training data preparation to independent GIS evaluation.

<br>











<br>

Buildings + Roads · PyTorch + ArcGIS Learn · Validation-Driven Optimization · Independent Final Testing

</div>

✨ Project at a Glance

<table>
<tr>
<td width="25%" align="center">

🏢

Buildings

Source-specific extraction from aerial, satellite, and drone imagery.

</td>
<td width="25%" align="center">

🛣️

Roads

Mixed-source RGB training with separate final evaluation by source.

</td>
<td width="25%" align="center">

🧠

Deep Learning

U-Net · DeepLabV3+ · Mask R-CNN · ConnectNet · MultiTask · SAM-LoRA

</td>
<td width="25%" align="center">

🗺️

GeoAI

ArcGIS Pro · ArcPy · GIS post-processing · spatial evaluation

</td>
</tr>
</table>

🎯 Thesis Objective

This repository contains the complete implementation developed for the master thesis:

Deep Learning-Based Extraction of Buildings and Roads from Multi-Source Remote Sensing Imagery

The research investigates how deep-learning architectures can extract geospatial features from heterogeneous remote-sensing imagery while preserving a controlled, reproducible, and scientifically separated Train / Validation / Final-Test methodology.

The project integrates:

multi-source remote sensing

semantic segmentation

instance segmentation

foundation-model adaptation

PyTorch training pipelines

ArcGIS Learn models

Optuna hyperparameter optimization

Master Optuna experiment optimization

validation-only threshold search

GIS post-processing

independent final testing

IoU, Precision, Recall, and F1 evaluation

one unified Streamlit research interface

🔥 Two Complete Research Approaches



Approach 1 — Building Extraction

Approach 2 — Road Extraction

🎯 Target

Building footprints

Road surfaces / road network

🛰️ Imagery

Aerial · Satellite · Drone

Aerial + Satellite + Drone

🧪 Development Strategy

Source-specific experiments

Mixed-source RGB training

🧠 Models

U-Net, DeepLabV3+, Mask R-CNN, SAM-LoRA

U-Net, DeepLabV3+, ConnectNet, MultiTask Road Extractor, SAM-LoRA

⚙️ Backends

PyTorch

PyTorch + ArcGIS Learn

🔬 Validation

Threshold + post-processing selection

Threshold + post-processing selection

🧊 Final Stage

Frozen source-specific testing

Same frozen model tested separately on Aerial, Satellite, and Drone

🧭 End-to-End Research Workflow

flowchart TD
    A["🛰️ Multi-Source Remote-Sensing Imagery<br/>Aerial · Satellite · Drone"]
    B["🏷️ Ground Truth Labels"]
    C["📦 Export Training Data<br/>Image Chips + Masks"]
    D["🧪 Development Dataset<br/>Train / Validation"]
    E["🧠 Model Training<br/>Baseline · Augmentation · Optuna · Master Optuna"]
    F["🎯 Validation-Only Selection<br/>Best Epoch · Threshold · GIS Post-processing"]
    G["🧊 Freeze Final Settings"]
    H["🔬 Independent Final Test"]
    I["🗺️ GIS Prediction + Evaluation<br/>IoU · Precision · Recall · F1"]

    A --> B --> C --> D --> E --> F --> G --> H --> I

Core scientific principle:
Train on training data, select on validation data, freeze the configuration, and evaluate only on independent final-test data.

🏢 Approach 1 — Building Extraction

The Building workflow evaluates multiple deep-learning architectures for automatic building-footprint extraction from aerial, satellite, and drone imagery.

Models

Model

Task

Backend

U-Net

Semantic Segmentation

PyTorch

DeepLabV3+

Semantic Segmentation

PyTorch

Mask R-CNN

Instance Segmentation

PyTorch

SAM-LoRA

Foundation Model Adaptation

PyTorch

Imagery Sources

Source

Resolution Used in Thesis

Aerial

0.30 m

Satellite

0.30 m

Drone

0.10 m

Building Pipeline

flowchart TD
    A["🛰️ Source Imagery"]
    B["🏢 Building Ground Truth"]
    C["📦 ArcGIS Training Export"]
    D["🧪 Train / Validation"]
    E["🧠 Model Training"]
    F["🎯 Validation Optimization"]
    G["Threshold Search"]
    H["GIS Post-processing"]
    I["🧊 Frozen Configuration"]
    J["🔬 Independent Test"]
    K["🏘️ Building Prediction Polygons"]
    L["🗺️ GIS Evaluation"]

    A --> B --> C --> D --> E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J --> K --> L

🛣️ Approach 2 — Road Extraction

The Road workflow introduces a mixed-source development strategy.

Aerial, satellite, and drone imagery are standardized into a common RGB development dataset, while final generalization is evaluated separately by source.

Models

Model

Backend

U-Net

PyTorch

DeepLabV3+

PyTorch

SAM-LoRA

PyTorch

ConnectNet

ArcGIS Learn

MultiTask Road Extractor

ArcGIS Learn

Primary Road Dataset

<div align="center">

CT_256_Roads_RGB

Property

Value

Total image tiles

7,110

Verified RGB tiles

7,110 / 7,110

Bands

3 — RGB

Train split

80%

Validation split

20%

Fixed split seed

42

Final external test included in development

No

</div>

The original exported data remain preserved. The RGB dataset is a controlled development copy for consistent model input.

Independent Road Evaluation

The same frozen road configuration is evaluated separately on:

<table>
<tr>
<td align="center" width="33%"><b>✈️ Aerial</b></td>
<td align="center" width="33%"><b>🛰️ Satellite</b></td>
<td align="center" width="33%"><b>🚁 Drone</b></td>
</tr>
</table>

The following remain unchanged:

checkpoint

model architecture

backbone

threshold

GIS post-processing settings

training configuration

No source-specific re-tuning is allowed after freezing.

⚙️ Optimization Framework

1. Baseline

Controlled model training using a deterministic Train / Validation split.

2. Data Augmentation

Optional training-only transformations may include:

horizontal flip

vertical flip

90° rotation

brightness adjustment

contrast adjustment

3. Optuna

Validation-driven hyperparameter optimization for supported models.

Typical parameters include:

learning rate

weight decay

loss parameters

model-specific hyperparameters

4. Master Optuna

Master Optuna treats every trial as a complete candidate experiment.

flowchart LR
    A["Candidate Configuration"]
    B["Complete Trial Training"]
    C["Best Validation Checkpoint"]
    D["Compare Candidate Trials"]
    E["Select Winning Trial"]
    F["Promote Exact Winning Checkpoint"]

    A --> B --> C --> D --> E --> F

The winning checkpoint is promoted directly.
No second retraining of the Master Optuna winner is required.

🧊 Validation → Freeze → Final Test

Stage

Role

Training

Learn model parameters

Validation

Select best epoch and supported optimization settings

Threshold Search

Select probability threshold using validation only

GIS Post-processing

Select cleanup settings using validation only

Freeze

Preserve the selected checkpoint and configuration

Final Test

Measure independent generalization only

📊 Evaluation Metrics

<table>
<tr>
<td width="25%" align="center">

IoU

Spatial overlap between prediction and Ground Truth.

</td>
<td width="25%" align="center">

Precision

How many predicted positives are correct.

</td>
<td width="25%" align="center">

Recall

How many Ground Truth positives are detected.

</td>
<td width="25%" align="center">

F1

Balance between Precision and Recall.

</td>
</tr>
</table>

Validation and final-test metrics are intentionally kept separate.

🗺️ ArcGIS Pro Integration

ArcGIS Pro is a core part of the complete GeoAI workflow.

It is used for:

imagery preparation

Ground Truth creation

training-data export

raster management

ArcGIS Learn training

deep-learning inference

Raster-to-Polygon conversion

coordinate-system handling

geodatabase operations

GIS post-processing

Ground Truth comparison

custom IoU evaluation

prediction visualization

The GIS and Python workflows form one integrated research pipeline.

🖥️ Unified Streamlit Research Platform

The project contains one shared Streamlit application for Buildings and Roads.

ui/
├── app.py
├── pages/
│   ├── home.py
│   ├── buildings.py
│   ├── roads.py
│   └── dashboard.py
├── components/
├── styles/
└── utils/

Application Flow

flowchart TD
    H["🏠 Home"]
    B["🏢 Buildings"]
    R["🛣️ Roads"]
    D["📊 Results Dashboard"]

    B1["Setup & Train"]
    B2["Validation Options"]
    B3["Results"]
    B4["Final Test"]
    B5["Map Results"]

    R1["Setup & Train"]
    R2["Validation Options"]
    R3["Results"]
    R4["Final Test"]
    R5["Map Results"]

    H --> B
    H --> R
    H --> D

    B --> B1 --> B2 --> B3 --> B4 --> B5
    R --> R1 --> R2 --> R3 --> R4 --> R5

The UI manages configuration, experiment execution, result inspection, and GIS visualization while the backend modules remain authoritative.

📁 Repository Architecture

GeoAI_Thesis_Codebase/
│
├── Buildings/
│   ├── config/
│   ├── scripts/
│   └── src/
│
├── Roads/
│   ├── config/
│   ├── scripts/
│   └── src/
│
├── shared/
│
├── ui/
│   ├── app.py
│   ├── components/
│   ├── pages/
│   ├── styles/
│   └── utils/
│
├── master_outputs/
│
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md

🧠 Model Backends

<table>
<tr>
<td width="50%" valign="top">

PyTorch

U-Net

DeepLabV3+

Mask R-CNN

SAM-LoRA

</td>
<td width="50%" valign="top">

ArcGIS Learn

ConnectNet

MultiTask Road Extractor

</td>
</tr>
</table>

The backends remain intentionally separated to preserve the native behavior of each framework.

🤖 SAM-LoRA

SAM-LoRA provides parameter-efficient adaptation of the Segment Anything Model.

flowchart TD
    A["Official Pretrained SAM"]
    B["LoRA Q/V Adapters"]
    C["Trainable Mask Decoder"]
    D["Task-Specific Segmentation Model"]

    A --> D
    B --> D
    C --> D

The official pretrained SAM checkpoint is stored externally and is not committed to the repository.

🧪 JSON-Driven Experiment Configuration

Experiments are controlled through configuration files instead of hard-coding training decisions.

Configuration Field

Example

experiment_id

example_experiment

model_type

unet

backbone

resnet50

source_type

satellite

tile_size

256

use_augmentation

false

use_optuna

false

use_master_optuna

false

use_threshold_search

false

use_postprocessing

false

This architecture improves reproducibility, traceability, and experiment management.

📦 Experiment Artifacts

A typical experiment stores its evidence under:

outputs/experiments/<experiment_id>/

Artifact

Purpose

best_model.pth

Best selected model checkpoint

last_checkpoint.pth

Last training checkpoint

training_history.csv

Epoch-level training history

training_history.json

Structured training history

training_summary.json

Final training summary

optuna_selection_summary.json

Standard Optuna selection evidence

master_optuna_selection_summary.json

Master Optuna winner evidence

validation_threshold_summary.json

Validation-only threshold selection

validation_postprocess_summary.json

Validation-only GIS post-processing selection

final_settings.json

Frozen configuration before independent testing

final_test/final_metrics.json

Independent final-test metrics

These artifacts preserve experiment provenance from training → validation → freezing → final testing.

🔁 Reproducibility

The codebase preserves:

Deterministic Train / Validation splitting

Fixed internal seed where required

JSON-driven experiment configurations

Model and backbone metadata

Training history

Validation metrics

Optuna trial information

Master Optuna winner information

Threshold-selection results

GIS post-processing settings

Checkpoint provenance

Frozen final settings

Independent final-test metrics

Strict validation / final-test separation

🧰 Technology Stack

<table>
<tr>
<td width="33%" valign="top">

🧠 Deep Learning

Python

PyTorch

TorchVision

U-Net

DeepLabV3+

Mask R-CNN

SAM-LoRA

Optuna

</td>
<td width="33%" valign="top">

🌍 GeoAI / GIS

ArcGIS Pro

ArcPy

ArcGIS Learn

Raster processing

Vector processing

GIS post-processing

Spatial evaluation

</td>
<td width="33%" valign="top">

🖥️ Application

Streamlit

Pandas

NumPy

PyDeck

Pillow

JSON configuration

</td>
</tr>
</table>

🔒 Repository Policy

Large datasets and machine-specific artifacts are intentionally excluded from Git.

Examples include:

*.pth
*.pt
*.ckpt
*.onnx
*.tif
*.tiff
*.img
*.jp2
*.gdb
*.shp
*.gpkg

The repository does not distribute:

original aerial imagery

original satellite imagery

original drone imagery

exported Classified Tiles

RCNN datasets

ArcGIS geodatabases

SAM pretrained weights

generated model checkpoints

experiment outputs

Optuna databases

This keeps the repository focused on source code, methodology, configuration, and reproducible experiment logic.

✅ Project Status

<div align="center">

MASTER THESIS IMPLEMENTATION COMPLETED

🏢 Approach 1 — Building Extraction

🛣️ Approach 2 — Road Extraction

🖥️ Unified Streamlit Platform

🗺️ GIS Evaluation

🧠 PyTorch + ArcGIS Learn

⚙️ Optuna + Master Optuna


<div align="center">

Train → Validate → Freeze → Independent Test

A controlled GeoAI workflow for reproducible building and road extraction from multi-source remote-sensing imagery.

</div>
