Multitask Spectral Property Prediction from Protein Embeddings

Overview
--------
This project trains a multitask neural network to predict spectral properties of fluorescent proteins from embedding vectors. The model uses a multi-head architecture with ranking-based training and optional classification tasks, followed by calibration to improve prediction accuracy.

The pipeline supports:
- Multitask regression (e.g., excitation, emission, brightness)
- Multi-head outputs per task
- Ranking-based training (Spearman-focused)
- Post-hoc calibration (MLP, isotonic, sigmoid, 5PL)
- Cross-validation evaluation
- Optional classification tasks (e.g., localization, structural flags)

Repository Structure
--------------------
project/
  driver.py                # Main training + evaluation script
  config.py                # Global constants (targets, task definitions)
  utils.py                 # Data loading and batching utilities
  models.py                # Neural network architectures
  training.py              # Training loops and loss functions
  calibration.py           # Calibration methods (MLP, isotonic, etc.)
  evaluation.py            # Metrics and plotting
  preprocessing.py         # Embedding normalization
  requirements.txt         # Python dependencies (recommended)
  README.md                # This file

Installation
------------
Create a Python environment (conda or venv recommended), then install dependencies:

pip install -r requirements.txt

If requirements.txt is not available:

pip install numpy pandas torch scikit-learn scipy matplotlib

Data Format
-----------
Input data should be a CSV file containing:
- Embedding vectors (numerical columns)
- Target values for spectral properties
- Optional classification labels

Example:
embeddings/max_embeddings_plus.csv

Data loading is handled in:
utils.load_data()

Running the Model
-----------------
Basic usage:

python driver.py --csv embeddings/max_embeddings_plus.csv

Key arguments:
--batch_size        (default: 128)
--epochs            (default: 100)
--lr                (default: 4e-4)
--margin            (default: 0.6)
--load_normalizer   (use saved normalization)
--final_train       (train on full dataset after CV)
--train_error_model (enable error predictor)
--output_json       (save results to JSON)

Example:

python driver.py \
  --csv embeddings/max_embeddings_plus.csv \
  --epochs 200 \
  --batch_size 256 \
  --output_json results.json

Model Description
-----------------
The main model (MultiTaskRegressorMultiHead_wClass) includes:
- Shared feature extraction layers
- Multiple heads per regression target
- Optional classification heads
- Ranking-based loss to optimize ordering

Training emphasizes:
- Spearman correlation
- Constraint-aware predictions
- Multi-head diversity

Calibration
-----------
Model outputs are calibrated using:
- MLP calibration (primary)
- Isotonic regression
- Sigmoid (Platt-style)
- Five-parameter logistic (5PL)

Calibration improves:
- Absolute accuracy (R²)
- Alignment with real values

Outputs
-------
The script produces:
- Cross-validation predictions
- Calibrated predictions
- Evaluation plots
- Optional JSON metrics
- Saved models and calibrators

Example outputs:
saved_models/
validation_predictions.csv
final_regressor.pt
mlp_calibrators.pkl

Notes
-----
- Some parts of the driver contain experimental or legacy code
- Error prediction module may require additional setup
- Auxiliary feature models (FP classifier, pLDDT predictor) are optional and not included
- Import paths may need adjustment depending on folder structure

Recommendations for Contributors
--------------------------------
- Keep modules modular
- Avoid adding logic directly into driver.py
- Clean unused imports and duplicate functions
- Document new features clearly

Contact / Contribution
----------------------
This is an active research codebase and may change frequently.
Contributions should prioritize clarity, modularity, and reproducibility.
