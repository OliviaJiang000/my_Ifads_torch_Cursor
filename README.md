# My LFADS-torch Cursor Implementation

This repository contains a complete implementation of LFADS (Latent Factor Analysis via Dynamical Systems) using PyTorch Lightning, specifically adapted for neural data analysis.

## Project Structure

```
├── lfads-torch/           # Main LFADS implementation
│   ├── lfads_torch/       # Core LFADS modules
│   ├── configs/           # Configuration files
│   ├── scripts/           # Training scripts
│   └── tutorials/         # Tutorial notebooks
├── nlb_tools/             # Neural Latents Benchmark tools
└── README.md              # This file
```

## Features

- **Complete LFADS Implementation**: Full PyTorch Lightning implementation with modern best practices
- **NLB Integration**: Neural Latents Benchmark evaluation metrics
- **Custom Data Support**: Supports custom neural datasets (demonstrated with 000128 dataset)
- **Flexible Configuration**: Hydra-based configuration system
- **Multiple Training Modes**: Single session, multi-session, and population-based training

## Model Performance

Successfully trained on 000128 dataset with:
- **Model Parameters**: 390K parameters
- **Training Results**: 
  - Initial validation loss: 0.239
  - Final validation loss: 0.190 (20% improvement)
  - BPS metric: 0.307 (good reconstruction quality)
- **Training Duration**: 100 epochs, stable convergence

## Key Components

### LFADS Core (`lfads_torch/`)
- **Model**: Main LFADS model implementation
- **Encoder/Decoder**: Variational autoencoder components
- **Priors**: Gaussian priors for latent variables
- **Metrics**: Bits per spike, R² score, and other neural metrics

### Configuration (`configs/`)
- **Datamodule**: Custom data loading configurations
- **Model**: Model hyperparameter configurations
- **Callbacks**: Training callbacks including NLB evaluation

### Training Scripts (`scripts/`)
- `run_my.py`: Main training script for custom datasets
- `run_single.py`: Single session training
- `run_multi.py`: Multi-session training

## Installation

1. Clone the repository:
```bash
git clone git@github.com:OliviaJiang000/my_Ifads_torch_Cursor.git
cd my_Ifads_torch_Cursor
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
cd lfads-torch
pip install -r requirements.txt
pip install -e .
```

## Usage

### Basic Training
```bash
cd lfads-torch/scripts
python run_my.py
```

### Custom Configuration
Modify configuration files in `configs/` directory:
- `datamodule/my_datamodule01.yaml`: Data loading settings
- `model/my_module01.yaml`: Model hyperparameters

## Data Format

The model expects HDF5 files with:
- `train_encod_data`: Training input data (samples × time × neurons)
- `train_recon_data`: Training reconstruction targets
- `valid_encod_data`: Validation input data
- `valid_recon_data`: Validation reconstruction targets
- `behavior`: Behavioral data (optional)

## Results

The trained model achieves:
- Excellent reconstruction performance (validation loss: 0.190)
- Good generalization (no overfitting observed)
- Stable training convergence
- Meaningful latent representations

## Compatibility

- **Python**: 3.9+
- **PyTorch**: 1.12+
- **PyTorch Lightning**: 2.0+ (with compatibility fixes)
- **Hydra**: 1.2+

## Contributing

This is a research project. For questions or contributions, please create an issue or submit a pull request.

## License

This project is based on the original LFADS-torch implementation. Please check individual component licenses.

## Acknowledgments

- Original LFADS-torch implementation
- Neural Latents Benchmark (NLB) tools
- PyTorch Lightning framework
