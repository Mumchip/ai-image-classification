# AI Project: Image Classification with Convolutional Neural Networks

This project implements an **image classifier** using a *convolutional neural network* (CNN) to classify images from the widely used **CIFAR-10** dataset.  The goal is to provide a solid starting point for building more advanced neural network models while introducing best practices for data loading, training loops and evaluation.

## Features

* **Self-contained dataset loading** – the script automatically downloads the CIFAR-10 dataset from PyTorch's `torchvision.datasets` if it is not present on disk.
* **Customizable CNN architecture** – defined in `model.py` and easily adjustable for experimenting with different network depths and layer configurations.
* **Training loop with progress reporting** – includes a clean training loop with periodic loss/accuracy printing so you can track performance during training.
* **Validation and test evaluation** – after each epoch the model is evaluated on the validation set, and final accuracy is reported on the test set.
* **Model checkpointing** – optionally save your trained model to a file for later re-use or inference.

## Requirements

The project requires Python 3.8+ and a recent version of PyTorch. All dependencies are listed in [`requirements.txt`](./requirements.txt). To install them using `pip` simply run:

```bash
pip install -r requirements.txt
```

> **Note:** Training on GPU is supported if a CUDA-capable device is available, but the code will fall back to CPU automatically.

## Usage

From the `ai_project` directory, run the training script with:

```bash
python main.py --epochs 10 --batch-size 64 --learning-rate 0.001
```

Key command-line options include:

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--batch-size` | Mini-batch size for stochastic gradient descent | 64 |
| `--learning-rate` | Learning rate for the optimizer | 0.001 |
| `--save-model` | Path to save the trained model (optional) | *no saving* |

After training completes, the script prints the final accuracy on the test set. If `--save-model` is provided, the trained weights will be stored at the specified path (e.g. `models/cifar10_cnn.pt`).

## Project structure

```text
ai_project/
├── README.md            # Project documentation (this file)
├── requirements.txt      # Python dependencies
├── model.py             # Definition of the CNN architecture
└── main.py              # Training and evaluation script
```

## Extending the project

This project is intentionally modular to encourage experimentation.  Here are a few ideas for extensions:

* **Data augmentation:** Improve generalization by adding random cropping, flips and color jitter to the training transforms.
* **Hyperparameter tuning:** Adjust the number of epochs, batch size, optimizer (e.g. Adam, SGD with momentum) or learning rate schedule.
* **Deeper architectures:** Replace the provided CNN with a deeper network (e.g. ResNet or DenseNet) to achieve higher accuracy.
* **Visualization:** Plot training and validation curves using Matplotlib or TensorBoard for better insight into the learning process.
* **Transfer learning:** Fine-tune a pre-trained model from the torchvision model zoo on CIFAR-10.

Feel free to fork this repository and build upon it for your own research or coursework.  Pull requests are welcome!
