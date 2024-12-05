# SpaDiT

## Setup

```bash
conda create -n diffusers python=3.10
conda activate diffusers
# set current directory to the root of the project
ODIR=$(pwd)
pip install -e .
cd examples/dreambooth/
pip install -r requirements.txt
pip install -r requirements_sd3.txt
accelerate config default
cd $ODIR
cd zlab
pip install -r requirements.txt
cd dreambooth
```

Run the notebook `load_data.ipynb` to download the data.

```bash
bash train_dreambooth_sd3.sh
```
