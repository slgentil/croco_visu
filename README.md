# Visualization tools for CROCO


## Install Miniconda:

Download Miniconda3 (i.e. for python3) from the [conda website](https://conda.io/miniconda.html)
Install Miniconda3 (See documentation on web site)

Put in your .cshrc
```
source /Users/slgentil/.miniconda2/etc/profile.d/conda.csh
```

## Installation environnement croco_visu
```
conda update conda
conda create -n croco_visu -c conda-forge python=2.7 wxpython xarray matplotlib netcdf4 scipy
conda activate croco_visu
```


## Clone croco_visu from the git repository:
```
git clone https://github.com/slgentil/croco_visu.git
```

## Lancement visualisation
```
python croco_gui.py
```