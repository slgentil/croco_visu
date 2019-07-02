# Python visualization tools for history files from CROCO


## Install Miniconda:

Download Miniconda3 (i.e. for python3) from the [conda website](https://conda.io/miniconda.html)
Install Miniconda3 (See documentation on web site)

Put in your .cshrc
```
source path_to_miniconda/etc/profile.d/conda.csh
```

## Installation environnement croco_visu
```
conda update conda
conda create -n croco_visu -c conda-forge python=3.7 wxpython xarray matplotlib netcdf4 scipy ffmpeg
```


## Clone croco_visu from the git repository:
```
git clone https://github.com/slgentil/croco_visu.git
cd croco_visu
```

## Lancement visualisation
```
conda activate croco_visu
python croco_gui_xarray.py
```