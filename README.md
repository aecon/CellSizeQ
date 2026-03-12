# Cell and Nucleus size quantification

Repurposing of images from high-content arrayed CRISPR screens, for quantification of cell/nuclei morphology changes.


## Installation

Clone the repository:
```
git clone https://github.com/aecon/CellSizeQ.git
```

Create a new conda environment with dependencies:

```
cd CellSizeQ
conda env create -f environment.yml
```


Then compile the C file:
```
cd size
make
```


## Usage

```
cd size
conda activate CellSizeQ
./process_all.sh  <path/to/directory/containing/HA_xx>
```

