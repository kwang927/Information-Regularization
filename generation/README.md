# Generation
The generated query for each regularization are provided in the pickle file.

## Table of Contents
* [Format](#file-format)

## File Format
The files are in form `{dataset_name}_{type}.pickle`.
The types include:
- `Ireg`: Instruction regularization
    - `Qreg_Ireg`: Query regularization on Instruction regularization
- `Dreg_{p}%`: Document p% regularization
    - `Qreg_Dreg_{p}%`: Query regularization on Document p% regularization
- `promptagator`: [Promptagator style](https://iclr.cc/virtual/2023/poster/10937)
