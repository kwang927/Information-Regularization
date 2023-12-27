# Generation
The generated quereis for each regularization are provided in the pickle files, in the form `{dataset_name}_{type}.pickle`. The prompt used to generate queries can be found in `prompt\`, in the form `{dataset_name}_{type}_prompt.pickle`.

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

The files contain dictionary of the following keys:
- `id`: the query id
- `query`: the synthetically generated query
- `breakdown`: the query after Query regularization. This key is only contained in Qreg files.
