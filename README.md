## DeepUSPS: Deep Learning-Empowered Unconstrained-Structural Protein Sequence Design

This is a kind of model to design unconstrained-structural protein sequences
![img](https://github.com/mazhichong/MZC/assets/91598973/1124ed01-bf27-4ef8-9352-7b9d20515569)
The architecture of the unconstrained-structural protein sequence design model-DeepUSPS

## Requirements
```
# Our version of Cuda is 11.1, version of cuDNN is 8.4.1, and operation system is Ubuntu(20.04)
# Once this is done, you can run the following commands to install the required environment:
conda env create -f environment.yml
```

## Usage
```
usage: hallucinate.py [-h] [-l LEN] [-s SEQ] [-o FAS] [--ocsv= CSV]
                      [--SPFESN= SPDIR] [--RTIDR= RTDIR] 
                      [--aa_weight= AA_WEIGHT]

optional arguments:
  -h, --help              show this help message and exit
  -l LEN, --len= LEN      sequence length (default: 100)
  -s SEQ, --seq= SEQ      starting sequence (default: )
  -o FAS, --ofas= FAS     save final sequence to a FASTA files (default: )
  --ocsv= CSV             save trajectory to a CSV files (default: )
  --SPFESN= SPDIR         path to SPFESN network weights (default: ../SPSP)
  --RTIDR= RTDIR          path to RTIDR network weights (default: ../RTRT)
  --aa_weight= AA_WEIGHT  weight for the aa composition biasing loss term (default: 0.0)
```

#### DeepUSPS a random protein of length 200
```
python ./DeepUSPS.py -l 200 -o seq.fa
```

#### DeepUSPS starting from a given sequence and save trajectory to a CSV files
```
python ./DeepUSPS.py \
	-s KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL  \  # PDB ID 6LYZ
	--aa_weight=1.0 \              # turn on amino acid composition biasing term
	--ocsv= seq.csv
```
## Acknowledgement
We refer to the paper of [De novo protein design by deep network hallucination](https://doi.org/10.1038/s41586-021-04184-w). We are grateful for the previous work of I Anishchenko, TM Chidyausiku, S Ovchinnikov, SJ Pellock, D Baker.

