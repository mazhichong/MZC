## DeepUSPS: Deep Learning-Empowered Unconstrained-Structural Protein Sequence Design
This is a kind of model to design unconstrained-structural protein sequences
![img](https://github.com/mazhichong/MZC/assets/91598973/882bfd3a-1d04-40b7-87cb-71d7f3f9da70)
The architecture of the unconstrained-structural protein sequence design model-DeepUSPS

## Requirements
```
absl-py==1.0.0
astor==0.8.1
astunparse==1.6.3
biopython==1.78
blast==0.2.1
blosc==1.5.1
brotlipy==0.7.0
cached-property==1.5.2
cachetools==4.2.4
certifi==2021.10.8
cffi==1.14.5
chardet==4.0.0 
charset-normalizer==2.0.10
click==8.0.4
coverage==6.2
cryptography==3.4.7 
cycler==0.10.0
dataclasses==0.8 
decorator==4.4.2
deepdish==0.3.7
dgl-cu110==0.6.1
einops==0.4.1
Flask==2.0.3
flatbuffers==1.12
gast==0.3.3 
google-auth==2.3.3
google-auth-oauthlib==0.4.6
google-pasta==0.2.0 
googledrivedownloader==0.4
grpcio==1.43.0
h5py==3.1.0
idna==3.3
iminuit==2.16.0
importlib-metadata==4.8.3
ipython-genutils==0.2.0
isodate==0.6.1
itsdangerous==2.0.1
Jinja2==3.0.1 
joblib==1.1.1
Keras==2.3.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
lie-learn==0.0.1.post1
Markdown==3.3.6
MarkupSafe==2.0.1
matplotlib==3.2.2
mkl-fft==1.0.12
mkl-random==1.1.1
networkx==2.5.1
numexpr==2.8.1
numpy==1.19.5
oauthlib==3.1.1
olefile==0.46
opt-einsum==3.3.0
packaging==20.9
pandas==0.20.3
Pillow==8.2.0
protobuf==3.19.3
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycparser==2.20
pydal==20200714.1
pyOpenSSL==20.0.1
pyparsing==2.4.7
pyrosetta==2020.10+release.46415fa
PySocks==1.7.1
python-dateutil==2.6.1
python-louvain==0.15
pytm==1.2.1
pytz==2017.2
PyYAML==3.12
rdflib==5.0.0
requests==2.27.1
requests-oauthlib==1.3.0
rsa==4.8
scikit-learn==0.24.2
scipy==1.2.1
six==1.16.0
tables==3.7.0
tensorboard==1.14.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==1.14.0
tensorflow-estimator==1.14.0
termcolor==1.1.0
threadpoolctl==3.1.0
timm==0.3.2
tmscore==0.0.1
tmscoring==0.4.post0
torch==1.9.0+cu111
torch-cluster==1.5.9
torch-geometric==1.7.2
torch-scatter==2.0.7
torch-sparse==0.6.10
torch-spline-conv==1.2.1
torchaudio==0.9.0
torchvision==0.10.0+cu111
tornado==4.5.2
tqdm==4.61.1
traitlets==4.3.2
typing-extensions=3.7.4.3
urllib3==1.26.8
utils==1.0.1
Werkzeug==2.0.2
wrapt==1.13.3
yacs==0.1.8
zipp==3.6.0
```

## Download and installation
```
# download package
git clone --recursive https://github.com/gjoni/trDesign
cd DeepUSPS

# download RTRT network weights
wget https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2

# download SPSP network weights
wget https://files.ipd.uw.edu/pub/trRosetta/bkgr2019_05.tar.bz2
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

