# BELHD: Improving Biomedical Entity Linking with Homonym Disambiguation

Code to reproduce experiments in:

```
@article{BelhdImprovinGarda2024,
  archiveprefix = {arXiv},
  author = {Garda, Samuele and Leser, Ulf},
  eprint = {2401.05125v1},
  month = {Jan},
  primaryclass = {cs.CL},
  title = {BELHD: Improving Biomedical Entity Linking with Homonoym Disambiguation},
  url = {http://arxiv.org/abs/2401.05125v1},
  year = {2024},
}
```

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=2 -->

- [Setup](#setup)
- [Results](#results)
  - [BELB](#belb)
  - [BioRED](#biored)
- [Run](#run)
  - [Homonym Disambiguation](#homonym-disambiguation)
  - [BELHD](#belhd)
  - [BELHD Ablations](#belhd-ablations)
  - [Ad-hoc solutions for homonyms](#ad-hoc-solutions-for-homonyms)
  - [Baselines](#baselines)

<!-- mdformat-toc end -->

## Setup<a name="setup"></a>

Install the [belb](https://github.com/sg-wbi/belb) library in your python environment:

```bash
git clone https://github.com/sg-wbi/belb
cd belb
pip install -e .
```

Then you need to install other requirements specific for BELHD:

```bash
(belhd) user $ pip install -r requirements.txt
```

## Results<a name="results"></a>

We stored predictions of all models and gold labels in the `data` directory.
Below you find the commands to reproduce all tables reported in the paper.

### BELB<a name="belb"></a>

Reproduce the results on BELB.

Main table:

```bash
(belhd) user $ python -m scripts.evaluate 
```

BELHD ablations:

```bash
(belhd) user $ python -m scripts.evaluate_ablations
```

Ad-hoc solutions for homonyms. Abbreviations:

```bash
(belhd) user $ python -m scripts.evaluate_ar
```

and species assignment:

```bash
(belhd) user $ python -m scripts.evaluate_sa 
```

### BioRED<a name="biored"></a>

```bash
(belhd) user $ python -m biored.evaluate 
```

## Run<a name="run"></a>

If you wish to use our code with BELB you first need to follow the [`belb`](https://github.com/sg-wbi/belb/tree/main)
instructions to setup a directory with all the data (corpora and KBs).

### Homonym Disambiguation<a name="homonym-disambiguation"></a>

To create KB versions with disambiguated homonyms:

```bash
(belhd) user $ python -m scripts.disambiguate_kbs --dir /path/to/belb/dir
```

We note that `belb` deals with large KBs and its code it's not optimized.
This step takes quite a while, especially for NCBI Gene.

### BELHD<a name="belhd"></a>

To train BELHD you need to convert BELB data into the required input format

Edit `data/configs/data.yaml`:

```yaml
belb_dir : 'path/to/belb/directory'
exp_dir : 'path/to/experiments/directory'
```

Prepare data with:

```bash
(belhd) user $ python -m scripts.tokenize_corpora
```

and

```bash
(belhd) user $ python -m scripts.tokenize_dkbs
```

Then you can use the helpers scripts `bin/train.sh` to train the models and `bin/predict.sh` to obtain the predictions
for each corpus.

### BELHD Ablations<a name="belhd-ablations"></a>

Run scripts `bin/train_ablations.sh` and `bin/predict_ablations.sh`

### Ad-hoc solutions for homonyms<a name="ad-hoc-solutions-for-homonyms"></a>

You need to first train BELHD without HD and with abbreviation resolution (`bin/train_nohd.sh`) and obtain the predictions (`bin/predict_nohd.sh`).
For this you need to create a version of the data with abbreviation resolution with:

```bash
(belhd) user $ python -m scripts.tokenize_corpora abbres=true
```

Similarly you need to rerun the [baselines](#Baselines) with abbreviation resolution.
Gene corpora with species assignment are stored in `./data/belb/species_assign`
(see [SpeciesAssignment.md](./data/docs/SpeciesAssignment.md) for details).

### Baselines<a name="baselines"></a>

For each baseline we use the original code. We provide detailed instruction on how to run them in separate files:

- BioSyn: [./baselines/biosyn/README.md](./baselines/biosyn/README.md)
- GenBioEL: [./baselines/genbioel/README.md](./baselines/genbioel/README.md)
- arboEL: [./baselines/arboel/README.md](./baselines/arboel/README.md)
