# SpeciesAssignment

To reproduce the results with [SpeciesAssignment](https://github.com/ncbi/SpeciesAssignment) you need first to
convert the BELB gene corpora into the [PubTator format](https://bioc.readthedocs.io/en/latest/pubtator.html).

## Prepare input

```bash
python -m scripts.convert_belb_to_pubtator --run input --belb_dir ~/data/belb
```

## Species NER

Download and extract the **standalone** of [GNorm2](https://github.com/ncbi/GNorm2)
Move the files in `data/species_assignment/text` into the GNorm2 directory ( e.g. `belb_input`).
The to recognize the species mentioned run:

```bash
java -Xmx60G -Xms30G -jar GNormPlus.jar belb_input belb_input_SR setup.SR.txt
```

Move the folder `belb_input_SR` into `data/species_assignment/text_species`

Now we are going to add gene annotations to the files:

```bash
python -m scripts.convert_belb_to_pubtator --run append --belb_dir ~/data/belb
```

## Assign Species

Download and extract  [SpeciesAssignment](https://github.com/ncbi/SpeciesAssignment).

Please follow their instruction on how to setup the environment to run the tool.

Then move the folder `data/species_assignment/text_species_gene` into `belb_input` in the SpeciesAssignment folder.

Then run:

```bash
cd  src
python Species_Assignment.py -i ../belb_input/gnormplus_test.PubTator -m ../speass_trained_models/SpeAss-PubmedBERT-SG.h5 -o ../results_gnormplus_test
python Species_Assignment.py -i ../belb_input/nlm_gene_test.PubTator -m ../speass_trained_models/SpeAss-PubmedBERT-SG.h5 -o ../results_nlm_gene_test
```

And then move the results in `data/species_assignment/text_species_gene_assign`
