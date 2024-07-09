
### Requirements

- Python 3.x
- numpy
- scikit-learn
- RDKit
- Tensorflow
- keras
- matplotlib
- transformers
- pandas

### Train(association task)

python CapBChemMoleFusion.py --dti stage1_dataset/ourdataset_association.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024

### Train(up-regulation task)

python CapBChemMoleFusion.py --dti stage2_dataset/ourdataset_up.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024

### Train(down-regulation task)

python CapBChemMoleFusion.py --dti stage2_dataset/ourdataset_down.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024