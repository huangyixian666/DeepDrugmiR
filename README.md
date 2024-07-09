
### Requirements

- Python 3.8
- numpy 1.22.0
- scikit-learn 1.3.0
- RDKit 2023.9.1
- Tensorflow 2.13.1
- keras 2.13.1
- matplotlib 3.7.4
- transformers 4.35.2
- pandas 1.2.1

### Train (Association task)

python CapBChemMoleFusion.py --dti stage1_dataset/ourdataset_association.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024

### Train (Up-regulation task)

python CapBChemMoleFusion.py --dti stage2_dataset/ourdataset_up.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024

### Train (Down-regulation task)

python CapBChemMoleFusion.py --dti stage2_dataset/ourdataset_down.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024
