# AbtCDR

## Environment Requirement

The code has been tested running under Python 3.8. The required packages are as follows:

scikit-learn==1.3.1
scipy==1.10.1
numpy==1.24.4
tqdm==4.66.1
torch==1.10.2

## Training
The instruction of commands has been clearly stated in the codes (see the parser function in utility/parser.py).

##### Random sample(rns)

```
python run.py --batch_size 1024 --gpu_id 0 --dataset=sport_cell --lr=0.0005 --n_interaction=5 --lambda_s=0.8 --lambda_t=0.8 
```
