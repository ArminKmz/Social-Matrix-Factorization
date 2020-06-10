# Social-Matrix-Factorization

Simple model for exploiting scoial networks data in recommender systems.

## Model


- <img src="https://latex.codecogs.com/svg.latex?R_{ij}"/>: Score rated by user <img src="https://latex.codecogs.com/svg.latex?i"/> to item <img src="https://latex.codecogs.com/svg.latex?j"/>
<br></br>
- <img src="https://latex.codecogs.com/svg.latex?U_i"/>: Hidden feature vector for user <img src="https://latex.codecogs.com/svg.latex?i"/>
<br></br>
- <img src="https://latex.codecogs.com/svg.latex?V_j"/>: Hidden feature vector for item <img src="https://latex.codecogs.com/svg.latex?j"/>
<br></br>
- <img src="https://latex.codecogs.com/svg.latex?W_{ii'}"/>: Hidden weight for user <img src="https://latex.codecogs.com/svg.latex?i'"/> impact on user <img src="https://latex.codecogs.com/svg.latex?i"/>



<p align="center">
  <img width="392" alt="SMF Model" src="https://user-images.githubusercontent.com/7484808/42945818-9e2b336e-8b7e-11e8-8664-7036597b9d1a.png"> 
  <br></br>
  <img width="296" alt="Model Distributions" src="https://user-images.githubusercontent.com/7484808/42946309-ceafe240-8b7f-11e8-8e44-42e19ee94e39.png">
</p>
  
<br></br>
<img src="https://latex.codecogs.com/svg.latex?R_{ij}"/> is estimated from predictive distribution approximated by varational infrence.

## Useage

To train model set hyperparameters and use ```.train()``` method. For load the model and predict scores from data initialize the model with ``` network=network, load=True``` and use ``` .predict(data) ``` method.
<br></br>
Take a look at ```main.py``` for train and load model.
<br></br>
To speed up you can limit each user outdegree with setting ```LIMIT``` in ```main.py```. Data format is available in ```dataset/readme.txt```

## Example

Output of ```main.py```:

<p align="center">
<img width="351" alt="train-rmse" src="https://user-images.githubusercontent.com/7484808/42948646-3118b0a6-8b85-11e8-9436-6d0bae95f6c4.png">
</p>

```
iteration 0  RMSE: 1.1623740799051603
iteration 1  RMSE: 0.974779153572551
iteration 2  RMSE: 0.9061886948956133
iteration 3  RMSE: 0.8767977965409327
iteration 4  RMSE: 0.8624643916488482
iteration 5  RMSE: 0.85406229979618
iteration 6  RMSE: 0.8484390298439565
iteration 7  RMSE: 0.8441520641876232
iteration 8  RMSE: 0.8405251123950408
iteration 9  RMSE: 0.8375187875896802
iteration 10  RMSE: 0.8351068584344935
RMSE on validation data: 1.022167855935175
```


