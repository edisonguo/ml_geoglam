# 1. Background

The vegetation fractional coverage provided by the [Geoglam](https://github.com/nci/geoglam) project is computed by solving a non-negative least squares (NNLS) problem given a 84-dimensional vector for each pixel of each tile of the MODIS dataset.

Observing that taking the 84-dimensional vectors as the feature vectors and the measured fractional coverage as targets, we can re-interpret the NNLS problem into a machine learning setup. In other words, our objective to find a function that approximates the what the NNLS outputs, where the approximator can be any family of supervised machine learning models. 

# 2. Methods

Despite the large number of supervised machine learning methods, we choose random forest for this project. The reason is two folds:
1. From training methodological point of view, random forest does not have many hyper-parameter to tune and is robust against overfitting. It also has proven track record of top results in various ML contents.
2. From system engineering point of view, the training and inference of random forest can easily be parallelized to take advantage of many cores.

Having said that, random forest is never said to be the ultimate method. We present it here as the starting point to investigate machine learning methods applicable to Geoglam fractional cover products.

### 2.1 Training algorithm

Given the built-in parallelism offered by random forest, we implemented a distributed training algorithm as follows:

```
rf_models = []
for t in training MODIS tiles
   let D = load time series of MODIS and Geoglam fractional cover images for t 
   for b in Geoglam bands (i.e. pv, npv, bare_soil)
       let rf = train a new random forest of n trees over D
       rf_models[b] = concatenate(rf_models[b], rf)
```
As the two `for` loops are independent, it is easily to distribute the random forest training across different cores and `concatenate` the trained models later.

##### Notes

1. We train different random forests for each Geoglam bands because we observed that the data distribution for different bands are quite different. We initially trained a single random forest for all three bands jointly, but the results were not good.

2. It is possible to concatenate all the MODIS tiles to form a single training dataset. However, we have out-of-memory issues as random forest is batch method that requires loading all training data into memory. Thus we divide the training set on a per tile basis and still manage to get good results and parallelism.

### 2.2 Inference algorithm

Inference is straight-forward as follows:

```
for t in test MODIS tiles
    for ts in timestamps in t
       let D = load MODIS image for ts
       for b in Geoglam bands (i.e. pv, npv, bare_soil)
           inference over D using rf_models[b]
```
The independence of the loops also suggest possibility of distributed inference.

# 3. Experiments

We use one year worth of six MODIS tiles as the training set. The six MODIS tiles are hand picked to cover the the areas of Australia that have good balanced distribution of `pv`, `npv` and `bare_soil` such as east coast, Perth, and so forth. This hand picking procedure for balanced training data distribution is important to achieve good results. The reason is that many areas of Australia have highly skewed distribution. For example, the tiles corresponding to the center of the country barely have any plant coverage. If our training data consist of those skewed tiles, it would be virtually impossible for the model to learn meaningful representations. 

The year we use for building the training set is 2018 and we use year 2019 up to March as the test set. 

The loss function is mean squared error. The maximum depth of each tree is 12 levels. Each round of forest training consists of 16 trees. Thus the final random forest has 16 * 6 (tiles) = 96 trees. We have three 96-treed forests one for each Geoglam bands.

The reason we train 16 trees for each round because the servers have 16 cores each.  Thus it is convenient to parallelize the forest training of 16 trees across 16 cores.

# 4. Results

We use R squared as the scoring function to measure model performance. 

##### 4.1 Training mean squared errors using 2018 data

tile   | pv     | npv | bare_soil
---    | ---    | --- | ---
h28v11 | 0.9939 | 0.9652 | 0.9888
h29v12 | 0.9941 | 0.9725 | 0.9900
h30v10 | 0.9948 | 0.9615 | 0.9901
h30v12 | 0.9967 | 0.9706 | 0.9943
h31v10 | 0.9948 | 0.9611 | 0.9778
h31v11 | 0.9970 | 0.9683 | 0.9916

##### 4.2 Test mean squared errors using 2019 data up to March

tile   | pv     | npv | bare_soil
---    | ---    | --- | ---
h28v11 | 0.9939 | 0.9652 | 0.9888
h29v12 | 0.9941 | 0.9725 | 0.9900
h30v10 | 0.9948 | 0.9615 | 0.9901
h30v12 | 0.9967 | 0.9706 | 0.9943
h31v10 | 0.9948 | 0.9611 | 0.9778
h31v11 | 0.9970 | 0.9683 | 0.9916
h27v11 | 0.9704 | 0.8479 | 0.9107
h27v12 | 0.9930 | 0.9173 | 0.9508
h28v10 | 0.0000 | 0.0000 | 0.0000
h28v11 | 0.9591 | 0.9338 | 0.9604
h28v12 | 0.9892 | 0.9336 | 0.9833
h29v10 | 0.9914 | 0.9324 | 0.9893
h29v11 | 0.9396 | 0.9281 | 0.9327
h29v12 | 0.9961 | 0.9717 | 0.9922
h30v10 | 0.9966 | 0.9620 | 0.9922
h30v11 | 0.9603 | 0.8670 | 0.9379
h30v12 | 0.9975 | 0.9645 | 0.9927
h31v10 | 0.9935 | 0.9606 | 0.9787
h31v11 | 0.9912 | 0.9590 | 0.9795
h31v12 | 0.9873 | 0.9737 | 0.8647

##### Observations

1. Both training and test achieve good R squared. This suggests both good fit of the training data and small generalization gap of the models.

2. The test tile `h28v10` has zero R squared. This is because this particular tile only contains one single valid pixel. 

# 5. Discussions and Future Work

We managed to achieve good results in computing fractional coverage using random forest for the Geoglam project. This proof of concept indicates feasibility of using machine learning methods for fractional coverage in addition to the traditional statistical methods. The random forest model here is never meant to be the ultimate model for fractional cover computation but provides a strong baseline to begin further research with. Possible future directions of exploration are as follows:

1. The raw MODIS data only have 7 bands per pixel. The rest of the 84 dimensions are as a result of feature engineering to build non-linear features. It is worth trying to apply deep learning methods directly over the raw 7 bands. This will dramatically improve feature computing speed while may maintain good accuracy.

2. The 84-d feature vector is computed for each pixel individually. But it is sensible that if the neighborhood of pixel has good coverage, the pixel in question should have good coverage too. Thus it is worth taking the neighborhood for each pixel into the account to compute the features. In this case, it is worth applying convolutional neural networks (CNN) to exploit the pixel neighborhood as well as hierarchical feature representations.
