# Industrial_IoT_for_Gear_Bearing_Failure
This is the project I did when I was interned as an Internet of Things (IoT) Developer at Sunflower Lab in Columbus, Ohio. I was involved in developing digital representations of physical objects for an industrial IoT application using ThingWorx (a cloud-based platform), and further data analysis using Python and its main libraries. 

Probably you have heard the buzzword Industry 4.0, where the objective is to make use of data to increase productivity while save time and cost. Therefore, in this project, I was focusing on building a demo to our customers to demonstrate how to make good use of the vast amounts of available data and extract useful information for anomaly detection and condition monitoring (taking gear bearing failure as an example) to make it possible to reduce costs and keep downtime to a minimum.

**Keywords**: Industry 4.0, anomaly detection, condition monitoring, principal component analysis, Mahalanobis distance metric, multivariate statistical analysis

## Outline
1. Understanding and Loading Data
2. Data Cleaning and Pre-Processing
3. Model Setup
4. Distance Metric
5. Anomaly Detection
6. Summary

## Understanding and Loading Data
In the demo project, we employed the dataset of experiments on bearings to detect gear bearing degradation on an engine, and give a warning that allows for predictive measures to be taken in order to avoid a gear failure. For the dataset, the citation and download link is as follows. You can also check the downloaded *Readme Document for IMS Bearing Data* in this repository for further information on the experiment and available data.

* J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/), NASA Ames Research Center, Moffett Field, CA.

Let's first import the packages we need for this project.
```
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes = True)
import matplotlib.pyplot as plt
%matplotlib inline
```

Then let's import the datasets and merge them together to form a single dataframe. We can then sort the data by index in chronological order and save it as a .csv file. To have an idea of the dataframe, we can check the first five rows by default. As stated in the *Readme Document for IMS Bearing Data*, four different channels were employed to measure four bearings at every 10 minutes. At the end of the test-to-failure experiment, outer race failure occurred in Bearing 1.
```
data_dir = '2nd_test'
total_data = pd.DataFrame()

for filename in os.listdir(data_dir):
    dataset = pd.read_csv(os.path.join(data_dir, filename), sep = '\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
    dataset_mean_abs.index = [filename]
    total_data = total_data.append(dataset_mean_abs)

total_data.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']

total_data.index = pd.to_datetime(total_data.index, format = '%Y.%m.%d.%H.%M.%S')
total_data = total_data.sort_index()
total_data.to_csv('total_dataset_BearingTest_2.csv')
total_data.head()
```
<img src="Figures/Gear Bearing Dataset.png">

## Data Cleaning and Pre-Processing
Before setting up the models, we need to split the dataframe into training data and test data. The total duration of data lasts for about one week. Thus, the training data is selected to be the first half week which represents the normal operating conditions. The rest half week containing final bearing failure is considered as the test data. We can plot the training data as below.
```
dataset_train = total_data['2004-02-12 10:32:39':'2004-02-16 01:02:39']
dataset_test = total_data['2004-02-16 01:02:39':]
dataset_train.plot(figsize = (12,6), color = ['r', 'g', 'b', 'k'])
```
<img src="Figures/Training Data.png">

In the meantime, we need to normalize the data to ensure that the input variables of the model are within the same scale.
```
scaler = preprocessing.MinMaxScaler()
training_data = pd.DataFrame(scaler.fit_transform(dataset_train), columns = dataset_train.columns, index = dataset_train.index)
# Random shuffle training data
training_data.sample(frac=1)
test_data = pd.DataFrame(scaler.transform(dataset_test), columns = dataset_test.columns, index = dataset_test.index)
```

## Model Setup
It is often quite challenging to accurately generalize datasets that have high dimensional features, which is well-known as the *curse of dimensionality*. To address this issue, there are several techniques for dimensionality reduction, and principal component analysis (PCA) is one of the most widely used techniques. For this project, let's compress the initial sensor readings into the two main principal components, where we can still keep at least 90% variance of the orignial datasets.
```
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, svd_solver = 'full')
training_data_PCA = pca.fit_transform(training_data)
training_data_PCA = pd.DataFrame(training_data_PCA)
training_data_PCA.index = training_data.index

test_data_PCA = pca.transform(test_data)
test_data_PCA = pd.DataFrame(test_data_PCA)
test_data_PCA.index = test_data.index
```

## Distance Metric
In the problem of estimating the probability of whether a data point belongs to a distribution or not, a common metric is to calculate the distance between that point and the centroid or center of mass of the sample points. Intuitively, the closer the point in question is to this center of mass, the more likely it is to belong to the set. Often the distribution of the sample points are not ideally spherical in 2D, but rather ellipsoidal. In such case, *Mahalanobis Distance* is used to calculate the distance of the test point from the center of mass divided by the width of the ellipsoid in the direction of the test point.

Mathematically, we need to first calculate the covariance matrix, and then compute the Mahalanobis distance as follows. We will then be able to detect outliers and calculate threshold value for classifying datapoint as anomaly.
```
def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    return covariance_matrix, inv_covariance_matrix

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md
```

## Anomaly Detection
Putting all these together, we can set up the model by first computing the covariance matrix and its inverse. Then we can calculate the Mahalanobis distance for the training data defined as 'normal conditions', and find the threshold value to flag datapoints as an anomaly.
```
data_train = np.array(training_data_PCA.values)
data_test = np.array(test_data_PCA.values)

cov_matrix, inv_cov_matrix  = cov_matrix(data_train)

mean_distr = data_train.mean(axis=0)

dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose = False)
dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose = False)
```

We can visualize the Mahalanobis distance itself with kernel density estimation plotted as well. Based on the plot, it seems reasonable that we use a threshold value about 3.8 for flagging an anomaly, which is three standard deviations from the center of the distribution.
```
plt.figure()
sns.distplot(dist_train, bins = 10, kde = True);
plt.xlim([0, 5])
plt.xlabel('Mahalanobis dist')
```
<img src="Figures/Mahalanobis Distance.png">

Now we can save the Mahalanobis distance, the threshold value and 'anomaly flag' variables for both train and test data in a dataframe. We can further plot the calculated anomaly metric using Mahalanobis distance, and check when it crosses the anomaly threshold at about 3.8 as shown below.
```
anomaly_train = pd.DataFrame()
anomaly_train['Mahalanobis distance'] = dist_train
anomaly_train['Threshold'] = threshold
# If Mob dist above threshold: Flag as anomaly
anomaly_train['Anomaly'] = anomaly_train['Mahalanobis distance'] > anomaly_train['Threshold']
anomaly_train.index = training_data_PCA.index
anomaly = pd.DataFrame()
anomaly['Mahalanobis distance'] = dist_test
anomaly['Threshold'] = threshold
# If Mob dist above threshold: Flag as anomaly
anomaly['Anomaly'] = anomaly['Mahalanobis distance'] > anomaly['Threshold']
anomaly.index = test_data_PCA.index
anomaly_alldata = pd.concat([anomaly_train, anomaly])
anomaly_alldata.plot(logy = True, figsize = (10,6), ylim = [1e-1,1e3], color = ['b','r'])
```
<img src="Figures/Test Data Validation.png">

It can be noticed from the above plot that the model we employed here is able to detect the anomaly almost three days ahead of the actual bearing failure. In this way, the engineers can be alerted to schedule a maintanence accordingly to avoid unprepared failures which may cause long downtime and extra losses.

## Summary
In this project, I present a demo about implementing principal component analysis and multivariate statistical analysis for anomaly detection and condition monitoring for industrial IoT application. Specifically, we are able to detect gear bearing degradation on an engine, and give a warning that allows for predictive measures to be taken in order to avoid a gear failure.
In this way, we can reduce costs, increase efficiency and keep downtime to a minimum. 

If you have further questions or comments, please email: luyan461@gmail.com.
