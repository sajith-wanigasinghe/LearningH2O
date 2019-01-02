import sys

import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# initiating H2O server instance
h2o.init()

# import data
file_path = "data/loan.csv"
data = h2o.import_file(file_path)

# encode the binary repsonse as a factor
data['bad_loan'] = data['bad_loan'].asfactor()  

# after encoding, this shows the two factor levels, '0' and '1'
data['bad_loan'].levels()  

# Partition data into 70%, 15%, 15% chunks
data_partitions = data.split_frame(ratios=[0.7, 0.15], seed=1)
train_data = data_partitions[0]
validation_data = data_partitions[1]
test_data = data_partitions[2]

# testing the splits
# print (train_data.nrow)
# print (validation_data.nrow)
# print (test_data.nrow)

y = 'bad_loan'
x = list(data.columns)

# remove the label(target) column from the dataset
x.remove(y)
# remove int_rate because if interest rate is 0 it indicates a bad loan
x.remove('int_rate')
# display the columns after the alteration
# print(x)

# GBM hyperparameters
gbm_params1 = {
        'learn_rate': [0.01, 0.1], 
        'max_depth': [3, 5, 9],
        'sample_rate': [0.8, 1.0],
        'col_sample_rate': [0.2, 0.5, 1.0]
    }

# Train and validate a grid of GBMs
gbm_grid1 = H2OGridSearch(
        model=H2OGradientBoostingEstimator,
        grid_id='gbm_grid1',
        hyper_params=gbm_params1
    )

gbm_grid1.train(
        x=x, 
        y=y,
        training_frame=train_data,
        validation_frame=validation_data,
        ntrees=100,
        seed=1
    )

gbm_gridperf = gbm_grid1.get_grid(sort_by='auc', decreasing=True)
print(gbm_gridperf)
