config={
    'DATASET_NAME':'credit_approval', # Please follow paper and use 'X' as dataset name, where X can be ='credit','dress', etc. As an example here X='credit_approval' is provided.
    'SEED':15, # variations in machine configurations can affect distributions
    'BATCH':100,
    'LR':0.0001,
    'EPOCH':1000,
    'REPRESENTATION_LAYER':32,
    'ssl_epochs':100,
    'ssl_corruption':0.5,
    'ssl':False, 
    'device':'cuda'
}