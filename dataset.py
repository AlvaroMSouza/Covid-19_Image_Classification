import kaggle 

kaggle.api.authenticate()

kaggle.api.dataset_download_files('khoongweihao/covid19-xray-dataset-train-test-sets', path='.', unzip=True)


