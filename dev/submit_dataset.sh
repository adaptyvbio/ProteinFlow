proteinflow unsplit --tag $1
mv data/proteinflow_$1/splits_dict data/
aws s3 sync data/proteinflow_$1 s3://proteinflow-datasets/$1/proteinflow_$1
aws s3 sync data/splits_dict s3://proteinflow-datasets/$1/proteinflow_{$1}_splits_dict
cd data && zip -r proteinflow_{$1}.zip proteinflow_$1 
aws s3 cp proteinflow_{$1}.zip s3://proteinflow-datasets/$1
mv splits_dict proteinflow_$1 && cd ..
proteinflow split --tag $1