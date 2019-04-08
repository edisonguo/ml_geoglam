set -xeu

h=$1
v=$2
model_path=$3
output_dir=$4

(set -xeu
cd geoglam

year_bgn=2019
year_end=2019
checkpoint=fc_prod
python_bin=/local/anaconda2-python2.7/bin/python

bash pipeline.sh $h $v $year_bgn $year_end $checkpoint $output_dir $python_bin $model_path

)
