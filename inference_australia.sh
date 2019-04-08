set -eu

exec &> >(tee australia_inference.log)
export PYTHONUNBUFFERED=1

tile_file=geoglam/tile_files/australian_tiles.csv
model_path=`pwd`/dist_forest_time_tile_full_84_features
output_dir=/local/geoglam_outputs

for tile in $(cat $tile_file)
do
    h=$(echo -n $tile|cut -d',' -f1)
    v=$(echo -n $tile|cut -d',' -f2)

    echo "processing: h$h v$v"
    time bash inference_geoglam.sh $h $v $model_path $output_dir
    echo "done processing: h$h v$v"
done
