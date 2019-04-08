set -eu

function visualise_mosaic() {
    local year=$1
    local band_idx=$2
    local output_file=$3
    local tile_file=$4
    local data_dir=$5

    data_files=$(awk -F, -v data_dir=$data_dir -v year=$year '{print data_dir"/FC.v310.MCD43A4.h"$1"v"$2"."year".006.nc"}' $tile_file)

    gdalbuildvrt -b $band_idx -sd 1 phot_veg.vrt $data_files
    gdalbuildvrt -b $band_idx -sd 2 nphot_veg.vrt $data_files
    gdalbuildvrt -b $band_idx -sd 3 bare_soil.vrt $data_files

    gdalbuildvrt -separate tiles.vrt bare_soil.vrt phot_veg.vrt nphot_veg.vrt

    gdalwarp --config GDAL_CACHEMAX 1024 -wm 1024 -multi -of GTiff -overwrite -ts 512 512 -te 106 -45 160 -2 -t_srs epsg:4326 tiles.vrt $output_file.warp.tif 
    gdal_translate -scale $output_file.warp.tif $output_file.tif 
}

gdalinfo --version|grep 'GDAL 2.4' || (
echo 'GDAL v2.4.0+ is required'
exit 1
)

output_dir=outputs
gt_dir=$output_dir/ground_truth
pred_dir=$output_dir/predicted

mkdir -p $gt_dir
mkdir -p $pred_dir

year=2019
data_dir=/local/geoglam_outputs/v310/tiles/8-day/cover
tile_file=../geoglam/tile_files/australian_tiles.csv 

for band_idx in {1..11}
do
    base_output_file=${year}_${band_idx}

    visualise_mosaic $year $band_idx $gt_dir/gt_${base_output_file} $tile_file /g/data2/tc43/modis-fc/v310/tiles/8-day/cover
    visualise_mosaic $year $band_idx $pred_dir/pred_${base_output_file} $tile_file $data_dir

    rm -f $gt_dir/*.warp.tif $pred_dir/*.warp.tif *.vrt
done



