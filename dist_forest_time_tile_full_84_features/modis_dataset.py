import glob
import os
import numpy as np
from osgeo import gdal
import netCDF4 as nc
import datetime

def input_mask(pq_mask_path):
    #pq_ds = gdal.Open('HDF4_EOS:EOS_GRID:"{}":MOD_Grid_BRDF:BRDF_Albedo_Ancillary'.format(pq_mask_path))
    pq_ds = gdal.Open('HDF4_EOS:EOS_GRID:"{}":MOD_Grid_BRDF:BRDF_Albedo_LandWaterType'.format(pq_mask_path))
    pq_raw = pq_ds.GetRasterBand(1).ReadAsArray()

    mask = 0b0000000000001111
    #pq = pq_raw>>4 & mask
    # in collection 6 I don't need to shift 4 places  
    pq = pq_raw & mask

    snow_ds = gdal.Open('HDF4_EOS:EOS_GRID:"{}":MOD_Grid_BRDF:Snow_BRDF_Albedo'.format(pq_mask_path))
    snow_pq = np.equal(snow_ds.ReadAsArray(), 0)

    # True if pq is 1, 2 or 4    
    pq = np.logical_and(snow_pq, np.logical_or(np.logical_or(np.equal(pq, np.ones(1)), np.equal(pq, np.ones(1)*2)), np.equal(pq, np.ones(1)*4)))
    #pq = np.logical_or(np.logical_or(np.equal(pq, np.ones(1)), np.equal(pq, np.ones(1)*2)), np.equal(pq, np.ones(1)*4))
    
    return pq.reshape(2400*2400)

def input_stack(tile_path):
    bands = [gdal.Open('HDF4_EOS:EOS_GRID:"{}":MOD_Grid_BRDF:Nadir_Reflectance_Band{}'.format(tile_path, b)) for b in range(1, 8)]

    bands_stack = np.empty((2400*2400, 84), dtype=np.float32)

    #add band
    for b in range(0, len(bands)):
        bands_stack[:, b] = np.nan_to_num( bands[b].GetRasterBand(1).ReadAsArray().reshape((2400*2400, ))*.0001 )

    ii = 7
    #add log(band)
    bands_stack[:, ii:ii+7] = np.nan_to_num( np.log(bands_stack[:, :7]) )
    
    #add band*log(band)
    bands_stack[:, ii+7:ii+14] = np.nan_to_num( bands_stack[:, :7] * bands_stack[:, ii:ii+7] )

    ii += 14
    #add band*next_band
    for b in range(7):
        for b2 in range(b+1, 7):
            bands_stack[:, ii] = np.nan_to_num( bands_stack[:, b] * bands_stack[:, b2] )
            ii += 1
    
    #add log(band)*log(next_band)
    for b in range(7):
        for b2 in range(b+1, 7):
            bands_stack[:, ii] = np.nan_to_num( bands_stack[:, b+7] * bands_stack[:, b2+7] )
            ii += 1
    
    #add (next_band-band)/(next_band+band)
    for b in range(7):
        for b2 in range(b+1, 7):
            bands_stack[:, ii] = np.nan_to_num( (bands_stack[:, b2]-bands_stack[:, b]) / (bands_stack[:, b2]+bands_stack[:, b]) )
            ii += 1
    
    #--------------------------------
    # this bit new, by JPG
    #bands_stack[:, -1] = np.ones((bands_stack.shape[0],), dtype=bands_stack.dtype)
    #--------------------------------
    
    return bands_stack

def get_masked_bands(arr, mask):
    res_masks = mask & (np.sum(arr[:, :7] < 0, axis=1) == 0) & (np.sum(arr[:, :7] > 1, axis=1) == 0)
    return arr[res_masks, :], res_masks

def get_x(h, v, year, month, day):
    root_path = "/g/data2/u39/public/data/modis/lpdaac-tiles-c6"
    tile_pattern = os.path.join(root_path, "MCD43A4.006/%d.%.2d.%.2d/MCD43A4.*.h%.2dv%.2d.006.*.hdf" % (year, month, day, h, v))
    tile_path = glob.glob(tile_pattern)[0]

    mask_pattern = os.path.join(root_path, "MCD43A2.006/%d.%.2d.%.2d/MCD43A2.*.h%.2dv%.2d.006.*.hdf" % (year, month, day, h, v))
    mask_path = glob.glob(mask_pattern)[0]

    mask = input_mask(mask_path)
    arr = input_stack(tile_path)

    masked_arr, arr_masks = get_masked_bands(arr, mask)
    return masked_arr, arr_masks

def get_y_timestamps(h, v, year):
    fc_path = '/g/data2/tc43/modis-fc/v310/tiles/8-day/cover/FC.v310.MCD43A4.h%.2dv%.2d.%d.006.nc' % (h, v, year)

    timestamps = []
    with nc.Dataset(fc_path, 'r', format='NETCDF4') as src:
        ts = nc.num2date(src['time'][:], src['time'].units, src['time'].calendar)
        for i in xrange(ts.shape[0]):
            timestamps.append(ts[i])

    return timestamps

def get_y(h, v, year, month, day, mask=None):
    fc_path = '/g/data2/tc43/modis-fc/v310/tiles/8-day/cover/FC.v310.MCD43A4.h%.2dv%.2d.%d.006.nc' % (h, v, year)

    fc = None
    bands = ["phot_veg", "nphot_veg", "bare_soil"]
    with nc.Dataset(fc_path, 'r', format='NETCDF4') as src:
        ts = nc.num2date(src['time'][:], src['time'].units, src['time'].calendar)
        dt = datetime.datetime(year, month, day)
        for i in xrange(ts.shape[0]):
            if dt == ts[i]:
                for ib, bnd in enumerate(bands):
                    data = src[bnd][i, ...].reshape(-1)
                    if fc is None:
                        fc = np.empty((data.shape[0], len(bands)), dtype=data.dtype)

                    fc[:, ib] = data

                if not mask is None:
                    fc = fc[mask, ...]
                break

    return fc


