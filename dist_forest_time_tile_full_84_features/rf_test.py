import sys
from sklearn.externals import joblib
import modis_dataset as mds
import csv

year = 2019
month = 3
day = 6

if len(sys.argv[1:]) < 1:
    raise Exception('please specify one or more band ids <0, 1, 2>')
band_ids = [int(b) for b in sys.argv[1:]]
print band_ids

tile_hv = []
with open('australia_tiles.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        h = int(row[0])
        v = int(row[1]) 
        tile_hv.append([h, v])

n_bands = len(band_ids)
rf_models = [None, None, None]
for ib in band_ids:
    rf_models[ib] = joblib.load('rf%d.model' % ib)

for i, hv in enumerate(tile_hv):
    h = hv[0]
    v = hv[1]

    x, masks = mds.get_x(h, v, year, month, day)
    y = mds.get_y(h, v, year, month, day, masks)           

    assert x.shape[0] == y.shape[0]
    print 'testing %d of %d, n_samples:%d' % (i+1, len(tile_hv), x.shape[0])
    if x.shape[0] == 0:
        print '  h%.2dv%.2d: no test samples' % (h, v)
        continue

    for ib in band_ids:
        score = rf_models[ib].score(x, y[:, ib])
        print '  (%d of %d bands): h%.2dv%.2d: test r squared: %.5f' % (ib+1, n_bands, h, v, score)
