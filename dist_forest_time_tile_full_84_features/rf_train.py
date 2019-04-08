import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import modis_dataset as mds
import csv
import numpy as np

from sklearn.ensemble import forest
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).choice(n_samples, n, replace=False))
        #forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

year = 2018

if len(sys.argv[1:]) < 1:
    raise Exception('please specify one or more band ids <0, 1, 2>')
band_ids = [int(b) for b in sys.argv[1:]]
print band_ids

tile_hv = []
with open('train_tiles.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        h = int(row[0])
        v = int(row[1]) 
        tile_hv.append([h, v])

n_bands = len(band_ids)
n_trees = 16
rf_models = [None, None, None]

for ib in band_ids:
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, bootstrap=True, criterion='mse', max_features=0.8, max_depth=12,
            oob_score=False, random_state=None, verbose=5, warm_start=True)
    rf_models[ib] = rf


for i, hv in enumerate(tile_hv):
    h = hv[0]
    v = hv[1]

    timestamps = mds.get_y_timestamps(h, v, year)
    #print timestamps

    last_month = -1
    for it, ts in enumerate(timestamps):
        month = ts.month 
        if month == last_month:
            continue

        last_month = month
        if (month-1) % 2 != 0:
            continue
        day = ts.day

        x_, masks = mds.get_x(h, v, year, month, day)
        y_ = mds.get_y(h, v, year, month, day, masks)           

        assert x_.shape[0] == y_.shape[0]

        if it == 0:
            x = x_
            y = y_
        else:
            x = np.vstack((x, x_))
            y = np.vstack((y, y_))

        print 'loading data: %d of %d tiles, h%.2dv%.2d, year:%d, month:%d, day:%d, x:%s, y:%s' % (i+1, len(tile_hv), \
                h, v, year, month, day, x.shape, y.shape)

    train_x = x
    set_rf_samples(int(train_x.shape[0]*0.7))

    for ib in band_ids:
        train_y = y[:, ib]

        rf = rf_models[ib]
        rf.n_estimators = (i+1) * n_trees
        rf.fit(train_x, train_y)

        score = rf.score(train_x, train_y)
        print '   (%d of %d bands):   h%.2dv%.2d: train r squared: %.5f' % (ib+1, n_bands, h, v, score)

for ib in band_ids:
    joblib.dump(rf_models[ib], 'rf%d.model' % ib)
