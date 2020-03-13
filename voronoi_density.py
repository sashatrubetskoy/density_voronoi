# Sasha Trubetskoy
# March 2020

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
import shapely.wkt

df = pd.read_csv('cell_counts.csv')
df = df.loc[df['dir_id']==1]
df = df.loc[(df['tx'] >= 156) & (df['ty'] >= 347) & (df['tx'] <= 323) & (df['ty'] <= 442)]

df['coord'] = [(tx, ty) for tx, ty in zip(df['tx'], df['ty'])]

xrng = df['tx'].max() - df['tx'].min()
yrng = df['ty'].max() - df['ty'].min()

def closest_point(point, points):
    return cdist([point], points).argmin()

def tx_to_lon(tx, zoom=10):
    return (tx/(2**zoom)*360-180)

def ty_to_lat(ty, zoom=10):
    n = np.pi - 2*np.pi*ty/(2**zoom)
    return (180/np.pi*np.arctan(0.5*(np.exp(n)-np.exp(-n))))

# dif_pcts = []
# min_rel_stds = []

k = 25

def get_score(centroids):
    cell = [closest_point(x, centroids) for x in df['coord']]
    cell_totals = df.groupby(cell)['count'].sum()
    # Allow 1 emptier cell
    second_smallest = cell_totals.nsmallest(2).iloc[-1]
    dif_pct = (cell_totals.max() - second_smallest) / cell_totals.min()
    score = -dif_pct # define score so that higher = better
    return score

# First we use a random brute force approach to find a decent starting point
#   for the iterator.
best_score = -999999
best_start_centroids = []
max_iter = 500
for i in range(max_iter):
    my_x0 = int(df['tx'].min()+0.2*xrng)
    my_x1 = int(df['tx'].max()-0.2*xrng)
    my_y0 = int(df['ty'].min()+0.2*yrng)
    my_y1 = int(df['ty'].max()-0.2*yrng)

    vor_x = np.random.choice(list(range(my_x0, my_x1)), k, replace=False)
    vor_y = np.random.choice(list(range(my_y0, my_y1)), k, replace=False)
    centroids = [np.array([x, y]) for x, y in zip(vor_x, vor_y)]
    
    # Weed out obviously bad iterations.
    #   If xs or ys are too tight
    if min([np.std(vor_x)/xrng, np.std(vor_y)/yrng]) < 0.1:
        continue

    cur_score = get_score(centroids)
    
    if not i%50:
        print('Still searching... ({} of {})'.format(i, max_iter))

    if cur_score > best_score:
        print('Best score: {}! (iter. {})'.format(round(cur_score, 3), i))
        best_score = cur_score
        best_start_centroids = centroids

    if cur_score > -0.5:
        break

# Then using this decent start we can run the iterator
centroids = best_start_centroids
poss_moves = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
last_score = get_score(centroids)
for iteration in range(30):
    print('Running iteration', iteration)
    for c in range(k):
        print('\tAdjusting centroid', c)
        best_score = -99999
        best_move_idx = -99
        for i, move in enumerate(poss_moves):
            print('\t\tTrying move {} of {}'.format(i, len(poss_moves)))
            new_centroids = [cent + move if j==c else cent for j, cent in enumerate(centroids)]
            cur_score = get_score(new_centroids)
            if cur_score > best_score:
                best_score = cur_score
                best_move_idx = i

        best_move = poss_moves[best_move_idx]
        centroids = [cent + best_move if j==c else cent for j, cent in enumerate(centroids)]

    score = get_score(centroids)
    print(iteration, round(score, 3), 'improvement of {}'.format(round(score-last_score, 3)))
    last_score=score
    if score > -0.1:
        break


best_centroids = centroids
cell = [closest_point(x, best_centroids) for x in df['coord']]
cell_totals = df.groupby(cell)['count'].sum()
print('Cell centroids (tx, ty) zoom 10:')
for i, centroid in enumerate(best_centroids):
    print(i, centroid)

print('Cell totals:')
print(cell_totals)

my_centroids = []
for c in best_centroids:
    x, y = c
    centroid = 'POINT ({} {})'.format(tx_to_lon(x+0.5), ty_to_lat(y+0.5))
    my_centroids.append(shapely.wkt.loads(centroid))
gdf = gpd.GeoDataFrame(geometry=my_centroids, crs={'init': 'epsg:4326'})
gdf.to_file("my_centroids.geojson", driver='GeoJSON')