# Sasha Trubetskoy
# March 2020

print('Making imports...')
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
import shapely.wkt

from sklearn.cluster import KMeans

import argparse

parser = argparse.ArgumentParser(description='Get density voronois')
parser.add_argument('-k', action="store", type=int, default=4)
parser.add_argument('-e', action="store", type=int, default=0)
parser.add_argument('-t', action="store", type=int, default=0.05)
args = parser.parse_args()

df = pd.read_csv('cell_counts.csv')
df = df.loc[df['dir_id']==3]
# df = df.loc[(df['tx'] >= 156) & (df['ty'] >= 347) & (df['tx'] <= 323) & (df['ty'] <= 442)]
df['coord'] = [(tx, ty) for tx, ty in zip(df['tx'], df['ty'])]

xrng = df['tx'].max() - df['tx'].min()
yrng = df['ty'].max() - df['ty'].min()

def get_cells(centroids, coords):
    dist_mat = -2 * np.dot(centroids, coords.T) + np.sum(coords**2,
        axis=1) + np.sum(centroids**2, axis=1)[:, np.newaxis]
    return np.argmin(dist_mat, axis=1)

def tx_to_lon(tx, zoom=10):
    return (tx/(2**zoom)*360-180)

def ty_to_lat(ty, zoom=10):
    n = np.pi - 2*np.pi*ty/(2**zoom)
    return (180/np.pi*np.arctan(0.5*(np.exp(n)-np.exp(-n))))

# dif_pcts = []
# min_rel_stds = []

k = args.k
n_empty_allowed = args.e

def get_score(centroids):
    cell = get_cells(df[['tx', 'ty']].values, np.array(centroids))
    cell_totals = df.groupby(cell)['count'].sum()
    second_smallest = cell_totals.nsmallest(1 + n_empty_allowed).iloc[-1]
    dif_pct = (cell_totals.max() - second_smallest) / cell_totals.min()
    score = -dif_pct # define score so that higher = better
    return score

print('Initializing centroids...')
# Initialize centroids based on proportional horizontal and vertical slicing
n_slices_per_axis = int(np.sqrt(k)) + 1 # better to have extra than not enough
goal = df['count'].sum() / n_slices_per_axis


# # INITIALIZATION BASED ON EXACT MIDDLE OF GROUPS
# x_counts = df.groupby('tx')['count'].sum()
# s1 = x_counts.cumsum() % goal
# s2 = abs((x_counts.cumsum() % goal) - goal)
# adif = pd.concat([s1, s2], axis=1).min(axis=1)
# x_breaks = adif[(adif.shift(1) > adif) & (adif.shift(-1) > adif)].index.tolist()
# x_breaks = [0] + x_breaks + [df['tx'].max()]

# y_counts = df.groupby('ty')['count'].sum()
# s1 = y_counts.cumsum() % goal
# s2 = abs((y_counts.cumsum() % goal) - goal)
# adif = pd.concat([s1, s2], axis=1).min(axis=1)
# y_breaks = adif[(adif.shift(1) > adif) & (adif.shift(-1) > adif)].index.tolist()
# y_breaks = [0] + y_breaks + [df['ty'].max()]

# x_centers = [0.5*(a+b) for a, b in zip(x_breaks[:-1], x_breaks[1:])]
# y_centers = [0.5*(a+b) for a, b in zip(y_breaks[:-1], y_breaks[1:])]


# # INITIALIZATION BASED ON "PEAKS" OF GROUPS
# x_counts = df.groupby('tx')['count'].sum()
# s1 = x_counts.cumsum() % goal
# s2 = abs((x_counts.cumsum() % goal) - goal)
# adif = pd.concat([s1, s2], axis=1).min(axis=1)
# x_centrs = adif[(adif.shift(1) < adif) & (adif.shift(-1) < adif)].index.tolist()

# y_counts = df.groupby('ty')['count'].sum()
# s1 = y_counts.cumsum() % goal
# s2 = abs((y_counts.cumsum() % goal) - goal)
# adif = pd.concat([s1, s2], axis=1).min(axis=1)
# y_centrs = adif[(adif.shift(1) < adif) & (adif.shift(-1) < adif)].index.tolist()

# centroid_candidates = [(x, y) for x in x_centrs for y in y_centrs]

# print('\t(picking best combo...)')
# best_score = -99999999
# best_start_centroids = []
# max_iter = 500
# for i in range(max_iter):
#     sel_idx = np.random.choice(list(range(len(centroid_candidates))), k, replace=False)
#     centroids = [centroid_candidates[idx] for idx in sel_idx]

#     cur_score = get_score(centroids)
    
#     if cur_score > best_score:
#         best_score = cur_score
#         best_start_centroids = centroids

#     if cur_score > -0.5:
#         break

# INITIALIZATION BASED ON K MEANS
my_points = []
for i, row in df.iterrows():
    w = round(np.sqrt((row['count'] / 1e5)))
    if w:
        my_points.extend([[row['tx'], row['ty']]]*int(w))

kmeans = KMeans(n_clusters=k).fit(np.array(my_points))
best_start_centroids = kmeans.cluster_centers_

# Then using this decent start we can run the iterator
print('Iteratively adjusting centroids...')
centroids = best_start_centroids
# poss_moves = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
poss_moves = [np.array([-1, -1]), np.array([1, 1]), np.array([1, -1]), np.array([-1, 1])]
# poss_moves = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1]), np.array([-1, -1]), np.array([1, 1])]
# poss_moves = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1]), np.array([0, 0])]
last_score = get_score(centroids)
last_improvement = 9999
for iteration in range(200):
    print('Running iteration', iteration)
    for c in range(k):
        # print('\tAdjusting centroid', c)
        best_score = -99999
        best_move_idx = -99
        for i, move in enumerate(poss_moves):
            # print('\t\tTrying move {} of {}'.format(i, len(poss_moves)))
            new_centroids = [cent + move if j==c else cent for j, cent in enumerate(centroids)]
            cur_score = get_score(new_centroids)
            if cur_score > best_score:
                best_score = cur_score
                best_move_idx = i

        best_move = poss_moves[best_move_idx]
        centroids = [cent + best_move if j==c else cent for j, cent in enumerate(centroids)]

    score = get_score(centroids)
    improvement = score-last_score
    print(iteration, round(score, 3), 'improvement of {}'.format(round(improvement, 3)))
    last_score=score
    last_improvement=improvement
    if score > args.t:
        break
    if last_improvement == 0 and improvement == 0:
        break


best_centroids = centroids
cell = get_cells(df[['tx', 'ty']].values, np.array(best_centroids))
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