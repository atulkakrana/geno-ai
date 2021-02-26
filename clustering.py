#!/usr/bin/env python

## env: binf
## functions to visualize embeddings
## quality via clustering


# %% IMPORTS
import os
import sys
import umap
import pickle
import hdbscan
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d 
import sklearn.cluster as cluster
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from dlcore import fasta_to_embed

# %% ENVIRONMENT
from os.path import expanduser
HOME = expanduser("~")

# %% INPUTS
MODEL = f'{HOME}/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.mod.clean-train-segmented_02_03_19_24.bin'
# MODEL = f"{HOME}/0.work/genomes/genomic_features-train-segmented_02_03_20_04.bin"

# %% FUNCTIONS
def embed_dct_to_array(fastdct_fl):
    '''
    input is fast dct
    output is a numpy array for clustring 
    '''

    ## initialize
    le = preprocessing.LabelEncoder()

    ## convert to array
    fastdct = pickle.load(open(fastdct_fl, 'rb'))
    lst     = list(fastdct.values())
    arr     = np.array(lst)

    ## generate classes
    names   = list(fastdct.keys())
    names   = [name.split("|")[-1] for name in names] ## very brittle logic, will work only for gene-level data from biomary website
    classes = [name[:3] for name in names]
    
    ## encode
    le.fit(classes)
    class_labs      = le.transform(classes)
    class_cnts      = Counter(classes)
    min_clust_size  = len(np.unique(class_labs))
    min_samples     = min(list(class_cnts.values()))


    print(f"Gene Name Classes:{class_cnts}")

    print(f"Shape of FASTA embeddings:{arr.shape}")
    return arr, names, class_labs, min_samples, min_clust_size

def umap_embeds(arr, names, class_labs, n_neighbors=15, min_dist=0.1, n_components = 2, metric = 'euclidean',
                    fig_size = (10, 8), plot = True):
    '''
    cluster via umap

    annotate: https://stackoverflow.com/questions/5147112/how-to-put-individual-tags-for-a-scatter-plot
    '''

    ## intialize
    reducer = umap.UMAP(n_neighbors =n_neighbors,
                        min_dist    =min_dist,
                        n_components=n_components,
                        metric      =metric)

    ## fit
    u = reducer.fit_transform(arr)

    ## annotate
    if plot == True:
        _ = plot_umap(u, names, class_labs, n_components, fig_size)


    return u

def plot_umap(u, names, class_labs, n_components, fig_size):
    '''
    plot lower dimensional embeddings
    '''

    
    if n_components == 2:
        ## plot
        plt.figure(figsize=fig_size)
        plt.scatter(u[:,0], u[:,1], c = class_labs, label = class_labs,
                                    s = 50,
                                    cmap=plt.get_cmap('Spectral'))
        ## annotate
        for label, x, y in zip(names, u[:, 0], u[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(-25, 15),
                        textcoords='offset points', ha='left', va='top')
        plt.title('UMAP Embedding Of Test Genes')

    
    elif n_components == 3:
        ## plot
        fig = plt.figure(figsize=fig_size)
        # ax  = plt.axes(projection='3d')
        ax  = fig.add_subplot(111, projection='3d')
        
        ax.scatter(u[:,0], u[:,1], u[:,2], c=class_labs, s = 60,
                        cmap=plt.get_cmap('Spectral'))
        ax.set_title('UMAP Embedding Of Test Genes') 
        plt.show() 
        
        
    else:
        print(f"Components = {n_components} not supported yet")
        print(f"Please set `plot` argument value to `False` - exiting")
        sys.exit()

    return u

def hdbscan_clust(umap_res,class_labs, min_samples =3,
                            min_clust_size = 5):
    '''
    Cluster embeddings via HDBscan

    Link: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
    '''

    ## use HDB scan to predict labels
    labels = hdbscan.HDBSCAN(
    min_samples     =min_samples,
    min_cluster_size=min_clust_size).fit_predict(umap_res)

    ## evaluate quality
    clustered = (labels >= 0)
    ars = adjusted_rand_score(class_labs[clustered], labels[clustered]),
    aim = adjusted_mutual_info_score(class_labs[clustered], labels[clustered])
    print(f"{ars}, {aim}")

    return labels

def corr_plots(embeds, names):
    '''
    Given full or lower-diensional embeddings
    generate a correlation plot
    '''

    ## compute metrics
    ## corr or dist, either
    ## can be used to plot
    ## make sure to adjust 
    ## color directions to 
    ## show high correlation and
    ## lower distance
    corr = np.corrcoef(embeds)    ## 10-dimensions
    dist = 1 - corr
    
    ## covert to DF and plot
    df   = pd.DataFrame(dist)
    df.columns  = names
    df['names'] = names
    df.set_index('names', inplace = True)
    cmap    = sns.color_palette("flare_r", as_cmap=True) 
    g       = sns.clustermap(df, row_cluster=True, col_cluster=True, 
                                 method = 'complete', metric='cosine',
                                 figsize=(30, 30), cmap=cmap)
    g.savefig("clust.png")
    print(f"Generated distance plot - lower the better")
    return None

# %% MAIN
## Generate FASTA embeddings
infas       = 'clust_gene_2.fas'
adct, apkl  = fasta_to_embed(infas, MODEL)
embeds, names, class_labs, min_samples, min_clust_size  = embed_dct_to_array(apkl)

# %% EMBEDS - REDUCED DIMENSIONS
size = (16, 12) ## default is (10,8)
umap_emb_2d     = umap_embeds(embeds, names, class_labs, n_components = 2, min_dist = 0.1, fig_size = size)
umap_emb_3d     = umap_embeds(embeds, names, class_labs, n_components = 3,  plot = False)
umap_emb_5d     = umap_embeds(embeds, names, class_labs, n_components = 5,  plot = False)
umap_emb_10d    = umap_embeds(embeds, names, class_labs, n_components = 10, plot = False)
umap_emb_20d    = umap_embeds(embeds, names, class_labs, n_components = 20, plot = False)

# %% CLUSTERMAP - PLOT
## 10-dimensions gives okay plot
## compared to full embeddings
_ = corr_plots(umap_emb_10d, names)

# %% CLUSTER - HDBSCAN
labels = hdbscan_clust(umap_emb_10d, class_labs, min_samples, min_clust_size)

# %% TEST

# %%
# corr = np.corrcoef(umap_res_10d)    ## 10-dimensions
# # corr = np.corrcoef(embeds)
# dist = 1 - corr

# df = pd.DataFrame(dist)
# df.columns      = names
# df['names']     = names
# df.set_index('names', inplace = True)
# cmap = sns.color_palette("flare_r", as_cmap=True) 
# g = sns.clustermap(df, row_cluster=True, col_cluster=True, 
#                     method = 'complete', metric='cosine',
#                     figsize=(30, 30), cmap=cmap)

# %% NOTES
## CDS seqences for clustering test downloaded from here: http://www.ensembl.org/biomart/martview/
