
## ENVIRONMENT


## IMPORTS
# %%
from scipy.io import mmread
import pandas as pd

# %%
## SETTINGS

# %%
## FUNCTIONS

# %%
dmat = mmread("data/sc-seq/e-enad-15/E-ENAD-15.aggregated_filtered_normalised_counts.mtx")
# %%
df  = pd.DataFrame.sparse.from_spmatrix(dmat)
# %%
df.
# %%
