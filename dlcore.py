#!/usr/bin/env python

## Stores functions required
## to generate the train set

# %% IMPORT



# %% FUNCTIONS
def hugo_to_ense(species, aset):
    '''
    Maps hugo symbols to a Ensembl IDs
    Input: A set of gene symbols
    Output: a dict of gene symbls as keys and ense IDs as values
    '''
    ## imports
    from ensembl import gen_query_url, sym_lookup

    ## output
    map_dct   = {} ## HUGO to ENSEMBL mapping dict
    unmap_lst = [] ## HUGO that can't be mapped to ENSEMBL ID

    ## inputs
    url = gen_query_url(endpoint = 'lookup/symbol')
    inp_len = len(aset)

    acount = 0 
    for idx, asym in enumerate(aset):
        print(f"\nMapping-{idx}/{inp_len}:{asym}")
        res_json = sym_lookup(url, species, asym)
        # print(f"Response:{res_json}")
        
        if 'error' in res_json:
            text = res_json.get('error')
            print(f"Error: {text}")
            eid = None
            unmap_lst.append(asym)
        else:
            eid = res_json.get('id')
        
        if eid is not None:
            acount+=1
        else:
            pass

        print(f"Mapped Ensembl:{eid}")
        map_dct[asym] = eid

    print(f"\nTotal Symbol for mapping:{len(aset)} | mapped:{acount}")
    return map_dct, unmap_lst

def update_labs_to_ensembl(labs_dct_pkl, species):
    '''
    Input: dct with gene symbols as keys and labels (str, or list) as values 
    Converts labels key from gene symbols to ensembl
    '''

    ## imports
    import pickle

    ## output 
    labs_dct_up = {}
    out_pkl     = "%s_ensembl.p" % (labs_dct_pkl.rpartition(".")[0])

    ## inputs
    labs_dct = pickle.load( open( labs_dct_pkl, "rb" ) )
    
    ## generate HUGO to ENSEMBL mappings
    map_lst  = [k.strip() for k in labs_dct.keys()]
    map_dct, unmap_lst  = hugo_to_ense(species, set(map_lst))

    ## update labels dict keys to ENSEMBL
    acount = 0
    for idx, (k,v) in enumerate(labs_dct.items()):
        eid = map_dct.get(k.strip())
        if eid is not None:
            labs_dct_up[eid] = v
        else:
            labs_dct_up[k]   = v
            acount +=1
    
    ## pickle dicts
    pickle.dump( labs_dct_up, open( out_pkl, "wb" ) )

    print(f"Total keys:{len(labs_dct)} | Unmapped:{acount}")
    return labs_dct_up, unmap_lst

# %% DEV
species = "human"
# test_set = set(["BRCA1", "ACT1"])
# map_dct, unmap_lst  = hugo_to_ense(species, test_set)

labs_dct_pkl = "labs_bin_dct.p"
labs_dct, unmap_lst = update_labs_to_ensembl(labs_dct_pkl, species)

# %% TEST
import pickle
labs_dct = pickle.load( open( "labs_dct_ensembl.p", "rb" ) )

# %% CHANGELOG
## v01 [12/26/2020]
