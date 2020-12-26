#!/usr/bin/env python

## ENSEMBL API 
## ACCESS AND FUNCTIONS


# %% IMPORTS
import requests 


# %% FUNCTIONS
def gen_query_url(endpoint):
    '''
    generates endpoint for symbol lookup
    '''
    scheme      = 'http'
    host        = 'rest.ensembl.org'
    port        = '80'
    # endpoint    = 'lookup/symbol'
    url         = '{}://{}/{}'.format(scheme, host, endpoint)

    return url 

def sym_lookup(url, sp, sym):
    '''
    Converts given HUGO symbol to ensembl ID

    src: http://rest.ensembl.org/documentation/info/symbol_lookup
    '''
    url         = url + f"/{sp}/{sym}" 
    payload     = {"content-type" : "application/json", "expand": 0} ## set expand to 1 for more info
    r           = requests.get(url=url, params=payload )
    # print(f"URL:{r.url}")
    # print(f"BODY:{r.raw}")

    r.encoding  = 'utf-8'
    jsn_res     = r.json()

    # print(f" JSON Res:{jsn_res}")
    return jsn_res

# %% TEST
# symbol  = "BRCA1"
# species = "human"
# url     = gen_query_url(endpoint = 'lookup/symbol')
# sym_lookup(url, species, symbol)

# %% CHANGELOG
## v0.1 [12/26/2020]
## added basic rest url generator and symbol query
