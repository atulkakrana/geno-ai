## code and functions for 
## calling Ensembl APIs

# %% IMPORTS
import requests, json
import sys


# %% ENSEMBL API
scheme      = 'https'
host        = 'rest.ensembl.org'
ENSE_API    = '{}://{}'.format(scheme, host)

# %% FUNCTIONS
def get_seq(id, flank, ENSE_API, endpoint = 'sequence'):
    '''
    https://rest.ensembl.org/documentation/info/sequence_id
    '''
    ENSE_SEQ_QUERY = ENSE_API + f"/sequence/{endpoint}/id/{id}"
    print(f"\nQuery:{ENSE_SEQ_QUERY}")
    r = requests.get(ENSE_SEQ_QUERY, headers={ "Content-Type" : "text/plain"})
 
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    else:
        seq = r.text
        print(f"ID:{id}")
        print(f"Seq snippet:{seq[:100]}")

    return seq

def ense_info(id, ENSE_API, endpoint = 'lookup'):
    '''
    gets information for given id
    '''

    ENSE_ID_QUERY = ENSE_API + f"/{endpoint}/id/{id}"
    print(f"\nQuery:{ENSE_ID_QUERY}")
    r = requests.get(ENSE_ID_QUERY, headers={ "Content-Type" : "application/json"})
 
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    else:
        decoded = r.json()
        print(repr(decoded))


    return decoded

def check_assembly(id, ENSE_API, endpoint = 'lookup'):
    '''
    https://rest.ensembl.org/documentation/info/assembly_info
    '''

    ENSE_QUERY = ENSE_API + f"/{endpoint}/id/{id}"
    print(f"\nQuery:{ENSE_QUERY}")
    r = requests.get(ENSE_QUERY, headers={ "Content-Type" : "application/json"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    else:
        decoded = r.json()
        print(repr(decoded))

    return decoded

def extract_seq(info, ENSE_API, endpoint = 'region'):
    '''
    https://rest.ensembl.org/documentation/info/sequence_region
    "/sequence/region/human/X:1000000..1000100:1?expand_5prime=60;expand_3prime=60"
    '''

    id      = info.get('id')
    species = info.get('species')
    chr     = info.get('seq_region_name')
    start   = info.get('start')
    end     = info.get('end')
    strand  = info.get('strand')

    if None in [species, chr, start, end. strand]:
        print(f"Info:{info}")
        print(f" There is a '`None` in required info")
        sys.exit()

    if strand == -1:
        ## reverse strand
        pass
    else:
        pass

    ENSE_REGION_QUERY = ENSE_API + f"/sequence/{endpoint}/{species}/{chr}:{start}..{end}:{strand}?"
    print(f"\nQuery:{ENSE_REGION_QUERY}")
    r = requests.get(ENSE_REGION_QUERY, headers={ "Content-Type" : "text/plain"})
 
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    else:
        seq = r.text
        print(f"ID:{id}")
        print(f"Seq snippet:{seq[:100]}")

    return seq

# %% TEST
qids    = ['ENSMUSG00000000001','ENSMUSG00000000028'] ## test ensembl ids
flank   = 100
for qid in qids:
    # seq = get_promotors(qid, flank, ENSE_API)
    # dct = ense_info(qid, ENSE_API)
    inf = check_assembly(qid, ENSE_API)
    reg = extract_seq(inf, ENSE_API)

# %% TEST

# %% CHANGELOG


# %% TO DO
## Check is negative strand is reversed and excised or we need to compute coords from left hand
