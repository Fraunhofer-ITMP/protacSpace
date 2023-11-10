import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import MACCSkeys
import pickle
from rdkit import DataStructs
from rdkit.Chem import Descriptors
import copy
import json
from rdkit.Chem import rdMolDescriptors
import numpy as np
import xlsxwriter
#MCS 

from collections import defaultdict
from pathlib import Path
from copy import deepcopy
import random

from ipywidgets import interact, fixed, widgets
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import PandasTools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import Descriptors, Draw, PandasTools

DrawingOptions.bondLineWidth=1.8
DrawingOptions.atomLabelFontSize=14
#DrawingOptions.includeAtomNumbers=False

IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 800,800

from itertools import groupby

def remove_salt(df,colname):
    temp = []
    for item in df[colname]:
        
        if '.' not in item:
            temp.append(item)
            
        if '.' in item:
            x = item.split('.')
            x = max(x, key=len)
            temp.append(x)
            #print(x)
    
    df[colname] = temp
    return(df)
	
def GetAttributes_1by1(e3,whead,linker):
    
    linker = Chem.MolFromSmiles(linker)
    e3 = Chem.MolFromSmiles(e3)
    whead = Chem.MolFromSmiles(whead)
    
    linker_df = GetAttributes(linker)
    #linker_df = GetAnchorPoints(linker_df)
    linker_df['Type'] = 'linker'
    
    e3_df = GetAttributes(e3)
    e3_df['Type'] = 'e3binder'
    #e3_df = GetAnchorPoints(e3_df)
    
    whead_df = GetAttributes(whead)
    whead_df['Type'] = 'warhead'
    #whead_df = GetAnchorPoints(whead_df)
    
    final_df = pd.concat([linker_df,e3_df,whead_df], axis=0)
    return(final_df)
    
def GetAttributes(mol):
    
    atomNum_list, atomIdx_list,symbol_list,degree_list, Hs = [],[],[],[],[]
    attr_dict = pd.DataFrame()

    for atom in mol.GetAtoms():
        atomNum_list.append(atom.GetAtomicNum()),atomIdx_list.append(atom.GetIdx()),
        symbol_list.append(atom.GetSymbol()),degree_list.append(atom.GetDegree()),
        Hs.append(atom.GetTotalNumHs())

    attr_dict['AtomNum'] = atomNum_list
    attr_dict['AtomIdx'] = atomIdx_list
    attr_dict['Symbol'] = symbol_list
    attr_dict['Degree'] = degree_list
    attr_dict['Hs'] = Hs
    
    return(attr_dict)

def GetAnchorPoints(df):
    filter_df = df[((df['Hs'] == 1) & (df['Degree'] == 1) & (df['Symbol'] == 'O') |
                    (df['Hs'] == 3) & (df['Degree'] == 1) & (df['Symbol'] == 'C') |
                    (df['Hs'] == 2) & (df['Degree'] == 1) & (df['Symbol'] == 'N') |
                    (df['Hs'] == 1) & (df['Degree'] == 2) & (df['Symbol'] == 'N'))]

    idx_list = list(filter_df['AtomIdx'])

    gb = groupby(enumerate(idx_list), key=lambda x: x[0] - x[1])

    # Repack elements from each group into list
    all_groups = ([i[1] for i in g] for _, g in gb)

    # Filter out one element lists #get lists of list for consecutive nums
    lists_of_list = list(filter(lambda x: len(x) > 1, all_groups))

    temp_df = pd.DataFrame()
    if lists_of_list:
        for sublist in lists_of_list:
            if len(sublist) == 3:
                temp_df = filter_df.loc[filter_df['AtomIdx'].isin(sublist)]

    final_df = pd.concat([filter_df, temp_df, temp_df]).drop_duplicates(keep=False)

    return (final_df)

def Generate_Protacs(linker,e3b,wh):
    
    protac_df = pd.DataFrame()
    
    raw = linker+'.'+e3b+'.'+wh
    #print(raw)
    raw = Chem.MolFromSmiles(raw)
    
    #return(raw)
    
    df_raw = GetAttributes(raw)
    #print(df_raw)
    component_df = GetAttributes_1by1(e3b,wh,linker)
    
    df_raw['Type'] = list(component_df['Type'])
    
    #get anchoring point indexes of linkers 
    
    link_idx = df_raw.loc[df_raw['Type'] == 'linker']
    
    #print(linker)
    
    filter_df = link_idx[((link_idx['Hs'] ==1) & (link_idx['Degree'] == 1) & (link_idx['Symbol'] == 'O') | 
            (link_idx['Hs'] ==3) & (link_idx['Degree'] == 1) & (link_idx['Symbol'] == 'C') | 
            (link_idx['Hs'] ==2) & (link_idx['Degree'] == 1) & (link_idx['Symbol'] == 'N'))]
    
    #link_idx = GetAnchorPoints(link_idx)
    
    #print(filter_df)
    
    if len(filter_df) == 2:
    
        anch_1 = filter_df.iloc[0]['AtomIdx']
        anch_2 = filter_df.iloc[1]['AtomIdx']


        #anch_1 = filter_df.iloc[0]['AtomIdx']
        #anch_2 = filter_df.iloc[1]['AtomIdx']

        e3_idx = df_raw.loc[df_raw['Type'] == 'e3binder']
        e3_idx = GetAnchorPoints(e3_idx)
        e3_idx = list(e3_idx['AtomIdx'])

        wh_idx = df_raw.loc[df_raw['Type'] == 'warhead']
        wh_idx = GetAnchorPoints(wh_idx)
        wh_idx = list(wh_idx['AtomIdx'])

        link_e3_list = []

        for item in e3_idx:

            mol = Chem.RWMol(raw)
            #print(item)
            temp_mol = mol

            temp_mol.AddBond(int(anch_1),int(item),order=Chem.rdchem.BondType.SINGLE)

            test_mol = temp_mol.GetMol()
            link_e3_list.append(test_mol)

        final_protac = []
        for link_e3 in link_e3_list:

            #test = Chem.RWMol(mol)

            for obj in wh_idx:
                #print(wh_idx[i])
                test = Chem.RWMol(link_e3)
                dummy = test

                dummy.AddBond(int(anch_2),int(obj))

                final_test = dummy.GetMol()
                final_protac.append(final_test)

            #print(Draw.MolsToGridImage(final_protac))

        link_wh_list = []

        for item in wh_idx:

            mol = Chem.RWMol(raw)
            #print(item)
            temp_mol = mol

            temp_mol.AddBond(int(anch_1),int(item))

            test_mol = temp_mol.GetMol()
            link_wh_list.append(test_mol)

        final_protac_reverse = []
        for mol in link_wh_list:

            #test = Chem.RWMol(mol)

            for obj in e3_idx:
                #print(wh_idx[i])
                test = Chem.RWMol(mol)
                dummy = test

                dummy.AddBond(int(anch_2),int(obj))

                final_test = dummy.GetMol()
                final_protac_reverse.append(final_test)

        final_merged = final_protac + final_protac_reverse
        
        protac_df['protac'] = [Chem.MolToSmiles(mol) for mol in final_merged]
        protac_df['e3binder'] = e3b
        protac_df['warhead'] = wh
        protac_df['linker'] = linker

        return(protac_df)
    
    
    else:
        return(protac_df)

    
    