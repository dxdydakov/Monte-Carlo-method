import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from timeit import timeit
from typing import Dict, List, Tuple, Union

import sys
import json
#import gc
from loguru import logger


class MCMixer():
    
    def __init__(
        self,
        cells_expr: pd.DataFrame,
        cells_annot: pd.DataFrame,
        num_points: int = 20000,
        total_coverage: int = 50e6,
        MC_coeff = 1,
        distribution: str = "exponential",
        dirichlet_coef: float = 1.0,
        dirichlet_dict: Dict[str, List[Union[int, float]]] = {},
        FP_TYPE: str = "float32",
        models_path: str = r"configs\models.config-2.json",
        genes_in_expression_path: str = r"configs\genes_v2.txt",
        genes_lenght_path: str = r"configs\genes_length.tsv",
        disable_logs: bool = False,
        seed: int = None,
    ) -> None:
        
        """
        Class MCMixer for mixes generation via Monte-Carlo Method.

        :param cells_expr: pd.DataFrame with expressions of sorted cells with genes as index and sample names as columns
        :param cells_annot: pd.DataFrame with sorted cells annotation with sample names as index and 'Cell_type', 'Dataset' columns.
        :param num_points: int, a number of artificial mixes to create as a linear combination.
        :param total_coverage: int, a total coverage for 1 sample, a number of realisations in Monte-Carlo sampling. Default is 1.
        :param MC_coeff: int, a coefficient for sampling MC_coeff transcripts from 1 linear combination.
        :param distribution: modelled cell fraction distribution ('uniform'/'exponential'/'normal').
        :param healthy_points_ratio: float. Not included in this version.
        :param dirichlet_coef: float, a coefficient for generation parameters for Dirichlet PDF.
        :param dirichlet_dict: dict, parameters for Dictichlet PDF for included cell types.
        :param FP_TYPE: str, nunpy data type for expression outputs.
        :param Models: dict with cell type names as keys and dicts with following keys as values:
                Genes: a list of genes used for that cell type training subsetting
                Mix: a list of cell type names to use as mix for that cell type
                Params: a dict with LGBM parameters
        :param genes_in_expression_path: str, path to txt file with genes used for subsetting
        :param gene_length: str, path to TSV file with gene lengths in bp
        """

        super().__init__()
        # Добавить checking TPM и ренормализацию. Пока ожидается, что в поданных датафрэймах всё ок.
        self.num_points = num_points
        self.total_coverage = total_coverage
        self.MC_coeff = MC_coeff
        self.distribution = distribution
        self.dirichlet_coef = dirichlet_coef
        self.dirichlet_dict = dirichlet_dict
        self.FP_TYPE = FP_TYPE
        self.disable_logs = disable_logs
        self.add_tumor = False
        self.rng = np.random.default_rng(None)

        with open(models_path, 'r') as f:
            self.Models = json.load(f)

        if (self.Models == {}):
            raise ValueError("`Models` and `Types_structure` must not be empty.")          
        
        with open(genes_in_expression_path, "r") as f:
            self.genes_in_expression = [gene_name.rstrip("\n") for gene_name in f]
            
        with open(genes_lenght_path, 'r') as f:
            self.genes_length = {k: float(v) for k,v in (map(str, line.split()) for line in f)}
        self.genes_length_df = pd.DataFrame(self.genes_length, index = ['length']).T
        
        if self.disable_logs:
            logger.remove()
        else:
            logger.remove()
            logger.add(sys.stderr)

        self.cells_expr = cells_expr[self.genes_in_expression]
        self.cells_expr = self.cells_expr.div(self.cells_expr.sum(axis=1), axis = 0) * 10**6

        self.cells_annot = cells_annot
    
    def _generate_expr_probability(self, samples_expr):
        
        samples = samples_expr.index.to_list()
        genes = samples_expr.columns
        self.genes_proba = genes
        
        for sample in samples:
            expr_tpm = samples_expr.loc[sample, :]
            
            length_adj_expr = expr_tpm * self.genes_length_df.loc[expr_tpm.index.values, :].values.reshape(-1)
            gene_probability_vector = length_adj_expr / length_adj_expr.sum()
            self.expr_proba[sample] = gene_probability_vector
        
        self.expr_proba = pd.DataFrame(self.expr_proba)
        return self.expr_proba, self.genes_proba


    def get_fractions(self, modeled_cell: str, cells_to_mix: List[str]):
        
        cells_to_mix_values = self.dirichlet_mixing(cells_to_mix)
        
        if self.distribution == "exponential":
            modeled_cell_values = self.exponential_with_uniform_distribution()
            if self.add_tumor:
                tumor_values = self.exponential_with_uniform_distribution()

        if self.add_tumor:
            cells_fractions = cells_to_mix_values * (1 - tumor_values)
            cells_fractions.loc["Tumor"] = tumor_values
            cells_fractions = cells_fractions * (1 - modeled_cell_values)
        else:
            cells_fractions = cells_to_mix_values * (1 - modeled_cell_values)
            
        cells_fractions.loc[modeled_cell] = modeled_cell_values

        return cells_fractions.T
    
    def dirichlet_mixing(self, cells_to_mix: List[str]):
        """
        Method generates the values of the proportion of mixed cells by dirichlet method.
        The method guarantees a high probability of the the presence of each cell type from 0 to 100%
        at the expense of enrichment of fractions close to zero.
        :param num_points: int number of how many mixes to create
        :param cells_to_mix: list of cell types to mix
        :returns: pandas dataframe with generated cell type fractions
        """
        
        if self.dirichlet_dict:
            # Create a list with Dirichlet coeffs values:
            coefs_dirichlet_list = [self.dirichlet_dict[ct] for ct in cells_to_mix]
            # Normalize by sum:
            coefs_dirichlet_list = [coefs_dirichlet_element / sum(coefs_dirichlet_list) for coefs_dirichlet_element in coefs_dirichlet_list]
            data_dirichlet = self.rng.dirichlet(coefs_dirichlet_list, size=self.num_points,).T.astype(self.FP_TYPE)
            
        else:
            data_dirichlet = self.rng.dirichlet([self.dirichlet_coef / len(cells_to_mix)] * len(cells_to_mix), size=self.num_points).T.astype(self.FP_TYPE)
        return pd.DataFrame(data_dirichlet,index=cells_to_mix,columns=range(self.num_points))
    
    
    def exponential_with_uniform_distribution(self) -> pd.Series:
        """Generates vector with of mixed uniform and exponential distrtribution truncated on [0,1] for cell mixing.

        Returns:
            pd.Series: Array with samples from given mixed exponential and uniform distribution.
        """
        x = self.rng.exponential(size=self.num_points // 2 + (self.num_points % 2))
        x = x / max(x)
        x = np.concatenate([x, self.rng.uniform(size=self.num_points // 2)])
        self.rng.shuffle(x)

        return pd.Series(x, index=range(self.num_points))
    
    def generate_lc(
        self,
        modeled_cell: str,
        extended: bool = False,
        proba: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        if extended:
            genes = self.genes_in_expression
        else:
            genes = self.Models[modeled_cell]['Genes']

        mix_expr = np.zeros((self.num_points, len(genes)), dtype = 'float64')

        cells_to_mix = self.Models[modeled_cell]['Mix']
        cell_types = [modeled_cell] + cells_to_mix
        fractions_df = self.get_fractions(modeled_cell, cells_to_mix) 

        for cell_type in cell_types:
            fractions_i = fractions_df[cell_type].values.reshape(-1, 1)                 
            real_sample_names = self.cells_annot[self.cells_annot['Cell_type'] == cell_type].index.to_list()
            samples_to_choose = np.random.choice(real_sample_names, size=self.num_points, replace=True)  
            expr_to_choose = self.cells_expr.loc[samples_to_choose, genes].to_numpy()
            mix_expr = np.add(mix_expr, np.multiply(fractions_i, expr_to_choose))

        if proba:
            genes_length = self.genes_length_df.loc[genes, :].values.reshape(1, -1)

            mix_expr = np.multiply(mix_expr, genes_length)
            proba = np.divide(mix_expr, np.sum(mix_expr, axis = 1).reshape(-1,1))
            return fractions_df, pd.DataFrame(proba, columns = genes)
        
        return fractions_df, pd.DataFrame(mix_expr, columns = genes)
    

    def generate_mc(
        self,
        modeled_cell: str,
        extended: bool = False,
        MC_coeff: int = None,
        coverage: int = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        if MC_coeff is None:
            MC_coeff = self.MC_coeff
        
        if coverage is None:
            coverage = self.total_coverage

        fractions_df, proba = self.generate_lc(modeled_cell, extended=True, proba = True)

        genes = proba.columns
        genes_length = self.genes_length_df.loc[genes, :].values.reshape(1, -1)

        fractions, proba = fractions_df.to_numpy(), proba.to_numpy()
        cells = fractions_df.columns
        
        del fractions_df

        fractions = np.repeat(fractions, MC_coeff, axis = 0)
        proba = np.repeat(proba, MC_coeff, axis = 0)

        if not extended:
            in_genes = self.Models[modeled_cell]['Genes']
            
            #print(in_genes)
            intesection = np.in1d(genes, in_genes)
            in_genes_index = np.where(intesection)[0]
            out_genes_index = np.where(~intesection)[0]

            in_genes_length = genes_length[0][in_genes_index]
            out_genes_length = genes_length[0][out_genes_index]

            in_proba = proba[:, in_genes_index]
            not_proba = proba[:, out_genes_index]

            #print(in_proba, not_proba)
            not_proba_sum = np.sum(not_proba, axis = 1).reshape(-1, 1)
            not_proba_conditional = np.divide(not_proba, not_proba_sum)

            #print(not_proba_sum)
            in_proba = np.hstack((in_proba, not_proba_sum))
            #print(in_proba)
            mc_results = self.rng.multinomial(coverage, in_proba)

            #print(mc_results)
            in_counts = mc_results[:, :-1]
            in_counts_adj = np.divide(in_counts, in_genes_length)
            in_counts_adj_sum = np.sum(in_counts_adj, axis = 1)

            #print(in_counts_adj_sum)
            not_counts = mc_results[:, -1].reshape(-1, 1)
            #print(not_counts)
            expected_not_counts = np.einsum('ij,ik-> ij', not_proba_conditional, not_counts)
            #print(expected_not_counts)
            expected_not_counts_adj = np.divide(expected_not_counts, out_genes_length)
            #print(expected_not_counts_adj)
            expected_not_counts_adj_sum = np.sum(expected_not_counts_adj, axis = 1)
            #print(expected_not_counts_adj_sum)


            #print(np.add(in_counts_adj_sum, expected_not_counts_adj_sum))
            in_TPMs = np.divide(in_counts_adj, np.add(in_counts_adj_sum, expected_not_counts_adj_sum).reshape(-1, 1)) * 10**6

            return (pd.DataFrame(fractions, columns = cells),
                    pd.DataFrame(in_TPMs, columns = in_genes))
        
        mc_results = self.rng.multinomial(coverage, proba)
        mc_results = np.divide(mc_results, genes_length)  
        mc_results = np.divide(mc_results, mc_results.sum(axis=1).reshape(-1,1)) * 10**6

        return (pd.DataFrame(fractions, columns = cells),
                pd.DataFrame(mc_results, columns = genes))