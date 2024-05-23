#!/usr/bin/env python3

import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import qtl.norm
import itertools as it
import numba as nb

from os import makedirs, path
from typing import Union, List
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.sparse import csr_matrix


# ADPBulk from https://github.com/noamteyssier/adpbulk/blob/main/adpbulk/adpbulk.py
class ADPBulk:
    def __init__(
            self,
            adat: anndata.AnnData,
            groupby: Union[List[str], str],
            method: str = "sum",
            name_delim: str = "-",
            group_delim: str = ".",
            use_raw: bool = False):
        """
        Class of Pseudo-Bulking `AnnData` objects based on categorical variables
        found in the `.obs` attribute
        inputs:
            adat: anndata.AnnData
                The `AnnData` object to process
            groupby: Union[List[str], str]
                The categories to group by. Can provide as a single value
                or a list of values.
            method: str
                The method to aggregate with (sum[default], mean, median)
            name_delim: str
                The delimiter to use when grouping multiple categories together.
                example: 'cat1{delim}cat2'
            group_delim: str
                The delimiter to use for adding the value to its category.
                example: 'cat{delim}value'
            use_raw: bool
                Whether to use the `.raw` attribute on the `AnnData` object
        """

        self.agg_methods = {
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median}

        self.adat = adat
        self.groupby = groupby
        self.method = method
        self.name_delim = name_delim
        self.group_delim = group_delim
        self.use_raw = use_raw

        self.group_idx = dict()
        self.groupings = list()
        self.grouping_masks = dict()

        self.meta = pd.DataFrame([])
        self.matrix = pd.DataFrame([])
        self.samples = pd.DataFrame([])

        self._isfit = False
        self._istransform = False

        self._validate()

    def _validate(self):
        """
        validates that the input is as expected
        """
        self._validate_anndata()
        self._validate_groups()
        self._validate_method()
        self._validate_raw()

    def _validate_anndata(self):
        """
        validates that the anndata object is as expected
        """
        if self.adat.X is None:
            raise AttributeError("Provided Matrix is None")
        if self.adat.obs is None:
            raise AttributeError("Provided Obs are None")
        if self.adat.var is None:
            raise AttributeError("Provided Var are None")

    def _validate_groups(self):
        """
        validates that the groups are as expected
        """
        # convert groups to list if provided as str
        if isinstance(self.groupby, str):
            self.groupby = [self.groupby]

        if isinstance(self.groupby, list):
            self.groupby = np.unique(self.groupby)
            for group in self.groupby:
                self._validate_group(group)
        else:
            raise TypeError("Groupby is not a list or str")

    def _validate_group(self, group):
        """
        confirms that provided group is a column in the observations
        """
        if group not in self.adat.obs.columns:
            raise ValueError(f"Provided group {group} not in observations")

    def _validate_method(self):
        """
        confirms that the method is known
        """
        if self.method not in self.agg_methods.keys():
            raise ValueError(
                f"Provided method {self.method} not in known methods {''.join(self.agg_methods)}")

    def _validate_raw(self):
        """
        if the `use_raw` flag is provided will confirm that
        the raw field is present
        """
        if self.use_raw and self.adat.raw is None:
            raise AttributeError(
                "use_raw provided, but no raw field is found in AnnData")


    def _fit_indices(self):
        """
        determines the indices for each of the provided groups
        """
        for group in self.groupby:
            unique_values = np.unique(self.adat.obs[group].values)
            self.group_idx[group] = {
                uv: set(np.flatnonzero(self.adat.obs[group].values == uv))
                    for uv in tqdm(unique_values, desc=f"fitting indices: {group}")}

    def _get_mask(self, pairs: tuple) -> np.ndarray:
        """
        retrieve the indices for the provided values from their respective groups
        calculates the global intersection between the sets
        """
        group_indices = []
        for j, key in enumerate(pairs):
            group_indices.append(self.group_idx[self.groupby[j]][key])
        mask = set.intersection(*group_indices)
        return np.array(list(mask))

    def _get_name(self, pairs: tuple) -> str:
        """
        create a name for the provided values based on their respective groups
        """
        name = self.name_delim.join([
                f"{self.groupby[i]}{self.group_delim}{pairs[i]}" for i in range(len(pairs))])
        return name

    def _get_agg(self, mask: np.ndarray) -> np.ndarray:
        """
        runs the aggregation function with the provided sample mask
        """
        if self.use_raw:
            mat = self.adat.raw.X[mask]
        else:
            mat = self.adat.X[mask]
        return self.agg_methods[self.method](mat, axis=0)

    def _get_var(self) -> np.ndarray:
        """
        return the var names using the given scheme (normal/raw)
        """
        if self.use_raw:
            return self.adat.raw.var.index.values
        else:
            return self.adat.var.index.values

    def _prepare_meta(self, pairs: tuple) -> dict:
        """
        defines the meta values for the pairs
        """
        values = {
            self.groupby[idx]: pairs[idx] for idx in np.arange(len(self.groupby))
            }
        values["SampleName"] = self._get_name(pairs)
        return values

    def _build_groupings(self):
        """
        generate the grouping iterable
        """
        if len(self.groupby) > 1:
            group_keys = [self.group_idx[g].keys() for g in self.groupby]
            self.groupings = list(it.product(*group_keys))
        else:
            self.groupings = self.group_idx[self.groupby[0]]

    def _build_masks(self):
        """
        generate the masks for each of the groupings and generates the
        metadata for each of the groupings
        """
        self.grouping_masks = dict()
        self.meta = []
        for pairs in self.groupings:
            if not isinstance(pairs, tuple):
                pairs = tuple([pairs])
            mask = self._get_mask(pairs)
            if mask.size > 0:
                self.grouping_masks[pairs] = mask
                self.meta.append(self._prepare_meta(pairs))

        if len(self.meta) == 0:
            raise ValueError("No combinations of the provided groupings found")
        self.meta = pd.DataFrame(self.meta)

    def fit(self):
        """
        fits the indices for each of the groups
        """
        self._fit_indices()
        self._build_groupings()
        self._build_masks()
        self._isfit = True

    def transform(self) -> pd.DataFrame:
        """
        performs the aggregation based on the fit indices
        """
        if not self._isfit:
            raise AttributeError("Please fit the object first")

        matrix = []
        for pairs in tqdm(self.groupings, desc="Aggregating Samples"):
            if not isinstance(pairs, tuple):
                pairs = tuple([pairs])
            if pairs in self.grouping_masks:
                matrix.append(self._get_agg(self.grouping_masks[pairs]))
        
        # stack all observations into single matrix
        matrix = np.vstack(matrix)

        self.matrix = pd.DataFrame(
            matrix,
            index=self.meta.SampleName.values,
            columns=self._get_var())

        self._istransform = True
        return self.matrix

    def fit_transform(self):
        """
        firs the indices and performs the aggregation based on those indices
        """
        self.fit()
        return self.transform()

    def get_meta(self) -> pd.DataFrame:
        """
        return the meta dataframe
        """
        if not self._isfit:
            raise AttributeError("Please fit the object first")
        return self.meta


def get_pca_linear_model_residuals(normalized_matrix):
    # scale data prior to PCA
    scaler = StandardScaler().set_output(transform="pandas")
    scaled_normalized_matrix = scaler.fit_transform(normalized_matrix)
    
    # fit PCA
    pca = PCA(n_components=10)
    pca_features = pca.fit_transform(scaled_normalized_matrix)
    
    # fit linear model
    reg = LinearRegression().fit(pca_features, normalized_matrix)
    predicted_vals = reg.predict(pca_features)

    # get residuals for log-TMM-normalized cell-type-specific pseudobulk
    model_residuals = normalized_matrix - predicted_vals

    # return transpose
    return model_residuals.T


def create_ct_df(in_df, cell_type, samples, dataset_name):
    ct_dic = {}
    df_cols = list(in_df)

    for s in samples:
        if dataset_name == "onek1k":
            sample_col = f"cell_type_name.{cell_type}-donor_id.{s}"
        elif dataset_name == "lupus":
            sample_col = f"ind_cov.{s}-onek1k_cell_type.{cell_type}"
        else:
            raise ValueError("Invalid dataset name")
        if sample_col in df_cols:
            ct_dic[s] = np.asarray(in_df[sample_col])
        else:
            ct_dic[s] = np.zeros(len(in_df))
            ct_dic[s][:] = np.nan

    ct_df = pd.DataFrame(ct_dic)
    ct_df.index = in_df.index
    return ct_df


def inverse_normal_transform(M):
    """Transform rows to a standard normal distribution"""
    R = stats.rankdata(M, axis=1)  # ties are averaged
    Q = stats.norm.ppf(R/(M.shape[1]+1))
    Q = pd.DataFrame(Q, index=M.index, columns=M.columns)
    return Q


def transfer_onek1k_cell_type_names(cell_by_gene_h5_fi, onek1k_adata):
    # transfer cell type names
    adata = anndata.read_h5ad(cell_by_gene_h5_fi)
    row_keys = onek1k_adata.obs.index
    filtered_row_keys = [key for key in row_keys if key in adata.obs.index]
    filtered_adata = adata[filtered_row_keys, :]
    filtered_onek1k_adata = onek1k_adata[filtered_adata.obs.index, :]
    assert filtered_adata.obs.index.equals(filtered_onek1k_adata.obs.index)
    filtered_onek1k_adata.obs["cell_type_name"] = filtered_adata.obs["new_names"]
    return filtered_onek1k_adata


def subsample_adata_and_save(
    onek1k_adata_filtered, n_subsample_cells, subset_adata_out_subset_dir, dataset_name
):
    for n in n_subsample_cells:
        if dataset_name == "onek1k":
            donor_col = "donor_id"
        elif dataset_name == "lupus":
            donor_col = "ind_cov"
        else:
            raise ValueError("Invalid dataset name")
        n = min(n, min(onek1k_adata_filtered.obs.groupby(donor_col).size()))
        subsampled_indices = []
        for individual, group in onek1k_adata_filtered.obs.groupby(donor_col):
            indices = group.index.values
            if len(indices) <= n:
                subsampled_indices.extend(indices)
            else:
                subsampled_indices.extend(np.random.choice(indices, n, replace=False))
        
        adata_subsample = onek1k_adata_filtered[subsampled_indices].copy()
        adata_subsample.write_h5ad(
            path.join(subset_adata_out_subset_dir, f"subsample_{n}.h5ad")
        )
    return None


def create_sum_agg_files(
    n,
    pseudobulk_matrix,
    n_subsample_cells,
    reduced_cell_types,
    write_samples,
    dataset_name
):
    ### sum agg ###
    gene_pseudobulk_dics = {}
    gene_pseudobulk_invnorm_dics = {}
    gene_pseudobulk_residuals_dics = {}
    gene_pseudobulk_residuals_invnorm_transformed_dics = {}

    for cell_type in tqdm(reduced_cell_types):
        # define cells
        if dataset_name == "onek1k":
            cell_type_cols = [col for col in pseudobulk_matrix if col.startswith(f"cell_type_name.{cell_type}-")]
        else:
            cell_type_cols = [col for col in pseudobulk_matrix if col.endswith(f"{cell_type}")]
        # subset to keep only cells of cell type
        cell_type_pseudobulk_matrix = pseudobulk_matrix[cell_type_cols]
        # create log+1 pseudobulk matrix (sumagg)
        log_pseudobulk_matrix = np.log2(cell_type_pseudobulk_matrix + 1)
        # inverse normal transform the log pseudobulk matrix
        inv_norm_log_pseudobulk_matrix = inverse_normal_transform(log_pseudobulk_matrix)
        # get residual expression using RINT-log pseudobulk matrix
        log_model_residuals = get_pca_linear_model_residuals(log_pseudobulk_matrix.T)
        # RINT the model residuals
        inv_norm_transformed_log_model_residuals = inverse_normal_transform(log_model_residuals)
        # define matrices to store log pseudobulk, RINT-log-pseudobulk, and RINT-log-residual-pseudobulk
        df_columns = list(log_pseudobulk_matrix)
        log_pseudobulk_matrix_df = create_ct_df(
            log_pseudobulk_matrix,
            cell_type,
            write_samples,
            dataset_name
        )
        inv_norm_transformed_log_pseudobulk_matrix_df = create_ct_df(
            inv_norm_log_pseudobulk_matrix,
            cell_type,
            write_samples,
            dataset_name
        )
        inv_norm_transformed_log_pseudobulk_model_residuals_df = create_ct_df(
            inv_norm_transformed_log_model_residuals,
            cell_type,
            write_samples,
            dataset_name
        )
        # assign processed cell-type data to dictionaries
        gene_pseudobulk_dics[cell_type] = log_pseudobulk_matrix_df
        gene_pseudobulk_invnorm_dics[cell_type] = inv_norm_transformed_log_pseudobulk_matrix_df
        gene_pseudobulk_residuals_invnorm_transformed_dics[cell_type] = inv_norm_transformed_log_pseudobulk_model_residuals_df

    # write output
    out_cell_type_log_pseudobulk_dir = f"../data/subsample_{dataset_name}_cell_type_pseudobulk/n_{n}/sum_agg/log_pseudobulk_matrices"
    makedirs(out_cell_type_log_pseudobulk_dir, exist_ok=True)
    out_cell_type_inv_norm_log_pseudobulk_dir = f"../data/subsample_{dataset_name}_cell_type_pseudobulk/n_{n}/sum_agg/invnorm_log_pseudobulk_matrices"
    makedirs(out_cell_type_inv_norm_log_pseudobulk_dir, exist_ok=True)
    out_cell_type_inv_norm_log_pseudobulk_residuals_dir = f"../data/subsample_{dataset_name}_cell_type_pseudobulk/n_{n}/sum_agg/invnorm_regression_residuals"
    makedirs(out_cell_type_inv_norm_log_pseudobulk_residuals_dir, exist_ok=True)

    for ct in reduced_cell_types:
        # process log tmm normalized
        gene_pseudobulk_dics[ct].to_csv(
            path.join(out_cell_type_log_pseudobulk_dir, f"{ct}.tsv"),
            sep="\t",
            index=True,
            index_label="gene_id"
        )
        gene_pseudobulk_invnorm_dics[ct].to_csv(
            path.join(out_cell_type_inv_norm_log_pseudobulk_dir, f"{ct}.tsv"),
            sep="\t",
            index=True,
            index_label="gene_id"
        )
        gene_pseudobulk_residuals_invnorm_transformed_dics[ct].to_csv(
            path.join(out_cell_type_inv_norm_log_pseudobulk_residuals_dir, f"{ct}.tsv"),
            sep="\t",
            index=True,
            index_label="gene_id"
        )
    return None


def create_log_tmm_files(
    n,
    pseudobulk_matrix,
    n_subsample_cells,
    reduced_cell_types,
    write_samples,
    dataset_name
):
    ### log TMM ###
    gene_pseudobulk_dics = {}
    gene_pseudobulk_invnorm_dics = {}
    gene_pseudobulk_residuals_dics = {}
    gene_pseudobulk_residuals_invnorm_transformed_dics = {}

    for cell_type in tqdm(reduced_cell_types):
        # define cells
        if dataset_name == "onek1k":
            cell_type_cols = [col for col in pseudobulk_matrix if col.startswith(f"cell_type_name.{cell_type}-")]
        else:
            cell_type_cols = [col for col in pseudobulk_matrix if col.endswith(f"{cell_type}")]
        # subset to keep only cells of cell type
        cell_type_pseudobulk_matrix = pseudobulk_matrix[cell_type_cols]
        # create TMM matrix
        normalized_matrix = qtl.norm.edger_cpm(cell_type_pseudobulk_matrix, normalized_lib_sizes=True)
        # create log+1 pseudobulk matrix (sumagg)
        log_pseudobulk_matrix = np.log2(normalized_matrix + 1)
        # inverse normal transform the log pseudobulk matrix
        inv_norm_log_pseudobulk_matrix = qtl.norm.inverse_normal_transform(log_pseudobulk_matrix)
        # get residual expression using RINT-log pseudobulk matrix
        log_model_residuals = get_pca_linear_model_residuals(log_pseudobulk_matrix.T)
        # RINT the model residuals
        inv_norm_transformed_log_model_residuals = qtl.norm.inverse_normal_transform(log_model_residuals)
        # define matrices to store log pseudobulk, RINT-log-pseudobulk, and RINT-log-residual-pseudobulk
        df_columns = list(log_pseudobulk_matrix)
        log_pseudobulk_matrix_df = create_ct_df(
            log_pseudobulk_matrix,
            cell_type,
            write_samples,
            dataset_name
        )
        inv_norm_transformed_log_pseudobulk_matrix_df = create_ct_df(
            inv_norm_log_pseudobulk_matrix,
            cell_type,
            write_samples,
            dataset_name
        )
        inv_norm_transformed_log_pseudobulk_model_residuals_df = create_ct_df(
            inv_norm_transformed_log_model_residuals,
            cell_type,
            write_samples,
            dataset_name
        )
        # assign processed cell-type data to dictionaries
        gene_pseudobulk_dics[cell_type] = log_pseudobulk_matrix_df
        gene_pseudobulk_invnorm_dics[cell_type] = inv_norm_transformed_log_pseudobulk_matrix_df
        gene_pseudobulk_residuals_invnorm_transformed_dics[cell_type] = inv_norm_transformed_log_pseudobulk_model_residuals_df

    out_cell_type_log_pseudobulk_dir = f"../data/subsample_{dataset_name}_cell_type_pseudobulk/n_{n}/log_tmm/log_pseudobulk_matrices"
    makedirs(out_cell_type_log_pseudobulk_dir, exist_ok=True)

    out_cell_type_inv_norm_log_pseudobulk_dir = f"../data/subsample_{dataset_name}_cell_type_pseudobulk/n_{n}/log_tmm/invnorm_log_pseudobulk_matrices"
    makedirs(out_cell_type_inv_norm_log_pseudobulk_dir, exist_ok=True)

    out_cell_type_inv_norm_log_pseudobulk_residuals_dir = f"../data/subsample_{dataset_name}_cell_type_pseudobulk/n_{n}/log_tmm/invnorm_regression_residuals"
    makedirs(out_cell_type_inv_norm_log_pseudobulk_residuals_dir, exist_ok=True)

    for ct in reduced_cell_types:
        # process log tmm normalized
        gene_pseudobulk_dics[ct].to_csv(
            path.join(out_cell_type_log_pseudobulk_dir, f"{ct}.tsv"),
            sep="\t",
            index=True,
            index_label="gene_id"
        )

        gene_pseudobulk_invnorm_dics[ct].to_csv(
            path.join(out_cell_type_inv_norm_log_pseudobulk_dir, f"{ct}.tsv"),
            sep="\t",
            index=True,
            index_label="gene_id"
        )

        gene_pseudobulk_residuals_invnorm_transformed_dics[ct].to_csv(
            path.join(out_cell_type_inv_norm_log_pseudobulk_residuals_dir, f"{ct}.tsv"),
            sep="\t",
            index=True,
            index_label="gene_id"
        )
    return None


def main():
    ### BOTH DATASETS ###
    # define subset out directory and create
    subset_adata_out_dirs = {
        "onek1k": "/gpfs/commons/groups/gursoy_lab/cwalker/projects/sc_privacy/ml/data/onek1k/adata/subsets",
        "lupus": "/gpfs/commons/groups/gursoy_lab/cwalker/projects/sc_privacy/data/lupus/adata_2/subsets"
    }
    for v in subset_adata_out_dirs.values():
        makedirs(v, exist_ok=True)
    # define cell types of interest
    reduced_cell_types = sorted(
        [
            'Mono_NC', 'Plasma', 'CD8_ET', 'CD4_NC', 'Mono_C', 'DC', 'CD8_NC', 'NK_R',
            'B_MEM', 'CD4_ET', 'CD8_S100B', 'B_IN', 'NK', 'CD4_SOX4'
        ]
    )
    # # define subsamples
    n_subsample_cells = [200, 400, 600, 800, 1000]

    ### ONEK1K ###
    # define input
    base_dir = "/gpfs/commons/groups/gursoy_lab/cwalker/projects/sc_privacy/ml/"
    data_dir = path.join(base_dir, "data")
    cell_by_gene_h5_dir = path.join(data_dir, "onek1k", "cellbygene")
    cell_by_gene_h5_fi = path.join(cell_by_gene_h5_dir, "onek1k_cell_by_gene.h5ad")
    orig_cell_by_gene_h5_dir = path.join(data_dir, "onek1k", "adata")
    orig_cell_by_gene_h5_fi = path.join(orig_cell_by_gene_h5_dir, "local.h5ad")
    # load data
    print("Loading OneK1K anndata object 1...")
    onek1k_adata = anndata.read_h5ad(orig_cell_by_gene_h5_fi)
    print("Loading OneK1K anndata object 2 and assigning cell type names...")
    filtered_onek1k_adata = transfer_onek1k_cell_type_names(
        cell_by_gene_h5_fi, onek1k_adata
    )
    # load onek1k samples to retain
    onek1k_hq_sample_fi = "/gpfs/commons/groups/gursoy_lab/xli/linking/output/GTEx_OneK1K/reprocessed/CD4_ET/quantified_genotype_samples.txt"
    with open(onek1k_hq_sample_fi, "r") as f:
        hq_onek1k_samples = [i.strip() for i in f.readlines()]
    print("Filtering OneK1K anndata object to samples with >= 1000 cells...")
    # retain only samples with at least 1000 cells
    sample_cell_counts = filtered_onek1k_adata.obs.groupby('donor_id').size()
    filtered_samples = sample_cell_counts[sample_cell_counts >= 1000].index
    out_metadata_dir = "../data/onek1k_variant_selection/metadata"
    with open(path.join(out_metadata_dir, "onek1k_1000_cell_samples.txt"), "w") as f:
        f.write("\n".join(filtered_samples))


    ### LUPUS ###
    # load integrated lupus data
    lupus_integrated_h5 = "/gpfs/commons/groups/gursoy_lab/xli/Lupus/integration/data/Lupus_corrected_integrated.h5ad"
    lupus_adata = sc.read_h5ad(lupus_integrated_h5)
    out_metadata_dir = "../data/onek1k_variant_selection/metadata/lupus"
    makedirs(out_metadata_dir, exist_ok=True)
    # load CZI lupus adata
    lupus_adata_file = "/gpfs/commons/groups/gursoy_lab/cwalker/projects/sc_privacy/data/lupus/adata_2/fd5e58b5-ddff-457f-b182-d18f99e36207.h5ad"
    czi_lupus_adata = anndata.read_h5ad(lupus_adata_file)
    # get list of samples
    lupus_samples = list(set(czi_lupus_adata.obs["ind_cov"]))
    # tweak index ids to make them identical between two adata objects and check
    # they're identical
    lupus_adata.obs.index = lupus_adata.obs.index.str.rstrip('-lupus')
    assert all(lupus_adata.obs.index == czi_lupus_adata.obs.index)
    # assign cell type names and delete now unused adata object
    czi_lupus_adata.obs["onek1k_cell_type"] = lupus_adata.obs["new_names"]
    del lupus_adata
    # retain only samples with at least 1000 cells
    sample_cell_counts = czi_lupus_adata.obs.groupby('ind_cov').size()
    filtered_samples = sample_cell_counts[sample_cell_counts >= 1000].index
    mask = czi_lupus_adata.obs['ind_cov'].isin(filtered_samples)
    czi_lupus_adata = czi_lupus_adata[mask].copy()
    # create and save subsampled anndata objects
    print("Subsampling and saving Lupus anndata objects...")
    subsample_adata_and_save(
        czi_lupus_adata,
        n_subsample_cells,
        subset_adata_out_dirs["lupus"],
        "lupus"
    )

    # create lupus pseudobulk files for each cell n subsample
    print("Pseudobulking subsampled anndatas...")
    for n in tqdm(n_subsample_cells):
        # load subsample adata and create pseudobulk
        subsample_adata_file = path.join(
            subset_adata_out_dirs["lupus"], f"subsample_{n}.h5ad"
        )
        ss_adata = sc.read_h5ad(subsample_adata_file)
        #ss_adata.raw = ss_adata
        adpb = ADPBulk(
            ss_adata, ["ind_cov", "onek1k_cell_type"], use_raw=True, method="mean"
        )
        pseudobulk_matrix = adpb.fit_transform()
        pseudobulk_matrix = pseudobulk_matrix.T
        gene_names = ss_adata.var.feature_name
        pseudobulk_matrix.index = gene_names
        # generate pseudobulk files
        create_sum_agg_files(
            n=n,
            pseudobulk_matrix=pseudobulk_matrix,
            n_subsample_cells=n_subsample_cells,
            reduced_cell_types=reduced_cell_types,
            write_samples=lupus_samples,
            dataset_name="lupus"
        )
        create_log_tmm_files(
            n=n,
            pseudobulk_matrix=pseudobulk_matrix,
            n_subsample_cells=n_subsample_cells,
            reduced_cell_types=reduced_cell_types,
            write_samples=lupus_samples,
            dataset_name="lupus"
        )




if __name__ == "__main__":
    main()
