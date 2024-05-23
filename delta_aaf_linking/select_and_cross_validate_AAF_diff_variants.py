#!/usr/bin/env python3

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path, makedirs
from glob import glob
from tqdm import tqdm
from sys import argv


def parse_pseudobulk_index(representative_pb_file):
    celltype_pseudobulk_matrix = pd.read_csv(
        representative_pb_file, sep="\t", index_col="gene_id"
    )
    return list(celltype_pseudobulk_matrix.index)


def load_cell_type_expression_dataframes(cell_types, df_dir):
    ct_dfs = {}
    for ct in tqdm(cell_types):
        ct_fi = path.join(df_dir, f"{ct}.tsv")
        ct_df = pd.read_csv(ct_fi, sep="\t", index_col=0)
        ct_dfs[ct] = ct_df
    return ct_dfs



def main():
    print("Init")
    curr_cohort = argv[1]
    aaf_threshold = float(argv[2])
    expression_threshold = float(argv[3])
    curr_chrom = argv[4]
    agg_method = argv[5]
    dataset_norm = argv[6]
    cell_types = [argv[7]]
    curr_cell_type = cell_types[0]
    chrom = curr_chrom


    # define working directories
    bd = "/gpfs/commons/groups/gursoy_lab/cwalker/projects/sc_privacy/ml/"
    vcf_dir = path.join(bd, "data/onek1k/vcfs")

    ### GENE FILES ###
    gene_id_fi = path.join(bd, "data/metadata/onek1k/unfiltered_retained_gene_ids.txt")
    gene_gtf_fi = path.join(bd, "data/gencode/Homo_sapiens.GRCh37.87.genes.gtf")

    ### PARSE GENE IDs ###
    with open(gene_id_fi, "r") as f:
        gene_ids = [i.strip() for i in f.readlines()]
    
    ### DEFINE OUTPUT DATA PATH ###
    data_dir = path.join(bd, f"data/onek1k_variant_selection/")
    haplotype_df_dir = path.join(data_dir, "haplotype_dataframes")
    genotype_df_dir = path.join(data_dir, "genotype_dataframes")

    ### define cohorts
    cohort_samples = {}

    print("Load cohorts")
    cohort_dir = path.join(data_dir, "cohort_sample_lists")
    
    curr_cohort_fi = path.join(cohort_dir, f"{curr_cohort}.tsv")
    cohort_samples[curr_cohort] = {"train": [], "test": []}
    with open(curr_cohort_fi, "r") as f:
        for line in f.readlines()[1:]:
            line = line.strip().split("\t")
            cohort_samples[curr_cohort][line[-1]].append(line[0])


    genotype_dataframe_files =  glob(path.join(genotype_df_dir, "*", "*"))

    print("Load genotype files")

    gene_chroms = [curr_chrom]

    chrom_genes = {}
    chrom_gene_files = {}

    chrom_gene_file_list = glob(path.join(genotype_df_dir, chrom, "*"))
    chrom_gene_list = [i.split("/")[-1].split(".")[0] for i in chrom_gene_file_list]
    chrom_gene_files[chrom] = chrom_gene_file_list
    chrom_genes[chrom] = chrom_gene_list

    gene_files = chrom_gene_files[chrom]
    gene = gene_files[0].split("/")[-1].split(".")[0]
    tmp_df = pd.read_parquet(gene_files[0])
    samples = list(tmp_df)
    
    chrom_gene_genotype_dataframes = {}
    for chrom in gene_chroms:
        chrom_gene_genotype_dataframes[chrom] = {}
        gene_files = chrom_gene_files[chrom]
        for gene_fi in tqdm(gene_files):
            gene = gene_fi.split("/")[-1].split(".")[0]
            chrom_gene_genotype_dataframes[chrom][gene] = pd.read_parquet(
                gene_fi
            )

    print("Parse expression files")

    if agg_method == "tmm":
        agg_subdir = "log_tmm"
    elif agg_method == "sum_agg":
        agg_subdir = "sum_agg"
    else:
        raise ValueError

    if dataset_norm == "rint_residual":
        norm_subdir = "invnorm_regression_residuals"
    elif dataset_norm == "rint":
        norm_subdir = "invnorm_log_pseudobulk_matrices"
    elif dataset_norm == "log":
        norm_subdir = "log_pseudobulk_matrices"
    else:
        raise ValueError

    tmm_normalised_pseudobulk_dir = path.join(
        bd,
        f"data/onek1k_cell_type_pseudobulk/{agg_subdir}/log_pseudobulk_matrices"
    )
    tmm_residual_normalised_pseudobulk_dir = path.join(
        bd,
        f"data/onek1k_cell_type_pseudobulk/{agg_subdir}/{norm_subdir}"
    )

    tmm_ct_pseudobulk_dataframes = load_cell_type_expression_dataframes(
        cell_types=cell_types,
        df_dir=tmm_normalised_pseudobulk_dir
    )
    tmm_residual_ct_pseudobulk_dataframes = load_cell_type_expression_dataframes(
        cell_types=cell_types,
        df_dir=tmm_residual_normalised_pseudobulk_dir
    )

    vcf_to_expr_sample_map = {}
    expr_to_vcf_sample_map = {}

    
    for expr_samp in list(tmm_residual_ct_pseudobulk_dataframes[curr_cell_type]):
        vcf_to_expr_sample_map[expr_samp] = expr_samp
        expr_to_vcf_sample_map[expr_samp] = expr_samp

    print("Define split subsets")
    curr_samples = {}
    curr_train_samples = cohort_samples[curr_cohort]["train"]
    n_val = int(round(len(curr_train_samples) * 0.25))
    num_k = 5
    for i in range(num_k):
        random.seed(i)
        curr_samples[f"val_{i}"] = random.sample(curr_train_samples, n_val)
        curr_samples[f"train_{i}"] = [
            item for item in curr_train_samples if item not in curr_samples[f"val_{i}"]
        ]

    curr_samples["test"] = cohort_samples[curr_cohort]["test"]

    count_gt_zero = 0
    count_zero = 0
    count_total = 0

    gene_gt_zero_aaf_diff_dfs = {}

    cross_val_final_dfs = {}
    null_cross_val_final_dfs = {}

    zero_threshold = round((len(curr_train_samples) - n_val) * 0.25)

    cohort_split_low_high_n = {}

    print("Main loop")
    for curr_gene in tqdm(chrom_genes[curr_chrom]):
        try:
            # residual expression
            curr_gene_df = pd.DataFrame(
                tmm_residual_ct_pseudobulk_dataframes[curr_cell_type].loc[curr_gene]
            )
            curr_gene_df.index = [
                expr_to_vcf_sample_map[k] for k in list(curr_gene_df.index)
            ]

            nonresid_curr_gene_df = pd.DataFrame(
                tmm_ct_pseudobulk_dataframes[curr_cell_type].loc[curr_gene]
            )
            nonresid_curr_gene_df.index = [
                expr_to_vcf_sample_map[k] for k in list(nonresid_curr_gene_df.index)
            ]
        except KeyError:
            continue
        train_nonresid = nonresid_curr_gene_df.loc[curr_train_samples]
        n_train_with_zero = len(train_nonresid[train_nonresid[curr_gene] == 0])
        if n_train_with_zero >= zero_threshold:
            continue
        count_total += 1
        high_low_dics = {}
        for cohort_split in list(curr_samples.keys()):
            high_low_dics[cohort_split] = {"low": [], "high": []}
            cohort_split_low_high_n[cohort_split] = {"low": 0, "high": 0}
            # Iterate through the DataFrame
            curr_gene_split_df = curr_gene_df.loc[curr_samples[cohort_split]]
            low_indices = curr_gene_split_df[curr_gene_split_df[curr_gene] <= -expression_threshold].index.tolist()
            high_low_dics[cohort_split]["low"].extend(low_indices)

            high_indices = curr_gene_split_df[curr_gene_split_df[curr_gene] >= expression_threshold].index.tolist()
            high_low_dics[cohort_split]["high"].extend(high_indices)

            cohort_split_low_high_n[cohort_split]["low"] = len(low_indices)
            cohort_split_low_high_n[cohort_split]["high"] = len(high_indices)

        curr_df = chrom_gene_genotype_dataframes[curr_chrom][curr_gene].copy()
        curr_df = curr_df.replace(2,1)

        cross_val_pass_dfs = {}
        null_cross_val_pass_dfs = {}

        for i in range(num_k):
            curr_dfs = {}
            curr_dfs[f"train_{i}"] = curr_df[curr_samples[f"train_{i}"]]
            n_train_samples = len(list(curr_dfs[f"train_{i}"]))
            low_aa_count_cutoff = int(n_train_samples * 0.05)
            high_aa_count_cutoff = n_train_samples - low_aa_count_cutoff
            curr_train_aa_count = curr_dfs[f"train_{i}"].sum(axis=1)

            filtered_train_df = curr_dfs[f"train_{i}"][
                (curr_train_aa_count >= low_aa_count_cutoff) & (curr_train_aa_count <= high_aa_count_cutoff)
            ]

            retained_variants = filtered_train_df.index.tolist()

            delta_aafs = {}
            null_delta_aafs = {}

            for cohort_split in [f"train_{i}", f"val_{i}", "test"]:
                split_df = curr_df[curr_samples[cohort_split]].loc[retained_variants]
                curr_low_aaf = split_df[high_low_dics[cohort_split]["low"]].sum(axis=1) / \
                    split_df[high_low_dics[cohort_split]["low"]].count(axis=1)
                curr_high_aaf = split_df[high_low_dics[cohort_split]["high"]].sum(axis=1) / \
                    split_df[high_low_dics[cohort_split]["high"]].count(axis=1)
                delta_aaf = curr_low_aaf - curr_high_aaf
                delta_aafs[cohort_split] = delta_aaf


                null_low_samples = random.sample(
                    curr_samples[cohort_split],
                    len(high_low_dics[cohort_split]["low"])
                )
                null_high_samples = random.sample(
                    curr_samples[cohort_split],
                    len(high_low_dics[cohort_split]["high"])
                )

                null_low_aaf = split_df[null_low_samples].sum(axis=1) / \
                    split_df[null_low_samples].count(axis=1)
                null_high_aaf = split_df[null_high_samples].sum(axis=1) / \
                    split_df[null_high_samples].count(axis=1)
                null_delta_aaf = null_low_aaf - null_high_aaf
                null_delta_aafs[cohort_split] = null_delta_aaf


            df = pd.DataFrame(delta_aafs)
            train_and_val_pass_df = df[((df[f"train_{i}"] >= aaf_threshold) & (df[f"val_{i}"] >= aaf_threshold)) | ((df[f"train_{i}"] <= -aaf_threshold) & (df[f"val_{i}"] <= -aaf_threshold))]
            cross_val_pass_dfs[i] = train_and_val_pass_df

            null_df = pd.DataFrame(null_delta_aafs)
            null_train_and_val_pass_df = null_df[((null_df[f"train_{i}"] >= aaf_threshold) & (null_df[f"val_{i}"] >= aaf_threshold)) | ((null_df[f"train_{i}"] <= -aaf_threshold) & (null_df[f"val_{i}"] <= -aaf_threshold))]
            null_cross_val_pass_dfs[i] = null_train_and_val_pass_df


        lengths_list = [len(df) for df in cross_val_pass_dfs.values()]
        if any(length == 0 for length in lengths_list):
            count_zero += 1
            continue
        elif len(cross_val_pass_dfs) == 0:
            count_zero += 1
            continue
        else:
            common_index = set(cross_val_pass_dfs[next(iter(cross_val_pass_dfs))].index)
            for df in cross_val_pass_dfs.values():
                common_index = common_index.intersection(df.index)
            if len(common_index) == 0:
                continue
            final_dfs = []
            for k, v in cross_val_pass_dfs.items():
                final_dfs.append(v.loc[list(common_index)])

            final_df = pd.concat(final_dfs, axis=1)
            final_df = final_df.loc[:, ~final_df.T.duplicated(keep='first')]
            print(f"Added df for gene: {curr_gene}")
            count_gt_zero += 1
            cross_val_final_dfs[curr_gene] = final_df

        lengths_list = [len(df) for df in null_cross_val_pass_dfs.values()]
        if any(length == 0 for length in lengths_list):
            count_zero += 1
            continue
        elif len(null_cross_val_pass_dfs) == 0:
            count_zero += 1
            continue
        else:
            common_index = set(null_cross_val_pass_dfs[next(iter(null_cross_val_pass_dfs))].index)
            for df in null_cross_val_pass_dfs.values():
                common_index = common_index.intersection(df.index)
            if len(common_index) == 0:
                continue
            null_final_dfs = []
            for k, v in null_cross_val_pass_dfs.items():
                null_final_dfs.append(v.loc[list(common_index)])

            null_final_df = pd.concat(null_final_dfs, axis=1)
            null_final_df = null_final_df.loc[:, ~null_final_df.T.duplicated(keep='first')]
            print(f"Added NULL df for gene: {curr_gene}")
            print(null_final_df)
            count_gt_zero += 1
            null_cross_val_final_dfs[curr_gene] = null_final_df


    print("Writing dataframes")
    cross_val_df_out_dir = path.join(
        data_dir,
        "cross_validation",
        agg_method,
        dataset_norm,
        f"k_{num_k}",
        curr_cohort,
        f"aaf_threshold_{aaf_threshold}",
        f"expression_threshold_{expression_threshold}",
        curr_cell_type,
        curr_chrom
    )
    makedirs(cross_val_df_out_dir, exist_ok=True)

    for gene_id, df in cross_val_final_dfs.items():
        df.to_csv(
            path.join(cross_val_df_out_dir, f"{gene_id}.tsv"),
            sep="\t"
        )

    null_cross_val_df_out_dir = path.join(
        data_dir,
        "null_cross_validation",
        agg_method,
        dataset_norm,
        f"k_{num_k}",
        curr_cohort,
        f"aaf_threshold_{aaf_threshold}",
        f"expression_threshold_{expression_threshold}",
        curr_cell_type,
        curr_chrom
    )
    makedirs(null_cross_val_df_out_dir, exist_ok=True)

    for gene_id, df in null_cross_val_final_dfs.items():
        df.to_csv(
            path.join(null_cross_val_df_out_dir, f"{gene_id}.tsv"),
            sep="\t"
        )


if __name__ == "__main__":
    main()
