#!/usr/bin/env python

import argparse
import glob
import pandas as pd
import random

from os import path, makedirs
from tqdm import tqdm
from collections import defaultdict


parser = argparse.ArgumentParser()


parser.add_argument(
    "--cell-type",
    help="e.g. CD4_NC, B_IN, etc.",
    default="CD4_NC",
    required=False,
    type=str,
    dest="cell_type",
)

parser.add_argument(
    "--aaf-threshold",
    help="e.g. 0.1, 0.5",
    default=0.1,
    required=True,
    type=float,
    dest="aaf_threshold",
)

parser.add_argument(
    "--expression-threshold",
    help="e.g. 0.1, 0.5",
    default=0.1,
    required=True,
    type=float,
    dest="expression_threshold",
)

parser.add_argument(
    "--chromosome",
    help="1, 2, ...",
    default="",
    required=True,
    type=str,
    dest="chromosome",
)


parser.add_argument(
    "--crossval-dir",
    help="Path to a directory to output cross-validated SNV files in",
    required=True,
    type=str,
    dest="crossval_df_dir",
)

parser.add_argument(
    "--experiment-dir",
    help="Path to a directory to output files in",
    required=True,
    type=str,
    dest="experiment_dir",
)

parser.add_argument(
    "--pseudobulk-matrix-dir",
    help="Path to a directory containing the log pseudobulk dataframes",
    required=True,
    type=str,
    dest="pseudobulk_matrix_dir",
)

parser.add_argument(
    "--norm-pseudobulk-matrix-dir",
    help="Path to a directory containing the normalized log pseudobulk dataframes",
    required=True,
    type=str,
    dest="norm_pseudobulk_matrix_dir",
)

parser.add_argument(
    "--true-genotype-dir",
    help="Path to a directory containing true genotype dataframes",
    required=True,
    type=str,
    dest="true_genotype_dir",
)

parser.add_argument(
    "--pred-genotype-dir",
    help="Path to a directory containing predicted genotype dataframes",
    required=True,
    type=str,
    dest="pred_genotype_dir",
)

parser.add_argument(
    "--haplotype-df-dir",
    help="Path to a directory containing predicted genotype dataframes",
    required=True,
    type=str,
    dest="haplotype_df_dir",
)

parser.add_argument(
    "--sample-file",
    help="Path to a file containing a new-line-delimited list of samples",
    required=True,
    type=str,
    dest="sample_file",
)


def create_df_for_prediction(cell_type_tsvs):
    df_dic = {
        "gene": [],
        "chromosome": [],
        "position": [],
        "mean_aaf_diff": [],
    }
    for tsv_file in tqdm(cell_type_tsvs):
        if tsv_file.endswith(".tsv"):
            gene = tsv_file.split("/")[-1][:-4]
        else:
            raise ValueError(f"File {tsv_file} does not end with '.tsv'")
        with open(tsv_file, "r") as f:
            for line in f.readlines()[1:]:
                line = line.strip().split("\t")
                line = [value for value in line if value != ""]
                for i in range(len(line)-1):
                    #print(line)
                    line[i+1] = float(line[i+1])

                # assumes the test val is in position 4 of the line
                del line[3]

                chrom, pos = line[0].split("_")
                del line[0]
                
                mean_aaf_diff = sum(line) / len(line)
                mean_aaf_diff = round(mean_aaf_diff, 3)
                df_dic["gene"].append(gene)
                df_dic["chromosome"].append(chrom)
                df_dic["position"].append(pos)
                df_dic["mean_aaf_diff"].append(mean_aaf_diff)

    return pd.DataFrame(df_dic)


def main():
    args = parser.parse_args()
    expression_threshold = args.expression_threshold
    aaf_threshold = args.aaf_threshold
    data_dir = args.experiment_dir

    haplotype_dir = args.haplotype_df_dir

    expression_dir = args.norm_pseudobulk_matrix_dir
    nonresid_expression_dir = args.pseudobulk_matrix_dir

    true_out_dir = args.true_genotype_dir
    prediction_out_dir = args.pred_genotype_dir

    # create output directories
    makedirs(true_out_dir, exist_ok=True)
    makedirs(prediction_out_dir, exist_ok=True)


    # load samples present in expression and genotype data
    with open(args.sample_file, "r") as f:
        samples = [i.strip() for i in f.readlines()]

    print("Parsing cross-validated variants...")
    # parse cross validated positions
    cell_type_tsvs = glob.glob(
        path.join(args.crossval_df_dir, f"chromosome_{args.chromosome}/*")
    )

    pred_df = create_df_for_prediction(cell_type_tsvs)

    # define genes for processing
    unique_genes = pred_df["gene"].unique().tolist()

    # load relevant genotypes
    haplotype_files = glob.glob(
        path.join(haplotype_dir, f"chromosome_{args.chromosome}/*")
    )
    haplotype_file_dic = {}
    for haplo_file in haplotype_files:
        if haplo_file.endswith(".parquet"):
            gene_id = haplo_file.split("/")[-1][:-8]
        else:
            raise ValueError(f"File {haplo_file} does not end with '.parquet'")
        haplotype_file_dic[gene_id] = haplo_file

    print("Parsing haplotype dataframes...")
    gene_haplotype_dfs = {}
    unique_genes_in_cohort = []
    for gene in tqdm(unique_genes):
        try:
            gene_haplotype_dfs[gene] = pd.read_parquet(haplotype_file_dic[gene])
            unique_genes_in_cohort.append(gene)
        except KeyError:
            print(f"{gene}: no variant .parquet found for this cohort.")
            continue

    # load expression
    print("Parsing expression data...")
    ct_expression_fi = path.join(expression_dir, f"{args.cell_type}.tsv")
    ct_expression_df = pd.read_csv(ct_expression_fi, index_col=0, header=0, sep="\t")
    nonresid_ct_expression_fi = path.join(
        nonresid_expression_dir, f"{args.cell_type}.tsv"
    )
    nonresid_ct_expression_df = pd.read_csv(
        nonresid_ct_expression_fi, index_col=0, header=0, sep="\t"
    )

    # filter expression to only genes with cross-validated SNPs
    ct_expression_df = ct_expression_df.loc[unique_genes_in_cohort][samples]
    nonresid_ct_expression_df = nonresid_ct_expression_df.loc[unique_genes_in_cohort][samples]

    thresh = args.expression_threshold

    predictions = []
    variant_ids = []

    gene_pos_dic = defaultdict(list)
    gene_id_dic = defaultdict(list)
   
    print("Predicting variants...")
    for index, variant_row in pred_df.iterrows():
        gene = variant_row["gene"]        
        chrom = variant_row["chromosome"]
        pos = variant_row["position"]
        mean_aaf_diff = variant_row["mean_aaf_diff"]
        variant_id = f"{gene}_{chrom}_{pos}"
        variant_pos = f"{chrom}_{pos}"
        if gene not in unique_genes_in_cohort:
            continue
        if variant_pos not in gene_haplotype_dfs[gene].index:
            continue

        if mean_aaf_diff < 0:
            lambda_func = lambda x: 0 if x < -thresh else (1 if x > thresh else ".")
        else:
            lambda_func = lambda x: 1 if x < -thresh else (0 if x > thresh else ".")

        variant_expression = ct_expression_df.loc[gene]
        variant_nonresid_expression = nonresid_ct_expression_df.loc[gene]

        variant_prediction = variant_expression.apply(lambda_func)
        variant_prediction[variant_nonresid_expression == 0] = "."


        gene_pos_dic[gene].append(variant_pos)
        gene_id_dic[gene].append(variant_id)

        predictions.append(list(variant_prediction))
        variant_ids.append(variant_id)


    print("Retrieving true haplotypes...")
    true_genotype_df = pd.DataFrame(columns=samples)

    for gene in unique_genes_in_cohort:
        gene_vars = gene_pos_dic[gene]
        gene_haplotype_df_expr_samples = gene_haplotype_dfs[gene][samples]
        gene_true_genotypes = gene_haplotype_df_expr_samples.loc[gene_vars]
        true_genotype_df = pd.concat(
            [true_genotype_df, gene_true_genotypes], ignore_index=False
        )
    
    # will happen if the query dataset, e.g. Lupus, did not have any of these genes
    # for the given chromosome in the single cell data
    if len(true_genotype_df) == 0:
        print("No expression data for this cohort from this chromosome")
        exit()

    haplo_map = {"0|0": 0, "0|1": 1, "1|0": 1, "1|1": 1}
    true_genotype_df.replace(haplo_map, inplace=True)

    predicted_genotype_df = pd.DataFrame(predictions)
    predicted_genotype_df.columns = samples
    predicted_genotype_df.index = variant_ids

    predicted_genotype_df = predicted_genotype_df
    true_genotype_df = true_genotype_df
    
    true_index = list(true_genotype_df.index)
    predicted_index = list(predicted_genotype_df.index)
    predicted_index_for_check = ["_".join(i.split("_")[1:]) for i in predicted_index]

    assert true_index == predicted_index_for_check
    
    true_genotype_df.index = predicted_index

    true_genotype_df.to_csv(
        path.join(true_out_dir, f"chromosome_{args.chromosome}.tsv"),
        sep="\t"
    )
    predicted_genotype_df.to_csv(
        path.join(prediction_out_dir, f"chromosome_{args.chromosome}.tsv"),
        sep="\t"
    )


if __name__ == "__main__":
    main()

