#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse

from os import path, makedirs
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Any, Tuple


parser = argparse.ArgumentParser()

parser.add_argument(
    "--cell-type",
    required=True,
    type=str,
    nargs="+",
    dest="cell_type",
    help="Cell type to use for linking."
)

parser.add_argument(
    "--aaf-threshold",
    required=True,
    type=float,
    dest="aaf_threshold",
)

parser.add_argument(
    "--expression-threshold",
    required=True,
    type=float,
    dest="expression_threshold",
)

parser.add_argument(
    "--k",
    required=False,
    default=5,
    type=int,
    dest="k",
)

parser.add_argument(
    "--experiment-dir",
    help="Path to a directory to output files in",
    required=True,
    type=str,
    dest="experiment_dir",
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
    "--auxiliary-tsv",
    required=False,
    type=str,
    dest="auxiliary_tsv",
    help="A TSV containing auxillary information for each sample."
)

# as a string to simplify job loop
parser.add_argument(
    "--freq-norm",
    required=False,
    default="none",
    type=str,
    dest="freq_norm",
)

parser.add_argument(
    "--cohort-file",
    help="Path to a file containing train/test sample split",
    required=True,
    type=str,
    dest="cohort_file",
)

parser.add_argument(
    "--results-dir",
    help="Path to a directory to output linking files in",
    required=True,
    type=str,
    dest="results_dir",
)



def linking_score(
        predicted_genotypes: np.ndarray, genotype_testset: pd.DataFrame, 
) -> Dict[str, float]:
    """Calculate linking scores between predicted genotypes and a test set.

    Parameters:
        predicted_genotypes (np.ndarray):
            Array of predicted genotypes.
        genotype_testset (pd.DataFrame):
            DataFrame containing the test set genotypes.

    Returns:
        Dictionary containing sample indices as keys and corresponding linking scores
        as values.
    """
    np.seterr(divide="ignore")
    n_samples = genotype_testset.shape[0]
    # Convert DataFrame to NumPy array for efficient computation
    genotypes_matrix = genotype_testset.to_numpy()
    # Create a mask for the predicted genotypes
    predicted_genotypes_mask = np.tile(predicted_genotypes, (n_samples, 1))
    # Create masks for matching genotypes
    matching_genotype_mask = (genotypes_matrix == predicted_genotypes_mask)
    # Count matching genotypes across samples
    count_across_samples = matching_genotype_mask.sum(axis=0)
    # Calculate scores
    scores = -np.log2(count_across_samples / n_samples)
    scores = np.where(np.isfinite(scores), scores, 0)
    # Sum the scores for each sample
    final_scores = np.dot(matching_genotype_mask, scores)
    # Convert to dictionary
    score_dic = pd.Series(final_scores, index=genotype_testset.index).to_dict()
    np.seterr(divide='warn')
    return score_dic


def linking_score_no_frequency_norm(
        predicted_genotypes: np.ndarray, genotype_testset: pd.DataFrame
) -> Dict[str, float]:
    """Calculate linking scores between predicted genotypes and a test set.

    Parameters:
        predicted_genotypes (np.ndarray):
            Array of predicted genotypes.
        genotype_testset (pd.DataFrame):
            DataFrame containing the test set genotypes.

    Returns:
        Dictionary containing sample indices as keys and corresponding linking scores
        as values.
    """
    np.seterr(divide="ignore")
    n_samples = genotype_testset.shape[0]
    # Convert DataFrame to NumPy array for efficient computation
    genotypes_matrix = genotype_testset.to_numpy()
    # Create a mask for the predicted genotypes
    predicted_genotypes_mask = np.tile(predicted_genotypes, (n_samples, 1))
    # Create masks for matching genotypes
    matching_genotype_mask = (genotypes_matrix == predicted_genotypes_mask)
    # Count matching genotypes across samples
    count_across_samples = matching_genotype_mask.sum(axis=0)
    count_across_samples = -np.log2(count_across_samples)
    # Sum the scores for each sample
    final_scores = np.dot(matching_genotype_mask, count_across_samples)
    # Convert to dictionary
    score_dic = pd.Series(final_scores, index=genotype_testset.index).to_dict()
    np.seterr(divide='warn')
    return score_dic



def map_norm_subdir(norm_method: str) -> str:
    """Map norm_method to the corresponding subdirectory.

    Parameters:
        norm_method (str):
            The normalization method.

    Returns:
        Subdirectory corresponding to the normalization method.
    """
    if norm_method == "rint_residual":
        norm_subdir = "invnorm_regression_residuals"
    elif norm_method == "rint":
        norm_subdir = "invnorm_log_pseudobulk_matrices"
    elif norm_method == "log":
        norm_subdir = "log_pseudobulk_matrices"
    return norm_subdir


def read_cohort_file(
    cohort_file_path
) -> Tuple[List[str], List[str]]:
    """Read train and test samples from a cohort file.

    Parameters:
        cohort_file_path (str):
            Path to a file containing train/test cohort info.
    Returns:
        Tuple containing lists of train_samples and test_samples.
    """
    train_samples, test_samples = [], []
    with open(cohort_file_path, "r") as f:
        for line in f.readlines()[1:]:
            line = line.strip().split("\t")
            if line[-1] == "train":
                train_samples.append(line[0])
            else:
                test_samples.append(line[0])
        # used for case where all samples are used for train and test for benchmarking
        if len(test_samples) == 0:
            test_samples = train_samples[:]
    return train_samples, test_samples


def read_genotypes(
    ct_true_genotype_dir: str,
    ct_pred_genotype_dir: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Read genotypes from true and predicted genotype directories.

    Parameters:
        ct_true_genotype_dir (str):
            Directory path containing true genotype files for a given cell type.
        ct_pred_genotype_dir (str):
            Directory path containing predicted genotype files for a given cell type.

    Returns:
        Tuple containing true_genotypes and pred_genotypes dictionaries.
    """
    true_genotypes = pd.DataFrame()
    pred_genotypes = pd.DataFrame()

    for chrom in tqdm(range(1, 21)):
        try:
            # Load true genotypes
            true_chrom_fi = path.join(ct_true_genotype_dir, f"chromosome_{chrom}.tsv")
            true_chrom_df = pd.read_csv(true_chrom_fi, sep="\t", index_col=0)
            true_genotypes = pd.concat(
                [true_genotypes, true_chrom_df], ignore_index=False
            )
            # Load predicted genotypes
            pred_chrom_fi = path.join(ct_pred_genotype_dir, f"chromosome_{chrom}.tsv")
            pred_chrom_df = pd.read_csv(
                pred_chrom_fi, sep="\t", index_col=0, low_memory=False
            )
            pred_genotypes = pd.concat(
                [pred_genotypes, pred_chrom_df], ignore_index=False
            )
        except Exception as e:
            print(f"Error processing chromosome {chrom}: {e}")
            continue

    return true_genotypes, pred_genotypes


def create_test_only_genotypes(
    true_genotypes: Dict[str, pd.DataFrame],
    pred_genotypes: Dict[str, pd.DataFrame],
    test_samples: List[str]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Create test-only dataframes from true and predicted genotypes.

    Parameters:
        true_genotypes (Dict[str, pd.DataFrame]):
            Dictionary of true genotypes.
        pred_genotypes (Dict[str, pd.DataFrame]):
            Dictionary of predicted genotypes.
        test_samples (List[str]):
            List of samples for which test-only dataframes are created.

    Returns:
        Tuple containing true_genotypes_test_only and pred_genotypes_test_only
        dataframes.
    """
    df_pred = pred_genotypes.copy()
    pred_genotypes_test_only = df_pred[test_samples]
    df_true = true_genotypes.copy()
    true_genotypes_test_only = df_true[test_samples]
    return true_genotypes_test_only, pred_genotypes_test_only


def main():
    # Parse CLI arguments to get cohort information
    args = parser.parse_args()

    results_dir = args.results_dir
    makedirs(results_dir, exist_ok=True)

    variant_selection_dir = args.experiment_dir

    ct_true_genotype_dir = args.true_genotype_dir
    ct_pred_genotype_dir = args.pred_genotype_dir

    # Define output paths
    gap_plot_out_file = path.join(results_dir, "gap_scores.tsv")
    correct_count_accuracy_out_file = path.join(
        results_dir, "correct_incorrect_accuracy.tsv"
    )

    # Define train and test samples for cohort
    _, test_samples = read_cohort_file(args.cohort_file)

    # Read auxillary information if specified
    if args.auxiliary_tsv:
        auxiliary_df = pd.read_csv(args.auxiliary_tsv, sep="\t")
        auxiliary_df = auxiliary_df.set_index("sample")
        print(auxiliary_df)

    # Read true and predicted genotypes
    true_genotypes, pred_genotypes = read_genotypes(
        ct_true_genotype_dir, ct_pred_genotype_dir
    )

    # Prune genotypes to test dataset only genotypes
    true_genotypes_test_only, pred_genotypes_test_only = create_test_only_genotypes(
        true_genotypes=true_genotypes,
        pred_genotypes=pred_genotypes,
        test_samples=test_samples
    )

    print(true_genotypes_test_only)
    print(pred_genotypes_test_only)

    # Fixed values for now defining which test datbase and which query genotypes to use
    test_database = "Predicted genotypes"
    which_query = "True genotypes"

    # Tally correctly/incorrectly linked samples
    correct = 0
    incorrect = 0

    # Retain gap and correct sample info
    gaps = []
    true_sample_gaps = []
    is_sample_correct = []
    linked_sample_ids = []
    n_test_samples = []

    # Process data from each cell type
    # Conditional to handle currently fixed test_database values
    if test_database == "Predicted genotypes":
        df = pred_genotypes_test_only.T
    elif test_database == "True genotypes":
        df = true_genotypes_test_only.T
    else:
        raise ValueError
    
    # Try to link each sample
    for sample in tqdm(test_samples, desc="Linking samples"):
        # Conditional to handle currently fixed which_query values
        if which_query == "True genotypes":
            query_arr = true_genotypes_test_only[sample]
        elif which_query == "Predicted genotypes":
            query_arr = pred_genotypes_test_only[sample]
        else:
            raise ValueError

        if args.freq_norm == "none":
            linked = linking_score_no_frequency_norm(query_arr.astype(str), df)
        else:
            linked = linking_score(query_arr.astype(str), df)

        if args.auxiliary_tsv:
            sample_aux_values = auxiliary_df.loc[sample]
            matching_indices = auxiliary_df[
                auxiliary_df.eq(sample_aux_values).all(axis=1)
            ].index.tolist()
            filtered_indices = list(set(matching_indices) & set(test_samples))

            linked = {key: value for key, value in linked.items() if key in filtered_indices}

        # Get array of linking scores
        linked_arr = np.asarray(list(linked.values()))

        # Get indices of linked samples
        linked_indices = list(np.where(linked_arr == linked_arr.max())[0])
        top_linked_individuals = [
            str(i) for i in list(map(list(linked.keys()).__getitem__, linked_indices))
        ]
        # Get top linked sample ID
        top_linked_individual = top_linked_individuals[0]
        
        # Note if the correct sample was linked
        if sample == top_linked_individual:
            anno = "correct"
            correct += 1
        else:
            anno = "incorrect"
            incorrect += 1

        # Retain annotation of if the sample is correctly linked
        is_sample_correct.append(anno)
        linked_sample_ids.append(top_linked_individual)
        n_test_samples.append(len(linked_arr))

        if len(linked_arr) > 1:
            # Note the highest and second highest linking scores
            sorted_linked_arr = sorted(linked_arr)
            tophit = sorted_linked_arr[-1]
            tophit2 = sorted_linked_arr[-2]

            # Retain the gap score between the highest-scoring and second
            # highest-scoring sample
            gap_score = float(tophit) / tophit2
            gaps.append(gap_score)

            # Retain the gap score between the true sample and highest-scoring sample
            try:
                true_sample_gap_score = float(tophit) / linked[sample]
            except ZeroDivisionError:
                true_sample_gap_score = 0
            true_sample_gaps.append(true_sample_gap_score)
        else:
            gaps.append(0)
            true_sample_gaps.append(0)

    # Create a dataframe for writing that contains gap information and a label
    # indicating if the correct sample was linked
    gap_plot_df = pd.DataFrame(
        {
            "linked_sample": linked_sample_ids,
            "gaps": gaps,
            "true_sample_gap_score": true_sample_gaps,
            "correct": is_sample_correct,
            "test_database_n": n_test_samples
        }
    )
    gap_plot_df.index = test_samples

    # Save gap information to tsv
    gap_plot_df.to_csv(
        gap_plot_out_file,
        sep="\t"
    )

    # Save file with correct & incorrect samples, & overall accuracy -- used for
    # quick referencing without tallying tsv
    accuracy = correct / float(correct + incorrect)
    with open(correct_count_accuracy_out_file, "w") as f:
        f.write(f"Correct\tIncorrect\tAccuracy\n{correct}\t{incorrect}\t{accuracy}")



if __name__ == "__main__":
    main()
