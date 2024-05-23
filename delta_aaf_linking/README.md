## Directory contents


**onek1k_crossval_variant_selection_AAF_diffs.py** </br>
Used to select variants that are amenable to prediction/linking by cross-validating differences in allele frequency from a training cohort. This script takes as arguments:
`--train-cohort`: a substring in a file path indicating the training cohort to use for variant selection (e.g. "300" was used to indicate the 300-individual training cohort)
`--aaf-threshold`: a threshold on the alternate allele frequency used in variant cross-validation
`--expression-threshol`: the absolute threshold on determining if an individual falls in the left or right tails of the pseudobulk expression distribution
`--chrom`: the chromosome to process genes from
`--gene-list`: a file containing the set of genes to process, one per line
`--cohort-file`: path to list of training samples, one per line
`--experiment-dir`: base directory to save output relative to
`--pseudobulk-matrix-dir`: directory containing the individual x gene un-normalized pseudobulk matrices
`--norm-pseudobulk-matrix-dir`: directory containing the individual x gene normalized pseudobulk matrices
`--crossval-dir`: directory in which cross-validate variant information is saved
`--null-crossval-dir`: directory in which null cross-validated variants are stored (determined through randomly sampling both sides of the pseudobulk distribution, used to assess significance of variants)


