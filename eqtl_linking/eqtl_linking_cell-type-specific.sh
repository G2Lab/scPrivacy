#!/bin/bash
#SBATCH --job-name=eqtl-linking                
#SBATCH --mem=1G 
#SBATCH --time=08:00:00               
#SBATCH --array=1-14
#SBATCH --output=../log/log_tmm/invnorm_regression_residuals/log_%a.log 


dir=/gpfs/commons/groups/gursoy_lab/xli/linking

ct=$(sed -n ${SLURM_ARRAY_TASK_ID}p  ${dir}/data/cell_types.txt)
echo ${ct}

sub_dir="log_tmm/invnorm_regression_residuals"
#"log_tmm/invnorm_log_pseudobulk_matrices"
#"sum_agg/invnorm_log_pseudobulk_matrices"
#"sum_agg/invnorm_regression_residuals"
echo ${sub_dir}


echo "\n### GTEx eQTLs - OneK1K cell-type specific pseudobulk expression"


mkdir ${dir}/output/GTEx_OneK1K/${sub_dir}/${ct}
cd ${dir}/output/GTEx_OneK1K/${sub_dir}/${ct}


PrivaSeq -compute_vulnerable_fraction ${dir}/data/eQTL/GTEx/GTEx_Whole_Blood_converted_eqtl.txt \
  ${dir}/data/genotype/OneK1K/OneK1K_genotype.txt \
  ${dir}/data/expression/OneK1K/${sub_dir}/${ct}.dat \
  ${dir}/data/genotype/OneK1K/OneK1K_genotype_samples.list \
  ${dir}/data/expression/OneK1K/${sub_dir}/${ct}_samples.list \
  ${dir}/data/auxiliary/OneK1K_sex.txt \
  ${dir}/data/auxiliary/OneK1K_pop.txt \
  ${dir}/data/auxiliary/OneK1K_family_id_relationship.txt \
  ${dir}/data/auxiliary/OneK1K_info_samples.txt



echo "\n### GTEx eQTLs - Lupus cell-type specific pseudobulk expression"


mkdir ${dir}/output/GTEx_Lupus/${sub_dir}/${ct}
cd ${dir}/output/GTEx_Lupus/${sub_dir}/${ct}


PrivaSeq -compute_vulnerable_fraction ${dir}/data/eQTL/GTEx/GTEx_Whole_Blood_converted_eqtl.txt \
  ${dir}/data/genotype/Lupus/Lupus_genotype_all.txt \
  ${dir}/data/expression/Lupus/${sub_dir}/${ct}.dat \
  ${dir}/data/genotype/Lupus/Lupus_genotype_all_samples.list \
  ${dir}/data/expression/Lupus/${sub_dir}/${ct}_samples.list \
  ${dir}/data/auxiliary/Lupus_sex.txt \
  ${dir}/data/auxiliary/Lupus_pop.txt \
  ${dir}/data/auxiliary/Lupus_family_id_relationship.txt \
  ${dir}/data/auxiliary/Lupus_info_samples.txt



echo "\n### OneK1K cell-type specific eQTLs - OneK1K cell-type specific pseudobulk expression"

mkdir ${dir}/output/OneK1K_OneK1K/${sub_dir}/${ct}
cd ${dir}/output/OneK1K_OneK1K/${sub_dir}/${ct}

PrivaSeq -compute_vulnerable_fraction ${dir}/data/eQTL/OneK1K/OneK1K_eqtl_${ct}.txt \
  ${dir}/data/genotype/OneK1K/OneK1K_genotype.txt \
  ${dir}/data/expression/OneK1K/${sub_dir}/${ct}.dat \
  ${dir}/data/genotype/OneK1K/OneK1K_genotype_samples.list \
  ${dir}/data/expression/OneK1K/${sub_dir}/${ct}_samples.list \
  ${dir}/data/auxiliary/OneK1K_sex.txt \
  ${dir}/data/auxiliary/OneK1K_pop.txt \
  ${dir}/data/auxiliary/OneK1K_family_id_relationship.txt \
  ${dir}/data/auxiliary/OneK1K_info_samples.txt


echo "\n### OneK1K cell-type specific eQTLs - Lupus cell-type specific pseudobulk expression"

mkdir ${dir}/output/OneK1K_Lupus/${sub_dir}/${ct}
cd ${dir}/output/OneK1K_Lupus/${sub_dir}/${ct}

PrivaSeq -compute_vulnerable_fraction ${dir}/data/eQTL/OneK1K/OneK1K_eqtl_${ct}.txt \
  ${dir}/data/genotype/Lupus/Lupus_genotype_all.txt \
  ${dir}/data/expression/Lupus/${sub_dir}/${ct}.dat \
  ${dir}/data/genotype/Lupus/Lupus_genotype_all_samples.list \
  ${dir}/data/expression/Lupus/${sub_dir}/${ct}_samples.list \
  ${dir}/data/auxiliary/Lupus_sex.txt \
  ${dir}/data/auxiliary/Lupus_pop.txt \
  ${dir}/data/auxiliary/Lupus_family_id_relationship.txt \
  ${dir}/data/auxiliary/Lupus_info_samples.txt

