#!/bin/bash
#SBATCH --job-name=linking-all-cells                  
#SBATCH --mem=1G 
#SBATCH --time=04:00:00               
#SBATCH --output=../log/log_all.log 


dir=/gpfs/commons/groups/gursoy_lab/xli/linking
echo '#### All Cells'

#sub_dir="original"
sub_dir="reprocessed"


echo "\n### GTEx eQTLs - OneK1K all-cells pseudobulk"

mkdir ${dir}/output/GTEx_OneK1K/${sub_dir}/all
cd ${dir}/output/GTEx_OneK1K/${sub_dir}/all

PrivaSeq -compute_vulnerable_fraction ${dir}/data/eQTL/GTEx/GTEx_Whole_Blood_converted_eqtl.txt \
  ${dir}/data/genotype/OneK1K/OneK1K_genotype.txt \
  ${dir}/data/expression/OneK1K/${sub_dir}/allCells.dat \
  ${dir}/data/genotype/OneK1K/OneK1K_genotype_samples.list \
  ${dir}/data/expression/OneK1K/${sub_dir}/allCells_samples.list \
  ${dir}/data/auxiliary/OneK1K_sex.txt \
  ${dir}/data/auxiliary/OneK1K_pop.txt \
  ${dir}/data/auxiliary/OneK1K_family_id_relationship.txt \
  ${dir}/data/auxiliary/OneK1K_info_samples.txt 


echo "\n### GTEx eQTLs - Lupus all-cells pseudobulk"

mkdir ${dir}/output/GTEx_Lupus/${sub_dir}/all
cd ${dir}/output/GTEx_Lupus/${sub_dir}/all

PrivaSeq -compute_vulnerable_fraction ${dir}/data/eQTL/GTEx/GTEx_Whole_Blood_converted_eqtl.txt \
  ${dir}/data/genotype/Lupus/Lupus_genotype_all.txt \
  ${dir}/data/expression/Lupus/${sub_dir}/allCells.dat \
  ${dir}/data/genotype/Lupus/Lupus_genotype_all_samples.list \
  ${dir}/data/expression/Lupus/${sub_dir}/allCells_samples.list \
  ${dir}/data/auxiliary/Lupus_sex.txt \
  ${dir}/data/auxiliary/Lupus_pop.txt \
  ${dir}/data/auxiliary/Lupus_family_id_relationship.txt \
  ${dir}/data/auxiliary/Lupus_info_samples.txt 




