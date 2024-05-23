import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sc.settings.verbosity = 2
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(4, 4), facecolor='white', fontsize=13)


lupus_adata = sc.read_h5ad("/gpfs/commons/groups/gursoy_lab/xli/Lupus/data/GSE174188_CLUES1_adjusted.h5ad")
onek1k_adata = sc.read_h5ad( "/gpfs/commons/groups/gursoy_lab/xli/OneK1K/data/OneK1K_980Indiv_counts.h5ad")


### Preprocess Onek1k count data

sc.pp.filter_cells(onek1k_adata, min_genes=200)
sc.pp.filter_genes(onek1k_adata, min_cells=3)

onek1k_adata = onek1k_adata[onek1k_adata.obs.nFeature_RNA < 3500, :]
onek1k_adata = onek1k_adata[onek1k_adata.obs['percent.mt'] < 9, :]

# Normalize 
sc.pp.normalize_total(onek1k_adata)
sc.pp.log1p(onek1k_adata)

# Filter by shared genes (highly variable genes in Lupus_
var_names = onek1k_adata.var_names.intersection(lupus_adata.var_names) #1996 shared genes 
onek1k_adata = onek1k_adata[:, var_names]
lupus_adata = lupus_adata[:, var_names]

# Regress and scale
sc.pp.regress_out(onek1k_adata, ['nCount_RNA', 'percent.mt'])
sc.pp.scale(onek1k_adata, max_value=10)


### Correct for batch effect

all_concat = onek1k_adata.concatenate(lupus_adata, batch_categories=['onek1k', 'lupus'])

sc.pp.combat(all_concat, key='batch')

onek1k_combat = all_concat[all_concat.obs['batch'] == 'onek1k']
lupus_combat = all_concat[all_concat.obs['batch'] == 'lupus']


### Onek1k: scanpy clustering 
# Train the model and graph (here PCA, neighbors, UMAP) on the reference data Onek1k

sc.pp.pca(onek1k_combat)
sc.pp.neighbors(onek1k_combat, n_neighbors=10, n_pcs=40)
sc.tl.umap(onek1k_combat)

sc.pl.umap(onek1k_combat, color='new_names', title=['Onek1k (filtered, processed, corrected): onek1k_labels'], legend_loc='on data', legend_fontsize=8, show=False)
plt.savefig("plots/onek1k_corrected_scanpy.png", bbox_inches='tight')


### Map labels and embeddings from Onek1k to Lupus using ingest

sc.tl.ingest(lupus_combat, onek1k_combat, obs='new_names')
lupus_combat.uns['new_names_colors'] = onek1k_combat.uns['new_names_colors']  # fix colors

sc.pl.umap(lupus_combat, color=['new_names', 'cg_cov'], title=['Lupus (corrected, ingest): onek1k_labels', 'lupus_labels'], legend_loc='on data', legend_fontsize=8, show=False)
plt.savefig('plots/lupus_corrected_ingest.png', bbox_inches='tight')


### Test batch-effect

new_concat = onek1k_combat.concatenate(lupus_combat, batch_categories=['onek1k', 'lupus'])
new_concat.obs.new_names = new_concat.obs.new_names.astype('category')
new_concat.obs.new_names.cat.reorder_categories(lupus_combat.obs.new_names.cat.categories)  # fix category ordering
new_concat.uns['new_names_colors'] = lupus_combat.uns['new_names_colors']  # fix category colors

sc.pl.umap(new_concat, color=['batch', 'new_names'], title=['dataset', 'onek1k_labels'], legend_loc='on data', legend_fontsize=8, show=False)
plt.savefig('plots/corrected_integrated_batch.png', bbox_inches='tight')


### Save data

#new_concat.write('data/concat.h5ad')
onek1k_combat.write('data/OneK1K_corrected_integrated.h5ad')
lupus_combat.write('data/Lupus_corrected_integrated.h5ad')




