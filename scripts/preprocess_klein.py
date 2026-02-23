import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path

adata = ad.read_h5ad('/rds/user/wz369/hpc-work/pseudodynamics_plus/data/klein/klein_addpop.h5ad')

# Map timepoints 2.0 → 0, 4.0 → 1, 6.0 → 2
tp_map = {2.0: 0, 4.0: 1, 6.0: 2}
adata.obs['samples'] = adata.obs['timepoint_tx_days'].astype(float).map(tp_map).astype(int)

# Split train (Well != 2) and test (Well == 2)
train_mask = adata.obs['Well'].astype(int) != 2
test_mask  = adata.obs['Well'].astype(int) == 2

representations = {
    'klein_pca50': ('X_pca', 50),
    'klein_pca30': ('X_pca', 30),
    'klein_dm5':   ('DM_EigenVectors_multiscaled', 5),
    'klein_dm10':  ('DM_EigenVectors', 10),
}

out_dir = Path('data/klein')
for name, (obsm_key, dim) in representations.items():
    for split, mask in [('train', train_mask), ('test', test_mask)]:
        X = adata[mask].obsm[obsm_key][:, :dim]
        samples = adata[mask].obs['samples'].values
        cols = ['samples'] + [f'x{i+1}' for i in range(dim)]
        df = pd.DataFrame(np.column_stack([samples, X]), columns=cols)
        df['samples'] = df['samples'].astype(int)
        df.to_csv(out_dir / f'{name}_{split}.csv', index=False)
        print(f"Saved {name}_{split}.csv: {df.shape}")
