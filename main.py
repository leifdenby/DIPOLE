import drmethods
import xarray as xr
import ipdb

filepath_emb = "/home/jovyan/embs_train_triplets10k.nc"

da_emb = xr.open_dataarray(filepath_emb)

da = da_emb.sel(tile_type="anchor").isel(tile_id=slice(0, 5000))
data = da.values * 1.0e3

embedding_dipole = drmethods.DIPOLE_mask_method(
    high_ptcloud=data,
    target_dim=2,
    lmr_edges=32,
    alpha=0.1,
    k=64,
    lr=1.0
)

da_dipole = xr.DataArray(embedding_dipole, dims=("tile_id", "dipole_dim"), coords=dict(tile_id=da.tile_id))

filepath_out = filepath_emb.replace(".nc", ".dipole.lmr_edges32.nc")
da_dipole.to_netcdf(filepath_out)
print(f"Saved output to `{filepath_out}`")
