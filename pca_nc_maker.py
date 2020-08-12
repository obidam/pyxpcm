import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)
import pyxpcm
from pyxpcm.models import pcm
# ln -s /Volumes/BSOSE-DISC/bsose_monthly bsose_monthly

def pcm_pca_out(time_i=42, K=4, maxvar=2, min_depth=300, interp=False):
    # Define features to use
    # Instantiate the PCM
    main_dir = '/Users/simon/bsose_monthly/'
    salt = main_dir + 'bsose_i106_2008to2012_monthly_Salt.nc'
    theta = main_dir + 'bsose_i106_2008to2012_monthly_Theta.nc'

    max_depth = 2000
    z = np.arange(-min_depth, -max_depth, -10.)
    features_pcm = {'THETA': z, 'SALT': z}
    features = {'THETA': 'THETA', 'SALT': 'SALT'}
    salt_nc = xr.open_dataset(salt).isel(time=time_i)
    theta_nc = xr.open_dataset(theta).isel(time=time_i)
    big_nc = xr.merge([salt_nc, theta_nc])
    both_nc = big_nc.where(big_nc.coords['Depth'] >
                           max_depth).drop(['iter', 'Depth',
                                            'rA', 'drF', 'hFacC'])

    attr_d = {}

    for coord in both_nc.coords:
        attr_d[coord] = both_nc.coords[coord].attrs

    lons_new = np.linspace(both_nc.XC.min(), both_nc.XC.max(), 60*4)
    lats_new = np.linspace(both_nc.YC.min(), both_nc.YC.max(), 60)
    # ds = both_nc # .copy(deep=True)
    if interp:
        ds = both_nc.interp(coords={'YC': lats_new, 'XC': lons_new})#, method='cubic')
    else:
        ds = both_nc

    m = pcm(K=K, features=features_pcm,
            maxvar=maxvar,
            timeit=True, timeit_verb=1)
    ds = m.add_pca_to_xarray(ds, features=features, dim='Z', inplace=True)

    #m.fit(ds, features=features, dim='Z') #, inplace=True)
    #m.predict(ds, features=features, dim='Z', inplace=True)
    #m.predict_proba(ds, features=features, dim='Z', inplace=True)
    #m.find_i_metric(ds, inplace=True)

    def sanitize():
        #    del ds.PCM_LABELS.attrs['_pyXpcm_cleanable']
        #    del ds.PCM_POST.attrs['_pyXpcm_cleanable']
        #    del ds.PCM_RANK.attrs['_pyXpcm_cleanable']
        del ds.PCA_VALUES.attrs['_pyXpcm_cleanable']

    for coord in attr_d:
        ds.coords[coord].attrs = attr_d[coord]

    sanitize()

    ds = ds.drop(['SALT', 'THETA'])

    ds = ds.expand_dims(dim='time', axis=None)

    ds = ds.assign_coords({"time":
                ("time", [salt_nc.coords['time'].values])})

    ds.coords['time'].attrs = salt_nc.coords['time'].attrs


    ds.to_netcdf('nc/pca/'+str(time_i)+'.nc', format='NETCDF4')
    m.to_netcdf('nc/m_pca/'+str(time_i)+'.nc')


def run_through_pca():
    for time_i in range(60):
        pcm_pca_out(time_i=time_i)


def merge_whole_density_netcdf():

    pca_ds = xr.open_mfdataset('nc/pca/*.nc',
                               concat_dim="time",
                               combine='by_coords',
                               chunks={'time': 1},
                               data_vars='minimal',
                               # parallel=True,
                               coords='minimal',
                               compat='override')

    # this is too intense for memory

    return pca_ds


def save_density_netcdf(pca_ds):

    xr.save_mfdataset([pca_ds], ['nc/pcm_pca.nc'], format='NETCDF4')


def take_derivative_pca(dimension="YC", typ='float32'):

    chunk_d = {'time': 1, 'YC': 588, 'XC': 2160}

    density_ds = xr.open_mfdataset('nc/pcm_pca.nc',
                                   # engine=engine,
                                   # decode_cf=False,
                                   chunks=chunk_d,
                                   combine='by_coords',
                                   data_vars='minimal',
                                   coords='minimal',
                                   compat='override',
                                   parallel=True
                                   ).astype(typ)

    grad_da = density_ds.PCA_VALUES.differentiate(dimension)
    #.astype(typ).chunk(chunks=chunk_d)

    name = 'PC_Gradient_' + dimension
    grad_ds = grad_da.to_dataset().rename_vars({'PCA_VALUES': name})
    grad_ds[name].attrs['long_name'] = 'PC Gradient ' + dimension
    grad_ds[name].attrs['units'] = 'box-1'

    # .astype(typ).chunk(chunks=chunk_d)
    xr.save_mfdataset([grad_ds],
                      ['nc/pc_grad_' + dimension + '.nc'],
                      format='NETCDF4')


def go_through_all():
    
    run_through_pca()
    pca_ds = merge_whole_density_netcdf()
    save_density_netcdf(pca_ds)
    take_derivative_pca()
    take_derivative_pca(dimension="XC")
