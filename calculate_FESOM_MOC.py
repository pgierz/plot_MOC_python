#!/usr/bin/env python3
"""
A script adaption of Thomas Rackow's AMOC Plotting Notebooks, for batch use.

Dr. Paul Gierz
May 2019
"""

import argparse
import logging
import datetime
import os
import re
import sys

try:
        from dask.diagnostics import ProgressBar
        import numpy as np
        import pandas as pd
        import pyfesom as pf
        import xarray as xr
        import matplotlib.pyplot as plt
except:
        raise ImportError("You need numpy, pandas, pyfesom, and xarray for this!")


# Set up logging to be a useful info
def set_up_logging(loglevel):
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level,
                        format='     - %(levelname)s : %(message)s')


def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate FESOM MOC from output data.')
        parser.add_argument('outdata_dir', help='The output directory of FESOM')
        parser.add_argument('mesh_dir', help='The mesh directory use for this run')
        parser.add_argument('mask_file', default='', nargs='?', 
                        help='If provided, uses only these element IDs to calculate the MOC (Default is to use all elements in the file')
        parser.add_argument('--log', default='ERROR', help="Logging level. Choose from DEBUG, INFO, ERROR")
        return parser.parse_args()


def _load_fesom_mfdataset(filepattern, outdata_dir):
        """
        Loads all FESOM files matching a particular pattern in a directory

        Parameters
        ----------
        filepattern : str
                The pattern to match
        outdata_dir : str
                Where to look

        Returns
        -------
        xarray.Dataset
                The dataset object with all information.
        """
        starting_time = datetime.datetime.now()
        r = re.compile(filepattern)
        for root, dirs, files in os.walk(outdata_dir):
                l = [os.path.join(root,x) for x in files if r.match(x)]
        # NOTE: Open all the vertical velocity data. This might take time
        # depending on how big the experiment is:
        mf_ds = xr.open_mfdataset(sorted(l)[-360:], chunks={'time': 12}, concat_dim='time') 
        if (datetime.datetime.now() - starting_time) > datetime.timedelta(seconds=3):
                print(" "*4, "    ...loaded %s files in %s" % (len(l), datetime.datetime.now() - starting_time))
        else:
                # print("    ...loaded %s files in %s" % (len(l), datetime.datetime.now() - starting_time))
                print(" "*4, "    ...loaded %s files." % len(l))
        return mf_ds


def load_velocity(outdata_dir):
        """
        Load all the FESOM velocities.

        Parameters
        ----------
        outdata_dir : str
                Where to look for the files

        Returns
        -------
        xarray.Dataset
                The dataset object with all the vertical velocity information.
        """
        # NOTE: We make a filepattern to match velocity files. This is assuming
        # coupled simulations using ESM-Tools:
        filepattern = 'wo_fesom_\d+.nc'
        return _load_fesom_mfdataset(filepattern, outdata_dir)

def mesh_to_dataset(mesh):
        """
        Generates an xarray dataset out of the mesh attributes; useful for
        merging this with other datarrays to attach mesh information to a
        dataset
        """


if __name__ == '__main__':
        master_start = datetime.datetime.now()
        print(80*"-")
        print("S T A R T   O F   calculate_FESOM_MOC.py".center(80))
        # Get user arguments
        args = parse_arguments()
        set_up_logging(args.log)
        print("\n")
        print("-"*5+"|", "(1)   L O A D I N G   D A T A")
        # Load the required data
        print(" "*4, "*   Loading velocity data...")
        vertical_velocity = load_velocity(args.outdata_dir)
        print(" "*4, "*   Loading mesh...")
        start_loadmesh = datetime.datetime.now()
        mesh = pf.load_mesh(args.mesh_dir, usepickle=False)
        if (datetime.datetime.now() - start_loadmesh) > datetime.timedelta(seconds=2):
                print(" "*4, "    ...loaded mesh in %s" % (datetime.datetime.now() - start_loadmesh))
        else:
                print(" "*4, "    ...done.")

        

        # Determine if the user gave a mask:
        if args.mask_file:
                with open(args.mask_file) as f:
                        element_ids = [int(l) for l in f.read().splitlines()]
        else:
                element_ids = slice(None)

        print("\n")
        print("-"*5+"|", "(2)   C A L C U L A T I N G   M E A N S")
        try:
                vertical_velocity = xr.open_dataset("w.nc")
                print(" "*4, "*   Loaded from file w.nc")
                print(" "*4, "    ...done.")
        except:
                # Test for now: make time means:
                with ProgressBar():
                        vertical_velocity = vertical_velocity.mean(dim='time').compute()
                        # Save the result to speed up development:
                        vertical_velocity.to_netcdf("w.nc")
                print("    ...done.")

        print("\n")
        print("-"*5+"|", "(3)   D E T E R M I N G   M E S H   P R O P E R T I E S")
        start_mesh_props = datetime.datetime.now()

        print(" "*4, "*   Determing mesh properties for binning...")
        for mesh_property in [
                        'mesh.path',
                        'mesh.alpha',
                        'mesh.beta',
                        'mesh.gamma',
                        'mesh.n2d',
                        'mesh.e2d',
                        'mesh.n3d',
                        'mesh.x2',
                        'mesh.y2',
                        'mesh.zlevs',
                        ]:
                logging.info("%s: %s" % (mesh_property, getattr(mesh, mesh_property.split(".")[-1])))

        # Which domain to integrate over:
        total_nlats = 90 #90 #180 #1800
        lats = np.linspace(-89.95,89.95,total_nlats)
        logging.info("lats: %s", lats)
        dlat = lats[1]-lats[0]
        logging.info("dlat: %s", dlat)

        # Get values needed in the computation
        nlevels = np.shape(mesh.zlevs)[0]
        depth = mesh.zlevs

        # The triangle areas
        el_area = mesh.voltri

        # Node indices for every element
        el_nodes = mesh.elem

        # Mesh coordinates
        nodes_x = mesh.x2 # -180 ... 180
        nodes_y = mesh.y2 # -90 ... 90
        logging.info('nodes_x.shape: %s', nodes_x.shape)
        logging.info('nodes_y.shape: %s', nodes_y.shape)

        # Compute lon/lat coordinate of an element required later for binning
        elem_x = nodes_x[el_nodes].sum(axis=1)/3.
        elem_y = nodes_y[el_nodes].sum(axis=1)/3.

        # Number of local vertical levels for every grid point
        nlevels_local = np.sum(mesh.n32[:, :]!=-999, axis=1)
        hittopo_at = np.min(nlevels_local[el_nodes[:,0:3]], axis=1) - 1 # hit topo at this level

        print(" "*4, "*   Attaching mesh information to velocity data...")

        # Compute positions of elements in zonalmean-array for binning
        pos = ((elem_y - lats[0]) / dlat).astype('int')
        if (datetime.datetime.now() - start_mesh_props) > datetime.timedelta(seconds=2):
                print(" "*9, "...done in %s" % (datetime.datetime.now() - start_mesh_props))
        else:
                print(" "*9, "...done.")

        print("\n")
        print("-"*5+"|", "(4)   A L L O C A T E   B I N N E D   M O C   A R R A Y")
        # Allocate binned array
        binned = np.zeros((nlevels, total_nlats)) * np.nan

        print("\n")
        print("-"*5+"|", "(5)   C R U N C H I N G   T H E   I N T E G R A T I O N")
        # To start from the bottom, use reversed(range(nlevels)):
        for lev in range(nlevels):
                logging.info("Binning for level: %s", mesh.zlevs[lev])
                W_ds = xr.Dataset(
                        {
                                "vertical_velocity": (
                                        ['nodes_2d'], 
                                        pf.fesom2depth(depth[lev], vertical_velocity.wo, mesh, verbose=False)),
                                "Latitude_nodes": (
                                        ['nodes_2d'],
                                        mesh.y2),
                                "Latitude_elements": (
                                        ['elements_2d'],
                                        elem_y),
                                "Longitude_nodes": (
                                        ['nodes_2d'],
                                        mesh.x2),
                                "Longitude_elements": (
                                        ['elements_2d'],
                                        elem_x),
                                'Bathymetry_elements': (
                                        ['elements_2d'],
                                        hittopo_at),
                        },
                        coords = {
                                'nodes_2d': (['nodes_2d'], range(mesh.n2d)),
                                'elements_2d': (['elements_2d'], range(mesh.e2d))
                                },
                        attrs = {
                                'vertical_velocity': {
                                        'units': 'm/s'
                                        },
                                'Latitude': {
                                        'units': 'degrees North'
                                        },
                                'Longitude': {
                                        'units': 'degrees East'
                                        },
                                'Bathymetry': {
                                        'comment': 'Level where the bathymetry starts.'
                                        }
                                },
                        )
                # Make the weighted average in the dataset:
                W = pf.fesom2depth(depth[lev], vertical_velocity.wo, mesh, verbose=False)
                W_elem_mean_weight = el_area * np.mean(W[el_nodes[:, 0:3]], axis=1)
                # Convert to Sv
                W_elem_mean_weight *= 10.**-6
                # Set to 0 where the bathymetery starts:
                W_elem_mean_weight[hittopo_at] = 0.

                # Store this in the array with the other information and convert to Sv:
                W_ds['vertical_volume_transport'] = xr.DataArray(
                        W_elem_mean_weight,
                        coords=[range(mesh.e2d)],
                        dims=['elements_2d'],
                        attrs = {'units': 'm**3/s'}
                        )

                logging.debug("W_ds: %s", W_ds)

                logging.info("Selecting which elements the user specified")
                logging.info("If nothing was specified, use all elements.")
                # Apply the mask specifying which **NODES** to select, and only
                # take values that are above the bathymetry:
                mask = W_ds.elements_2d.isin(element_ids)
                logging.debug(mask)
                W_elem_mean_weigh = W_ds.vertical_volume_transport.where(mask)
                for bins in range(pos.min(), pos.max()+1):
                        indices=np.logical_and(pos==bins, lev<=hittopo_at)
                        if np.sum(indices)==0.:
                                binned[lev, bins]=np.nan # only topo
                        else:
                                binned[lev, bins]=np.nansum(W_elem_mean_weigh[indices])

        # How big is the binned array now?
        # print("-   INFO: binned.shape = ", binned.shape)
        # Generate cummulative sum:
        binned = np.ma.cumsum(np.ma.masked_invalid(binned), axis=1)

        print("\n")
        print("-"*5+"|", "(6)   S A V I N G   R E S U L T S")
        binned_ds = xr.Dataset(
                        {
                                "MOC": (('Depth', 'Latitude'), binned,)
                                },
                        coords={
                                'Latitude': (['Latitude'], np.linspace(-89.95,89.95,total_nlats)),
                                'Depth': (['Depth'], -depth,),
                                }
                        )
        binned_ds.to_netcdf("test.nc")

        f, ax = plt.subplots(1, 1)
        binned_ds.MOC.plot.contourf(ax=ax, levels=range(-2, 26, 2))
        f.savefig("test.png")
        print("\n")
        print("F I N I S H E D !".center(80))
        print(80*"-")
        print("Total time %s" % (datetime.datetime.now() - master_start))
