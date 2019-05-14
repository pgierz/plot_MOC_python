#!/usr/bin/env python3
"""
A script adaption of Thomas Rackow's AMOC Plotting Notebooks, for batch use.

Dr. Paul Gierz
May 2019
"""

import argparse
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
except:
        raise ImportError("You need numpy, pandas, pyfesom, and xarray for this!")

def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate FESOM MOC from output data.')
        parser.add_argument('outdata_dir', help='The output directory of FESOM')
        parser.add_argument('mesh_dir', help='The mesh directory use for this run')
        parser.add_argument('mask_file', default='', nargs='?', 
                        help='If provided, uses only these element IDs to calculate the MOC (Default is to use all elements in the file')
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
        mf_ds = xr.open_mfdataset(sorted(l), chunks={'time': 12}, concat_dim='time') 
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


if __name__ == '__main__':
        print(80*"-")
        print("S T A R T   O F   calculate_FESOM_MOC.py".center(80))
        master_start = datetime.datetime.now()
        # Get user arguments
        args = parse_arguments()
        print("\n")
        print("-"*5+"|", "(1)   L O A D I N G   D A T A")
        # Load the required data
        print(" "*4, "*   Loading velocity data...")
        vertical_velocity = load_velocity(args.outdata_dir)
        print(" "*4, "*   Loading mesh...")
        start_loadmesh = datetime.datetime.now()
        mesh = pf.load_mesh(args.mesh_dir)
        if (datetime.datetime.now() - start_loadmesh) > datetime.timedelta(seconds=2):
                print(" "*4, "    ...loaded mesh in %s" % (datetime.datetime.now() - start_loadmesh))
        else:
                print(" "*4, "    ...done.")

        # Determine if the user gave a mask:
        if args.mask_file:
                with open(args.mask_file) as f:
                        node_ids = [int(l) for l in f.read().splitlines()]
        else:
                node_ids = slice(None)

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

        for mesh_property in ['mesh.path', 'mesh.alpha', 'mesh.beta', 'mesh.gamma', 'mesh.n2d', 'mesh.e2d', 'mesh.n3d']:
                print("     - INFO : %s: %s" % (mesh_property, getattr(mesh, mesh_property.split(".")[-1])))

        # Which domain to integrate over:
        nlats = 180 #90 #180 #1800
        lats = np.linspace(-89.95,89.95,nlats)
        dlat = lats[1]-lats[0]

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

        # Compute lon/lat coordinate of an element required later for binning
        elem_x = nodes_x[el_nodes].sum(axis=1)/3.
        elem_y = nodes_y[el_nodes].sum(axis=1)/3.

        # Number of local vertical levels for every grid point
        nlevels_local = np.sum(mesh.n32[:, :]!=-999, axis=1)
        hittopo_at = np.min(nlevels_local[el_nodes[:,0:3]], axis=1) - 1 # hit topo at this level

        # Compute positions of elements in zonalmean-array for binning
        pos = ((elem_y - lats[0]) / dlat).astype('int')
        if (datetime.datetime.now() - start_mesh_props) > datetime.timedelta(seconds=2):
                print("Done in %s" % (datetime.datetime.now() - start_mesh_props))
        else:
                print("Done.")

        print("\n")
        print("-"*5+"|", "(4)   A L L O C A T E   B I N N E D   M O C   A R R A Y")
        # Allocate binned array
        binned = np.zeros((nlevels, nlats)) * np.nan

        print("\n")
        print("-"*5+"|", "(5)   C R U N C H I N G   T H E   I N T E G R A T I O N")
        for lev in range(nlevels):
                W = pf.fesom2depth(depth[lev], vertical_velocity.wo, mesh, verbose=False)
                # How big is W now?
                # print("-   INFO: W.shape = ", W.shape)
                # Compute Area Weighted mean over the elements:
                W_elem_mean_weight = el_area * np.mean(W[el_nodes[:, 0:3]], axis=1)

                # Select the elements in the mask (default will take all node ids)
                # W_elem_mean_weight = W_elem_mean_weight[node_ids]

                # Convert to Sv
                W_elem_mean_weight *= 10.**-6
                # Set to 0 where the bathymetery starts:
                W_elem_mean_weight[hittopo_at] = 0.

                # For every bin, select elements that lie within the bin, and then sum them:
                for bins in range(pos.min(), pos.max()+1):
                        # Generate which indices to use for the calculation:
                        # elements at this poition, and enough levels:
                        indices = np.logical_and(pos==bins, lev<=hittopo_at)
                        # Select out the elements you want to actually integrate
                        # print("-   INFO: indicies has shape", indices.shape)
                        if np.sum(indices) == 0:
                                binned[lev, bins] = np.nan #  Only topography at this bin
                        else:
                                binned[lev, bins] = np.nansum(W_elem_mean_weight[indices])
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
                                'Latitude': (['Latitude'], lats),
                                'Depth': (['Depth'], depth,),
                                }
                        )
        binned_ds.to_netcdf("test.nc")

        print("\n")
        print("F I N I S H E D !".center(80))
        print(80*"-")
        print("Total time %s" % (datetime.datetime.now() - master_start))
