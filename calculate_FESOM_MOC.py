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

try:
    import numpy as np
    import pyfesom as pf
    import xarray as xr
except:
    raise ImportError("You need numpy, pyfesom, and xarray for this!")


# Set up logging to be a useful info
step_number = 1


def show_step(message):
    """ Prints a message about what step is currently being done """
    global step_number
    print("-"*5+"|", "("+step_number+")   "+"".join([c+" " for c in message.upper().replace(" ", 3*" ")]))
    step_number += 1


def show_substep(message):
    """ Prints a message about the part of the step currently being done """
    print(" "*4, "*   ", message.capitalize(), "...")


def set_up_logging(loglevel):
    """ Sets up logging formatting """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level,
                        format='     - %(levelname)s : %(message)s')

def time_function_execution(func):
    """ Decorates a function so that it prints information about execution time """
    def timed_func(*args, **kwargs):
        starting_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        if (datetime.datetime.now() - starting_time) > datetime.timedelta(seconds=3):
            print(" "*4, "    ...finished in %s" % (datetime.datetime.now() - starting_time))
        else:
            print(" "*4, "    ...finished.")
        return result
    return timed_func


# Determine arguments:
def parse_arguments():
    """
    Parse user arguments to the script

    Gets the user arguments. Currently, the following arguments are **required**:
    #. outdata_dir : Where the outdata is stored
    #. mesh_dir : Where pyfesom can find information regarding the used mesh
    #. ofile : The path and name of the file to save the results to
    The following arguments are optional:
    #. mask_file : only calculate MOC for a particular region
    """
    parser = argparse.ArgumentParser(description='Generate FESOM MOC from output data.')

    parser.add_argument(
        'outdata_dir',
        help='The output directory of FESOM'
        )
    parser.add_argument(
        'mesh_dir',
        help='The mesh directory use for this run'
        )
    parser.add_argument(
        'ofile',
        help='Name and path of the output file with the calculated MOC'
        )
    parser.add_argument(
        'mask_file',
        default='',
        nargs='?',
        help='If provided, uses only these element IDs to calculate the MOC (Default is to use all elements in the file)'
        )
    parser.add_argument(
        '--log',
        default='ERROR',
        help="Logging level. Choose from DEBUG, INFO, ERROR"
        )

    return parser.parse_args()


@time_function_execution
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
    regex = re.compile(filepattern)
    # FIXME: This rewrites the list for every level you go down...?
    for root, _, files in os.walk(outdata_dir):
        matching_files = [os.path.join(root, x) for x in files if regex.match(x)]
    mf_ds = xr.open_mfdataset(sorted(matching_files)[-360:], chunks={'time': 12}, concat_dim='time')
    return mf_ds


@time_function_execution
def load_mesh(mesh_path):
    """Wrapper function to load the mesh with timing"""
    return pf.load_mesh(mesh_path, usepickle=False)


@time_function_execution
def load_mask(mesh, mask_file=None):
    """Loads the mask provided by the user, or gives back a True mask for all elements"""
    if mask_file:
        with open(args.mask_file) as user_mask_for_moc:
            element_ids = [int(l-1) for l in user_mask_for_moc.read().splitlines()]
        return np.array([1 if element_id in element_ids else 0 for element_id in range(mesh.elems)])
    return np.ones(mesh.elems)


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
    filepattern = r'wo_fesom_\d+.nc'
    return _load_fesom_mfdataset(filepattern, outdata_dir)

if __name__ == '__main__':
    master_start = datetime.datetime.now()
    print(80*"-")
    print("S T A R T   O F   calculate_FESOM_MOC.py".center(80))

    # Get user arguments
    args = parse_arguments()
    set_up_logging(args.log)

    show_step('loading data')
    show_substep('loading velocity data')
    vertical_velocity = load_velocity(args.outdata_dir)
    show_substep('loading mesh')
    mesh = load_mesh(args.mesh_dir)
    show_substep('loading mask (if provided)')
    mask = load_mask(mesh, args.mask_file)

    if args.calculate_mean:
        print("\n")
        show_step('calculating means')
        vertical_velocity = vertical_velocity.mean(dim='time').compute()
        print("    ...done.")

    # Which domain to integrate over:
    total_nlats = 90
    lats = np.linspace(-89.95, 89.95, total_nlats)
    dlat = lats[1] - lats[0]

    # Get values needed in the computation
    nlevels = np.shape(mesh.zlevs)[0]

    # Compute lon/lat coordinate of an element required later for binning
    elem_x = mesh.x2[mesh.elem].sum(axis=1)/3.
    elem_y = mesh.y2[mesh.elem].sum(axis=1)/3.

    # Number of local vertical levels for every grid point
    nlevels_local = np.sum(mesh.n32[:, :] != -999, axis=1)
    hittopo_at = np.min(nlevels_local[mesh.elem[:,0:3]], axis=1) - 1 # hit topo at this level

    # Compute positions of elements in zonalmean-array for binning
    pos = ((elem_y - lats[0]) / dlat).astype('int')

    # Allocate binned array
    binned = np.zeros((nlevels, total_nlats)) * np.nan

    show_step('crunching the integration')
    # To start from the bottom, use reversed(range(nlevels)):
    for lev in range(nlevels):
        show_substep("Binning for level: %s" % mesh.zlevs[lev])
        # Make the weighted average in the dataset:
        W = pf.fesom2depth(mesh.zlevs[lev], vertical_velocity.wo, mesh, verbose=False)

        logging.info("Generating weighted mean.")
        W_elem_mean_weight = mesh.voltri * np.mean(W[mesh.elem[:, 0:3]], axis=1)

        logging.info("Selecting which elements the user specified.")
        logging.info("If nothing was specified, use all elements.")
        W_elem_mean_weighted_masked = W_elem_mean_weight * mask
        # Convert to Sv
        W_elem_mean_weight *= 10.**-6
        # Set to 0 where the bathymetery starts:
        W_elem_mean_weight[hittopo_at] = 0.

        for bins in range(pos.min(), pos.max()+1):
            indices = np.logical_and(pos == bins, lev <= hittopo_at)
            if np.sum(indices) == 0.:
                binned[lev, bins] = np.nan # only topo
            else:
                binned[lev, bins] = np.nansum(W_elem_mean_weight[indices])

    # Generate cummulative sum:
    binned = np.ma.cumsum(np.ma.masked_invalid(binned), axis=1)

    show_step('saving results')
    binned_ds = xr.Dataset(
        {
            "MOC": (('Depth', 'Latitude'), binned,)
        },
        coords={
            'Latitude': (['Latitude'], np.linspace(-89.95, 89.95, total_nlats)),
            'Depth': (['Depth'], -mesh.zlevs,),
        }
    )
    binned_ds.to_netcdf(args.ofile)

    print("\n")
    print("F I N I S H E D !".center(80))
    print(80*"-")
    print("Total time %s" % (datetime.datetime.now() - master_start))
