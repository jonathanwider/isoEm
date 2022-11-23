import os.path
import numpy as np
import pickle
import json
import torch
import hashlib
from sklearn.model_selection import train_test_split


def flatten(l):
    """
    flatten a 2d list into a 1d list
    """
    return [item for sublist in l for item in sublist]


def check_dict_conditions(dic, conditions, use_prints=False):
    """
    Test whether dict "dic" fullfills the given conditions "conditions".
    The latter are given as a array of key-value pairs, values may be lists/tuples.
    If prints==True, information on mismatche can be printed.
    """
    for key, value in conditions.items():
        if key in dic.keys():
            if not np.array(dic[key] == value).all():
                if use_prints:
                    print("Difference:", key, "Original:", dic[key], "Condition:", value)
                return False
        else:
            if key != "RESULTS_INTERPOLATED":
                if use_prints:
                    print("Key not contained in dataset:", key)
                return False
    return True


def test_if_folder_exists(dir_name):
    """
    Test if a given filename already exists. Because files are named by hashes this
    implies that we already created a file for the given configuration
    """
    if os.path.exists(dir_name):
        return True
    else:
        return False


class jsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, type) or isinstance(obj, torch.device):
            return str(type)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_hash_from_description(description):
    """
    Create a hash value that can than be used to identify duplicates of folders.
    """
    # to be able to hash, we need tuples instead of lists and frozensets instead of dicts.
    p = json.dumps(description, ensure_ascii=False, sort_keys=True, indent=None, separators=(',', ':'), cls=jsonEncoder)
    return hashlib.sha256(p.encode('utf-8')).hexdigest()[:30]


def get_years_months(t, units, calendar):
    """
    The data sets come with different time units and calendars.
    We write a function that returns the 'true' time steps.
    @param t: Timesteps to be converted
    @param units: String containing a description of the time units used. Can be extracted via netCDF4.
    @param calendar: String containing a description of the calendar used. Can be extracted via netCDF4.
    @return: 
    """
    if units == "months since 801-1-15 00:00:00":
        years = 801 + t // 12
        months = 0 + t % 12
    elif units == "days since 0801-01-30 00:00:00.000000":
        if calendar == "360_day":
            y_ref = 801
            m_ref = 0
            years = y_ref + (t // 360)  # in the dataset the year (2954) corresponds to year 850
            months = (t+m_ref*30) % 360 // 30
    elif units == "month as %Y%m.%f":
        timesteps = [t_.split(".")[0] for t_ in t.astype(str)]
        years = [int(t_[:-2]) for t_ in timesteps]
        months = [int(t_[-2:])-1 for t_ in timesteps]  # want months to start with zero instead of one
    elif units == "day as %Y%m%d.%f":
        timesteps = [t_.split(".")[0] for t_ in t.astype(str)]
        years = [int(t_[:-4]) for t_ in timesteps]
        months = [int(t_[-4:-2])-1 for t_ in timesteps]  # want months to start with zero instead of one
    elif units == "months since 850-1-15 00:00:00":
        years = 850 + t // 12
        months = 0 + t % 12
    elif units == "days since 0850-01-01 00:00:00":
        if calendar == "365_day":
            years = 850 + (t-1) // 365  # data gets saved at the end of each month.
                                        # Thus subtract one, so that data gets saved in same year
            ds = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 0]
            months = [np.argmin(np.abs(ds - t_ % 365)) for t_ in t]
    elif units == "days since 2350-12-01 00:00:00":
        if calendar == "360_day":
            y_ref = 654
            m_ref = 11
            years = - y_ref + 850 + (t // 360)  # in the dataset the year (2954) corresponds to year 850
            months = (t+m_ref*30) % 360 // 30
    elif units == "days since 0000-01-01 00:00:00":
        return None, None
    else:
        raise NotImplementedError("Invalid date format")
    return np.array(years), np.array(months)


def add_dates(y1, m1, y2, m2):
    m_res = (m1 + m2) % 11
    y_res = y1 + y2 + (m1 + m2) // 11
    return y_res, m_res


def get_year_mon_day_from_timesteps(time_steps, ref_date):
    """
    Extract the month (0: January,..., 11: December) of given timesteps, assuming the time is given as days
    since the reference date and a 360 day calendar with month length 30.

    @param time_steps: Array of timesteps
    @param ref_date: Reference date
    @return: Array containing month of each timestep.
    """
    ref_year = ref_date.year
    ref_month = ref_date.month - 1  # months in calendar 1-12, we want 0-11
    ref_day = ref_date.day

    ref_n_days = ref_year*360 + ref_month*30 + ref_day

    total_days = ref_n_days + time_steps

    year = total_days // 360
    month = (total_days - year * 360) // 30
    day = (total_days - year * 360 - month * 30) // 1
    return year, month, day


def train_test_split_by_indices(description, variable):
    assert "INDICES_TRAIN" in description.keys()
    assert "INDICES_TEST" in description.keys()
    return variable[description["INDICES_TRAIN"], ...], variable[description["INDICES_TEST"], ...]


def train_test_split_by_proportion(description, variable, seed):
    assert "TEST_FRACTION" in description.keys()
    assert "DO_SHUFFLE" in description.keys()
    return train_test_split(variable, test_size=description["TEST_FRACTION"], random_state=seed, shuffle=description["DO_SHUFFLE"])


def load_longitudes_latitudes(description, dset):
    assert "latitude" in dset.variables.keys() or "lat" in dset.variables.keys()
    assert "longitude" in dset.variables.keys() or "lon" in dset.variables.keys()
    try:
        lat = tuple(dset.variables["latitude"][description["LATITUDES_SLICE"][0]:description["LATITUDES_SLICE"][1]].data)
    except KeyError:
        lat = tuple(dset.variables["lat"][description["LATITUDES_SLICE"][0]:description["LATITUDES_SLICE"][1]].data)
    try:
        lon = tuple(dset.variables["longitude"][:].data)
    except KeyError:
        lon = tuple(dset.variables["lon"][:].data)
    return lat, lon


def load_units_cals(description, dsets):
    """
    Extract time units and calendars of a list of datasets.
    @param description: Dataset description
    @param dsets: List of datasets
    @return: Lists of units and calendars
    """
    try:
        units = [ds.variables["t"].units for ds in dsets if ds.variables["t"][:].data.shape[0] > 1]
        cals = [ds.variables["t"].calendar for ds in dsets if ds.variables["t"][:].data.shape[0] > 1]
    except KeyError:
        units = [ds.variables["time"].units for ds in dsets if ds.variables["time"][:].data.shape[0] > 1]
        cals = [ds.variables["time"].calendar for ds in dsets if ds.variables["time"][:].data.shape[0] > 1]

    return units, cals


def cartesian_to_spherical(data):
    """
    convert cartesian coordinates to spherical coordinates
    Use answer to:
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    # takes list xyz (single coord)
    x = data[..., 0]
    y = data[..., 1]
    z = data[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    # format in HadCM3: lat:(-90,90), lon(0,360)
    theta = 90 - np.arccos(z / r) * 180 / np.pi  # to degrees
    phi = 180 + np.arctan2(y, x) * 180 / np.pi
    return np.array([theta, phi]).transpose((1, 0))  # careful, this will only work if the shape is correct