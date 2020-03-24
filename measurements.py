#!/usr/bin/env python3

import pandas as pd

def clamp_to_monotonic_increasing(table):
    # I can't find an efficient pandas way to do an arbitrary fold-then-map or (shift; zip-then-map)
    z = table.iloc[0, :]
    for i in range(1, table.shape[0]):
        table.iloc[i, :].clip(lower=z, inplace=True)
        z = table.iloc[i, :]

def extract_measurements(table, config):
    """Match out the row for the country, get rid of the initial metadata
    columns and drop any leading or trailing NaN values
    """
    return table.loc[table['Country/Region'] == config['country']].loc[:, config['start_index']:].dropna(axis=1).sum(axis=0)

def reached_n_infections(meas, n):
    """Return information about when the given country reached n confirmed
    cases.
    """
    return int(max(0, meas.shape[0] - meas.loc[meas['Confirmed'] > n - 1, :].shape[0])), 'Confirmed', n

def reached_n_deaths(meas, n):
    """Return information about when the given country reached n deaths.
    """
    return int(max(0, meas.shape[0] - meas.loc[meas['Dead'] > n - 1, :].shape[0])), 'Dead', n

def truncate_initial(meas, config):
    """Truncate the measurement set before the earliest date which meets the
    infection or death threshold set above.
    """
    confirmedt0 = reached_n_infections(meas, config['start_at_infections'])
    deatht0 = reached_n_deaths(meas, config['start_at_deaths'])
    return min(confirmedt0, deatht0, key=lambda x: x[0])

def form_table(config, confirmed, recovered, dead):
    """Form a combined table with all the statistics for a single country,
    including correct column labels for easier indexing and plotting later.
    """
    c0 = extract_measurements(confirmed, config)
    r0 = extract_measurements(recovered, config)
    d0 = extract_measurements(dead, config)
    data = pd.concat([c0, r0, d0], axis=1)
    data.columns = ['Confirmed', 'Recovered', 'Dead']
    newt0, reason, limit = truncate_initial(data, config)
    # Provide info about how the start of the data window was chosen.
    country = config['country']
    print(f'{country} reached {limit} "{reason}" on {data.index[newt0]}; dropping preceding data')
    ret = data.iloc[newt0:, :]
    clamp_to_monotonic_increasing(ret)
    return ret

def display_measurements(measurements, populations, name):
    title=f'Coronavirus over time in {name} (total population {populations[name]})'
    measurements.plot(grid=True,
                      title=title, style=['-', '--', ':'])
