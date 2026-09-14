"""Bind released normalization before constructing the RLDS transformation graph."""
from copy import deepcopy


def bind_release_statistics(dataset_kwargs, statistics):
    result = deepcopy(dataset_kwargs)
    for entry in result:
        name = entry['name']
        if name not in statistics:
            raise ValueError(f'Missing released normalization for {name}')
        stats = statistics[name]
        for field, dim in (('action', 7), ('proprio', 8)):
            for quantile in ('q01', 'q99'):
                if len(stats.get(field, {}).get(quantile, [])) != dim:
                    raise ValueError(f'Invalid {name} {field}.{quantile} released normalization')
        if stats.get('num_transitions', 0) <= 0:
            raise ValueError(f'Invalid released sample count for {name}')
        entry['dataset_statistics'] = deepcopy(stats)
    return result
