from functools import partial
from libam.datasets._dataset import Dataset
from libam.datasets._loaders import load_edge_list, load_alpine
from libam.datasets._parsers import parse_edge_list, parse_alpine


def _edge_list_dataset(filename: str) -> Dataset:
    """Build a Dataset form a single-file edge list"""
    return Dataset(
        filename,
        loader=load_edge_list,
        parser=partial(parse_edge_list),
    )


def _alpine_dataset(name: str, features: bool) -> Dataset:
    """Build a Dataset for an Alpine-format graph pair living under ``{name}/``."""
    members = [f"{name}_s_edge.txt", f"{name}_t_edge.txt"]
    if features:
        members += [f"{name}_s_feat.txt", f"{name}_t_feat.txt"]
    members.append(f"{name}_ground_True.txt")
    return Dataset(
        f"{name}/",
        loader=lambda path: load_alpine(path, name=name, features=features),
        parser=partial(parse_alpine),
        members=members,
    )


bio_celegans = _edge_list_dataset("bio-celegans.txt")
bio_dmela = _edge_list_dataset("bio-dmela.txt")
ca_astro_ph = _edge_list_dataset("CA-AstroPh.txt")
ca_erdos992 = _edge_list_dataset("ca-Erdos992.txt")
ca_gr_qc = _edge_list_dataset("ca-GrQc.txt")
ca_netscience = _edge_list_dataset("ca-netscience.txt")
in_arenas = _edge_list_dataset("in-arenas.txt")
inf_euroroad = _edge_list_dataset("inf-euroroad.txt")
inf_power = _edge_list_dataset("inf-power.txt")
soc_facebook = _edge_list_dataset("soc-facebook.txt")
soc_hamsterster = _edge_list_dataset("soc-hamsterster.txt")
socfb_bowdoin47 = _edge_list_dataset("socfb-Bowdoin47.txt")
socfb_hamilton46 = _edge_list_dataset("socfb-Hamilton46.txt")
socfb_haverford76 = _edge_list_dataset("socfb-Haverford76.txt")
socfb_swarthmore42 = _edge_list_dataset("socfb-Swarthmore42.txt")

cora = _alpine_dataset("cora", features=True)
douban = _alpine_dataset("douban", features=True)
acm_dblp = _alpine_dataset("acm_dblp", features=True)
allmv_tmdb = _alpine_dataset("allmv_tmdb", features=True)
fb_tw = _alpine_dataset("fb_tw", features=True)
ppi = _alpine_dataset("ppi", features=True)
foursquare = _alpine_dataset("foursquare", features=False)
phone = _alpine_dataset("phone", features=False)