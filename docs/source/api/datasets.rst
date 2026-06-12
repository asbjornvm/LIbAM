Datasets
========

A number of real-world graphs are bundled with the library and exposed through the
``libam.datasets`` submodule. Importing the submodule gives you ready-to-use
:class:`~libam.datasets.Dataset` objects, with no manual download or unpacking steps required.

The first time you access a dataset, its file is fetched over the network from the
`libam-datasets <https://github.com/asbjorn/libam-datasets>`_ repository,
and cached on disk (under the per-user pooch cache, e.g. ``~/.cache/libam``). 
Subsequent accesses load directly from that cache, so a dataset is only ever downloaded once.

Loading a dataset
-----------------

Every dataset exposes two methods. Use :meth:`~libam.datasets.Dataset.graph` to get the
raw graph as a NetworkX object, or :meth:`~libam.datasets.Dataset.graphpair` to get a
:class:`~libam.graph.GraphPair` ready for alignment:

.. code-block:: python

    from libam import datasets

    # Downloaded and cached on first access, loaded from cache thereafter.
    # NetworkX.Graph, can use to create a GraphPair
    g = datasets.bio_celegans.graph()

    # libam.GraphPair, can be used directly in algorithms
    pair = datasets.bio_celegans.graphpair().permute().add_noise(target_noise=0.05)

Available datasets
------------------

The following datasets are available as attributes of ``libam.datasets``.

Single-graph datasets (structure only), often used to build mirrored graph pairs
:meth:`~libam.datasets.Dataset.graphpair` together with ``permute()`` and ``add_noise()``:

- ``bio_celegans``
- ``bio_dmela``
- ``ca_astro_ph``
- ``ca_erdos992``
- ``ca_gr_qc``
- ``ca_netscience``
- ``in_arenas``
- ``inf_euroroad``
- ``inf_power``
- ``soc_facebook``
- ``soc_hamsterster``
- ``socfb_bowdoin47``
- ``socfb_hamilton46``
- ``socfb_haverford76``
- ``socfb_swarthmore42``

Paired datasets, each holding a source graph, a target graph and a
ground-truth mapping. These include node features unless noted:

- ``cora``
- ``douban``
- ``acm_dblp``
- ``allmv_tmdb``
- ``fb_tw``
- ``ppi``
- ``foursquare`` (no node features)
- ``phone`` (no node features)

The ``Dataset`` class
---------------------

.. autoclass:: libam.datasets.Dataset
   :members:
