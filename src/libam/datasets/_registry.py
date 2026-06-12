"""Pooch-backed registry for the bundled datasets.

Dataset files are stored in the libam-datasets GitHub repository. The first time a file is
requested it is downloaded and cached on disk, subsiquent requests are served from the cache.
Set the ``LIBAM_DATA`` environment variable to point the cache at an existing directory of
dataset files (used by the offline examples, which read straight from ``./data``).
"""

import pooch

_REGISTRY = pooch.create(
    path=pooch.os_cache("libam"),
    base_url="https://raw.githubusercontent.com/asbjornvm/libam-datasets/main/",
    env="LIBAM_DATA",
    registry={
        "bio-celegans.txt": "sha256:b45c3d0a1a3002e3b282488769b7147dc0e3000b8206be21cfc90162523d55e8",
        "bio-dmela.txt": "sha256:72a1dc6c579b4d3e4f292e4dd72bae474220b71f58b41cab2f7a95e019af50e6",
        "CA-AstroPh.txt": "sha256:7ee9cc5cb3049e3f33674b0687ca5f31b937a9f4569f6cc90f9ad12fd8f42de0",
        "ca-Erdos992.txt": "sha256:e5b49e3b32b9e67395d586251123aa3c002bb04c520b3636d237b7f051e12c40",
        "ca-GrQc.txt": "sha256:12bb77970e2360cc61eb76088d8b31e0c1e70909f004ea5e6bc33ab037182a70",
        "ca-netscience.txt": "sha256:9eb2e0778ec43000fc97ec36f880755a042b989da5d2a272198f6dcafd2e55fd",
        "in-arenas.txt": "sha256:cec1a687baba65eb73479e79f000d9c1e3f57366ee43baac06eeb027bc320ab2",
        "inf-euroroad.txt": "sha256:ca10603b12fe75ab3dc7a651407d06ffceb0ba41a1fb9ff936d27f0dfa5ffd2e",
        "inf-power.txt": "sha256:0e232930364b9278dde7a8ffed2fcaa783f94d4fbb98f6411dfb7958d135f7b8",
        "soc-facebook.txt": "sha256:91f100e0acc35a940a9a29064c5550911e39143a9d91f0de20bb33a53caeb740",
        "soc-hamsterster.txt": "sha256:a98171b77d6020699f9bdc4f7c7c91fc054a6312ddf28123d96ea3f100b8b069",
        "socfb-Bowdoin47.txt": "sha256:53e60f49e617b65ab2849cf3c2576b1d6728196106b6b7ec6244618c3cf172cc",
        "socfb-Hamilton46.txt": "sha256:8469c3d24a45599307321f9524020b3425f9575f14813a99873a79d0581fd103",
        "socfb-Haverford76.txt": "sha256:b13050e0a069050a02c4253910316029820ef11eba9ce529ae0f80ab47a63f9d",
        "socfb-Swarthmore42.txt": "sha256:68066e43619b0685b69872deb92c95d3999883e9943e1e24c7552bfa58a1b93e",
        "cora/cora_s_edge.txt": "sha256:96a525b75335d0616b58937f248ad43669f01eb0194d499a5c5ad28ab92ccbb9",
        "cora/cora_t_edge.txt": "sha256:35d2307160319147aace5092e6d26906ca343821bb50e90c4627812e834b1434",
        "cora/cora_s_feat.txt": "sha256:3bc41fe9b0ce5c40f39c0bcc95feb03bba1b2ac2a3dd5eafa83de3ba3c3c7190",
        "cora/cora_t_feat.txt": "sha256:6c9cad745d148e60915c5cdd115b4729447b4878f868af4d455d0069ce24049b",
        "cora/cora_ground_True.txt": "sha256:3ca8dc5440a8fc4285d4fd5107e5bab671e8bff0f4af68eb82e3d50ece291238",
        "douban/douban_s_edge.txt": "sha256:e6772481b573d3087687d5885727ca1c5c72576a26ea43db161f4ddb1ffedab7",
        "douban/douban_t_edge.txt": "sha256:0c1748ab8b9d30e1fcf3effbcc28e0e97311f762e7c535cd11454b45b0e1b415",
        "douban/douban_s_feat.txt": "sha256:b5c0db8b72a74d155345cbe1b1f18acb690ba24d39567a22110a740fc30022b9",
        "douban/douban_t_feat.txt": "sha256:d33145f2d1cb155693337ed26e7048e19baf9ce6ecee9df98b59a83bb8499fc1",
        "douban/douban_ground_True.txt": "sha256:866aa71ea7ffe8029dc9e744a37425851f4a6872b1ea960cc365b1f95eb903c9",
        "acm_dblp/acm_dblp_s_edge.txt": "sha256:e2c890a6e18c48f02064d54705d44192a93266761e1edb9a32fd542f8431375f",
        "acm_dblp/acm_dblp_t_edge.txt": "sha256:b008015929e630905fcc55e4b88a9ed1db4d6b4d155f72b42cb6458435a02a49",
        "acm_dblp/acm_dblp_s_feat.txt": "sha256:0ec700d5a515ecf54a438aca6151ecb7b104e63edafc4d86e7a443e3f2d04dcb",
        "acm_dblp/acm_dblp_t_feat.txt": "sha256:2515fb31e97ec31695f92ba87a5bb731546298afb01a6c07c11982d2b3c4c887",
        "acm_dblp/acm_dblp_ground_True.txt": "sha256:f3a15d3bbc2e716ca7268bc751960af2fa523a309de3ad6b90d78c1c2556adc7",
        "allmv_tmdb/allmv_tmdb_s_edge.txt": "sha256:364288e7ca98235f6ceaa13afd371ac97f3313d0a10f6af70a855290c9fc281e",
        "allmv_tmdb/allmv_tmdb_t_edge.txt": "sha256:e796ba9e7246119eec0e92063eee52ade38a8df8ee4ccfed3daed91f3105de67",
        "allmv_tmdb/allmv_tmdb_s_feat.txt": "sha256:a5cd818cecee2704589f33a9f0eed255bd1ea78034000d72f43b943ae2249b0e",
        "allmv_tmdb/allmv_tmdb_t_feat.txt": "sha256:824112b3c01c27cccd9dcce8f63b79d82b1b2863bac7d9d0595c234322640e43",
        "allmv_tmdb/allmv_tmdb_ground_True.txt": "sha256:19033a55d913497b7b8e25d590fc85057906e7efa0f4fceff1429089d960648d",
        "fb_tw/fb_tw_s_edge.txt": "sha256:3b0c39fccc471fa31d3de1b74a7d943ab41c622dfdb17cf7f32e77d59c297988",
        "fb_tw/fb_tw_t_edge.txt": "sha256:ccb469a800637a83900ed3857e2673c3a5174de8ff03c4c168133d7ef8ef3014",
        "fb_tw/fb_tw_s_feat.txt": "sha256:7777fe4d62962557139c463ef0ccbc16c1d31b23151749e4ffa2b1666e03fba2",
        "fb_tw/fb_tw_t_feat.txt": "sha256:7777fe4d62962557139c463ef0ccbc16c1d31b23151749e4ffa2b1666e03fba2",
        "fb_tw/fb_tw_ground_True.txt": "sha256:e5bb8ea2ef470074576082efba623fd1e3492882a612b630deef03f69299a312",
        "ppi/ppi_s_edge.txt": "sha256:12fcdf2d294928497dac775aa22efdb4ebe8aa820c6e155735f628182633ae9e",
        "ppi/ppi_t_edge.txt": "sha256:421dab822e3231f4836f3a46f65f8ba5d68d0e4359c313fd6602413cef87fcd3",
        "ppi/ppi_s_feat.txt": "sha256:c92beee6d8bb4734d29ad0bcd85574b49ebfe98fc2bf15f834ef08aebff45e94",
        "ppi/ppi_t_feat.txt": "sha256:c8b551dfe269f9e9277b5ef2a6791de7debd1d1d255f46aa47b4e7cefa7a117a",
        "ppi/ppi_ground_True.txt": "sha256:ae1a59bd3fe141fb4cd1a802b022a70733c636be8ab0ef8a59da14e92b1121c8",
        "foursquare/foursquare_s_edge.txt": "sha256:a8db8511f9e67e6ba7d584baafea0fcb6053c9940d0eb734ef3d77a0b75d645d",
        "foursquare/foursquare_t_edge.txt": "sha256:4d855fb657b56ab79c773c7ea1e5fcb6ec7d223c83dd287ea9f4c22210868d60",
        "foursquare/foursquare_ground_True.txt": "sha256:0a91ec13bb97fcb9eebcfca24b30e67bac74bd77662e6e2d933310f63d009911",
        "phone/phone_s_edge.txt": "sha256:baec471903ace0b82a376738ead7bf6414e555e8190b7520ab78b8ac3bc0f6d3",
        "phone/phone_t_edge.txt": "sha256:202c672ad8a95e90a6cad3a6811062cc12862ade1613be627c069a00f9eb57cf",
        "phone/phone_ground_True.txt": "sha256:6da74f06998df5e4d7355b72f90ce4f39db0feebe203b683a929903113fb1037",
    },
)


def fetch(filename: str) -> str:
    """Return the local path to ``filename``, downloading and caching it if needed."""
    return _REGISTRY.fetch(filename)