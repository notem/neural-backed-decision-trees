from nbdt.utils import DATASETS, METHODS, Colors, fwd
from nbdt.graph import (
    build_semirandom_graph,
    prune_single_successor_nodes,
    augment_graph,
    build_induced_graph,
)
from nbdt.hierarchy import(
    print_graph_stats,
    assert_all_wnids_in_graph,
    match_wnid_leaves,
    match_wnid_nodes,
    print_stats,
    generate_hierarchy_vis_from,
)
from nbdt.thirdparty.nx import (
    write_graph,
    get_roots,
    get_root,
    read_graph,
    get_leaves,
    get_depth,
)
from pathlib import Path


def generate_hierarchy(
    wnid_groups, path,
    seed=0,
    branching_factor=2,
    extra=0,
    no_prune=False,
    **kwargs,
):
    wnids = [wnid for wnids in wnid_groups for wnid in wnids]
    G = build_semirandom_graph(wnid_groups, seed=seed, branching_factor=branching_factor)
    print_graph_stats(G, "matched")
    assert_all_wnids_in_graph(G, wnids)

    if not no_prune:
        G = prune_single_successor_nodes(G)
        print_graph_stats(G, "pruned")
        assert_all_wnids_in_graph(G, wnids)

    if extra > 0:
        G, n_extra, n_imaginary = augment_graph(G, extra, True)
        print(f"[extra] \t Extras: {n_extra} \t Imaginary: {n_imaginary}")
        print_graph_stats(G, "extra")
        assert_all_wnids_in_graph(G, wnids)

    write_graph(G, path)

    Colors.green("==> Wrote tree to {}".format(path))
    return path



def test_hierarchy(wnids, path):
    print("==> Reading from {}".format(path))

    G = read_graph(path)

    G_name = Path(path).stem

    leaves_seen, wnid_set1 = match_wnid_leaves(wnids, G, G_name)
    print_stats(leaves_seen, wnid_set1, G_name, "leaves")

    leaves_seen, wnid_set2 = match_wnid_nodes(wnids, G, G_name)
    print_stats(leaves_seen, wnid_set2, G_name, "nodes")

    num_roots = len(list(get_roots(G)))
    if num_roots == 1:
        Colors.green("Found just 1 root.")
    else:
        Colors.red(f"Found {num_roots} roots. Should be only 1.")

    if len(wnid_set1) == len(wnid_set2) == 0 and num_roots == 1:
        Colors.green("==> All checks pass!")
    else:
        Colors.red("==> Test failed")



if __name__ == "__main__":

    class_list = "../../data/wf-undefended/classes.list"
    subpage_count = 5
    hierarchy_path = "./hierarchies/UndefendedSubpages.json"
    wnids_path = "./wnids/UndefendedSubpages.txt"
    vis_path = "./graph.html"

    with open(class_list, 'r') as fi:
        class_names = [line for line in fi][:2]
    wnid_groups = [[f"f{i*subpage_count+j:08d}" for j in range(subpage_count)] for i in range(len(class_names))]
    wnids = [wnid for wnids in wnid_groups for wnid in wnids]
    with open(wnids_path, 'w') as fi:
        fi.write('\n'.join(wnids))

    generate_hierarchy(wnid_groups, hierarchy_path, branching_factor=2, extra=0)
    test_hierarchy(wnids, hierarchy_path)
    G = read_graph(hierarchy_path)
    generate_hierarchy_vis_from(G, None, vis_path)
