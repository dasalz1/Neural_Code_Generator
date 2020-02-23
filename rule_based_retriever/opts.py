import os, pathlib, math


# path_local = "/Users/minalee/contextual_autocomplete"
path_local = str(pathlib.Path.cwd())
assert (os.path.exists(path_local), f'{path_local} does not exist, please update directory in opts.py')
# path_server = "/afs/cs.stanford.edu/u/minalee/scr/contextual_autocomplete"
path = path_local

path_data = os.path.join(path, "data")


def system_opts(parser):
    group = parser.add_argument_group("Basic")
    group.add("--debug",
              action="store_true")
    group.add("--save",
              default = 1,
              help = 'save data')
    group.add("--overwrite",
              action="store_true")
    group.add("--verbose",
              action="store_true")

    group = parser.add_argument_group("Path")
    group.add("--path",
              type=str,
              default = str(pathlib.Path.cwd()  / 'github_data' / 'tensorflow-vgg'))
              # default="/Users/minalee/contextual_autocomplete/github/tensorflow-vgg")

    group = parser.add_argument_group("Setting")
    group.add("--unit",
              type=str,
              default="line",
              choices=["line"])
    group.add("--min_len",
              type=int,
              default=10)

    group.add("--distance_metric_x",
              type=str,
              default="nc",
              choices=["j", "l", "h", "s", "c", "nc"])
    group.add("--distance_threshold_x",
              type=float,
              default=math.inf)
    group.add("--distance_metric_y",
              type=str,
              default="c",
              choices=["j", "l", "h", "s", "c", "nc"])
    group.add("--distance_threshold_y",
              type=float,
              default=math.inf)


    group.add("--n_leftmost_tokens",
              type=int,
              default=1)
    group.add("--max_num_candidates",
              type=int,
              default=1)

    group.add("--check_exact_match_suffix",
              action="store_true")
