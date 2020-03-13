import feather, pathlib2


data_dir = pathlib2.Path.cwd() / 'github_data' / 'neural_ret_files'
df = feather.read_dataframe(data_dir / 'check_domain_line_pairs.feather')

