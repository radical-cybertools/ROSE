import os
import argparse

parser = argparse.ArgumentParser(description='Prepare for data dir with a given seed')
parser.add_argument('--seed', type=int, required=True, help='An integer seed')

args = parser.parse_args()
seed = args.seed

root_dir = "/lus/eagle/projects/RECUP/twang/exalearn_stage2/data/"
dir_list = ["base", "test", "study", "stream_phase_1", "stream_phase_2", "AL_phase_1", "AL_phase_2", "AL_phase_3"]
config_list = ["config_1001460_cubic.txt", "config_1522004_trigonal.txt", "config_1531431_tetragonal.txt"] 

cubic_config = """[Global_Params]
path_in = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/cif_file_in/'
symmetry = 'cubic'
name = 'Ba2BiO5'
cif = '1001460.cif'
instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'
path_out = '{}/'
name_out = 'cubic_1001460'
tmin = 1.36
tmax = 18.919
tstep = 0.0009381"""

trigonal_config="""[Global_Params]
path_in = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/cif_file_in/'
symmetry = 'trigonal'
name = 'LaMnO3'
cif = '1522004.cif'
instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'
path_out = '{}/'
name_out = 'trigonal_1522004'
tmin = 1.36
tmax = 18.919
tstep = 0.0009381"""

tetragonal_config="""[Global_Params]
path_in = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/cif_file_in/'
symmetry = 'tetragonal'
name = 'KNbO3'
cif = '1531431.cif'
instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'
path_out = '{}/'
name_out = 'tetragonal_1531431'
tmin = 1.36
tmax = 18.919
tstep = 0.0009381"""

config_content = [cubic_config, trigonal_config, tetragonal_config]

for dir_name in dir_list:
    dir_name = root_dir + "seed_{}/".format(seed) + dir_name
    os.makedirs(dir_name, exist_ok=True)

    config_path = os.path.join(dir_name, "config")
    os.makedirs(config_path, exist_ok=True)
    data_path   = os.path.join(dir_name, "data")
    os.makedirs(data_path, exist_ok=True)

    for i, config_file in enumerate(config_list):
        config_file = os.path.join(config_path, config_file)
        with open(config_file, 'w') as file:
            file.write(config_content[i].format(data_path))
