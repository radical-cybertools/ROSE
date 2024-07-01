import numpy as np
import pandas as pd
from glob import glob

path = "/scratch/pb_data"

train_csvs = np.array([])
test_csvs = np.array([])
sources = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 19, 20]
for i in sources:
    source_path = str(i) + "SourcesRdm"
    tr_f_files = glob(path + "/" + source_path + "/train/F*")
    tr_c_files = glob(path + "/" + source_path + "/train/C*")

    ts_f_files = glob(path + "/" + source_path + "/test/F*")
    ts_c_files = glob(path + "/" + source_path + "/test/C*")

    tr_f_files.sort()
    tr_c_files.sort()

    ts_f_files.sort()
    ts_c_files.sort()

    print("Train files: ", len(tr_f_files), len(tr_c_files))
    print("Test files: ", len(ts_f_files), len(ts_c_files))

    field_names = []
    cell_names = []
    
    fields = np.array([tr_f.split("/")[-1].split("_")[-1].split(".")[0] for tr_f in tr_f_files])
    cells = np.array([tr_f.split("/")[-1].split("_")[-1].split(".")[0] for tr_f in tr_c_files])
    
    inter = np.intersect1d(fields, cells)
    
    print(len(fields), len(cells), len(inter))
    for n in inter:
        field_names.append("Field_" + n + ".dat")
        cell_names.append("Cell_" + n + ".dat")

    field_names = np.array(field_names).reshape(-1, 1)
    cell_names = np.array(cell_names).reshape(-1, 1)
    
    train_source = np.array([source_path]).reshape(-1, 1)
    train_source = np.repeat(train_source, len(field_names), axis=0)

    tr_val = np.hstack((field_names, cell_names, train_source))

    field_names = []
    cell_names = []
    
    fields = np.array([tr_f.split("/")[-1].split("_")[-1].split(".")[0] for tr_f in ts_f_files])
    cells = np.array([tr_f.split("/")[-1].split("_")[-1].split(".")[0] for tr_f in ts_c_files])
    
    inter = np.intersect1d(fields, cells)
    
    print(len(fields), len(cells), len(inter))
    for n in inter:
        field_names.append("Field_" + n + ".dat")
        cell_names.append("Cell_" + n + ".dat")

    field_names = np.array(field_names).reshape(-1, 1)
    cell_names = np.array(cell_names).reshape(-1, 1)
    
    test_source = np.array([source_path]).reshape(-1, 1)
    test_source = np.repeat(test_source, len(field_names), axis=0)

    ts_val = np.hstack((field_names, cell_names, test_source))


    # print(tr_val)
    # exit()
    # df_train = pd.read_csv(path + "/" + str(i) + "SourcesRdm/train.csv", sep=",")
    # df_test = pd.read_csv(path + "/" + str(i) + "SourcesRdm/test.csv", sep=",")

    # tr_val = df_train.to_numpy()
    # train_source = np.array([source_path]).reshape(-1, 1)
    # train_source = np.repeat(train_source, len(tr_val), axis=0)
    # tr_val = np.hstack((tr_val, train_source))
    # print(tr_val)
    # exit()

    # ts_val = df_test.to_numpy()
    # test_source = np.array([source_path]).reshape(-1, 1)
    # test_source = np.repeat(test_source, len(ts_val), axis=0)
    # ts_val = np.hstack((ts_val, test_source))

    # print(train_csvs.shape, test_csvs.shape)
    if i == 1:
        train_csvs = np.array(tr_val)
        test_csvs = np.array(ts_val)
    else:
        train_csvs = np.vstack((train_csvs, tr_val))
        test_csvs = np.vstack((test_csvs, ts_val))
    
    # print(len(tr_val), len(ts_val), tr_val[:2], ts_val[:2])
    # print(train_csvs.shape, test_csvs.shape)
# exit()13
print(train_csvs.shape)
df = pd.DataFrame(train_csvs)
df.columns = ["Cell", "Field", "Prefix"]
print(df.head(), df.tail())
# print(df.shape)
df.to_csv(path + "/train_all.csv", header=True, index=False)

df = pd.DataFrame(test_csvs)
df.columns = ["Cell", "Field", "Prefix"]
print(df.head(), df.tail())
# print(df.shape)
# print(test_csvs.shape)
df.to_csv(path + "/test_all.csv", header=True, index=False)
exit()

print(len(train_csvs), len(test_csvs))
np.savetxt(path + "/train_all.csv", train_csvs, delimiter=",")
np.savetxt(path + "/test_all.csv", test_csvs, delimiter=",")