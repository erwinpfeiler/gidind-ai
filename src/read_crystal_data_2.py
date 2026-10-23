import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from multiprocessing import Process
#from CifFile import ReadCif
import os
from tqdm import tqdm
from time import sleep
from urllib.error import URLError

from config_loader import TargetConfig


def parse_entry_to_float(entry):
            if isinstance(entry, str):
                while "(" in entry:
                    entry = entry[:-1]
                entry = np.float64(entry)
            return entry

def find_cif_files(folder_path):
    cif_files = []
    # Walk through all the directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Check each file in the current directory
        
        for filename in filenames:
            # Check if the file has a .cif extension
            if filename.endswith('.cif'):
                # Construct the full path to the .cif file
                file_path = os.path.join(dirpath, filename)
                cif_files.append(file_path)
                
    return cif_files


#path = ~/cod_archive/cod-cifs-mysql/cif/1
        
CELL_COLUMNS = ["a", "b", "c", "alpha", "beta", "gamma"]
REDUCED_COLUMNS = [name + "_red" for name in CELL_COLUMNS]


def checked_niggli_cell(cell):
    """Reduce in float64, rejecting invalid geometry and the legacy failure sentinel."""
    from niggli import cell_to_niggli

    cell = np.asarray(cell, dtype=np.float64)
    def volume(values):
        cosines = np.cos(np.deg2rad(values[3:]))
        determinant = 1 + 2*np.prod(cosines) - np.sum(cosines**2)
        return np.prod(values[:3]) * np.sqrt(max(0.0, determinant))
    if (cell.shape != (6,) or not np.isfinite(cell).all()
            or np.any(cell[:3] <= 0) or np.any(cell[3:] <= 0)
            or np.any(cell[3:] >= 180) or volume(cell) <= 1e-10):
        raise ValueError("Invalid unit cell")
    reduced = np.asarray(cell_to_niggli(cell), dtype=np.float64)
    if (not np.isfinite(reduced).all() or np.any(reduced[:3] <= 0)
            or not np.isclose(volume(cell), volume(reduced), rtol=1e-7, atol=1e-9)):
        raise ValueError("Niggli reduction failed or changed cell volume")
    return reduced


def load_cod_to_pickle(path, path_to_save):
    """Parse a locally downloaded COD CIF archive (this function does not download it).

    Keep conventional and reduced parameters and COD/file provenance. Global
    three-decimal deduplication takes place after merging all archive shards.
    """
    import warnings
    from pymatgen.io.cif import CifFile

    rows, failures = [], 0
    for cif_path in sorted(find_cif_files(path)):
        try:
            blocks = CifFile.from_file(cif_path).data
            for block_name, block in blocks.items():
                data = block.data
                sg = data.get("_space_group_IT_number", data.get("_space_group_it_number",
                     data.get("_symmetry_Int_Tables_number")))
                if sg is None:
                    raise ValueError("Missing space-group number")
                cell = np.array([float(parse_entry_to_float(data["_cell_" +
                    ("length_" if i < 3 else "angle_") + name]))
                    for i, name in enumerate(CELL_COLUMNS)], dtype=np.float64)
                reduced = checked_niggli_cell(cell)
                rows.append(dict(zip(CELL_COLUMNS, cell)) | dict(zip(REDUCED_COLUMNS, reduced)) |
                            {"space_group_number": int(sg), "cod_id": str(block_name),
                             "source_file": str(cif_path)})
        except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError, OSError):
            failures += 1
    if not rows:
        raise ValueError(f"No valid COD unit cells parsed from {path}; rejected {failures} files")
    if failures:
        warnings.warn(f"Rejected {failures} COD CIF files; retained {len(rows)} cells", RuntimeWarning)
    table = pd.DataFrame(rows)
    table.to_csv(str(path_to_save) + ".csv", index=False)
    table.to_pickle(str(path_to_save) + ".pkl")


def filter_cod_dataframes(pd_cod_paths:list):
    target_config = TargetConfig()
    pd_cods = list()
    for pd_cod_path in pd_cod_paths:
        pd_cods.append(pd.read_pickle(pd_cod_path))
        
    pd_cod = pd.concat(pd_cods, ignore_index=True)
        
    print(f"Found {len(pd_cods)} pandas dataframes")
    print(f"Overall there are {len(pd_cod)} samples")
    
    space_groups = pd.to_numeric(pd_cod["space_group_number"], errors="coerce")
    pd_cod = pd_cod[space_groups.between(1, 230) & space_groups.eq(space_groups.round())].copy()
    pd_cod["space_group_number"] = space_groups.loc[pd_cod.index].astype(int)
    pd_cod[CELL_COLUMNS] = pd_cod[CELL_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    
    merged_lattice_data = {}
    
    merged_lattice_data["triclinic"] = pd_cod[(pd_cod["space_group_number"] >= 1) & 
                                              (pd_cod["space_group_number"] <= 2)]
    merged_lattice_data["monoclinic"] = pd_cod[(pd_cod["space_group_number"] >= 3) & 
                                               (pd_cod["space_group_number"] <= 15)]
    merged_lattice_data["orthorhombic"] = pd_cod[(pd_cod["space_group_number"] >= 16) & 
                                              (pd_cod["space_group_number"] <= 74)]
    merged_lattice_data["tetragonal"] = pd_cod[(pd_cod["space_group_number"] >= 75) & 
                                              (pd_cod["space_group_number"] <= 142)]
    merged_lattice_data["rhombohedral"] = pd_cod[(pd_cod["space_group_number"] == 146) | 
                                                 (pd_cod["space_group_number"] == 148) |
                                                 (pd_cod["space_group_number"] == 155) |
                                                 (pd_cod["space_group_number"] == 160) |
                                                 (pd_cod["space_group_number"] == 161) |
                                                 (pd_cod["space_group_number"] == 166) |
                                                 (pd_cod["space_group_number"] == 167)]
    merged_lattice_data["hexagonal"] = pd_cod[(pd_cod["space_group_number"] == 143) | 
                                              (pd_cod["space_group_number"] == 144) |
                                              (pd_cod["space_group_number"] == 145) |
                                              (pd_cod["space_group_number"] == 147) |
                                              (pd_cod["space_group_number"] == 149) |
                                              (pd_cod["space_group_number"] == 150) |
                                              (pd_cod["space_group_number"] == 151) |
                                              (pd_cod["space_group_number"] == 152) |
                                              (pd_cod["space_group_number"] == 153) |
                                              (pd_cod["space_group_number"] == 154) |
                                              (pd_cod["space_group_number"] == 156) |
                                              (pd_cod["space_group_number"] == 157) |
                                              (pd_cod["space_group_number"] == 158) |
                                              (pd_cod["space_group_number"] == 159) |
                                              (pd_cod["space_group_number"] == 162) |
                                              (pd_cod["space_group_number"] == 163) |
                                              (pd_cod["space_group_number"] == 164) |
                                              (pd_cod["space_group_number"] == 165) |
                                              ((pd_cod["space_group_number"] >= 168) &
                                               (pd_cod["space_group_number"] <= 194))]
    merged_lattice_data["cubic"] = pd_cod[(pd_cod["space_group_number"] >= 195) & 
                                              (pd_cod["space_group_number"] <= 230)]
    
    # filter hexagonal entries which are written with rhombohedral unit cells
    # this should actually not happen but I want to make sure
    copy_df = merged_lattice_data["hexagonal"]
    copy_df = copy_df[(copy_df["gamma"] >= 119.9) & (copy_df["gamma"] <= 120.1)]
    merged_lattice_data["hexagonal"] = copy_df
    
    # transform rhombohedral entries to a representation with a rhombohedron
    # as unit cell instead of the non-primitive hexagon
    rhomb_df = merged_lattice_data["rhombohedral"]
    for idx, data in rhomb_df.iterrows():
        if data["gamma"] > 119.0 and data["alpha"] < 91.0 and data["beta"] < 91.0:
            a_h = np.mean([data["a"], data["b"]])
            c_h = data["c"]
            #print(f"hex in rhomb a_h = {a_h}, c_h = {c_h}")
        elif data["beta"] > 119.0 and data["gamma"] < 91.0 and data["alpha"] < 91.0:
            a_h = np.mean([data["c"], data["a"]])
            c_h = data["b"]
        elif data["alpha"] > 119.0 and data["beta"] < 91.0 and data["gamma"] < 91.0:
            a_h = np.mean([data["b"], data["c"]])
            c_h = data["a"]
        else:
            continue
            
        a_r = np.float64(np.sqrt(3*np.square(a_h) + np.square(c_h))/3)
        alpha_r = np.float64(np.rad2deg(np.arccos(0.5*(2*np.square(c_h) - 3*np.square(a_h))/(np.square(c_h) + 3*np.square(a_h)))))
        if np.isnan(alpha_r):
            print(f"NaN from a_h: {a_h}, c_h: {c_h}")
            print(f"cos arg = {0.5*(2*np.square(c_h) - 3*np.square(a_h))/(np.square(c_h) + 3*np.square(a_h))}")
        data["a"], data["b"], data["c"] = a_r, a_r, a_r
        data["alpha"], data["beta"], data["gamma"] = alpha_r, alpha_r, alpha_r
        #print("data = ", data)
        merged_lattice_data["rhombohedral"].at[idx, "a"] = a_r
        merged_lattice_data["rhombohedral"].at[idx, "b"] = a_r
        merged_lattice_data["rhombohedral"].at[idx, "c"] = a_r
        merged_lattice_data["rhombohedral"].at[idx, "alpha"] = alpha_r
        merged_lattice_data["rhombohedral"].at[idx, "beta"] = alpha_r
        merged_lattice_data["rhombohedral"].at[idx, "gamma"] = alpha_r

    print_data_sizes = False
              
    seen_reduced_keys = set()
    for key in merged_lattice_data.keys():
        df = merged_lattice_data[key]
        if print_data_sizes:
            print("==========================================")
            print("Key = ", key)
            print(f"Initial #samples = {len(df)}")
        #print("Dataframe = ")
        df = df[(df["a"] >= target_config.min_distance) & (df["a"] <= target_config.max_distance) &
                (df["b"] >= target_config.min_distance) & (df["b"] <= target_config.max_distance) &
                (df["c"] >= target_config.min_distance) & (df["c"] <= target_config.max_distance) &
                (df["alpha"] >= target_config.min_angle) & (df["alpha"] <= target_config.max_angle) &
                (df["beta"] >= target_config.min_angle) & (df["beta"] <= target_config.max_angle) &
                (df["gamma"] >= target_config.min_angle) & (df["gamma"] <= target_config.max_angle)]
        if print_data_sizes: print(f"After constraints:{len(df)}")
        df = remove_duplicates_from_df(df, seen_keys=seen_reduced_keys)
        if print_data_sizes: print(f"After removing duplicates: {len(df)}")
        merged_lattice_data[key] = df
        if print_data_sizes: print("===========================================")
    return merged_lattice_data


def remove_duplicates_from_df(df: pd.DataFrame, *, seen_keys=None) -> pd.DataFrame:
    """Reduce before splitting; globally unique keys round all six values to 3 dp.

    Preserve unrounded reduced values and conventional coordinates. The latter
    are required by the existing symmetry-specific GMM parameterization (e.g.
    hexagonal a,c); substituting sorted reduced lengths would corrupt that fit.
    A rounded-key match is the definition here, not a pairwise distance tolerance.
    """
    seen = set() if seen_keys is None else seen_keys
    rows, reduced_rows = [], []
    rejected = 0
    for index, cell in enumerate(df[CELL_COLUMNS].to_numpy(dtype=np.float64)):
        try:
            reduced = checked_niggli_cell(cell)
        except (ValueError, TypeError, ArithmeticError, RuntimeError):
            rejected += 1
            continue
        key = tuple(np.round(reduced, 3))
        if key not in seen:
            seen.add(key)
            rows.append(index)
            reduced_rows.append(reduced)
    result = df.iloc[rows].copy()
    result[REDUCED_COLUMNS] = np.asarray(reduced_rows, dtype=np.float64).reshape(-1, 6)
    if rejected:
        import warnings
        warnings.warn(f"Rejected {rejected} cells with invalid Niggli reductions", RuntimeWarning)
    return result



if False:
    processes = list()
    for path_idx in range(1,10):
        path = "./cod-cifs-mysql/cif/" + str(path_idx)
        path_to_save = "./cod_cif_files/" + str(path_idx)

        p = Process(target=load_cod_to_pickle,
                args=(path, path_to_save))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__=="__main__":
    merged_lattice_data = filter_cod_dataframes(["./cod_cif_files/" + str(path_idx) + ".pkl" for path_idx in range(1,10)])


""" 4.10.2024:
Found 9 pandas dataframes
Overall there are 424916 samples
==========================================
Key =  triclinic
Initial #samples = 101386
After constraints:99293
After removing duplicates: 97489
===========================================
==========================================
Key =  monoclinic
Initial #samples = 199996
After constraints:170689
After removing duplicates: 167043
===========================================
==========================================
Key =  orthorhombic
Initial #samples = 70043
After constraints:57653
After removing duplicates: 55702
===========================================
==========================================
Key =  tetragonal
Initial #samples = 16071
After constraints:11590
After removing duplicates: 10705
===========================================
==========================================
Key =  rhombohedral
Initial #samples = 8941
After constraints:5388
After removing duplicates: 4814
===========================================
==========================================
Key =  hexagonal
Initial #samples = 12125
After constraints:9100
After removing duplicates: 8144
===========================================
==========================================
Key =  cubic
Initial #samples = 16341
After constraints:13900
After removing duplicates: 2026
===========================================
"""

if False:
    lattice_name = "triclinic"
    print("real data (triclinic): ")
    print(merged_lattice_data[lattice_name])
    triclinic_generator = TriclinicGenerator(use_niggli=False, return_niggli=True)
    triclinic_generator.load_data(merged_lattice_data[lattice_name])
    if True:
        triclinic_generator.optimize_crystal_distribution()
        
    triclinic_generator.learn_crystal_distribution(30)
    triclinic_sample = triclinic_generator.sample_crystals(10000)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")



if False:
    lattice_name = "orthorhombic"
    print("real data (orthorhombic): ")
    print(merged_lattice_data[lattice_name])
    triclinic_generator = OrthorhombicGenerator(use_niggli=False, return_niggli=True)
    triclinic_generator.load_data(merged_lattice_data[lattice_name])
    triclinic_generator.optimize_crystal_distribution()
    
    triclinic_generator.learn_crystal_distribution(25)
    triclinic_sample = triclinic_generator.sample_crystals(10000)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")

if False:
    lattice_name = "cubic"
    print("real data (cubic): ")
    print(merged_lattice_data[lattice_name])
    cubic_generator = CubicGenerator(use_niggli=False, return_niggli=True)
    cubic_generator.load_data(merged_lattice_data[lattice_name])
    cubic_generator.optimize_crystal_distribution()
    
    cubic_generator.learn_crystal_distribution(20)
    triclinic_sample = cubic_generator.sample_crystals(10000)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")


if False:
    lattice_name = "tetragonal"
    print("real data (tetragonal): ")
    print(merged_lattice_data[lattice_name])
    tetragonal_generator = TetragonalGenerator(use_niggli=False, return_niggli=True)
    tetragonal_generator.load_data(merged_lattice_data[lattice_name])
    tetragonal_generator.optimize_crystal_distribution()
    
    tetragonal_generator.learn_crystal_distribution(15)
    triclinic_sample = tetragonal_generator.sample_crystals(10000)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")
    
    
if False:
    lattice_name = "rhombohedral"
    print("real data (rhombohedral): ")
    print(merged_lattice_data[lattice_name])
    crystal_generator = RhombohedralGenerator(use_niggli=False, return_niggli=True)
    crystal_generator.load_data(merged_lattice_data[lattice_name])
    crystal_generator.optimize_crystal_distribution()
    
    
    crystal_generator.learn_crystal_distribution(12)
    triclinic_sample = crystal_generator.sample_crystals(10000)
    
    if False:
        print("training data:")
        print(crystal_generator.train_crystals)
        valid_sample = crystal_generator.safe_valid_sample(100)
        print(f"valid_sample.shape = {valid_sample.shape}")
        print(valid_sample)
        
        test_sample = crystal_generator.safe_test_sample(100)
        print(f"test_sample.shape = {test_sample.shape}")
        print(test_sample)
        
        quit(0)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")
    
    
if False:
    lattice_name = "hexagonal"
    print("real data (hexagonal): ")
    print(merged_lattice_data[lattice_name])
    crystal_generator = HexagonalGenerator(use_niggli=False, return_niggli=True)
    crystal_generator.load_data(merged_lattice_data[lattice_name])
    crystal_generator.optimize_crystal_distribution()
   
    crystal_generator.learn_crystal_distribution(12)
    
    triclinic_sample = crystal_generator.sample_crystals(10000)
    
    train_samples = crystal_generator.safe_train_sample(100)
    valid_samples = crystal_generator.safe_valid_sample(100)
    test_samples = crystal_generator.safe_test_sample(100)

    print("Training samples from GMM")
    for idx, sample in enumerate(train_samples):
        print(f"{sample[0]:7.2f} {sample[1]:7.2f} {sample[2]:7.2f} {sample[3]:7.2f} {sample[4]:7.2f} {sample[5]:7.2f}")
        
    print("Validation samples from GMM")
    for idx, sample in enumerate(valid_samples):
        print(f"{sample[0]:7.2f} {sample[1]:7.2f} {sample[2]:7.2f} {sample[3]:7.2f} {sample[4]:7.2f} {sample[5]:7.2f}")
        
    print("Test samples from GMM")
    for idx, sample in enumerate(test_samples):
        print(f"{sample[0]:7.2f} {sample[1]:7.2f} {sample[2]:7.2f} {sample[3]:7.2f} {sample[4]:7.2f} {sample[5]:7.2f}")

    #quit(0)
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")
    
if False:
    lattice_name = "monoclinic"
    print("real data (hexagonal): ")
    print(merged_lattice_data[lattice_name])
    tetragonal_generator = MonoclinicGenerator(use_niggli=False, return_niggli=True)
    tetragonal_generator.load_data(merged_lattice_data[lattice_name])
    tetragonal_generator.optimize_crystal_distribution()
    
    tetragonal_generator.learn_crystal_distribution(50)
    triclinic_sample = tetragonal_generator.sample_crystals(10000)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["b_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,1], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_b_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["a_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,0], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_a_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["b_red"], merged_lattice_data[lattice_name]["c_red"], bins=np.linspace(3, 40, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,1], triclinic_sample[:,2], bins=np.linspace(3, 40, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_b_red_VS_c_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["beta_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,4], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_beta_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["alpha_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,3], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_alpha_red_VS_gamma_red.png")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    axs[0].hist2d(merged_lattice_data[lattice_name]["beta_red"], merged_lattice_data[lattice_name]["gamma_red"], bins=np.linspace(50, 130, 100),
                  density=True)
    axs[0].set_title("Database")
    axs[1].hist2d(triclinic_sample[:,4], triclinic_sample[:,5], bins=np.linspace(50, 130, 100),
                  density = True)
    axs[1].set_title("GMM")
    plt.savefig("./gmm_hist/"+lattice_name+"_beta_red_VS_gamma_red.png")


    for i in range(min(triclinic_sample.shape[0],100)):
        print(f"{triclinic_sample[i, 0]:5.2f} {triclinic_sample[i, 1]:5.2f} {triclinic_sample[i, 2]:5.2f} {triclinic_sample[i, 3]:7.2f} {triclinic_sample[i, 4]:7.2f} {triclinic_sample[i, 5]:7.2f}")
    