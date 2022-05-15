import os


def create_directory(current_directory):
    top_folders = ["experiment_small_data", "experiment_tiny"]
    sub_folders = ["test", "train", "val"]
    try:
        for top_folder in top_folders:
            top_path = os.path.join(current_directory, top_folder)
            os.mkdir(top_path)
            for sub_folder in sub_folders:
                sub_path = os.path.join(top_folder, sub_folder)
                os.mkdir(sub_path)
    except:
        print("Folder exist already")
        pass
