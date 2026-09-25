# data/

Raw challenge data is NOT copied here (~2.5 GB). It lives at:

    6ab10eb3b23ba_student_resource/student_resource/dataset/
        train/train_source{1,2,3}.tsv, train/train_ground_truth.tsv
        test/test_source{1,2,3}.tsv

All files are tab-separated: read with `pd.read_csv(path, sep="\t")`.

This folder is for derived/intermediate data only and is git-ignored.
