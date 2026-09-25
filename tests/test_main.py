import pytest

import main


def test_load_config_skips_comments_and_env_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nA=1\n  B = two words  \nC=\nD=x=y\n")
    monkeypatch.setenv("A", "from-env")
    monkeypatch.setenv("UNLISTED", "ignored")
    assert main.load_config(env) == {"A": "from-env", "B": "two words", "C": "", "D": "x=y"}


def test_load_config_falls_back_to_example(tmp_path):
    (tmp_path / ".env.example").write_text("A=example\n")
    assert main.load_config(tmp_path / ".env") == {"A": "example"}
    (tmp_path / ".env").write_text("A=local\n")
    assert main.load_config(tmp_path / ".env") == {"A": "local"}


N = 600  # businesses in the fixture: enough per set for TF-IDF max_df and for matcher negatives


def _dataset(tmp_path, n_sets, sets, blocker="tfidf", matcher="", train_sets="1", tune_sets="2"):
    train = tmp_path / "ds" / "train"
    train.mkdir(parents=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    name = lambda i: f"shop {i * 7919 % 10007} traders"  # a distinct business per i; S2-i is S1-i's match
    (train / "train_source1.tsv").write_text(header + "".join(f"S1-{i}\t{name(i)}\t{i} Main St\tIndia\n" for i in range(N)))
    (train / "train_source2.tsv").write_text(header + "".join(f"S2-{i}\t{name(i)}\t{i} Main St\tIndia\n" for i in range(N)))
    # Look-alikes that are NOT matches, so blocking returns wrong candidates too (the matcher needs negatives).
    (train / "train_source3.tsv").write_text(
        header + "".join(f"S3-{i}\t{name(i)} group\t{i + 7} Market Rd\tIndia\n" for i in range(N)))
    (train / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" +
                                                  "".join(f"S1-{i}\tS2-{i}\n" for i in range(N)))
    (tmp_path / ".env").write_text(f"DATA_DIR={tmp_path / 'ds'}\nN_SETS={n_sets}\nSETS={sets}\nSPLIT=train\n"
                                   "NROWS=\nTRANSLITERATOR=anyascii\nNORMALIZER=rules\n"
                                   f"BLOCKER={blocker}\nBLOCK_K=5\nDEVICE=cpu\n"
                                   "EMBED_MODEL=intfloat/multilingual-e5-small\n"
                                   f"MATCHER={matcher}\nTRAIN_SETS={train_sets}\nTUNE_SETS={tune_sets}\n")


def test_main_runs_from_env_and_creates_missing_sets(tmp_path, monkeypatch, capsys):
    _dataset(tmp_path, 10, "0,1")
    monkeypatch.chdir(tmp_path)
    main.main()
    assert (tmp_path / "data/splits/10_sets/set_9/source1.tsv").exists()
    out = capsys.readouterr().out
    assert "writing the 10 training sets" in out
    assert "top 5: recall 1.000" in out  # every S1 finds its identical S2 record


def test_union_blocker_reports_each_member(tmp_path, monkeypatch, capsys):
    _dataset(tmp_path, 10, "0,1", blocker="tfidf+embedding")
    monkeypatch.chdir(tmp_path)
    main.main()
    out = capsys.readouterr().out
    assert "found by tfidf: recall 1.000" in out
    assert "found by embedding: recall 1.000" in out


def test_unknown_blocker(tmp_path, monkeypatch):
    _dataset(tmp_path, 10, "0", blocker="tfidf+bm25")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match=r"\['bm25'\] not in"):
        main.main()


def test_n_sets_gets_its_own_folder(tmp_path, monkeypatch):
    _dataset(tmp_path, 4, "3")
    monkeypatch.chdir(tmp_path)
    main.main()
    assert sorted(p.name for p in (tmp_path / "data/splits/4_sets").iterdir()) == ["set_0", "set_1", "set_2", "set_3"]


@pytest.mark.parametrize("sets", ["10", "0,12", "-1"])
def test_sets_out_of_range(tmp_path, monkeypatch, sets):
    _dataset(tmp_path, 10, sets)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="between 0 and N_SETS-1"):
        main.main()


def test_matcher_reports_f05_and_writes_outputs(tmp_path, monkeypatch, capsys):
    _dataset(tmp_path, 3, "0", matcher="xgb", train_sets="1", tune_sets="2")
    monkeypatch.chdir(tmp_path)
    main.main()
    out = capsys.readouterr().out
    assert "F0.5 on sets [0]:" in out and "baseline, predict nothing" in out
    written = (tmp_path / "output/matching_results.tsv").read_text().splitlines()
    assert written[0] == "source1_entity_id\tmatched_entity_ids"
    assert (tmp_path / "output/candidate_pairs.tsv").exists()


@pytest.mark.parametrize("train_sets, tune_sets", [("", "2"), ("1", ""), ("1", "1"), ("0", "2"), ("1", "5")])
def test_matcher_set_roles_validated(tmp_path, monkeypatch, train_sets, tune_sets):
    _dataset(tmp_path, 3, "0", matcher="xgb", train_sets=train_sets, tune_sets=tune_sets)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="TRAIN_SETS"):
        main.main()
