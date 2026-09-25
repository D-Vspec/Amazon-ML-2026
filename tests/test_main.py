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


def _dataset(tmp_path, n_sets, sets):
    train = tmp_path / "ds" / "train"
    train.mkdir(parents=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    s1 = [f"S1-{i}" for i in range(50)]
    (train / "train_source1.tsv").write_text(header + "".join(f"{x}\tAcme Pvt Ltd\t1 Main St\tIndia\n" for x in s1))
    (train / "train_source2.tsv").write_text(header + "".join(f"S2-{i}\tAcme\tx\tIndia\n" for i in range(50)))
    (train / "train_source3.tsv").write_text(header + "".join(f"S3-{i}\tAcme\tx\tIndia\n" for i in range(50)))
    (train / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" +
                                                  "".join(f"S1-{i}\tS2-{i}\n" for i in range(50)))
    (tmp_path / ".env").write_text(f"DATA_DIR={tmp_path / 'ds'}\nN_SETS={n_sets}\nSETS={sets}\nSPLIT=train\n"
                                   "NROWS=\nTRANSLITERATOR=anyascii\nNORMALIZER=rules\n")


def test_main_runs_from_env_and_creates_missing_sets(tmp_path, monkeypatch, capsys):
    _dataset(tmp_path, 10, "0,1")
    monkeypatch.chdir(tmp_path)
    main.main()
    assert (tmp_path / "data/splits/10_sets/set_9/source1.tsv").exists()
    assert "writing the 10 training sets" in capsys.readouterr().out


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
