from pygame_screen_record.EventRegister import (
    read,
    single_compress,
    single_decompress,
    json,
    SavedEvent,
    astuple,
)
from copy import copy


def test_compress():
    file = "events_ex1.json"
    try:
        uncomp = read(file)["events"]
        _comp = [single_compress(SavedEvent(*ev)) for ev in uncomp]
        comp = [astuple(x) for x in _comp]
        decomp = [astuple(single_decompress(copy(ev))) for ev in _comp]
        print("File:", len(json.dumps(uncomp)))
        print("Compressed:", comp_size := len(json.dumps(comp)))
        print("Decompressed:", decomp_size := len(json.dumps(decomp)))
        print(
            "Compression_rate:",
            str(comp_rate := (comp_size / decomp_size) * 100) + "%",
            "(Pretty Good)" if comp_rate <= 60 else "",
        )
    except FileNotFoundError:
        print(
            f"Test compress failed: Likely {file} missing in current working directory"
        )


if __name__ == "__main__":
    test_compress()
