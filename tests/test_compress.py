import sys,os
os.chdir(os.path.dirname(__file__))
sys.path.append("../")

from EventRegister import read, single_compress, single_decompress, json

def test_compress():
    try:
        uncomp = read("events.json")["events"]
        comp=[single_compress(ev) for ev in uncomp]
        decomp = [single_decompress(ev) for ev in comp]
        print()
        print("File:", len(json.dumps(uncomp)))
        print("Compressed:", comp_size:=len(json.dumps(comp)))
        print("Decompressed:", decomp_size:=len(json.dumps(decomp)))
        print("Compression_rate:", str(comp_rate:=(comp_size/decomp_size)*100)[:2]+"%","(Pretty Good)" if comp_rate <= 60 else "")
    except FileNotFoundError:
        print("Test compress failed: Likely file 'events.json' missing in current working directory")

if __name__ == "__main__":
    test_compress()
