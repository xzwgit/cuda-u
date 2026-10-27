#!/usr/bin/env python3
"""embed_cubins.py — 把补丁后的 cubin 生成为 C++ 字节数组源文件。
用法: embed_cubins.py <out.cpp> <name1>=<file1> [name2>=<file2> ...]
"""
import sys

def c_name(path: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in path)

def main():
    out, pairs = sys.argv[1], sys.argv[2:]
    with open(out, "w") as f:
        f.write("// 由 tools/embed_cubins.py 生成 — 补丁后的隐藏格式 cubin, 勿手改\n")
        f.write("#include <cstdint>\n#include <cstddef>\n\n")
        for spec in pairs:
            name, path = spec.split("=", 1)
            data = open(path, "rb").read()
            f.write(f"extern const unsigned char {name}[] = {{\n")
            for i in range(0, len(data), 20):
                chunk = ",".join(str(b) for b in data[i:i+20])
                f.write(f"{chunk},\n")
            f.write("};\n")
            f.write(f"const size_t {name}_size = sizeof({name});\n\n")
    print(f"{out}: {len(pairs)} 个 cubin 已嵌入")

if __name__ == "__main__":
    main()
