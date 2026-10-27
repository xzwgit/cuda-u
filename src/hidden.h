// hidden.h — 隐藏格式测量的对外接口
#ifndef CUDA_U_HIDDEN_H
#define CUDA_U_HIDDEN_H

#include <string>
#include "report.h"

// SM120/121 专属: 经 cubin 补丁测量 FP8 E3M4 / FP4 E0M3 发射率。
// 其他架构直接置 N/A 并返回 true。
bool runHiddenFormats(int device, Report &r, std::string &err);

#endif
