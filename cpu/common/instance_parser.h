#pragma once

#include "jssp_types.h"

// Resolve instance identifier or path to an existing file path
bool resolve_instance_file(const char* input_arg, char* out_path, int path_size);

// Parse Lawrence / Taillard instance file into JSSPInstance structure
bool parse_instance(const char* filepath, JSSPInstance& inst);
