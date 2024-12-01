#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using PythonCall
using Dewdrop

src = pyimport("src")
models = pyimport("models")
