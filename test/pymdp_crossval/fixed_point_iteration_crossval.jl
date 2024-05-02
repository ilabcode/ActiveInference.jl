using Pkg
using PyCall
using ActiveInference

Pkg.add(path = raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\Continued_ActiveInference\Dev_Branch\ActiveInference.jl")

ENV["PYTHON"] = raw"C:\Users\jonat\miniconda3\envs\ActiveInferenceEnv\python.exe"
#Pkg.build("PyCall")

pyversion = pyimport("sys").version
pyexecutable = pyimport("sys").executable
println("Using Python version: $pyversion at $pyexecutable")

pymdp = PyCall.pyimport("pymdp")


pymdp.utils.onehot(1, 5)


cd(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\Continued_ActiveInference\Dev_Branch\ActiveInference.jl\test")
Pkg.activate(".")
Pkg.add("PyCall")


