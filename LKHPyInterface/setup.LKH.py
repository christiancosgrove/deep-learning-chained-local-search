from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os

# If all in one script, download and untar and rename LKH dir here.

# %cd /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/LKHPyInterface

sourceFilesToRebuild = ["ReadParameters", "ReadProblem", "CreateCandidateSet", "ReadPenalties"]
# allSourceFiles = list(filter(lambda x: x.split(".")[-1] == "c", os.listdir("LKHProgram/SRC")))
# allSourceFiles.remove("LKHmain.c")

LKHsourceFiles = [] # reuse objects files instead
LKHsourceFiles += [f + ".c" for f in sourceFilesToRebuild]
LKHsourceFiles = ["LKHProgram/SRC/" + f for f in LKHsourceFiles]

extra_objects = os.listdir("LKHProgram/SRC/OBJ")
[extra_objects.remove(f) for f in (["LKHmain.o"] + [f + ".o" for f in sourceFilesToRebuild])]
extra_objects = ["LKHProgram/SRC/OBJ/" + f for f in extra_objects]

sourcefiles = ["LKH.pyx", *LKHsourceFiles]
include_dirs = ["LKHProgram/SRC/INCLUDE"]
ext_modules = [Extension("LKH", sourcefiles, extra_objects=extra_objects, include_dirs=include_dirs)]

setup(
  name = 'LKH',
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(ext_modules)
)
