# %cd /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/deep-learning-chained-local-search/LKHPyInterface
# !python setup.LKH.py build_ext --inplace
import os
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp, strcpy, strlen
# import ctypes
import numpy as np
cimport numpy as np

cdef extern from "LKHProgram/SRC/LKHmain.c":
    int LKHmain(char *parameterFileLines, int numParameterLines, char *problemFileLines, int numProblemLines, int* tour, int tourlen, int useInitialTour, int printDebug)

cdef char ** to_cstring_array(strings):
    cdef char **ret = <char **> malloc(len(strings) * sizeof(char *))
    for i in range(len(strings)):
        stringCharArr = <bytes> (strings[i].encode())
        ret[i] = <char *> malloc((strlen(stringCharArr) + 1) * sizeof(char)) #null character at end
        strcpy(ret[i], stringCharArr) #copy because original python strings seem to be getting garbage collected
    return ret

cdef char * to_cstring(string):
    string = string.encode()
    cdef char *stringCharArr
    cdef char *ret
    stringCharArr = <bytes> string
    ret = <char *> malloc((strlen(stringCharArr) + 1) * sizeof(char))
    strcpy(ret, stringCharArr)
    return ret

cpdef run(problemString, params, np.ndarray[int, ndim=1, mode="c"] input, useInitialTour, printDebug=0):
    # cdef char * argv[2];
    # parameterFileArg = os.path.join(os.getcwd(), inputFilename).encode()
    # argv[1] = <bytes> parameterFileArg

    parameterFileLines = [key + " = " + str(value) for (key, value) in params.items()]

    parameterString = ""
    for string in parameterFileLines:
        parameterString += string + "\n"

    plines = len(problemString.split('\n'))
    status = LKHmain(to_cstring(parameterString), len(parameterFileLines), to_cstring(problemString), plines, &input[0], input.shape[0], useInitialTour, printDebug)
    if status != 11:
        raise RuntimeError("LKH failed!")
    return input

# # How to do it inline with IPython, not practical in this case
# %%cython -I /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/LKHPyInterface
# cdef extern from "hello.c":
#     int f()
#
# cpdef myf():
#     return f()
#
# myf()
