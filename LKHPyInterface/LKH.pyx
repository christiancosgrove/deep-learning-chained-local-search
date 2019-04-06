# %cd /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/LKHPyInterface
# !python setup.LKH.py build_ext --inplace
import os
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp, strcpy, strlen

cdef extern from "LKHProgram/SRC/LKHmain.c":
    int LKHmain(char *parameterFileLines, int numParameterLines, char *problemFileLines, int numProblemLines)

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


cpdef run(inputFilename):
    # cdef char * argv[2];
    # parameterFileArg = os.path.join(os.getcwd(), inputFilename).encode()
    # argv[1] = <bytes> parameterFileArg

    parameterFileLines = ["PROBLEM_FILE = placeholder","RUNS = 1"]
    parameterString = ""
    for string in parameterFileLines:
        parameterString += string + "\n"


    with open("LKHProgram/TSP100.tsp", "r") as f:
        problemFileLines = f.readlines()
    problemString = ""
    for string in problemFileLines:
        problemString += string + "\n"

    status = LKHmain(to_cstring(parameterString), len(parameterFileLines), to_cstring(problemString), len(problemFileLines))
    return status

# # How to do it inline with IPython, not practical in this case
# %%cython -I /Users/cameronfranz/Documents/Learning/Projects/DiscreteOptiDLClass/LKHPyInterface
# cdef extern from "hello.c":
#     int f()
#
# cpdef myf():
#     return f()
#
# myf()
