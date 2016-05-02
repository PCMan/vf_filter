#!/usr/bin/env python3
# C definitions for using wfdb in Cython.

# use wfdb APIs
cdef extern from "<wfdb/wfdb.h>":
    ctypedef long WFDB_Time
    ctypedef int WFDB_Sample
    ctypedef double	WFDB_Frequency
    ctypedef char* FSTRING
    ctypedef WFDB_Frequency FFREQUENCY
    ctypedef int FINT
    ctypedef unsigned int WFDB_Annotator
    ctypedef double	WFDB_Gain
    ctypedef unsigned int WFDB_Group

    ctypedef struct WFDB_Anninfo:
        char *name
        int stat

    ctypedef struct WFDB_Siginfo:
        char *fname
        char *desc
        char *units
        WFDB_Gain gain
        WFDB_Sample initval
        WFDB_Group group
        int fmt
        int spf
        int bsize
        int adcres
        int adczero
        int baseline
        long nsamp
        int cksum

    ctypedef struct WFDB_Annotation:
        WFDB_Time time
        char anntyp
        signed char subtyp
        unsigned char chan
        signed char num
        unsigned char *aux

    FINT annopen(char *record, WFDB_Anninfo *aiarray, unsigned int nann)
    FINT isigopen(char *record, WFDB_Siginfo *siarray, int nsig)
    FFREQUENCY sampfreq(char *record)
    FINT getvec(WFDB_Sample *vector)
    FINT getann(WFDB_Annotator a, WFDB_Annotation *annot)
    FSTRING annstr(int annotation_code)
    void wfdbquit()
    void wfdbquiet()

    # constants
    int WFDB_DEFGAIN
    int WFDB_READ

# end of wfdb API declarations
