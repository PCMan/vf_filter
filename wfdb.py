'''Wrapper for ecgcodes.h

Generated with:
/usr/bin/ctypesgen.py -l libwfdb.so /usr/include/wfdb/ecgcodes.h /usr/include/wfdb/ecgmap.h /usr/include/wfdb/wfdb.h /usr/include/wfdb/wfdblib.h -o wfdb.py

Do not modify this file.
'''

__docformat__ =  'restructuredtext'

# Begin preamble

import ctypes, os, sys
from ctypes import *

_int_types = (c_int16, c_int32)
if hasattr(ctypes, 'c_int64'):
    # Some builds of ctypes apparently do not have c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (c_int64,)
for t in _int_types:
    if sizeof(t) == sizeof(c_size_t):
        c_ptrdiff_t = t
del t
del _int_types

class c_void(Structure):
    # c_void_p is a buggy return type, converting to int, so
    # POINTER(None) == c_void_p is actually written as
    # POINTER(c_void), so it can be treated as a real pointer.
    _fields_ = [('dummy', c_int)]

def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

class UserString:
    def __init__(self, seq):
        if isinstance(seq, basestring):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq)
    def __str__(self): return str(self.data)
    def __repr__(self): return repr(self.data)
    def __int__(self): return int(self.data)
    def __long__(self): return long(self.data)
    def __float__(self): return float(self.data)
    def __complex__(self): return complex(self.data)
    def __hash__(self): return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)
    def __contains__(self, char):
        return char in self.data

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.__class__(self.data[index])
    def __getslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, basestring):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other))
    def __radd__(self, other):
        if isinstance(other, basestring):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other) + self.data)
    def __mul__(self, n):
        return self.__class__(self.data*n)
    __rmul__ = __mul__
    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self): return self.__class__(self.data.capitalize())
    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))
    def count(self, sub, start=0, end=sys.maxint):
        return self.data.count(sub, start, end)
    def decode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())
    def encode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())
    def endswith(self, suffix, start=0, end=sys.maxint):
        return self.data.endswith(suffix, start, end)
    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))
    def find(self, sub, start=0, end=sys.maxint):
        return self.data.find(sub, start, end)
    def index(self, sub, start=0, end=sys.maxint):
        return self.data.index(sub, start, end)
    def isalpha(self): return self.data.isalpha()
    def isalnum(self): return self.data.isalnum()
    def isdecimal(self): return self.data.isdecimal()
    def isdigit(self): return self.data.isdigit()
    def islower(self): return self.data.islower()
    def isnumeric(self): return self.data.isnumeric()
    def isspace(self): return self.data.isspace()
    def istitle(self): return self.data.istitle()
    def isupper(self): return self.data.isupper()
    def join(self, seq): return self.data.join(seq)
    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))
    def lower(self): return self.__class__(self.data.lower())
    def lstrip(self, chars=None): return self.__class__(self.data.lstrip(chars))
    def partition(self, sep):
        return self.data.partition(sep)
    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))
    def rfind(self, sub, start=0, end=sys.maxint):
        return self.data.rfind(sub, start, end)
    def rindex(self, sub, start=0, end=sys.maxint):
        return self.data.rindex(sub, start, end)
    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))
    def rpartition(self, sep):
        return self.data.rpartition(sep)
    def rstrip(self, chars=None): return self.__class__(self.data.rstrip(chars))
    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)
    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)
    def splitlines(self, keepends=0): return self.data.splitlines(keepends)
    def startswith(self, prefix, start=0, end=sys.maxint):
        return self.data.startswith(prefix, start, end)
    def strip(self, chars=None): return self.__class__(self.data.strip(chars))
    def swapcase(self): return self.__class__(self.data.swapcase())
    def title(self): return self.__class__(self.data.title())
    def translate(self, *args):
        return self.__class__(self.data.translate(*args))
    def upper(self): return self.__class__(self.data.upper())
    def zfill(self, width): return self.__class__(self.data.zfill(width))

class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""
    def __init__(self, string=""):
        self.data = string
    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")
    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + sub + self.data[index+1:]
    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index+1:]
    def __setslice__(self, start, end, sub):
        start = max(start, 0); end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start]+sub.data+self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start]+sub+self.data[end:]
        else:
            self.data =  self.data[:start]+str(sub)+self.data[end:]
    def __delslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]
    def immutable(self):
        return UserString(self.data)
    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, basestring):
            self.data += other
        else:
            self.data += str(other)
        return self
    def __imul__(self, n):
        self.data *= n
        return self

class String(MutableString, Union):

    _fields_ = [('raw', POINTER(c_char)),
                ('data', c_char_p)]

    def __init__(self, obj=""):
        if isinstance(obj, (str, unicode, UserString)):
            self.data = str(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(POINTER(c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj)

        # Convert from c_char_p
        elif isinstance(obj, c_char_p):
            return obj

        # Convert from POINTER(c_char)
        elif isinstance(obj, POINTER(c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(cast(obj, POINTER(c_char)))

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)
    from_param = classmethod(from_param)

def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)

# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to c_void_p.
def UNCHECKED(type):
    if (hasattr(type, "_type_") and isinstance(type._type_, str)
        and type._type_ != "P"):
        return type
    else:
        return c_void_p

# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self,func,restype,argtypes):
        self.func=func
        self.func.restype=restype
        self.argtypes=argtypes
    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func
    def __call__(self,*args):
        fixed_args=[]
        i=0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i+=1
        return self.func(*fixed_args+list(args[i:]))

# End preamble

_libs = {}
_libdirs = []

# Begin loader

# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import os.path, re, sys, glob
import ctypes
import ctypes.util

def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []

class LibraryLoader(object):
    def __init__(self):
        self.other_dirs=[]

    def load_library(self,libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self,path):
        """Given a path to a library, load it."""
        try:
            # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
            # of the default RTLD_LOCAL.  Without this, you end up with
            # libraries not being loadable, resulting in "Symbol not found"
            # errors
            if sys.platform == 'darwin':
                return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
            else:
                return ctypes.cdll.LoadLibrary(path)
        except OSError,e:
            raise ImportError(e)

    def getpaths(self,libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname

        else:
            for path in self.getplatformpaths(libname):
                yield path

            path = ctypes.util.find_library(libname)
            if path: yield path

    def getplatformpaths(self, libname):
        return []

# Darwin (Mac OS X)

class DarwinLibraryLoader(LibraryLoader):
    name_formats = ["lib%s.dylib", "lib%s.so", "lib%s.bundle", "%s.dylib",
                "%s.so", "%s.bundle", "%s"]

    def getplatformpaths(self,libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir,name)

    def getdirs(self,libname):
        '''Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        '''

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [os.path.expanduser('~/lib'),
                                          '/usr/local/lib', '/usr/lib']

        dirs = []

        if '/' in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))

        dirs.extend(self.other_dirs)
        dirs.append(".")

        if hasattr(sys, 'frozen') and sys.frozen == 'macosx_app':
            dirs.append(os.path.join(
                os.environ['RESOURCEPATH'],
                '..',
                'Frameworks'))

        dirs.extend(dyld_fallback_library_path)

        return dirs

# Posix

class PosixLibraryLoader(LibraryLoader):
    _ld_so_cache = None

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = []
        for name in ("LD_LIBRARY_PATH",
                     "SHLIB_PATH", # HPUX
                     "LIBPATH", # OS/2, AIX
                     "LIBRARY_PATH", # BE/OS
                    ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")

        try: directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError: pass

        directories.extend(['/lib', '/usr/lib', '/lib64', '/usr/lib64'])

        cache = {}
        lib_re = re.compile(r'lib(.*)\.s[ol]')
        ext_re = re.compile(r'\.s[ol]$')
        for dir in directories:
            try:
                for path in glob.glob("%s/*.s[ol]*" % dir):
                    file = os.path.basename(path)

                    # Index by filename
                    if file not in cache:
                        cache[file] = path

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname)
        if result: yield result

        path = ctypes.util.find_library(libname)
        if path: yield os.path.join("/lib",path)

# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try: return getattr(self.cdll,name)
        except AttributeError:
            try: return getattr(self.windll,name)
            except AttributeError:
                raise

class WindowsLibraryLoader(LibraryLoader):
    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll"]

    def load_library(self, libname):
        try:
            result = LibraryLoader.load_library(self, libname)
        except ImportError:
            result = None
            if os.path.sep not in libname:
                for name in self.name_formats:
                    try:
                        result = getattr(ctypes.cdll, name % libname)
                        if result:
                            break
                    except WindowsError:
                        result = None
            if result is None:
                try:
                    result = getattr(ctypes.cdll, libname)
                except WindowsError:
                    result = None
            if result is None:
                raise ImportError("%s not found." % libname)
        return result

    def load(self, path):
        return _WindowsLibrary(path)

    def getplatformpaths(self, libname):
        if os.path.sep not in libname:
            for name in self.name_formats:
                dll_in_current_dir = os.path.abspath(name % libname)
                if os.path.exists(dll_in_current_dir):
                    yield dll_in_current_dir
                path = ctypes.util.find_library(name % libname)
                if path:
                    yield path

# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin":   DarwinLibraryLoader,
    "cygwin":   WindowsLibraryLoader,
    "win32":    WindowsLibraryLoader
}

loader = loaderclass.get(sys.platform, PosixLibraryLoader)()

def add_library_search_dirs(other_dirs):
    loader.other_dirs = other_dirs

load_library = loader.load_library

del loaderclass

# End loader

add_library_search_dirs([])

# Begin libraries

_libs["libwfdb.so"] = load_library("libwfdb.so")

# 1 libraries
# End libraries

# No modules

WFDB_Sample = c_int # /usr/include/wfdb/wfdb.h: 71

WFDB_Time = c_long # /usr/include/wfdb/wfdb.h: 72

WFDB_Date = c_long # /usr/include/wfdb/wfdb.h: 73

WFDB_Frequency = c_double # /usr/include/wfdb/wfdb.h: 74

WFDB_Gain = c_double # /usr/include/wfdb/wfdb.h: 75

WFDB_Group = c_uint # /usr/include/wfdb/wfdb.h: 76

WFDB_Signal = c_uint # /usr/include/wfdb/wfdb.h: 77

WFDB_Annotator = c_uint # /usr/include/wfdb/wfdb.h: 78

# /usr/include/wfdb/wfdb.h: 150
class struct_WFDB_siginfo(Structure):
    pass

struct_WFDB_siginfo.__slots__ = [
    'fname',
    'desc',
    'units',
    'gain',
    'initval',
    'group',
    'fmt',
    'spf',
    'bsize',
    'adcres',
    'adczero',
    'baseline',
    'nsamp',
    'cksum',
]
struct_WFDB_siginfo._fields_ = [
    ('fname', String),
    ('desc', String),
    ('units', String),
    ('gain', WFDB_Gain),
    ('initval', WFDB_Sample),
    ('group', WFDB_Group),
    ('fmt', c_int),
    ('spf', c_int),
    ('bsize', c_int),
    ('adcres', c_int),
    ('adczero', c_int),
    ('baseline', c_int),
    ('nsamp', c_long),
    ('cksum', c_int),
]

# /usr/include/wfdb/wfdb.h: 167
class struct_WFDB_calinfo(Structure):
    pass

struct_WFDB_calinfo.__slots__ = [
    'low',
    'high',
    'scale',
    'sigtype',
    'units',
    'caltype',
]
struct_WFDB_calinfo._fields_ = [
    ('low', c_double),
    ('high', c_double),
    ('scale', c_double),
    ('sigtype', String),
    ('units', String),
    ('caltype', c_int),
]

# /usr/include/wfdb/wfdb.h: 176
class struct_WFDB_anninfo(Structure):
    pass

struct_WFDB_anninfo.__slots__ = [
    'name',
    'stat',
]
struct_WFDB_anninfo._fields_ = [
    ('name', String),
    ('stat', c_int),
]

# /usr/include/wfdb/wfdb.h: 181
class struct_WFDB_ann(Structure):
    pass

struct_WFDB_ann.__slots__ = [
    'time',
    'anntyp',
    'subtyp',
    'chan',
    'num',
    'aux',
]
struct_WFDB_ann._fields_ = [
    ('time', WFDB_Time),
    ('anntyp', c_char),
    ('subtyp', c_char),
    ('chan', c_ubyte),
    ('num', c_char),
    ('aux', POINTER(c_ubyte)),
]

# /usr/include/wfdb/wfdb.h: 191
class struct_WFDB_seginfo(Structure):
    pass

struct_WFDB_seginfo.__slots__ = [
    'recname',
    'nsamp',
    'samp0',
]
struct_WFDB_seginfo._fields_ = [
    ('recname', c_char * (50 + 1)),
    ('nsamp', WFDB_Time),
    ('samp0', WFDB_Time),
]

WFDB_Siginfo = struct_WFDB_siginfo # /usr/include/wfdb/wfdb.h: 198

WFDB_Calinfo = struct_WFDB_calinfo # /usr/include/wfdb/wfdb.h: 199

WFDB_Anninfo = struct_WFDB_anninfo # /usr/include/wfdb/wfdb.h: 200

WFDB_Annotation = struct_WFDB_ann # /usr/include/wfdb/wfdb.h: 201

WFDB_Seginfo = struct_WFDB_seginfo # /usr/include/wfdb/wfdb.h: 202

FSTRING = String # /usr/include/wfdb/wfdb.h: 218

FCONSTSTRING = String # /usr/include/wfdb/wfdb.h: 219

FDATE = WFDB_Date # /usr/include/wfdb/wfdb.h: 220

FDOUBLE = c_double # /usr/include/wfdb/wfdb.h: 221

FFREQUENCY = WFDB_Frequency # /usr/include/wfdb/wfdb.h: 222

FINT = c_int # /usr/include/wfdb/wfdb.h: 223

FLONGINT = c_long # /usr/include/wfdb/wfdb.h: 224

FSAMPLE = WFDB_Sample # /usr/include/wfdb/wfdb.h: 225

FSITIME = WFDB_Time # /usr/include/wfdb/wfdb.h: 226

FVOID = None # /usr/include/wfdb/wfdb.h: 227

# /usr/include/wfdb/wfdb.h: 265
if hasattr(_libs['libwfdb.so'], 'annopen'):
    annopen = _libs['libwfdb.so'].annopen
    annopen.argtypes = [String, POINTER(WFDB_Anninfo), c_uint]
    annopen.restype = FINT

# /usr/include/wfdb/wfdb.h: 267
if hasattr(_libs['libwfdb.so'], 'isigopen'):
    isigopen = _libs['libwfdb.so'].isigopen
    isigopen.argtypes = [String, POINTER(WFDB_Siginfo), c_int]
    isigopen.restype = FINT

# /usr/include/wfdb/wfdb.h: 268
if hasattr(_libs['libwfdb.so'], 'osigopen'):
    osigopen = _libs['libwfdb.so'].osigopen
    osigopen.argtypes = [String, POINTER(WFDB_Siginfo), c_uint]
    osigopen.restype = FINT

# /usr/include/wfdb/wfdb.h: 270
if hasattr(_libs['libwfdb.so'], 'osigfopen'):
    osigfopen = _libs['libwfdb.so'].osigfopen
    osigfopen.argtypes = [POINTER(WFDB_Siginfo), c_uint]
    osigfopen.restype = FINT

# /usr/include/wfdb/wfdb.h: 271
if hasattr(_libs['libwfdb.so'], 'wfdbinit'):
    wfdbinit = _libs['libwfdb.so'].wfdbinit
    wfdbinit.argtypes = [String, POINTER(WFDB_Anninfo), c_uint, POINTER(WFDB_Siginfo), c_uint]
    wfdbinit.restype = FINT

# /usr/include/wfdb/wfdb.h: 274
if hasattr(_libs['libwfdb.so'], 'findsig'):
    findsig = _libs['libwfdb.so'].findsig
    findsig.argtypes = [String]
    findsig.restype = FINT

# /usr/include/wfdb/wfdb.h: 275
if hasattr(_libs['libwfdb.so'], 'getspf'):
    getspf = _libs['libwfdb.so'].getspf
    getspf.argtypes = []
    getspf.restype = FINT

# /usr/include/wfdb/wfdb.h: 276
if hasattr(_libs['libwfdb.so'], 'setgvmode'):
    setgvmode = _libs['libwfdb.so'].setgvmode
    setgvmode.argtypes = [c_int]
    setgvmode.restype = FVOID

# /usr/include/wfdb/wfdb.h: 277
if hasattr(_libs['libwfdb.so'], 'getgvmode'):
    getgvmode = _libs['libwfdb.so'].getgvmode
    getgvmode.argtypes = []
    getgvmode.restype = FINT

# /usr/include/wfdb/wfdb.h: 278
if hasattr(_libs['libwfdb.so'], 'setifreq'):
    setifreq = _libs['libwfdb.so'].setifreq
    setifreq.argtypes = [WFDB_Frequency]
    setifreq.restype = FINT

# /usr/include/wfdb/wfdb.h: 279
if hasattr(_libs['libwfdb.so'], 'getifreq'):
    getifreq = _libs['libwfdb.so'].getifreq
    getifreq.argtypes = []
    getifreq.restype = FFREQUENCY

# /usr/include/wfdb/wfdb.h: 280
if hasattr(_libs['libwfdb.so'], 'getvec'):
    getvec = _libs['libwfdb.so'].getvec
    getvec.argtypes = [POINTER(WFDB_Sample)]
    getvec.restype = FINT

# /usr/include/wfdb/wfdb.h: 281
if hasattr(_libs['libwfdb.so'], 'getframe'):
    getframe = _libs['libwfdb.so'].getframe
    getframe.argtypes = [POINTER(WFDB_Sample)]
    getframe.restype = FINT

# /usr/include/wfdb/wfdb.h: 282
if hasattr(_libs['libwfdb.so'], 'putvec'):
    putvec = _libs['libwfdb.so'].putvec
    putvec.argtypes = [POINTER(WFDB_Sample)]
    putvec.restype = FINT

# /usr/include/wfdb/wfdb.h: 283
if hasattr(_libs['libwfdb.so'], 'getann'):
    getann = _libs['libwfdb.so'].getann
    getann.argtypes = [WFDB_Annotator, POINTER(WFDB_Annotation)]
    getann.restype = FINT

# /usr/include/wfdb/wfdb.h: 284
if hasattr(_libs['libwfdb.so'], 'ungetann'):
    ungetann = _libs['libwfdb.so'].ungetann
    ungetann.argtypes = [WFDB_Annotator, POINTER(WFDB_Annotation)]
    ungetann.restype = FINT

# /usr/include/wfdb/wfdb.h: 285
if hasattr(_libs['libwfdb.so'], 'putann'):
    putann = _libs['libwfdb.so'].putann
    putann.argtypes = [WFDB_Annotator, POINTER(WFDB_Annotation)]
    putann.restype = FINT

# /usr/include/wfdb/wfdb.h: 286
if hasattr(_libs['libwfdb.so'], 'isigsettime'):
    isigsettime = _libs['libwfdb.so'].isigsettime
    isigsettime.argtypes = [WFDB_Time]
    isigsettime.restype = FINT

# /usr/include/wfdb/wfdb.h: 287
if hasattr(_libs['libwfdb.so'], 'isgsettime'):
    isgsettime = _libs['libwfdb.so'].isgsettime
    isgsettime.argtypes = [WFDB_Group, WFDB_Time]
    isgsettime.restype = FINT

# /usr/include/wfdb/wfdb.h: 288
if hasattr(_libs['libwfdb.so'], 'tnextvec'):
    tnextvec = _libs['libwfdb.so'].tnextvec
    tnextvec.argtypes = [WFDB_Signal, WFDB_Time]
    tnextvec.restype = FSITIME

# /usr/include/wfdb/wfdb.h: 289
if hasattr(_libs['libwfdb.so'], 'iannsettime'):
    iannsettime = _libs['libwfdb.so'].iannsettime
    iannsettime.argtypes = [WFDB_Time]
    iannsettime.restype = FINT

# /usr/include/wfdb/wfdb.h: 290
if hasattr(_libs['libwfdb.so'], 'ecgstr'):
    ecgstr = _libs['libwfdb.so'].ecgstr
    ecgstr.argtypes = [c_int]
    ecgstr.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 291
if hasattr(_libs['libwfdb.so'], 'strecg'):
    strecg = _libs['libwfdb.so'].strecg
    strecg.argtypes = [String]
    strecg.restype = FINT

# /usr/include/wfdb/wfdb.h: 292
if hasattr(_libs['libwfdb.so'], 'setecgstr'):
    setecgstr = _libs['libwfdb.so'].setecgstr
    setecgstr.argtypes = [c_int, String]
    setecgstr.restype = FINT

# /usr/include/wfdb/wfdb.h: 293
if hasattr(_libs['libwfdb.so'], 'annstr'):
    annstr = _libs['libwfdb.so'].annstr
    annstr.argtypes = [c_int]
    annstr.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 294
if hasattr(_libs['libwfdb.so'], 'strann'):
    strann = _libs['libwfdb.so'].strann
    strann.argtypes = [String]
    strann.restype = FINT

# /usr/include/wfdb/wfdb.h: 295
if hasattr(_libs['libwfdb.so'], 'setannstr'):
    setannstr = _libs['libwfdb.so'].setannstr
    setannstr.argtypes = [c_int, String]
    setannstr.restype = FINT

# /usr/include/wfdb/wfdb.h: 296
if hasattr(_libs['libwfdb.so'], 'anndesc'):
    anndesc = _libs['libwfdb.so'].anndesc
    anndesc.argtypes = [c_int]
    anndesc.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 297
if hasattr(_libs['libwfdb.so'], 'setanndesc'):
    setanndesc = _libs['libwfdb.so'].setanndesc
    setanndesc.argtypes = [c_int, String]
    setanndesc.restype = FINT

# /usr/include/wfdb/wfdb.h: 298
if hasattr(_libs['libwfdb.so'], 'setafreq'):
    setafreq = _libs['libwfdb.so'].setafreq
    setafreq.argtypes = [WFDB_Frequency]
    setafreq.restype = FVOID

# /usr/include/wfdb/wfdb.h: 299
if hasattr(_libs['libwfdb.so'], 'getafreq'):
    getafreq = _libs['libwfdb.so'].getafreq
    getafreq.argtypes = []
    getafreq.restype = FFREQUENCY

# /usr/include/wfdb/wfdb.h: 300
if hasattr(_libs['libwfdb.so'], 'iannclose'):
    iannclose = _libs['libwfdb.so'].iannclose
    iannclose.argtypes = [WFDB_Annotator]
    iannclose.restype = FVOID

# /usr/include/wfdb/wfdb.h: 301
if hasattr(_libs['libwfdb.so'], 'oannclose'):
    oannclose = _libs['libwfdb.so'].oannclose
    oannclose.argtypes = [WFDB_Annotator]
    oannclose.restype = FVOID

# /usr/include/wfdb/wfdb.h: 302
if hasattr(_libs['libwfdb.so'], 'wfdb_isann'):
    wfdb_isann = _libs['libwfdb.so'].wfdb_isann
    wfdb_isann.argtypes = [c_int]
    wfdb_isann.restype = FINT

# /usr/include/wfdb/wfdb.h: 303
if hasattr(_libs['libwfdb.so'], 'wfdb_isqrs'):
    wfdb_isqrs = _libs['libwfdb.so'].wfdb_isqrs
    wfdb_isqrs.argtypes = [c_int]
    wfdb_isqrs.restype = FINT

# /usr/include/wfdb/wfdb.h: 304
if hasattr(_libs['libwfdb.so'], 'wfdb_setisqrs'):
    wfdb_setisqrs = _libs['libwfdb.so'].wfdb_setisqrs
    wfdb_setisqrs.argtypes = [c_int, c_int]
    wfdb_setisqrs.restype = FINT

# /usr/include/wfdb/wfdb.h: 305
if hasattr(_libs['libwfdb.so'], 'wfdb_map1'):
    wfdb_map1 = _libs['libwfdb.so'].wfdb_map1
    wfdb_map1.argtypes = [c_int]
    wfdb_map1.restype = FINT

# /usr/include/wfdb/wfdb.h: 306
if hasattr(_libs['libwfdb.so'], 'wfdb_setmap1'):
    wfdb_setmap1 = _libs['libwfdb.so'].wfdb_setmap1
    wfdb_setmap1.argtypes = [c_int, c_int]
    wfdb_setmap1.restype = FINT

# /usr/include/wfdb/wfdb.h: 307
if hasattr(_libs['libwfdb.so'], 'wfdb_map2'):
    wfdb_map2 = _libs['libwfdb.so'].wfdb_map2
    wfdb_map2.argtypes = [c_int]
    wfdb_map2.restype = FINT

# /usr/include/wfdb/wfdb.h: 308
if hasattr(_libs['libwfdb.so'], 'wfdb_setmap2'):
    wfdb_setmap2 = _libs['libwfdb.so'].wfdb_setmap2
    wfdb_setmap2.argtypes = [c_int, c_int]
    wfdb_setmap2.restype = FINT

# /usr/include/wfdb/wfdb.h: 309
if hasattr(_libs['libwfdb.so'], 'wfdb_ammap'):
    wfdb_ammap = _libs['libwfdb.so'].wfdb_ammap
    wfdb_ammap.argtypes = [c_int]
    wfdb_ammap.restype = FINT

# /usr/include/wfdb/wfdb.h: 310
if hasattr(_libs['libwfdb.so'], 'wfdb_mamap'):
    wfdb_mamap = _libs['libwfdb.so'].wfdb_mamap
    wfdb_mamap.argtypes = [c_int, c_int]
    wfdb_mamap.restype = FINT

# /usr/include/wfdb/wfdb.h: 311
if hasattr(_libs['libwfdb.so'], 'wfdb_annpos'):
    wfdb_annpos = _libs['libwfdb.so'].wfdb_annpos
    wfdb_annpos.argtypes = [c_int]
    wfdb_annpos.restype = FINT

# /usr/include/wfdb/wfdb.h: 312
if hasattr(_libs['libwfdb.so'], 'wfdb_setannpos'):
    wfdb_setannpos = _libs['libwfdb.so'].wfdb_setannpos
    wfdb_setannpos.argtypes = [c_int, c_int]
    wfdb_setannpos.restype = FINT

# /usr/include/wfdb/wfdb.h: 313
if hasattr(_libs['libwfdb.so'], 'timstr'):
    timstr = _libs['libwfdb.so'].timstr
    timstr.argtypes = [WFDB_Time]
    timstr.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 314
if hasattr(_libs['libwfdb.so'], 'mstimstr'):
    mstimstr = _libs['libwfdb.so'].mstimstr
    mstimstr.argtypes = [WFDB_Time]
    mstimstr.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 315
if hasattr(_libs['libwfdb.so'], 'strtim'):
    strtim = _libs['libwfdb.so'].strtim
    strtim.argtypes = [String]
    strtim.restype = FSITIME

# /usr/include/wfdb/wfdb.h: 316
if hasattr(_libs['libwfdb.so'], 'datstr'):
    datstr = _libs['libwfdb.so'].datstr
    datstr.argtypes = [WFDB_Date]
    datstr.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 317
if hasattr(_libs['libwfdb.so'], 'strdat'):
    strdat = _libs['libwfdb.so'].strdat
    strdat.argtypes = [String]
    strdat.restype = FDATE

# /usr/include/wfdb/wfdb.h: 318
if hasattr(_libs['libwfdb.so'], 'adumuv'):
    adumuv = _libs['libwfdb.so'].adumuv
    adumuv.argtypes = [WFDB_Signal, WFDB_Sample]
    adumuv.restype = FINT

# /usr/include/wfdb/wfdb.h: 319
if hasattr(_libs['libwfdb.so'], 'muvadu'):
    muvadu = _libs['libwfdb.so'].muvadu
    muvadu.argtypes = [WFDB_Signal, c_int]
    muvadu.restype = FSAMPLE

# /usr/include/wfdb/wfdb.h: 320
if hasattr(_libs['libwfdb.so'], 'aduphys'):
    aduphys = _libs['libwfdb.so'].aduphys
    aduphys.argtypes = [WFDB_Signal, WFDB_Sample]
    aduphys.restype = FDOUBLE

# /usr/include/wfdb/wfdb.h: 321
if hasattr(_libs['libwfdb.so'], 'physadu'):
    physadu = _libs['libwfdb.so'].physadu
    physadu.argtypes = [WFDB_Signal, c_double]
    physadu.restype = FSAMPLE

# /usr/include/wfdb/wfdb.h: 322
if hasattr(_libs['libwfdb.so'], 'sample'):
    sample = _libs['libwfdb.so'].sample
    sample.argtypes = [WFDB_Signal, WFDB_Time]
    sample.restype = FSAMPLE

# /usr/include/wfdb/wfdb.h: 323
if hasattr(_libs['libwfdb.so'], 'sample_valid'):
    sample_valid = _libs['libwfdb.so'].sample_valid
    sample_valid.argtypes = []
    sample_valid.restype = FINT

# /usr/include/wfdb/wfdb.h: 324
if hasattr(_libs['libwfdb.so'], 'calopen'):
    calopen = _libs['libwfdb.so'].calopen
    calopen.argtypes = [String]
    calopen.restype = FINT

# /usr/include/wfdb/wfdb.h: 325
if hasattr(_libs['libwfdb.so'], 'getcal'):
    getcal = _libs['libwfdb.so'].getcal
    getcal.argtypes = [String, String, POINTER(WFDB_Calinfo)]
    getcal.restype = FINT

# /usr/include/wfdb/wfdb.h: 326
if hasattr(_libs['libwfdb.so'], 'putcal'):
    putcal = _libs['libwfdb.so'].putcal
    putcal.argtypes = [POINTER(WFDB_Calinfo)]
    putcal.restype = FINT

# /usr/include/wfdb/wfdb.h: 327
if hasattr(_libs['libwfdb.so'], 'newcal'):
    newcal = _libs['libwfdb.so'].newcal
    newcal.argtypes = [String]
    newcal.restype = FINT

# /usr/include/wfdb/wfdb.h: 328
if hasattr(_libs['libwfdb.so'], 'flushcal'):
    flushcal = _libs['libwfdb.so'].flushcal
    flushcal.argtypes = []
    flushcal.restype = FVOID

# /usr/include/wfdb/wfdb.h: 329
if hasattr(_libs['libwfdb.so'], 'getinfo'):
    getinfo = _libs['libwfdb.so'].getinfo
    getinfo.argtypes = [String]
    getinfo.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 330
if hasattr(_libs['libwfdb.so'], 'putinfo'):
    putinfo = _libs['libwfdb.so'].putinfo
    putinfo.argtypes = [String]
    putinfo.restype = FINT

# /usr/include/wfdb/wfdb.h: 331
if hasattr(_libs['libwfdb.so'], 'setinfo'):
    setinfo = _libs['libwfdb.so'].setinfo
    setinfo.argtypes = [String]
    setinfo.restype = FINT

# /usr/include/wfdb/wfdb.h: 332
if hasattr(_libs['libwfdb.so'], 'wfdb_freeinfo'):
    wfdb_freeinfo = _libs['libwfdb.so'].wfdb_freeinfo
    wfdb_freeinfo.argtypes = []
    wfdb_freeinfo.restype = FVOID

# /usr/include/wfdb/wfdb.h: 333
if hasattr(_libs['libwfdb.so'], 'newheader'):
    newheader = _libs['libwfdb.so'].newheader
    newheader.argtypes = [String]
    newheader.restype = FINT

# /usr/include/wfdb/wfdb.h: 334
if hasattr(_libs['libwfdb.so'], 'setheader'):
    setheader = _libs['libwfdb.so'].setheader
    setheader.argtypes = [String, POINTER(WFDB_Siginfo), c_uint]
    setheader.restype = FINT

# /usr/include/wfdb/wfdb.h: 335
if hasattr(_libs['libwfdb.so'], 'setmsheader'):
    setmsheader = _libs['libwfdb.so'].setmsheader
    setmsheader.argtypes = [String, POINTER(POINTER(c_char)), c_uint]
    setmsheader.restype = FINT

# /usr/include/wfdb/wfdb.h: 336
if hasattr(_libs['libwfdb.so'], 'getseginfo'):
    getseginfo = _libs['libwfdb.so'].getseginfo
    getseginfo.argtypes = [POINTER(POINTER(WFDB_Seginfo))]
    getseginfo.restype = FINT

# /usr/include/wfdb/wfdb.h: 337
if hasattr(_libs['libwfdb.so'], 'wfdbgetskew'):
    wfdbgetskew = _libs['libwfdb.so'].wfdbgetskew
    wfdbgetskew.argtypes = [WFDB_Signal]
    wfdbgetskew.restype = FINT

# /usr/include/wfdb/wfdb.h: 338
if hasattr(_libs['libwfdb.so'], 'wfdbsetiskew'):
    wfdbsetiskew = _libs['libwfdb.so'].wfdbsetiskew
    wfdbsetiskew.argtypes = [WFDB_Signal, c_int]
    wfdbsetiskew.restype = FVOID

# /usr/include/wfdb/wfdb.h: 339
if hasattr(_libs['libwfdb.so'], 'wfdbsetskew'):
    wfdbsetskew = _libs['libwfdb.so'].wfdbsetskew
    wfdbsetskew.argtypes = [WFDB_Signal, c_int]
    wfdbsetskew.restype = FVOID

# /usr/include/wfdb/wfdb.h: 340
if hasattr(_libs['libwfdb.so'], 'wfdbgetstart'):
    wfdbgetstart = _libs['libwfdb.so'].wfdbgetstart
    wfdbgetstart.argtypes = [WFDB_Signal]
    wfdbgetstart.restype = FLONGINT

# /usr/include/wfdb/wfdb.h: 341
if hasattr(_libs['libwfdb.so'], 'wfdbsetstart'):
    wfdbsetstart = _libs['libwfdb.so'].wfdbsetstart
    wfdbsetstart.argtypes = [WFDB_Signal, c_long]
    wfdbsetstart.restype = FVOID

# /usr/include/wfdb/wfdb.h: 342
if hasattr(_libs['libwfdb.so'], 'wfdbputprolog'):
    wfdbputprolog = _libs['libwfdb.so'].wfdbputprolog
    wfdbputprolog.argtypes = [String, c_long, WFDB_Signal]
    wfdbputprolog.restype = FINT

# /usr/include/wfdb/wfdb.h: 343
if hasattr(_libs['libwfdb.so'], 'wfdbquit'):
    wfdbquit = _libs['libwfdb.so'].wfdbquit
    wfdbquit.argtypes = []
    wfdbquit.restype = FVOID

# /usr/include/wfdb/wfdb.h: 344
if hasattr(_libs['libwfdb.so'], 'sampfreq'):
    sampfreq = _libs['libwfdb.so'].sampfreq
    sampfreq.argtypes = [String]
    sampfreq.restype = FFREQUENCY

# /usr/include/wfdb/wfdb.h: 345
if hasattr(_libs['libwfdb.so'], 'setsampfreq'):
    setsampfreq = _libs['libwfdb.so'].setsampfreq
    setsampfreq.argtypes = [WFDB_Frequency]
    setsampfreq.restype = FINT

# /usr/include/wfdb/wfdb.h: 346
if hasattr(_libs['libwfdb.so'], 'getcfreq'):
    getcfreq = _libs['libwfdb.so'].getcfreq
    getcfreq.argtypes = []
    getcfreq.restype = FFREQUENCY

# /usr/include/wfdb/wfdb.h: 347
if hasattr(_libs['libwfdb.so'], 'setcfreq'):
    setcfreq = _libs['libwfdb.so'].setcfreq
    setcfreq.argtypes = [WFDB_Frequency]
    setcfreq.restype = FVOID

# /usr/include/wfdb/wfdb.h: 348
if hasattr(_libs['libwfdb.so'], 'getbasecount'):
    getbasecount = _libs['libwfdb.so'].getbasecount
    getbasecount.argtypes = []
    getbasecount.restype = FDOUBLE

# /usr/include/wfdb/wfdb.h: 349
if hasattr(_libs['libwfdb.so'], 'setbasecount'):
    setbasecount = _libs['libwfdb.so'].setbasecount
    setbasecount.argtypes = [c_double]
    setbasecount.restype = FVOID

# /usr/include/wfdb/wfdb.h: 350
if hasattr(_libs['libwfdb.so'], 'setbasetime'):
    setbasetime = _libs['libwfdb.so'].setbasetime
    setbasetime.argtypes = [String]
    setbasetime.restype = FINT

# /usr/include/wfdb/wfdb.h: 351
if hasattr(_libs['libwfdb.so'], 'wfdbquiet'):
    wfdbquiet = _libs['libwfdb.so'].wfdbquiet
    wfdbquiet.argtypes = []
    wfdbquiet.restype = FVOID

# /usr/include/wfdb/wfdb.h: 352
if hasattr(_libs['libwfdb.so'], 'wfdbverbose'):
    wfdbverbose = _libs['libwfdb.so'].wfdbverbose
    wfdbverbose.argtypes = []
    wfdbverbose.restype = FVOID

# /usr/include/wfdb/wfdb.h: 353
if hasattr(_libs['libwfdb.so'], 'wfdberror'):
    wfdberror = _libs['libwfdb.so'].wfdberror
    wfdberror.argtypes = []
    wfdberror.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 354
if hasattr(_libs['libwfdb.so'], 'setwfdb'):
    setwfdb = _libs['libwfdb.so'].setwfdb
    setwfdb.argtypes = [String]
    setwfdb.restype = FVOID

# /usr/include/wfdb/wfdb.h: 355
if hasattr(_libs['libwfdb.so'], 'getwfdb'):
    getwfdb = _libs['libwfdb.so'].getwfdb
    getwfdb.argtypes = []
    getwfdb.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 356
if hasattr(_libs['libwfdb.so'], 'resetwfdb'):
    resetwfdb = _libs['libwfdb.so'].resetwfdb
    resetwfdb.argtypes = []
    resetwfdb.restype = FVOID

# /usr/include/wfdb/wfdb.h: 357
if hasattr(_libs['libwfdb.so'], 'setibsize'):
    setibsize = _libs['libwfdb.so'].setibsize
    setibsize.argtypes = [c_int]
    setibsize.restype = FINT

# /usr/include/wfdb/wfdb.h: 358
if hasattr(_libs['libwfdb.so'], 'setobsize'):
    setobsize = _libs['libwfdb.so'].setobsize
    setobsize.argtypes = [c_int]
    setobsize.restype = FINT

# /usr/include/wfdb/wfdb.h: 359
if hasattr(_libs['libwfdb.so'], 'wfdbfile'):
    wfdbfile = _libs['libwfdb.so'].wfdbfile
    wfdbfile.argtypes = [String, String]
    wfdbfile.restype = FSTRING

# /usr/include/wfdb/wfdb.h: 360
if hasattr(_libs['libwfdb.so'], 'wfdbflush'):
    wfdbflush = _libs['libwfdb.so'].wfdbflush
    wfdbflush.argtypes = []
    wfdbflush.restype = FVOID

# /usr/include/wfdb/wfdb.h: 361
if hasattr(_libs['libwfdb.so'], 'wfdbmemerr'):
    wfdbmemerr = _libs['libwfdb.so'].wfdbmemerr
    wfdbmemerr.argtypes = [c_int]
    wfdbmemerr.restype = FVOID

# /usr/include/wfdb/wfdb.h: 362
if hasattr(_libs['libwfdb.so'], 'wfdbversion'):
    wfdbversion = _libs['libwfdb.so'].wfdbversion
    wfdbversion.argtypes = []
    wfdbversion.restype = FCONSTSTRING

# /usr/include/wfdb/wfdb.h: 363
if hasattr(_libs['libwfdb.so'], 'wfdbldflags'):
    wfdbldflags = _libs['libwfdb.so'].wfdbldflags
    wfdbldflags.argtypes = []
    wfdbldflags.restype = FCONSTSTRING

# /usr/include/wfdb/wfdb.h: 364
if hasattr(_libs['libwfdb.so'], 'wfdbcflags'):
    wfdbcflags = _libs['libwfdb.so'].wfdbcflags
    wfdbcflags.argtypes = []
    wfdbcflags.restype = FCONSTSTRING

# /usr/include/wfdb/wfdb.h: 365
if hasattr(_libs['libwfdb.so'], 'wfdbdefwfdb'):
    wfdbdefwfdb = _libs['libwfdb.so'].wfdbdefwfdb
    wfdbdefwfdb.argtypes = []
    wfdbdefwfdb.restype = FCONSTSTRING

# /usr/include/wfdb/wfdb.h: 366
if hasattr(_libs['libwfdb.so'], 'wfdbdefwfdbcal'):
    wfdbdefwfdbcal = _libs['libwfdb.so'].wfdbdefwfdbcal
    wfdbdefwfdbcal.argtypes = []
    wfdbdefwfdbcal.restype = FCONSTSTRING

__off_t = c_long # /usr/include/bits/types.h: 131

__off64_t = c_long # /usr/include/bits/types.h: 132

# /usr/include/libio.h: 245
class struct__IO_FILE(Structure):
    pass

FILE = struct__IO_FILE # /usr/include/stdio.h: 48

_IO_lock_t = None # /usr/include/libio.h: 154

# /usr/include/libio.h: 160
class struct__IO_marker(Structure):
    pass

struct__IO_marker.__slots__ = [
    '_next',
    '_sbuf',
    '_pos',
]
struct__IO_marker._fields_ = [
    ('_next', POINTER(struct__IO_marker)),
    ('_sbuf', POINTER(struct__IO_FILE)),
    ('_pos', c_int),
]

struct__IO_FILE.__slots__ = [
    '_flags',
    '_IO_read_ptr',
    '_IO_read_end',
    '_IO_read_base',
    '_IO_write_base',
    '_IO_write_ptr',
    '_IO_write_end',
    '_IO_buf_base',
    '_IO_buf_end',
    '_IO_save_base',
    '_IO_backup_base',
    '_IO_save_end',
    '_markers',
    '_chain',
    '_fileno',
    '_flags2',
    '_old_offset',
    '_cur_column',
    '_vtable_offset',
    '_shortbuf',
    '_lock',
    '_offset',
    '__pad1',
    '__pad2',
    '__pad3',
    '__pad4',
    '__pad5',
    '_mode',
    '_unused2',
]
struct__IO_FILE._fields_ = [
    ('_flags', c_int),
    ('_IO_read_ptr', String),
    ('_IO_read_end', String),
    ('_IO_read_base', String),
    ('_IO_write_base', String),
    ('_IO_write_ptr', String),
    ('_IO_write_end', String),
    ('_IO_buf_base', String),
    ('_IO_buf_end', String),
    ('_IO_save_base', String),
    ('_IO_backup_base', String),
    ('_IO_save_end', String),
    ('_markers', POINTER(struct__IO_marker)),
    ('_chain', POINTER(struct__IO_FILE)),
    ('_fileno', c_int),
    ('_flags2', c_int),
    ('_old_offset', __off_t),
    ('_cur_column', c_ushort),
    ('_vtable_offset', c_char),
    ('_shortbuf', c_char * 1),
    ('_lock', POINTER(_IO_lock_t)),
    ('_offset', __off64_t),
    ('__pad1', POINTER(None)),
    ('__pad2', POINTER(None)),
    ('__pad3', POINTER(None)),
    ('__pad4', POINTER(None)),
    ('__pad5', c_size_t),
    ('_mode', c_int),
    ('_unused2', c_char * (((15 * sizeof(c_int)) - (4 * sizeof(POINTER(None)))) - sizeof(c_size_t))),
]

# /usr/include/wfdb/wfdblib.h: 204
class struct_netfile(Structure):
    pass

struct_netfile.__slots__ = [
    'url',
    'data',
    'mode',
    'base_addr',
    'cont_len',
    'pos',
    'err',
    'fd',
]
struct_netfile._fields_ = [
    ('url', String),
    ('data', String),
    ('mode', c_int),
    ('base_addr', c_long),
    ('cont_len', c_long),
    ('pos', c_long),
    ('err', c_long),
    ('fd', c_int),
]

# /usr/include/wfdb/wfdblib.h: 215
class struct_WFDB_FILE(Structure):
    pass

struct_WFDB_FILE.__slots__ = [
    'fp',
    'netfp',
    'type',
]
struct_WFDB_FILE._fields_ = [
    ('fp', POINTER(FILE)),
    ('netfp', POINTER(struct_netfile)),
    ('type', c_int),
]

netfile = struct_netfile # /usr/include/wfdb/wfdblib.h: 226

WFDB_FILE = struct_WFDB_FILE # /usr/include/wfdb/wfdblib.h: 227

# /usr/include/wfdb/wfdblib.h: 312
if hasattr(_libs['libwfdb.so'], 'wfdb_fclose'):
    wfdb_fclose = _libs['libwfdb.so'].wfdb_fclose
    wfdb_fclose.argtypes = [POINTER(WFDB_FILE)]
    wfdb_fclose.restype = c_int

# /usr/include/wfdb/wfdblib.h: 313
if hasattr(_libs['libwfdb.so'], 'wfdb_open'):
    wfdb_open = _libs['libwfdb.so'].wfdb_open
    wfdb_open.argtypes = [String, String, c_int]
    wfdb_open.restype = POINTER(WFDB_FILE)

# /usr/include/wfdb/wfdblib.h: 314
if hasattr(_libs['libwfdb.so'], 'wfdb_checkname'):
    wfdb_checkname = _libs['libwfdb.so'].wfdb_checkname
    wfdb_checkname.argtypes = [String, String]
    wfdb_checkname.restype = c_int

# /usr/include/wfdb/wfdblib.h: 315
if hasattr(_libs['libwfdb.so'], 'wfdb_striphea'):
    wfdb_striphea = _libs['libwfdb.so'].wfdb_striphea
    wfdb_striphea.argtypes = [String]
    wfdb_striphea.restype = None

# /usr/include/wfdb/wfdblib.h: 316
if hasattr(_libs['libwfdb.so'], 'wfdb_g16'):
    wfdb_g16 = _libs['libwfdb.so'].wfdb_g16
    wfdb_g16.argtypes = [POINTER(WFDB_FILE)]
    wfdb_g16.restype = c_int

# /usr/include/wfdb/wfdblib.h: 317
if hasattr(_libs['libwfdb.so'], 'wfdb_g32'):
    wfdb_g32 = _libs['libwfdb.so'].wfdb_g32
    wfdb_g32.argtypes = [POINTER(WFDB_FILE)]
    wfdb_g32.restype = c_long

# /usr/include/wfdb/wfdblib.h: 318
if hasattr(_libs['libwfdb.so'], 'wfdb_p16'):
    wfdb_p16 = _libs['libwfdb.so'].wfdb_p16
    wfdb_p16.argtypes = [c_uint, POINTER(WFDB_FILE)]
    wfdb_p16.restype = None

# /usr/include/wfdb/wfdblib.h: 319
if hasattr(_libs['libwfdb.so'], 'wfdb_p32'):
    wfdb_p32 = _libs['libwfdb.so'].wfdb_p32
    wfdb_p32.argtypes = [c_long, POINTER(WFDB_FILE)]
    wfdb_p32.restype = None

# /usr/include/wfdb/wfdblib.h: 320
if hasattr(_libs['libwfdb.so'], 'wfdb_parse_path'):
    wfdb_parse_path = _libs['libwfdb.so'].wfdb_parse_path
    wfdb_parse_path.argtypes = [String]
    wfdb_parse_path.restype = c_int

# /usr/include/wfdb/wfdblib.h: 321
if hasattr(_libs['libwfdb.so'], 'wfdb_addtopath'):
    wfdb_addtopath = _libs['libwfdb.so'].wfdb_addtopath
    wfdb_addtopath.argtypes = [String]
    wfdb_addtopath.restype = None

# /usr/include/wfdb/wfdblib.h: 322
if hasattr(_libs['libwfdb.so'], 'wfdb_error'):
    _func = _libs['libwfdb.so'].wfdb_error
    _restype = None
    _argtypes = [String]
    wfdb_error = _variadic_function(_func,_restype,_argtypes)

# /usr/include/wfdb/wfdblib.h: 323
if hasattr(_libs['libwfdb.so'], 'wfdb_fopen'):
    wfdb_fopen = _libs['libwfdb.so'].wfdb_fopen
    wfdb_fopen.argtypes = [String, String]
    wfdb_fopen.restype = POINTER(WFDB_FILE)

# /usr/include/wfdb/wfdblib.h: 324
if hasattr(_libs['libwfdb.so'], 'wfdb_fprintf'):
    _func = _libs['libwfdb.so'].wfdb_fprintf
    _restype = c_int
    _argtypes = [POINTER(WFDB_FILE), String]
    wfdb_fprintf = _variadic_function(_func,_restype,_argtypes)

# /usr/include/wfdb/wfdblib.h: 325
if hasattr(_libs['libwfdb.so'], 'wfdb_setirec'):
    wfdb_setirec = _libs['libwfdb.so'].wfdb_setirec
    wfdb_setirec.argtypes = [String]
    wfdb_setirec.restype = None

# /usr/include/wfdb/wfdblib.h: 326
if hasattr(_libs['libwfdb.so'], 'wfdb_getirec'):
    wfdb_getirec = _libs['libwfdb.so'].wfdb_getirec
    wfdb_getirec.argtypes = []
    if sizeof(c_int) == sizeof(c_void_p):
        wfdb_getirec.restype = ReturnString
    else:
        wfdb_getirec.restype = String
        wfdb_getirec.errcheck = ReturnString

# /usr/include/wfdb/wfdblib.h: 329
if hasattr(_libs['libwfdb.so'], 'wfdb_clearerr'):
    wfdb_clearerr = _libs['libwfdb.so'].wfdb_clearerr
    wfdb_clearerr.argtypes = [POINTER(WFDB_FILE)]
    wfdb_clearerr.restype = None

# /usr/include/wfdb/wfdblib.h: 330
if hasattr(_libs['libwfdb.so'], 'wfdb_feof'):
    wfdb_feof = _libs['libwfdb.so'].wfdb_feof
    wfdb_feof.argtypes = [POINTER(WFDB_FILE)]
    wfdb_feof.restype = c_int

# /usr/include/wfdb/wfdblib.h: 331
if hasattr(_libs['libwfdb.so'], 'wfdb_ferror'):
    wfdb_ferror = _libs['libwfdb.so'].wfdb_ferror
    wfdb_ferror.argtypes = [POINTER(WFDB_FILE)]
    wfdb_ferror.restype = c_int

# /usr/include/wfdb/wfdblib.h: 332
if hasattr(_libs['libwfdb.so'], 'wfdb_fflush'):
    wfdb_fflush = _libs['libwfdb.so'].wfdb_fflush
    wfdb_fflush.argtypes = [POINTER(WFDB_FILE)]
    wfdb_fflush.restype = c_int

# /usr/include/wfdb/wfdblib.h: 333
if hasattr(_libs['libwfdb.so'], 'wfdb_fgets'):
    wfdb_fgets = _libs['libwfdb.so'].wfdb_fgets
    wfdb_fgets.argtypes = [String, c_int, POINTER(WFDB_FILE)]
    if sizeof(c_int) == sizeof(c_void_p):
        wfdb_fgets.restype = ReturnString
    else:
        wfdb_fgets.restype = String
        wfdb_fgets.errcheck = ReturnString

# /usr/include/wfdb/wfdblib.h: 334
if hasattr(_libs['libwfdb.so'], 'wfdb_fread'):
    wfdb_fread = _libs['libwfdb.so'].wfdb_fread
    wfdb_fread.argtypes = [POINTER(None), c_size_t, c_size_t, POINTER(WFDB_FILE)]
    wfdb_fread.restype = c_size_t

# /usr/include/wfdb/wfdblib.h: 335
if hasattr(_libs['libwfdb.so'], 'wfdb_fseek'):
    wfdb_fseek = _libs['libwfdb.so'].wfdb_fseek
    wfdb_fseek.argtypes = [POINTER(WFDB_FILE), c_long, c_int]
    wfdb_fseek.restype = c_int

# /usr/include/wfdb/wfdblib.h: 336
if hasattr(_libs['libwfdb.so'], 'wfdb_ftell'):
    wfdb_ftell = _libs['libwfdb.so'].wfdb_ftell
    wfdb_ftell.argtypes = [POINTER(WFDB_FILE)]
    wfdb_ftell.restype = c_long

# /usr/include/wfdb/wfdblib.h: 337
if hasattr(_libs['libwfdb.so'], 'wfdb_fwrite'):
    wfdb_fwrite = _libs['libwfdb.so'].wfdb_fwrite
    wfdb_fwrite.argtypes = [POINTER(None), c_size_t, c_size_t, POINTER(WFDB_FILE)]
    wfdb_fwrite.restype = c_size_t

# /usr/include/wfdb/wfdblib.h: 338
if hasattr(_libs['libwfdb.so'], 'wfdb_getc'):
    wfdb_getc = _libs['libwfdb.so'].wfdb_getc
    wfdb_getc.argtypes = [POINTER(WFDB_FILE)]
    wfdb_getc.restype = c_int

# /usr/include/wfdb/wfdblib.h: 339
if hasattr(_libs['libwfdb.so'], 'wfdb_putc'):
    wfdb_putc = _libs['libwfdb.so'].wfdb_putc
    wfdb_putc.argtypes = [c_int, POINTER(WFDB_FILE)]
    wfdb_putc.restype = c_int

# /usr/include/wfdb/wfdblib.h: 343
if hasattr(_libs['libwfdb.so'], 'wfdb_sampquit'):
    wfdb_sampquit = _libs['libwfdb.so'].wfdb_sampquit
    wfdb_sampquit.argtypes = []
    wfdb_sampquit.restype = None

# /usr/include/wfdb/wfdblib.h: 344
if hasattr(_libs['libwfdb.so'], 'wfdb_sigclose'):
    wfdb_sigclose = _libs['libwfdb.so'].wfdb_sigclose
    wfdb_sigclose.argtypes = []
    wfdb_sigclose.restype = None

# /usr/include/wfdb/wfdblib.h: 345
if hasattr(_libs['libwfdb.so'], 'wfdb_osflush'):
    wfdb_osflush = _libs['libwfdb.so'].wfdb_osflush
    wfdb_osflush.argtypes = []
    wfdb_osflush.restype = None

# /usr/include/wfdb/wfdblib.h: 346
if hasattr(_libs['libwfdb.so'], 'wfdb_freeinfo'):
    wfdb_freeinfo = _libs['libwfdb.so'].wfdb_freeinfo
    wfdb_freeinfo.argtypes = []
    wfdb_freeinfo.restype = None

# /usr/include/wfdb/wfdblib.h: 347
if hasattr(_libs['libwfdb.so'], 'wfdb_oinfoclose'):
    wfdb_oinfoclose = _libs['libwfdb.so'].wfdb_oinfoclose
    wfdb_oinfoclose.argtypes = []
    wfdb_oinfoclose.restype = None

# /usr/include/wfdb/wfdblib.h: 350
if hasattr(_libs['libwfdb.so'], 'wfdb_anclose'):
    wfdb_anclose = _libs['libwfdb.so'].wfdb_anclose
    wfdb_anclose.argtypes = []
    wfdb_anclose.restype = None

# /usr/include/wfdb/wfdblib.h: 351
if hasattr(_libs['libwfdb.so'], 'wfdb_oaflush'):
    wfdb_oaflush = _libs['libwfdb.so'].wfdb_oaflush
    wfdb_oaflush.argtypes = []
    wfdb_oaflush.restype = None

# /usr/include/wfdb/ecgcodes.h: 33
try:
    NOTQRS = 0
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 34
try:
    NORMAL = 1
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 35
try:
    LBBB = 2
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 36
try:
    RBBB = 3
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 37
try:
    ABERR = 4
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 38
try:
    PVC = 5
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 39
try:
    FUSION = 6
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 40
try:
    NPC = 7
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 41
try:
    APC = 8
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 42
try:
    SVPB = 9
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 43
try:
    VESC = 10
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 44
try:
    NESC = 11
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 45
try:
    PACE = 12
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 46
try:
    UNKNOWN = 13
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 47
try:
    NOISE = 14
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 48
try:
    ARFCT = 16
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 49
try:
    STCH = 18
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 50
try:
    TCH = 19
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 51
try:
    SYSTOLE = 20
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 52
try:
    DIASTOLE = 21
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 53
try:
    NOTE = 22
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 54
try:
    MEASURE = 23
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 55
try:
    PWAVE = 24
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 56
try:
    BBB = 25
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 57
try:
    PACESP = 26
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 58
try:
    TWAVE = 27
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 59
try:
    RHYTHM = 28
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 60
try:
    UWAVE = 29
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 61
try:
    LEARN = 30
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 62
try:
    FLWAV = 31
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 63
try:
    VFON = 32
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 64
try:
    VFOFF = 33
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 65
try:
    AESC = 34
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 66
try:
    SVESC = 35
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 67
try:
    LINK = 36
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 68
try:
    NAPC = 37
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 69
try:
    PFUS = 38
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 70
try:
    WFON = 39
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 71
try:
    PQ = WFON
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 72
try:
    WFOFF = 40
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 73
try:
    JPT = WFOFF
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 74
try:
    RONT = 41
except:
    pass

# /usr/include/wfdb/ecgcodes.h: 78
try:
    ACMAX = 49
except:
    pass

# /usr/include/wfdb/ecgmap.h: 130
try:
    APUNDEF = 0
except:
    pass

# /usr/include/wfdb/ecgmap.h: 131
try:
    APSTD = 1
except:
    pass

# /usr/include/wfdb/ecgmap.h: 132
try:
    APHIGH = 2
except:
    pass

# /usr/include/wfdb/ecgmap.h: 133
try:
    APLOW = 3
except:
    pass

# /usr/include/wfdb/ecgmap.h: 134
try:
    APATT = 4
except:
    pass

# /usr/include/wfdb/ecgmap.h: 135
try:
    APAHIGH = 5
except:
    pass

# /usr/include/wfdb/ecgmap.h: 136
try:
    APALOW = 6
except:
    pass

# /usr/include/wfdb/wfdb.h: 33
try:
    WFDB_MAJOR = 10
except:
    pass

# /usr/include/wfdb/wfdb.h: 34
try:
    WFDB_MINOR = 5
except:
    pass

# /usr/include/wfdb/wfdb.h: 35
try:
    WFDB_RELEASE = 24
except:
    pass

# /usr/include/wfdb/wfdb.h: 36
try:
    WFDB_NETFILES = 1
except:
    pass

# /usr/include/wfdb/wfdb.h: 37
try:
    WFDB_NETFILES_LIBCURL = 1
except:
    pass

# /usr/include/wfdb/wfdb.h: 87
try:
    WFDB_INVALID_SAMPLE = (-32768)
except:
    pass

# /usr/include/wfdb/wfdb.h: 94
try:
    WFDB_MAXANN = 2
except:
    pass

# /usr/include/wfdb/wfdb.h: 95
try:
    WFDB_MAXSIG = 32
except:
    pass

# /usr/include/wfdb/wfdb.h: 96
try:
    WFDB_MAXSPF = 4
except:
    pass

# /usr/include/wfdb/wfdb.h: 97
try:
    WFDB_MAXRNL = 50
except:
    pass

# /usr/include/wfdb/wfdb.h: 98
try:
    WFDB_MAXUSL = 50
except:
    pass

# /usr/include/wfdb/wfdb.h: 99
try:
    WFDB_MAXDSL = 100
except:
    pass

# /usr/include/wfdb/wfdb.h: 102
try:
    WFDB_READ = 0
except:
    pass

# /usr/include/wfdb/wfdb.h: 103
try:
    WFDB_WRITE = 1
except:
    pass

# /usr/include/wfdb/wfdb.h: 104
try:
    WFDB_AHA_READ = 2
except:
    pass

# /usr/include/wfdb/wfdb.h: 105
try:
    WFDB_AHA_WRITE = 3
except:
    pass

# /usr/include/wfdb/wfdb.h: 106
try:
    WFDB_APPEND = 4
except:
    pass

# /usr/include/wfdb/wfdb.h: 125
try:
    WFDB_NFMTS = 11
except:
    pass

# /usr/include/wfdb/wfdb.h: 128
try:
    WFDB_DEFFREQ = 250.0
except:
    pass

# /usr/include/wfdb/wfdb.h: 129
try:
    WFDB_DEFGAIN = 200.0
except:
    pass

# /usr/include/wfdb/wfdb.h: 130
try:
    WFDB_DEFRES = 12
except:
    pass

# /usr/include/wfdb/wfdb.h: 133
try:
    WFDB_LOWRES = 0
except:
    pass

# /usr/include/wfdb/wfdb.h: 134
try:
    WFDB_HIGHRES = 1
except:
    pass

# /usr/include/wfdb/wfdb.h: 136
try:
    WFDB_GVPAD = 2
except:
    pass

# /usr/include/wfdb/wfdb.h: 142
try:
    WFDB_AC_COUPLED = 0
except:
    pass

# /usr/include/wfdb/wfdb.h: 143
try:
    WFDB_DC_COUPLED = 1
except:
    pass

# /usr/include/wfdb/wfdb.h: 144
try:
    WFDB_CAL_SQUARE = 2
except:
    pass

# /usr/include/wfdb/wfdb.h: 145
try:
    WFDB_CAL_SINE = 4
except:
    pass

# /usr/include/wfdb/wfdb.h: 146
try:
    WFDB_CAL_SAWTOOTH = 6
except:
    pass

# /usr/include/wfdb/wfdb.h: 147
try:
    WFDB_CAL_UNDEF = 8
except:
    pass

# /usr/include/wfdb/wfdblib.h: 112
try:
    DEFWFDB = '. /tmp/yaourt-tmp-pcman/aur-wfdb/src/wfdb-10.5.24/build/database http://physionet.org/physiobank/database'
except:
    pass

# /usr/include/wfdb/wfdblib.h: 143
try:
    DEFWFDBCAL = 'wfdbcal'
except:
    pass

# /usr/include/wfdb/wfdblib.h: 156
try:
    DEFWFDBANNSORT = 1
except:
    pass

# /usr/include/wfdb/wfdblib.h: 166
try:
    DEFWFDBGVMODE = WFDB_LOWRES
except:
    pass

# /usr/include/wfdb/wfdblib.h: 197
try:
    TRUE = 1
except:
    pass

# /usr/include/wfdb/wfdblib.h: 200
try:
    FALSE = 0
except:
    pass

# /usr/include/wfdb/wfdblib.h: 222
try:
    WFDB_LOCAL = 0
except:
    pass

# /usr/include/wfdb/wfdblib.h: 223
try:
    WFDB_NET = 1
except:
    pass

# /usr/include/wfdb/wfdblib.h: 252
try:
    CACHEDIR = '/tmp'
except:
    pass

# /usr/include/wfdb/wfdblib.h: 253
try:
    CACHESIZE = 100
except:
    pass

# /usr/include/wfdb/wfdblib.h: 254
try:
    ENTRYSIZE = 20
except:
    pass

# /usr/include/wfdb/wfdblib.h: 256
try:
    NF_PAGE_SIZE = 32768
except:
    pass

# /usr/include/wfdb/wfdblib.h: 259
try:
    NF_NO_ERR = 0
except:
    pass

# /usr/include/wfdb/wfdblib.h: 260
try:
    NF_EOF_ERR = 1
except:
    pass

# /usr/include/wfdb/wfdblib.h: 261
try:
    NF_REAL_ERR = 2
except:
    pass

# /usr/include/wfdb/wfdblib.h: 264
try:
    NF_CHUNK_MODE = 0
except:
    pass

# /usr/include/wfdb/wfdblib.h: 265
try:
    NF_FULL_MODE = 1
except:
    pass

WFDB_siginfo = struct_WFDB_siginfo # /usr/include/wfdb/wfdb.h: 150

WFDB_calinfo = struct_WFDB_calinfo # /usr/include/wfdb/wfdb.h: 167

WFDB_anninfo = struct_WFDB_anninfo # /usr/include/wfdb/wfdb.h: 176

WFDB_ann = struct_WFDB_ann # /usr/include/wfdb/wfdb.h: 181

WFDB_seginfo = struct_WFDB_seginfo # /usr/include/wfdb/wfdb.h: 191

netfile = struct_netfile # /usr/include/wfdb/wfdblib.h: 204

WFDB_FILE = struct_WFDB_FILE # /usr/include/wfdb/wfdblib.h: 215

# No inserted files

