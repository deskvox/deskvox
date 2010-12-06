AC_DEFUN([CHECK_CUDA],[ 

# ------------------------------------------------------------------------------
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   AC_HELP_STRING([--with-cuda=PATH], [prefix where CUDA is installed [default=/usr/local/cuda]]),
   [cuda_dir=$withval],
   [cuda_dir=/usr/local/cuda]
)

AC_ARG_WITH([cuda-libs],
	AC_HELP_STRING([--with-cuda-libs],[location of CUDA libraries]),
	[cuda_libdir=$withval],
	[cuda_libdir=$cuda_dir/lib]
)

AC_ARG_WITH([cuda-include],
	AC_HELP_STRING([--with-cuda-include],[location of CUDA headers]),
	[cuda_incdir=$withval],
	[cuda_incdir=$cuda_dir/include]
)

CHECK_64BITS

CUDA_INCLUDES="-I$cuda_incdir"
CUDA_LIBS="-L$cuda_libdir -lcudart"

if test "$ARCH_64BITS" = "yes"; then
    NVCC="$cuda_dir/bin/nvcc -m64"
else
    NVCC="$cuda_dir/bin/nvcc -m32"
fi

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CUDA_INCLUDES $CPPFLAGS"
LDFLAGS="$CUDA_LIBS $LDFLAGS"
AC_CHECK_HEADERS([cuda.h], [have_cuda_h=yes])
AC_CHECK_LIB(cudart, cudaFree, [have_cuda_lib=yes])

if test "$have_cuda_h" = "yes" -a "$have_cuda_lib" = yes; then
    AC_SUBST(CUDA_INCLUDES)
    AC_SUBST(CUDA_LIBS)
    AC_SUBST(NVCC)
    AC_DEFINE(HAVE_CUDA, 1, [CUDA framework])
    ac_have_cuda=yes
fi

# ------------------------------------------------------------------------------
# Setup nvcc flags
# ------------------------------------------------------------------------------
if test x$DEBUG = xtrue; then
   NVCCFLAGS="-g -G --ptxas-options=-v"
else
   NVCCFLAGS="-O3 -use_fast_math --ptxas-options=-v"
fi
AC_SUBST(NVCCFLAGS)

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
